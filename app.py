import os
import re
import json
import time
import uuid
import warnings
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

import boto3
import gradio as gr
import mlflow
from mlflow import get_current_active_span
from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback
import requests
import urllib3

# Suppress noisy warnings from self-signed certs (MLFLOW_TRACKING_INSECURE_TLS handles security)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message="AI_JUDGE is deprecated", category=FutureWarning)

# ── Configuration ──────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("API_KEY")
STUDENT_ENDPOINT = os.getenv(
    "STUDENT_ENDPOINT",
    "http://student-llm-predictor.sridharproject.svc.cluster.local:8080/v1"
)
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow-sridharproject.apps.sridhartest-pool-7f6n4.aws.rh-ods.com"
)

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio.sridharproject.svc.cluster.local:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin123")
TEACHER_BUCKET = os.getenv("TEACHER_BUCKET", "mlflow-artifacts")
TEACHER_PREFIX = os.getenv("TEACHER_PREFIX", "teacher-interactions/")

TEACHER_MODEL = "llama-3.3-70b-versatile"
STUDENT_MODEL_LABEL = "Llama-3.2-1B-Student (KServe)"
MODEL_VERSION = os.getenv("MODEL_VERSION", "unknown")

groq_client = None
if GROQ_API_KEY:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY)

_s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    verify=False,
)

# ── MLflow Setup ───────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Distillation-Eval-Hub")


# ── Teacher (70B via Groq) ─────────────────────────────────────────────────────
def call_teacher(message: str) -> str:
    if not groq_client:
        return "ERROR: GROQ_API_KEY not set. Cannot reach 70B Teacher."
    completion = groq_client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful and concise assistant."},
            {"role": "user", "content": message},
        ],
        temperature=0.7,
        max_tokens=512,
    )
    return completion.choices[0].message.content


# ── Student (1B via KServe / vLLM) ────────────────────────────────────────────
@mlflow.trace(name="student_inference", span_type="LLM")
def call_student(message: str) -> str:
    span = get_current_active_span()
    prompt = f"### Instruction:\n{message}\n\n### Response:\n"
    if span:
        span.set_attributes({
            "model": "student-1b",
            "endpoint": STUDENT_ENDPOINT,
            "model_version": MODEL_VERSION,
            "prompt_length": len(prompt),
        })
    try:
        resp = requests.post(
            f"{STUDENT_ENDPOINT}/completions",
            json={
                "model": "/mnt/models",
                "prompt": prompt,
                "max_tokens": 256,
                "temperature": 0.7,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"].strip()
    except requests.exceptions.ConnectionError:
        return "ERROR: Cannot reach Student model at KServe endpoint. Is the InferenceService running?"
    except Exception as e:
        return f"ERROR: Student call failed — {e}"


# ── Grader (70B grades the Student) ───────────────────────────────────────────
@mlflow.trace(name="teacher_assessment", span_type="LLM")
def grade_response(question: str, answer: str) -> tuple:
    if not groq_client:
        return None, ""
    span = get_current_active_span()
    try:
        prompt = f"""You are a strict grader. Rate this answer from 1 to 10.

Question: {question}

Answer: {answer}

Reply with exactly two lines:
Line 1: SCORE: <number>
Line 2: REASON: <one short sentence>"""
        completion = groq_client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150,
        )
        text = (completion.choices[0].message.content or "").strip()
        score, reason = None, ""
        for line in text.split("\n"):
            if line.upper().startswith("SCORE:"):
                m = re.search(r"\d+", line)
                if m:
                    score = min(10, max(1, int(m.group())))
                break
        for line in text.split("\n"):
            if line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[-1].strip()
                break
        if span:
            span.set_attributes({
                "grader_model": TEACHER_MODEL,
                "teacher_score": score or -1,
                "teacher_reason": reason,
                "model_version": MODEL_VERSION,
            })
        return score, reason
    except Exception as e:
        return None, f"Grading failed: {e}"


# ── Chat Handler ───────────────────────────────────────────────────────────────
def chat(message: str, model_choice: str):
    use_teacher = "70B" in model_choice

    if use_teacher:
        reply = call_teacher(message)
        log_training_pair(message, reply)
        return reply
    else:
        return _handle_student(message)


@mlflow.trace(name="student_interaction")
def _handle_student(message: str):
    """Student path — traced in MLflow. Child spans: student_inference + teacher_assessment."""
    span = get_current_active_span()
    if span:
        span.set_attributes({
            "question": message,
            "model_version": MODEL_VERSION,
            "turn_type": "student",
        })
    reply = call_student(message)
    score, reason = grade_response(message, reply)
    if score is not None:
        # Log as a proper MLflow Assessment — populates the Assessments tab in Traces UI
        trace_id = mlflow.get_active_trace_id()
        if trace_id:
            mlflow.log_assessment(
                trace_id=trace_id,
                assessment=Feedback(
                    name="teacher_score",
                    value=score,
                    rationale=reason or None,
                    source=AssessmentSource(
                        source_type=AssessmentSourceType.LLM_JUDGE,
                        source_id=TEACHER_MODEL,
                    ),
                    metadata={"model_version": MODEL_VERSION, "scale": "1-10"},
                ),
            )
        # Also log as a chartable MLflow Run metric
        with mlflow.start_run(
            run_name=f"student-turn-{int(time.time())}",
            tags={"model_version": MODEL_VERSION, "turn_type": "student"},
        ):
            mlflow.log_metric("teacher_score", score)
            mlflow.log_param("question", message[:500])
            mlflow.log_param("teacher_reason", reason)
    suffix = ""
    if score is not None:
        suffix = f"\n\n**Grade: {score}/10**"
        if reason:
            suffix += f" — {reason}"
    return reply + suffix


# ── Teacher Interaction Logger (writes to MinIO only) ──────────────────────────
def log_training_pair(question: str, answer: str):
    """Write one JSON object per teacher Q&A to MinIO under teacher-interactions/."""
    ts = time.time()
    date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    obj_id = uuid.uuid4().hex[:12]
    key = f"{TEACHER_PREFIX}{date_str}/{int(ts)}_{obj_id}.json"

    pair = {
        "instruction": question,
        "output": answer,
        "text": f"### Instruction:\n{question}\n\n### Response:\n{answer}",
        "timestamp": ts,
    }

    _s3_client.put_object(
        Bucket=TEACHER_BUCKET,
        Key=key,
        Body=json.dumps(pair).encode(),
        ContentType="application/json",
    )


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="Distillation Flywheel") as demo:
    gr.Markdown("# 70B → 1B Distillation Flywheel")
    gr.Markdown(
        "Chat with the **70B Teacher** (Groq) to generate training data, "
        "or test the **1B Student** (KServe) and see it graded in real-time."
    )

    model_selector = gr.Radio(
        choices=[
            "Llama-3.3-70B Teacher (Groq)",
            "Llama-3.2-1B Student (KServe)",
        ],
        value="Llama-3.2-1B Student (KServe)",
        label="Select Model",
    )

    chatbot = gr.Chatbot(height=480)
    msg = gr.Textbox(placeholder="Ask anything...", label="Your message")
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history, model_choice):
        reply = chat(message, model_choice)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
        return "", chat_history

    msg.submit(respond, [msg, chatbot, model_selector], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme="glass")
