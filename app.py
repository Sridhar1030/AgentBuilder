import os
import re
import json
import time
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import mlflow
from mlflow import get_current_active_span
import requests

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

TEACHER_MODEL = "llama-3.3-70b-versatile"
STUDENT_MODEL_LABEL = "Llama-3.2-1B-Student (KServe)"

groq_client = None
if GROQ_API_KEY:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY)

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
def call_student(message: str) -> str:
    prompt = f"### Instruction:\n{message}\n\n### Response:\n"
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
def grade_response(question: str, answer: str) -> tuple:
    if not groq_client:
        return None, ""
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
        return score, reason
    except Exception as e:
        return None, f"Grading failed: {e}"


# ── Chat Handler ───────────────────────────────────────────────────────────────
@mlflow.trace(name="chat_interaction")
def chat(message: str, history: list, model_choice: str):
    span = get_current_active_span()
    use_teacher = "70B" in model_choice

    if use_teacher:
        reply = call_teacher(message)
        if span:
            span.set_attributes({
                "model": "teacher-70b",
                "source": "groq",
                "question": message,
                "response": reply[:1000],
                "is_training_candidate": True,
            })
        log_training_pair(message, reply)
        return reply
    else:
        reply = call_student(message)
        score, reason = grade_response(message, reply)
        if span:
            span.set_attributes({
                "model": "student-1b",
                "source": "kserve",
                "question": message,
                "student_reply": reply[:500],
                "teacher_score": score or -1,
                "teacher_reason": reason,
            })
        suffix = ""
        if score is not None:
            suffix = f"\n\n**Grade: {score}/10**"
            if reason:
                suffix += f" — {reason}"
        return reply + suffix


# ── Training-Data Logger (captures 70B "gold" pairs) ──────────────────────────
GOLD_LOG = os.getenv("GOLD_LOG_PATH", "gold_pairs.jsonl")

def log_training_pair(question: str, answer: str):
    pair = {
        "instruction": question,
        "output": answer,
        "text": f"### Instruction:\n{question}\n\n### Response:\n{answer}",
        "timestamp": time.time(),
    }
    with open(GOLD_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(pair) + "\n")
    mlflow.log_dict(pair, f"gold_pairs/{int(time.time())}.json")


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
        reply = chat(message, chat_history, model_choice)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
        return "", chat_history

    msg.submit(respond, [msg, chatbot, model_selector], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme="glass")
