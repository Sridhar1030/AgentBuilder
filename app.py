import os
import re
from dotenv import load_dotenv
load_dotenv()
import gradio as gr
import mlflow
from mlflow import get_current_active_span
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# —— Groq grader (same LLM as in datagen) ———
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("API_KEY")
groq_client = None
if GROQ_API_KEY:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY)
GRADER_MODEL = "llama-3.3-70b-versatile"

# --- 2. SETUP MLFLOW ---
# This creates a local SQLite database file to act as your MLflow server
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Distillation-Eval-Hub")



def grade_slm_response(question: str, answer: str) -> tuple:
    """Send Q&A to Groq; returns (score out of 10, short reason) or (None, '') if grading skipped."""
    if not groq_client:
        return None, ""
    try:
        prompt = f"""You are a strict grader. A student model was asked this question and gave this answer.

Question: {question}

Student's answer: {answer}

Rate the answer from 1 to 10 (10 = excellent, accurate, helpful; 1 = wrong or unhelpful). Reply with exactly two lines:
Line 1: SCORE: <number>
Line 2: REASON: <one short sentence>"""
        completion = groq_client.chat.completions.create(
            model=GRADER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150,
        )
        text = (completion.choices[0].message.content or "").strip()
        score = None
        reason = ""
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


print("Loading Base Model... (This takes a few seconds)")
# 1. Load the Base Model (Unquantized so it works on any laptop CPU/GPU)
base_model_id = "unsloth/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto", # Will auto-detect if you have Mac M-chip, PC GPU, or just CPU
    torch_dtype=torch.float32 # Safe for all laptops
)

print("Attaching your custom Student Knowledge...")
# 2. Attach your fine-tuned adapters
model = PeftModel.from_pretrained(base_model, "./student_model")

# 3. Create the text generator
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.7
)

# 4. The Gradio Chat Logic
@mlflow.trace(name="student_chat_interaction")
def chat(message, history):
    # Match the exact prompt format you trained on
    prompt = f"### Instruction:\n{message}\n\n### Response:\n"
    
    # Generate the response
    result = pipe(prompt)[0]['generated_text']
    
    # Clean up the output to only show the assistant's reply
    reply = result.split("### Response:\n")[-1]
    
    # Grade this Q&A with Groq (same teacher as in datagen)
    score, reason = grade_slm_response(message, reply)
    if score is not None:
        span = get_current_active_span()
        if span:
            span.set_attributes({
                "teacher_score": score,
                "teacher_reason": reason,
                "question": message,
                "student_reply": reply[:500],
            })
        reply += f"\n\n📊 **Grade: {score}/10**"
        if reason:
            reply += f" — {reason}"
    
    return reply

# 5. Launch the UI
print("Launching UI...")
demo = gr.ChatInterface(
    fn=chat,
    title="🚀 My 1B Distilled Student",
    description="Trained via Llama-3-70B distillation. Each reply is graded 1–10 by Groq (same teacher). Set GROQ_API_KEY or API_KEY to enable grading.",
    theme="glass"
)

if __name__ == "__main__":
    demo.launch()