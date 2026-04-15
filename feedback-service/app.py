"""
Feedback API Service

Receives thumbs-up/down signals from the VS Code extension.
On thumbs-down: calls the Ollama teacher to generate a better review,
then writes a DPO preference pair to MinIO for the next pipeline run.
"""

import json
import os
import time
import uuid

import boto3
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Code Review Feedback API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama.sridharproject.svc.cluster.local:11434")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://minio.sridharproject.svc.cluster.local:9000")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "minioadmin123")
S3_BUCKET = os.environ.get("S3_BUCKET", "mlflow-artifacts")
PENDING_PREFIX = "preferences/human-feedback/pending/"
POSITIVE_PREFIX = "feedback/positive/"

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)


class FeedbackRequest(BaseModel):
    diff: str
    slm_response: str
    rating: str  # "up" or "down"


def call_teacher(diff: str) -> str:
    """Ask the teacher LLM to review the same diff."""
    api_url = OLLAMA_URL.rstrip("/") + "/v1/chat/completions"
    system_prompt = (
        "You are a senior code reviewer. Review the git diff provided. "
        "For each issue use: **Severity**: Critical | Warning | Suggestion, "
        "**Issue**: one-line description, **Fix**: how to fix it, with a code suggestion. "
        "Be precise. No false positives."
    )
    for attempt in range(3):
        try:
            resp = requests.post(
                api_url,
                json={
                    "model": TEACHER_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Review these code changes:\n\n```diff\n{diff}\n```"},
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.3,
                },
                timeout=600,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            wait = 2 ** attempt * 5
            print(f"[teacher] Attempt {attempt+1}/3 failed: {e}, retrying in {wait}s")
            time.sleep(wait)
    raise HTTPException(status_code=502, detail="Teacher LLM unreachable after 3 attempts")


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    ts = int(time.time())
    uid = uuid.uuid4().hex[:8]

    if req.rating == "down":
        teacher_response = call_teacher(req.diff)

        pair = {
            "prompt": req.diff,
            "chosen": teacher_response,
            "rejected": req.slm_response,
            "source": "human_feedback",
            "score_gap": None,
            "timestamp": ts,
        }

        key = f"{PENDING_PREFIX}fb-{ts}-{uid}.jsonl"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(pair).encode(),
        )
        print(f"[feedback] Wrote DPO pair to s3://{S3_BUCKET}/{key}")

        return {
            "status": "ok",
            "action": "dpo_pair_created",
            "teacher_response": teacher_response,
            "s3_key": key,
        }

    elif req.rating == "up":
        record = {
            "diff": req.diff,
            "slm_response": req.slm_response,
            "rating": "positive",
            "timestamp": ts,
        }
        key = f"{POSITIVE_PREFIX}fb-{ts}-{uid}.json"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(record).encode(),
        )
        print(f"[feedback] Stored positive signal at s3://{S3_BUCKET}/{key}")

        return {"status": "ok", "action": "positive_recorded"}

    else:
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")


@app.get("/health")
def health():
    return {"status": "ok"}
