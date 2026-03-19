#!/usr/bin/env python3
"""
Standalone synthetic gold-data generator.

Calls Groq to produce (question, answer) pairs in the same schema used by
extract_gold_data, then uploads the result as a date-partitioned JSONL file
to MinIO under synthetic/date=YYYY-MM-DD/run_<id>.jsonl.

Each run creates a new file — multiple runs accumulate for auditable history.
The pipeline's extract_gold_data step merges all objects under the synthetic
prefix automatically.

Usage:
    python scripts/generate_synthetic_gold.py --num-pairs 100

Environment variables (or .env):
    GROQ_API_KEY          — required
    S3_ENDPOINT           — MinIO endpoint   (default: http://localhost:9000)
    S3_ACCESS_KEY         — MinIO access key  (default: minioadmin)
    S3_SECRET_KEY         — MinIO secret key  (default: minioadmin123)
"""

import argparse
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone

import boto3
import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_S3_ENDPOINT = "http://minio.sridharproject.svc.cluster.local:9000"
DEFAULT_BUCKET = "mlflow-artifacts"
DEFAULT_PREFIX = "synthetic/"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
BATCH_SIZE = 10

SEED_TOPICS = [
    "knowledge distillation",
    "LoRA and QLoRA fine-tuning",
    "transformer architecture",
    "attention mechanisms",
    "model quantization",
    "KServe model serving",
    "vLLM inference engine",
    "MLflow experiment tracking",
    "Kubeflow pipelines",
    "OpenShift AI platform",
    "supervised fine-tuning (SFT)",
    "reinforcement learning from human feedback",
    "prompt engineering best practices",
    "retrieval-augmented generation",
    "large language model evaluation",
    "model compression techniques",
    "transfer learning",
    "tokenization strategies",
    "GPU memory optimization",
    "batch inference and throughput",
]

SYSTEM_PROMPT = """\
You are a dataset generator for training a small AI assistant.
Given a topic, produce exactly {batch_size} diverse question-answer pairs.
Each answer should be 2-4 sentences, accurate, and instructional.

Return ONLY a JSON array — no markdown fences, no commentary:
[
  {{"question": "...", "answer": "..."}},
  ...
]
"""


def generate_batch(
    api_key: str,
    model: str,
    topic: str,
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    """Call Groq and return a list of {question, answer} dicts."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(batch_size=batch_size),
            },
            {
                "role": "user",
                "content": f"Topic: {topic}",
            },
        ],
        "max_tokens": 2048,
        "temperature": 0.8,
    }

    resp = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]

    pairs = json.loads(raw.strip())
    if not isinstance(pairs, list):
        raise ValueError(f"Expected JSON array, got {type(pairs)}")
    return pairs


def to_gold_format(question: str, answer: str) -> dict:
    return {
        "instruction": question,
        "output": answer,
        "text": f"### Instruction:\n{question}\n\n### Response:\n{answer}",
    }


def upload_to_s3(
    records: list[dict],
    bucket: str,
    prefix: str,
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
):
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    run_id = uuid.uuid4().hex[:12]
    key = f"{prefix}date={date_str}/run_{run_id}.jsonl"

    body = "\n".join(json.dumps(r) for r in records)

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        verify=False,
    )
    s3.put_object(Bucket=bucket, Key=key, Body=body.encode())
    print(f"Uploaded {len(records)} records to s3://{bucket}/{key}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic gold training data via Groq")
    parser.add_argument("--num-pairs", type=int, default=100, help="Total pairs to generate")
    parser.add_argument("--groq-model", default="llama-3.3-70b-versatile")
    parser.add_argument("--bucket", default=os.getenv("SYNTHETIC_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--prefix", default=os.getenv("SYNTHETIC_PREFIX", DEFAULT_PREFIX))
    parser.add_argument("--s3-endpoint", default=os.getenv("S3_ENDPOINT", DEFAULT_S3_ENDPOINT))
    parser.add_argument("--s3-access-key", default=os.getenv("S3_ACCESS_KEY", "minioadmin"))
    parser.add_argument("--s3-secret-key", default=os.getenv("S3_SECRET_KEY", "minioadmin123"))
    parser.add_argument("--seed-topics-file", default=None, help="Optional file with one topic per line")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between Groq requests")
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit("ERROR: GROQ_API_KEY environment variable is required")

    topics = list(SEED_TOPICS)
    if args.seed_topics_file:
        with open(args.seed_topics_file) as f:
            topics = [line.strip() for line in f if line.strip()]

    gold_records: list[dict] = []
    generated = 0

    print(f"Generating {args.num_pairs} synthetic pairs using {args.groq_model} ...")

    while generated < args.num_pairs:
        topic = random.choice(topics)
        remaining = args.num_pairs - generated
        batch_size = min(BATCH_SIZE, remaining)

        try:
            pairs = generate_batch(api_key, args.groq_model, topic, batch_size)
        except Exception as exc:
            print(f"  Warning: batch failed for topic '{topic}': {exc}")
            time.sleep(args.delay * 2)
            continue

        for p in pairs:
            q = p.get("question", "").strip()
            a = p.get("answer", "").strip()
            if q and a:
                gold_records.append(to_gold_format(q, a))
                generated += 1
                if generated >= args.num_pairs:
                    break

        print(f"  {generated}/{args.num_pairs} pairs collected (topic: {topic})")
        time.sleep(args.delay)

    upload_to_s3(
        gold_records,
        args.bucket,
        args.prefix,
        args.s3_endpoint,
        args.s3_access_key,
        args.s3_secret_key,
    )
    print(f"Done — {len(gold_records)} synthetic gold pairs written.")


if __name__ == "__main__":
    main()
