#!/usr/bin/env python3
"""
Step 1.3 — Context-Grounded Kubeflow Q&A Generator

Reads data/chunks.jsonl and generates 3-5 Q&A pairs per chunk using a Groq
teacher model.  The system prompt requires answers to be *grounded* in the
provided passage — the teacher must not hallucinate beyond the context.

Output: data/kubeflow_qa.jsonl  (same {instruction, output, text} gold schema)
Optionally uploads directly to MinIO under synthetic/kubeflow/date=YYYY-MM-DD/.

Usage:
    python scripts/generate_kubeflow_qa.py [--target-pairs 700] [--upload]

Environment variables:
    GROQ_API_KEY   — required
    S3_ENDPOINT    — MinIO endpoint   (for --upload)
    S3_ACCESS_KEY  — MinIO access key (for --upload)
    S3_SECRET_KEY  — MinIO secret key (for --upload)
"""

import argparse
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

DEFAULT_S3_ENDPOINT = "http://minio.sridharproject.svc.cluster.local:9000"
DEFAULT_BUCKET = "mlflow-artifacts"
UPLOAD_PREFIX = "synthetic/kubeflow/"

SYSTEM_PROMPT = """\
You are a training-data generator for a Kubeflow AI assistant.

You will receive a PASSAGE from Kubeflow documentation. Your job:
1. Read the passage carefully.
2. Produce exactly {n_pairs} question-answer pairs.
3. Every answer MUST be grounded in the passage — do not add facts that
   are not present or clearly implied by the text.
4. Questions should be diverse: mix conceptual ("What is…"), procedural
   ("How do you…"), and comparative ("What is the difference between…").
5. Answers should be 2-5 sentences, precise, and instructional.
6. Do NOT use markdown fences. Do NOT add any text outside the JSON.
7. Escape any double quotes inside strings with a backslash.

Return ONLY a valid JSON array:
[{{"question":"...","answer":"..."}}]
"""


def call_groq(
    api_key: str,
    model: str,
    chunk_text: str,
    n_pairs: int = 3,
) -> list[dict]:
    """Send chunk context to Groq and parse the returned Q&A pairs."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(n_pairs=n_pairs)},
            {"role": "user", "content": f"PASSAGE:\n\n{chunk_text}"},
        ],
        "max_tokens": 2048,
        "temperature": 0.7,
    }

    for attempt in range(6):
        resp = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 429:
            wait = min(2 ** attempt * 5, 120)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break
    else:
        resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]

    raw = raw.strip()
    try:
        pairs = json.loads(raw)
    except json.JSONDecodeError:
        import re
        bracket = raw.find("[")
        if bracket >= 0:
            raw = raw[bracket:]
        last_brace = raw.rfind("}")
        if last_brace >= 0:
            raw = raw[: last_brace + 1] + "]"
        raw = re.sub(r',\s*]', ']', raw)
        pairs = json.loads(raw)

    if not isinstance(pairs, list):
        raise ValueError(f"Expected JSON array, got {type(pairs)}")
    return pairs


def to_gold(question: str, answer: str) -> dict:
    return {
        "instruction": question,
        "output": answer,
        "text": f"### Instruction:\n{question}\n\n### Response:\n{answer}",
    }


def upload_to_s3(records, bucket, prefix, endpoint, access_key, secret_key):
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_id = uuid.uuid4().hex[:12]
    key = f"{prefix}date={date_str}/run_{run_id}.jsonl"
    body = "\n".join(json.dumps(r) for r in records)

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        verify=False,
    )
    s3.put_object(Bucket=bucket, Key=key, Body=body.encode())
    print(f"Uploaded {len(records)} records → s3://{bucket}/{key}")


def main():
    parser = argparse.ArgumentParser(description="Generate grounded Kubeflow Q&A from chunks")
    parser.add_argument("--input", default="data/chunks.jsonl")
    parser.add_argument("--output", default="data/kubeflow_qa.jsonl")
    parser.add_argument("--target-pairs", type=int, default=700,
                        help="Stop after reaching this many pairs")
    parser.add_argument("--pairs-per-chunk", type=int, default=3)
    parser.add_argument("--groq-model", default="llama-3.3-70b-versatile")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="Seconds between Groq requests")
    parser.add_argument("--upload", action="store_true",
                        help="Upload results to MinIO after generation")
    parser.add_argument("--bucket", default=os.getenv("SYNTHETIC_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--s3-endpoint", default=os.getenv("S3_ENDPOINT", DEFAULT_S3_ENDPOINT))
    parser.add_argument("--s3-access-key", default=os.getenv("S3_ACCESS_KEY", "minioadmin"))
    parser.add_argument("--s3-secret-key", default=os.getenv("S3_SECRET_KEY", "minioadmin123"))
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit("ERROR: GROQ_API_KEY environment variable is required")

    if not Path(args.input).exists():
        sys.exit(f"ERROR: Chunks file not found: {args.input}")

    chunks = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks from {args.input}")

    random.shuffle(chunks)

    gold_records: list[dict] = []
    seen_questions: set[str] = set()
    errors = 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if Path(args.output).exists():
        with open(args.output) as existing:
            for line in existing:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    gold_records.append(rec)
                    seen_questions.add(rec["instruction"].lower().strip())
        print(f"Resuming from {len(gold_records)} existing pairs")

    with open(args.output, "a" if gold_records else "w") as fout:
        for i, chunk in enumerate(chunks):
            if len(gold_records) >= args.target_pairs:
                break

            try:
                pairs = call_groq(
                    api_key, args.groq_model,
                    chunk["chunk_text"],
                    n_pairs=args.pairs_per_chunk,
                )
            except Exception as exc:
                errors += 1
                print(f"  [{i+1}] FAIL chunk {chunk['chunk_id']}: {exc}")
                time.sleep(args.delay * 2)
                continue

            for p in pairs:
                q = p.get("question", "").strip()
                a = p.get("answer", "").strip()
                if q and a and len(a) > 30 and q.lower() not in seen_questions:
                    rec = to_gold(q, a)
                    gold_records.append(rec)
                    seen_questions.add(q.lower())
                    fout.write(json.dumps(rec) + "\n")
                    if len(gold_records) >= args.target_pairs:
                        break

            print(f"  [{i+1}/{len(chunks)}] {len(gold_records)}/{args.target_pairs} pairs "
                  f"(chunk: {chunk['chunk_id']}, project: {chunk['project']})")
            time.sleep(args.delay)

    print(f"\nGenerated {len(gold_records)} Q&A pairs ({errors} chunk failures)")
    print(f"Output: {args.output}")

    if args.upload:
        print("Uploading to MinIO...")
        upload_to_s3(
            gold_records, args.bucket, UPLOAD_PREFIX,
            args.s3_endpoint, args.s3_access_key, args.s3_secret_key,
        )

    return len(gold_records)


if __name__ == "__main__":
    main()
