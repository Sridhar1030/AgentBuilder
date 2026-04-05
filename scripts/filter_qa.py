#!/usr/bin/env python3
"""
Step 1.5 — Quality Filter for Kubeflow Q&A Data

Reads the grounded Q&A (Step 1.3) and question bank (Step 1.4) outputs,
applies quality filters, deduplicates, merges, and writes the final dataset.

Filters applied:
  1. Length — drop answers < 40 chars or > 2000 chars
  2. Format — drop entries missing instruction/output fields
  3. Dedup  — remove near-duplicate questions (normalised Jaccard > 0.85)
  4. Hollow — drop answers that are generic refusals or empty hedging

Optionally uploads the filtered set to MinIO under synthetic/kubeflow/.

Output: data/kubeflow_filtered.jsonl

Usage:
    python scripts/filter_qa.py [--upload]
"""

import argparse
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

DEFAULT_S3_ENDPOINT = "http://minio.sridharproject.svc.cluster.local:9000"
DEFAULT_BUCKET = "mlflow-artifacts"
UPLOAD_PREFIX = "synthetic/kubeflow/"

MIN_ANSWER_LEN = 40
MAX_ANSWER_LEN = 2000
MIN_INSTRUCTION_LEN = 10
JACCARD_THRESHOLD = 0.85

REFUSAL_PATTERNS = [
    r"^i'?m (?:sorry|not sure|unable)",
    r"^i (?:cannot|can't|don't know)",
    r"^as an ai",
    r"^unfortunately",
    r"^i do not have",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def normalise(text: str) -> set[str]:
    """Lowercase, strip punctuation, split to token set."""
    text = re.sub(r"[^\w\s]", "", text.lower())
    return set(text.split())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL_RE.match(answer.strip()))


def load_jsonl(path: str) -> list[dict]:
    records = []
    if not Path(path).exists():
        print(f"  [WARN] File not found: {path}")
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def deduplicate(records: list[dict], threshold: float) -> list[dict]:
    """Remove near-duplicate questions using Jaccard similarity."""
    seen_sets: list[set[str]] = []
    kept: list[dict] = []

    for rec in records:
        q_set = normalise(rec.get("instruction", ""))
        if not q_set:
            continue
        is_dup = False
        for prev in seen_sets:
            if jaccard(q_set, prev) > threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(rec)
            seen_sets.append(q_set)

    return kept


def upload_to_s3(records, bucket, prefix, endpoint, access_key, secret_key):
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_id = uuid.uuid4().hex[:12]
    key = f"{prefix}date={date_str}/filtered_{run_id}.jsonl"
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
    parser = argparse.ArgumentParser(description="Filter and deduplicate Kubeflow Q&A data")
    parser.add_argument("--qa-file", default="data/kubeflow_qa.jsonl",
                        help="Grounded Q&A from Step 1.3")
    parser.add_argument("--qbank-file", default="data/kubeflow_qbank.jsonl",
                        help="Question bank Q&A from Step 1.4")
    parser.add_argument("--output", default="data/kubeflow_filtered.jsonl")
    parser.add_argument("--jaccard-threshold", type=float, default=JACCARD_THRESHOLD)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--bucket", default=os.getenv("SYNTHETIC_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--s3-endpoint", default=os.getenv("S3_ENDPOINT", DEFAULT_S3_ENDPOINT))
    parser.add_argument("--s3-access-key", default=os.getenv("S3_ACCESS_KEY", "minioadmin"))
    parser.add_argument("--s3-secret-key", default=os.getenv("S3_SECRET_KEY", "minioadmin123"))
    args = parser.parse_args()

    qa_records = load_jsonl(args.qa_file)
    qbank_records = load_jsonl(args.qbank_file)
    raw_total = len(qa_records) + len(qbank_records)
    print(f"Loaded: grounded Q&A = {len(qa_records)}, question bank = {len(qbank_records)} → {raw_total} total")

    all_records = qa_records + qbank_records

    # 1. Format validation
    valid = []
    for rec in all_records:
        instr = rec.get("instruction", "").strip()
        outp = rec.get("output", "").strip()
        if not instr or not outp:
            continue
        if len(instr) < MIN_INSTRUCTION_LEN:
            continue
        if "text" not in rec:
            rec["text"] = f"### Instruction:\n{instr}\n\n### Response:\n{outp}"
        valid.append(rec)
    dropped_format = raw_total - len(valid)
    print(f"After format check: {len(valid)} (dropped {dropped_format})")

    # 2. Length filter
    length_ok = [
        r for r in valid
        if MIN_ANSWER_LEN <= len(r["output"]) <= MAX_ANSWER_LEN
    ]
    dropped_len = len(valid) - len(length_ok)
    print(f"After length filter: {len(length_ok)} (dropped {dropped_len})")

    # 3. Refusal filter
    no_refusals = [r for r in length_ok if not is_refusal(r["output"])]
    dropped_refusal = len(length_ok) - len(no_refusals)
    print(f"After refusal filter: {len(no_refusals)} (dropped {dropped_refusal})")

    # 4. Deduplication
    deduped = deduplicate(no_refusals, args.jaccard_threshold)
    dropped_dup = len(no_refusals) - len(deduped)
    print(f"After dedup (Jaccard > {args.jaccard_threshold}): {len(deduped)} (dropped {dropped_dup})")

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for rec in deduped:
            f.write(json.dumps(rec) + "\n")

    print(f"\nFinal dataset: {len(deduped)} Q&A pairs")
    print(f"Reduction: {raw_total} → {len(deduped)} ({raw_total - len(deduped)} removed)")
    print(f"Output: {args.output}")

    if args.upload:
        print("Uploading filtered dataset to MinIO...")
        upload_to_s3(
            deduped, args.bucket, UPLOAD_PREFIX,
            args.s3_endpoint, args.s3_access_key, args.s3_secret_key,
        )


if __name__ == "__main__":
    main()
