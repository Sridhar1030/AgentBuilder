#!/usr/bin/env python3
"""
generate_preference_bank.py
---------------------------
One-time batch job that pre-generates a large bank of DPO preference pairs
by pulling diverse Go/Python diffs from the HuggingFace dataset and having
the teacher model produce both "chosen" (concise PR comment) and "rejected"
(verbose essay-style) responses.

Runs as a Kubernetes Job. No GPU needed -- just API calls to Ollama teacher.
Saves results to MinIO as JSONL.

Output: s3://mlflow-artifacts/preferences/static-bank/preference-bank.jsonl
Format: {"prompt": "...", "chosen": "...", "rejected": "...", "source": "static_bank"}
"""

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import requests
from datasets import load_dataset

sys.stdout.reconfigure(line_buffering=True)

# ── Config ──────────────────────────────────────────────────────────────
TEACHER_URL = os.environ.get(
    "TEACHER_URL",
    "http://ollama.sridharproject.svc.cluster.local:11434/v1/chat/completions",
)
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://minio.sridharproject.svc.cluster.local:9000")
S3_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
S3_SECRET = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin123")
BUCKET = "mlflow-artifacts"
OUTPUT_KEY = "preferences/static-bank/preference-bank.jsonl"

HF_DATASET = "ronantakizawa/github-codereview"
TARGET_PAIRS = int(os.environ.get("TARGET_PAIRS", "500"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
MIN_DIFF_LEN = 150
MAX_DIFF_LEN = 4000
MIN_QUALITY = 0.7

GOOD_SYSTEM = (
    "You are a senior engineer leaving a review comment on a GitHub Pull Request. "
    "Be concise and actionable -- 2-4 sentences max. "
    "If there's an issue, name it and suggest a fix (with code if helpful). "
    "If the code is clean, say 'LGTM' or 'No issues'. "
    "Do NOT write an essay. Do NOT explain basics the author already knows."
)

BAD_SYSTEM = (
    "You are reviewing code and must write a very detailed, thorough explanation. "
    "Cover every possible angle: style, naming, error handling, performance, "
    "security, testing, documentation, and maintainability. Write at least 3-4 "
    "paragraphs. Explain the theory behind each suggestion. Be educational and "
    "verbose. Reference general programming principles even if they're obvious."
)


def ts():
    return time.strftime("%H:%M:%S")


def teacher_call(messages, max_tokens=512, temperature=0.7):
    for attempt in range(6):
        try:
            resp = requests.post(
                TEACHER_URL,
                json={
                    "model": TEACHER_MODEL,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=600,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = min(2 ** attempt * 10, 120)
                print(f"  [{ts()}] Teacher {resp.status_code}, retry in {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except (requests.ConnectionError, requests.Timeout) as e:
            wait = min(2 ** attempt * 10, 120)
            print(f"  [{ts()}] Teacher error: {e}, retry in {wait}s")
            time.sleep(wait)
    return ""


def build_instruction(row):
    file_path = row.get("file_path", "unknown")
    language = row.get("language", "unknown")
    diff = row.get("diff_context", "") or ""
    if not diff:
        before = row.get("before_code", "")
        after = row.get("after_code", "")
        if before or after:
            lines = [f"- {l}" for l in before.split("\n")]
            lines += [f"+ {l}" for l in after.split("\n")]
            diff = "\n".join(lines)
    return (
        f"Review the following code diff and identify any issues:\n\n"
        f"File: {file_path}\n"
        f"Language: {language}\n\n"
        f"```diff\n{diff}\n```"
    )


def generate_pair(prompt):
    """Generate one DPO pair: concise chosen + verbose rejected."""
    chosen = teacher_call(
        [{"role": "system", "content": GOOD_SYSTEM}, {"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )
    rejected = teacher_call(
        [{"role": "system", "content": BAD_SYSTEM}, {"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.7,
    )
    if chosen and rejected and len(rejected) > len(chosen):
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected, "source": "static_bank"}
    return None


def pull_diffs_from_hf(target_count):
    """Pull diverse Go/Python diffs from HuggingFace dataset."""
    print(f"[{ts()}] Loading HF dataset: {HF_DATASET} ...")
    ds = load_dataset(HF_DATASET, split="train")
    print(f"[{ts()}] Dataset loaded: {len(ds)} total rows")

    seen_hashes = set()
    seen_files = set()
    prompts = []

    go_count = 0
    py_count = 0
    go_target = target_count // 2
    py_target = target_count - go_target

    for row in ds:
        if len(prompts) >= target_count:
            break

        lang = row.get("language", "")
        if lang == "Go" and go_count >= go_target:
            continue
        if lang == "Python" and py_count >= py_target:
            continue
        if lang not in ("Go", "Python"):
            continue
        if row.get("is_negative", False):
            continue

        qs = row.get("quality_score", 0) or 0
        if qs < MIN_QUALITY:
            continue

        diff = row.get("diff_context", "") or ""
        if len(diff) < MIN_DIFF_LEN or len(diff) > MAX_DIFF_LEN:
            continue

        fp = row.get("file_path", "")
        if not fp or fp in seen_files:
            continue

        instruction = build_instruction(row)
        h = hashlib.md5(instruction.encode()).hexdigest()
        if h in seen_hashes:
            continue

        seen_hashes.add(h)
        seen_files.add(fp)
        prompts.append(instruction)

        if lang == "Go":
            go_count += 1
        else:
            py_count += 1

    print(f"[{ts()}] Pulled {len(prompts)} unique diffs (Go: {go_count}, Python: {py_count})")
    return prompts


def main():
    print("=" * 70)
    print("  GENERATE DPO PREFERENCE BANK")
    print("=" * 70)
    print(f"  Teacher:       {TEACHER_MODEL}")
    print(f"  Teacher URL:   {TEACHER_URL}")
    print(f"  Target pairs:  {TARGET_PAIRS}")
    print(f"  Workers:       {NUM_WORKERS}")
    print(f"  Output:        s3://{BUCKET}/{OUTPUT_KEY}")
    print("=" * 70)

    prompts = pull_diffs_from_hf(TARGET_PAIRS)
    if not prompts:
        print("ERROR: No diffs pulled from HF. Exiting.")
        sys.exit(1)

    print(f"\n[{ts()}] Generating {len(prompts)} preference pairs with {NUM_WORKERS} workers...")
    pairs = []
    failed = 0

    s3_client = boto3.client(
        "s3", endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_KEY, aws_secret_access_key=S3_SECRET,
    )

    def save_checkpoint():
        if not pairs:
            return
        body = "\n".join(json.dumps(p) for p in pairs)
        s3_client.put_object(Bucket=BUCKET, Key=OUTPUT_KEY, Body=body.encode())
        print(f"  [{ts()}] Checkpoint saved: {len(pairs)} pairs to s3://{BUCKET}/{OUTPUT_KEY}")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(generate_pair, p): i for i, p in enumerate(prompts)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                if result:
                    pairs.append(result)
                    if len(pairs) % 25 == 0:
                        print(f"  [{ts()}] Progress: {len(pairs)}/{len(prompts)} pairs generated, {failed} failed")
                        save_checkpoint()
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                print(f"  [{ts()}] Pair {idx} error: {e}")

    print(f"\n[{ts()}] Generation complete: {len(pairs)} pairs ({failed} failed)")

    if not pairs:
        print("ERROR: No pairs generated. Exiting.")
        sys.exit(1)

    body = "\n".join(json.dumps(p) for p in pairs)

    print(f"[{ts()}] Uploading to s3://{BUCKET}/{OUTPUT_KEY} ({len(body)} bytes, {len(pairs)} pairs)...")
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_KEY,
        aws_secret_access_key=S3_SECRET,
    )
    s3.put_object(Bucket=BUCKET, Key=OUTPUT_KEY, Body=body.encode())
    print(f"[{ts()}] Uploaded successfully.")

    # Save a summary alongside
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_pairs": len(pairs),
        "failed": failed,
        "target": TARGET_PAIRS,
        "teacher_model": TEACHER_MODEL,
        "hf_dataset": HF_DATASET,
        "go_pairs": sum(1 for p in pairs if "Language: Go" in p["prompt"]),
        "python_pairs": sum(1 for p in pairs if "Language: Python" in p["prompt"]),
    }
    s3.put_object(
        Bucket=BUCKET,
        Key="preferences/static-bank/summary.json",
        Body=json.dumps(summary, indent=2).encode(),
    )
    print(f"[{ts()}] Summary saved to s3://{BUCKET}/preferences/static-bank/summary.json")
    print(json.dumps(summary, indent=2))
    print(f"\n[{ts()}] DONE.")


if __name__ == "__main__":
    main()
