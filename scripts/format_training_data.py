#!/usr/bin/env python3
"""
format_training_data.py
───────────────────────
Merges GitHub-mined and HF-supplement data into a single ChatML-formatted
training JSONL ready for SFT with Qwen2.5-Coder-1.5B-Instruct.

Also adds negative examples (clean code diffs → "No issues found") and
creates the 50-diff evaluation diff-bank.json.

Requirements:
    pip install transformers boto3

Usage:
    python scripts/format_training_data.py \
        --mined data/kubeflow_reviews.json \
        --hf-supplement data/hf_supplement.json \
        --output data/code_review_train.jsonl \
        --diff-bank data/diff-bank.json

To upload to MinIO after:
    python scripts/format_training_data.py --upload \
        --s3-endpoint http://minio:9000 \
        --s3-access-key minioadmin \
        --s3-secret-key minioadmin123
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a senior code reviewer specializing in Go, Python, and Kubernetes. "
    "Review the given code diff and identify any issues related to bugs, security, "
    "performance, reliability, style, or Kubernetes best practices. "
    "If the code is clean, say so."
)

NO_ISSUE_RESPONSE = (
    "No issues found. The code change is clean and follows best practices."
)


def build_instruction(row: dict) -> str:
    """Build the user instruction from a data record."""
    file_path = row.get("file_path", "unknown")
    language = row.get("language", "unknown")
    diff = row.get("diff_context", "") or row.get("diff_hunk", "")

    if not diff:
        before = row.get("before_code", "")
        after = row.get("after_code", "")
        if before or after:
            diff_lines = []
            for line in before.split("\n"):
                diff_lines.append(f"- {line}")
            for line in after.split("\n"):
                diff_lines.append(f"+ {line}")
            diff = "\n".join(diff_lines)

    return (
        f"Review the following code diff and identify any issues:\n\n"
        f"File: {file_path}\n"
        f"Language: {language}\n\n"
        f"```diff\n{diff}\n```"
    )


def build_response(row: dict) -> str:
    """Build the assistant response from a data record."""
    return row.get("reviewer_comment", "").strip()


def format_chatml(instruction: str, response: str) -> str:
    """Format as ChatML text for SFT training."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )


def load_data(path: str) -> list[dict]:
    if not path or not Path(path).exists():
        print(f"  Skipping {path} (not found)")
        return []
    with open(path) as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} records from {path}")
    return data


def create_negative_examples(data: list[dict], count: int) -> list[dict]:
    """Create negative examples from existing data by keeping
    the diff but replacing the comment with 'No issues found'."""
    candidates = [r for r in data if r.get("diff_context") or r.get("before_code")]
    random.shuffle(candidates)
    negatives = []
    for row in candidates[:count]:
        neg = dict(row)
        neg["reviewer_comment"] = NO_ISSUE_RESPONSE
        neg["is_negative"] = True
        neg["comment_type"] = "negative"
        negatives.append(neg)
    return negatives


def main(args):
    random.seed(42)

    print("Loading data sources...")
    mined = load_data(args.mined)
    hf = load_data(args.hf_supplement)

    # Tag sources
    for r in mined:
        r["_source"] = "github_mined"
    for r in hf:
        r["_source"] = "hf_supplement"

    all_data = mined + hf
    print(f"\nTotal positive examples: {len(all_data)}")

    # Add negative examples
    neg_count = min(int(len(all_data) * 0.2), 3000)
    negatives = create_negative_examples(all_data, neg_count)
    for r in negatives:
        r["_source"] = r.get("_source", "synthetic") + "_negative"
    print(f"Added {len(negatives)} negative examples")

    all_data.extend(negatives)
    random.shuffle(all_data)
    print(f"Total training examples: {len(all_data)}")

    # Format as ChatML JSONL
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for row in all_data:
            instruction = build_instruction(row)
            response = build_response(row)
            if not response:
                continue
            text = format_chatml(instruction, response)
            record = {
                "text": text,
                "source": row.get("_source", "unknown"),
            }
            f.write(json.dumps(record) + "\n")

    line_count = sum(1 for _ in open(out_path))
    print(f"\nTraining JSONL written: {out_path} ({line_count} lines)")

    # Stats
    sources = Counter(r.get("_source", "unknown") for r in all_data)
    print(f"\nBy source: {dict(sources)}")
    langs = Counter(r.get("language", "?") for r in all_data)
    print(f"By language: {dict(langs)}")
    types = Counter(r.get("comment_type", "?") for r in all_data)
    print(f"By comment_type: {dict(types)}")

    # Build diff-bank.json (50 held-out diffs for evaluation)
    if args.diff_bank:
        print(f"\nBuilding diff-bank.json...")
        positive_data = [r for r in mined + hf if not r.get("is_negative")]
        random.shuffle(positive_data)

        # Pick 30 from mined, 20 from HF (or whatever's available)
        mined_eval = [r for r in positive_data if r.get("_source") == "github_mined"][:30]
        hf_eval = [r for r in positive_data if r.get("_source") == "hf_supplement"][:20]
        eval_diffs = mined_eval + hf_eval

        diff_bank = {
            "all_questions": [build_instruction(r) for r in eval_diffs]
        }

        db_path = Path(args.diff_bank)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(db_path, "w") as f:
            json.dump(diff_bank, f, indent=2)
        print(f"Diff bank written: {db_path} ({len(eval_diffs)} diffs)")
        print(f"  From mined: {len(mined_eval)}, from HF: {len(hf_eval)}")

    # Upload to MinIO
    if args.upload:
        import boto3

        s3 = boto3.client(
            "s3",
            endpoint_url=args.s3_endpoint,
            aws_access_key_id=args.s3_access_key,
            aws_secret_access_key=args.s3_secret_key,
        )
        bucket = "mlflow-artifacts"
        prefix = "synthetic/code-review/"

        s3.upload_file(str(out_path), bucket, f"{prefix}code_review_train.jsonl")
        print(f"\nUploaded training data to s3://{bucket}/{prefix}code_review_train.jsonl")

        if args.diff_bank and Path(args.diff_bank).exists():
            s3.upload_file(str(args.diff_bank), bucket, f"{prefix}diff-bank.json")
            print(f"Uploaded diff bank to s3://{bucket}/{prefix}diff-bank.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format training data for Code Review SLM")
    parser.add_argument("--mined", default="data/kubeflow_reviews.json")
    parser.add_argument("--hf-supplement", default="data/hf_supplement.json")
    parser.add_argument("--output", "-o", default="data/code_review_train.jsonl")
    parser.add_argument("--diff-bank", default="data/diff-bank.json")
    parser.add_argument("--upload", action="store_true", help="Upload to MinIO after formatting")
    parser.add_argument("--s3-endpoint", default="http://minio.sridharproject.svc.cluster.local:9000")
    parser.add_argument("--s3-access-key", default="minioadmin")
    parser.add_argument("--s3-secret-key", default="minioadmin123")
    main(parser.parse_args())
