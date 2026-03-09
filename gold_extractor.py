"""
Gold Extractor — pulls 70B teacher Q&A pairs from MLflow traces
and writes them into a consolidated training file.

Usage:
    python gold_extractor.py                       # local default
    python gold_extractor.py --output s3://bucket/gold_pairs.jsonl
    python gold_extractor.py --threshold 200       # only export if >= 200 pairs
"""

import os
import json
import argparse
from dotenv import load_dotenv
load_dotenv()

import mlflow

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "Distillation-Eval-Hub"


def extract_gold_pairs(min_threshold: int = 0) -> list[dict]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"Experiment '{EXPERIMENT_NAME}' not found.")
        return []

    traces = mlflow.search_traces(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.`mlflow.traceName` = 'chat_interaction'",
    )

    gold_pairs = []
    for _, trace_row in traces.iterrows():
        attrs = trace_row.get("request_metadata", {}) or {}
        if not isinstance(attrs, dict):
            continue
        if attrs.get("model") != "teacher-70b":
            continue
        if attrs.get("is_training_candidate") != "True":
            continue

        question = attrs.get("question", "")
        response = attrs.get("response", "")
        if question and response:
            gold_pairs.append({
                "instruction": question,
                "output": response,
                "text": f"### Instruction:\n{question}\n\n### Response:\n{response}",
            })

    print(f"Found {len(gold_pairs)} gold teacher pairs.")

    if min_threshold and len(gold_pairs) < min_threshold:
        print(f"Below threshold ({min_threshold}). Skipping export.")
        return []

    return gold_pairs


def save_pairs(pairs: list[dict], output_path: str):
    if output_path.startswith("s3://"):
        import boto3
        parts = output_path.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        body = "\n".join(json.dumps(p) for p in pairs)
        s3 = boto3.client("s3")
        s3.put_object(Bucket=bucket, Key=key, Body=body.encode())
        print(f"Uploaded {len(pairs)} pairs to {output_path}")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2)
        print(f"Saved {len(pairs)} pairs to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract gold training pairs from MLflow")
    parser.add_argument("--output", default="train_data_v2.json")
    parser.add_argument("--threshold", type=int, default=0)
    args = parser.parse_args()

    pairs = extract_gold_pairs(min_threshold=args.threshold)
    if pairs:
        save_pairs(pairs, args.output)
