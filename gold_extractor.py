"""
Gold Extractor — pulls 70B teacher Q&A pairs from MLflow traces
and writes them into a consolidated training file.

Usage:
    python gold_extractor.py
    python gold_extractor.py --output s3://mlflow-artifacts/gold/train.jsonl
    python gold_extractor.py --threshold 200
"""

import os
import json
import argparse
from dotenv import load_dotenv
load_dotenv()

import mlflow

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow-sridharproject.apps.sridhartest-pool-7f6n4.aws.rh-ods.com"
)
EXPERIMENT_NAME = "Distillation-Eval-Hub"


def extract_gold_pairs(min_threshold: int = 0) -> list[dict]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"Experiment '{EXPERIMENT_NAME}' not found.")
        return []

    traces = mlflow.search_traces(
        experiment_ids=[experiment.experiment_id],
        max_results=1000,
    )

    gold_pairs = []
    for _, row in traces.iterrows():
        req = row.get("request")
        resp = row.get("response")
        if not req or not resp:
            continue

        if isinstance(req, str):
            req = json.loads(req)

        model_choice = req.get("model_choice", "")
        if "70B" not in model_choice and "Teacher" not in model_choice:
            continue

        question = req.get("message", "")
        answer = resp if isinstance(resp, str) else str(resp)

        if question and answer:
            gold_pairs.append({
                "instruction": question,
                "output": answer,
                "text": f"### Instruction:\n{question}\n\n### Response:\n{answer}",
            })

    print(f"Found {len(gold_pairs)} gold teacher pairs out of {len(traces)} total traces.")

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
        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123"),
        )
        s3.put_object(Bucket=bucket, Key=key, Body=body.encode())
        print(f"Uploaded {len(pairs)} pairs to {output_path}")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"Saved {len(pairs)} pairs to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract gold training pairs from MLflow")
    parser.add_argument("--output", default="gold_train.jsonl")
    parser.add_argument("--threshold", type=int, default=0)
    args = parser.parse_args()

    pairs = extract_gold_pairs(min_threshold=args.threshold)
    if pairs:
        save_pairs(pairs, args.output)
