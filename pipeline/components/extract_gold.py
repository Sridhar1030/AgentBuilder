"""
KFP Component 1 — Extract Gold Data

Pulls 70B Teacher Q&A traces from MLflow, filters for teacher-generated pairs,
and writes them as train.jsonl to MinIO.
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["mlflow==3.10.0", "boto3", "pandas"],
)
def extract_gold_data(
    mlflow_tracking_uri: str,
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    experiment_name: str,
    output_s3_path: str,
    min_threshold: int = 0,
) -> str:
    """Extract teacher traces from MLflow, upload as JSONL to MinIO."""
    import json
    import os
    import mlflow

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise RuntimeError(f"Experiment '{experiment_name}' not found in MLflow")

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
        raise RuntimeError(
            f"Only {len(gold_pairs)} pairs found, below threshold {min_threshold}"
        )

    import boto3

    parts = output_s3_path.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    body = "\n".join(json.dumps(p) for p in gold_pairs)

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )
    s3.put_object(Bucket=bucket, Key=key, Body=body.encode())
    print(f"Uploaded {len(gold_pairs)} pairs to {output_s3_path}")

    return output_s3_path
