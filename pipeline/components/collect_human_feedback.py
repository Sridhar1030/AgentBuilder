"""
KFP Component -- Collect Human Feedback

Reads DPO preference pairs created by the VS Code extension's feedback API
from MinIO (preferences/human-feedback/pending/), then moves consumed files
to preferences/human-feedback/used/ so they are never reused.

Runs in parallel with extract_preferences since they are independent.
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["boto3"],
)
def collect_human_feedback(
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    bucket: str = "mlflow-artifacts",
) -> str:
    """Read pending human-feedback DPO pairs and move them to used/."""
    import json
    import uuid

    import boto3

    print("=" * 60)
    print("COLLECT HUMAN FEEDBACK STEP")
    print("=" * 60)

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )

    pending_prefix = "preferences/human-feedback/pending/"
    used_prefix = "preferences/human-feedback/used/"

    pairs = []
    consumed_keys = []
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=pending_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".jsonl"):
                continue
            try:
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
                for line in body.strip().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    pair = json.loads(line)
                    if pair.get("prompt") and pair.get("chosen") and pair.get("rejected"):
                        pairs.append(pair)
                consumed_keys.append(key)
                print(f"  Read {key}")
            except Exception as e:
                print(f"  Warning: skipping {key}: {e}")

    print(f"  Found {len(pairs)} human feedback pairs from {len(consumed_keys)} files")

    for key in consumed_keys:
        filename = key.split("/")[-1]
        dest_key = f"{used_prefix}{filename}"
        s3.copy_object(
            Bucket=bucket,
            CopySource={"Bucket": bucket, "Key": key},
            Key=dest_key,
        )
        s3.delete_object(Bucket=bucket, Key=key)
        print(f"  Moved {key} -> {dest_key}")

    if not pairs:
        print("  No human feedback pairs found. Returning empty path.")
        return ""

    run_id = uuid.uuid4().hex[:12]
    out_key = f"preferences/human-feedback-batch-{run_id}.jsonl"
    body = "\n".join(json.dumps(p) for p in pairs)
    s3.put_object(Bucket=bucket, Key=out_key, Body=body.encode())

    out_path = f"s3://{bucket}/{out_key}"
    print(f"  Wrote {len(pairs)} pairs to {out_path}")
    return out_path
