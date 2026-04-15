"""
KFP Component -- Merge Preferences

Merges DPO preference pairs from multiple sources (extract_preferences output
and human feedback) into a single JSONL file for the DPO training step.
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["boto3"],
)
def merge_preferences(
    pipeline_pref_path: str,
    human_feedback_path: str,
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    bucket: str = "mlflow-artifacts",
) -> str:
    """Merge pipeline-generated and human-feedback preference pairs."""
    import json
    import uuid

    import boto3

    print("=" * 60)
    print("MERGE PREFERENCES STEP")
    print("=" * 60)
    print(f"  Pipeline prefs:  {pipeline_pref_path}")
    print(f"  Human feedback:  {human_feedback_path or '(none)'}")

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )

    all_pairs = []

    def read_jsonl_from_s3(s3_path: str, label: str) -> int:
        if not s3_path:
            print(f"  {label}: empty path, skipping")
            return 0
        parts = s3_path.replace("s3://", "").split("/", 1)
        obj = s3.get_object(Bucket=parts[0], Key=parts[1])
        lines = [l for l in obj["Body"].read().decode().strip().split("\n") if l.strip()]
        count = 0
        for line in lines:
            pair = json.loads(line)
            if pair.get("prompt") and pair.get("chosen") and pair.get("rejected"):
                all_pairs.append(pair)
                count += 1
        print(f"  {label}: {count} pairs")
        return count

    pipeline_count = read_jsonl_from_s3(pipeline_pref_path, "Pipeline preferences")
    human_count = read_jsonl_from_s3(human_feedback_path, "Human feedback")

    print(f"  Total merged: {len(all_pairs)} pairs ({pipeline_count} pipeline + {human_count} human)")

    if not all_pairs:
        print("  No pairs to merge. Returning pipeline path as-is.")
        return pipeline_pref_path

    run_id = uuid.uuid4().hex[:12]
    out_key = f"preferences/merged-{run_id}.jsonl"
    body = "\n".join(json.dumps(p) for p in all_pairs)
    s3.put_object(Bucket=bucket, Key=out_key, Body=body.encode())

    out_path = f"s3://{bucket}/{out_key}"
    print(f"  Wrote merged preferences to {out_path}")
    return out_path
