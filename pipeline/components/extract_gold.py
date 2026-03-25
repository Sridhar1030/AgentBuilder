"""
KFP Component 1 -- Extract Gold Data

Reads teacher Q&A from MinIO (incremental since last cursor), merges with
synthetic data from MinIO, writes combined gold JSONL, and advances the cursor.
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["boto3"],
)
def extract_gold_data(
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    teacher_bucket: str,
    teacher_prefix: str,
    synthetic_bucket: str,
    synthetic_prefix: str,
    cursor_key: str,
    output_s3_path: str,
    min_threshold: int = 0,
) -> str:
    """Read teacher interactions (incremental) and synthetic data from MinIO, write merged gold JSONL."""
    import json
    import random
    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )

    # 1. Read cursor
    last_ts = 0.0
    try:
        obj = s3.get_object(Bucket=teacher_bucket, Key=cursor_key)
        cursor_data = json.loads(obj["Body"].read().decode("utf-8"))
        last_ts = float(cursor_data.get("last_processed_timestamp", 0))
        print(f"Cursor loaded: last_processed_timestamp={last_ts}")
    except s3.exceptions.NoSuchKey:
        print("No cursor found -- processing all teacher interactions.")
    except Exception as exc:
        print(f"Warning: could not read cursor: {exc}. Processing all teacher interactions.")

    # 2. List and read teacher interactions (incremental)
    teacher_pairs = []
    max_ts = last_ts
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=teacher_bucket, Prefix=teacher_prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith(".cursor.json") or key == teacher_prefix:
                continue
            try:
                body = s3.get_object(Bucket=teacher_bucket, Key=key)["Body"].read().decode("utf-8")
                record = json.loads(body)
                ts = float(record.get("timestamp", 0))
                if ts <= last_ts:
                    continue

                instr = record.get("instruction", "").strip()
                outp = record.get("output", "").strip()
                if not instr or not outp:
                    continue

                text = record.get("text") or f"### Instruction:\n{instr}\n\n### Response:\n{outp}"
                teacher_pairs.append({
                    "instruction": instr,
                    "output": outp,
                    "text": text,
                })
                if ts > max_ts:
                    max_ts = ts
            except Exception as exc:
                print(f"Warning: skipping {key}: {exc}")

    teacher_pairs.sort(key=lambda p: p["instruction"])
    print(f"Teacher: {len(teacher_pairs)} new interactions (since ts={last_ts})")

    # 3. Read synthetic data (all objects under synthetic prefix)
    synthetic_pairs = []
    for page in paginator.paginate(Bucket=synthetic_bucket, Prefix=synthetic_prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key == synthetic_prefix:
                continue
            try:
                body = s3.get_object(Bucket=synthetic_bucket, Key=key)["Body"].read().decode("utf-8")
                for line in body.strip().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    instr = record.get("instruction", "").strip()
                    outp = record.get("output", "").strip()
                    if not instr or not outp:
                        continue
                    text = record.get("text") or f"### Instruction:\n{instr}\n\n### Response:\n{outp}"
                    synthetic_pairs.append({
                        "instruction": instr,
                        "output": outp,
                        "text": text,
                    })
            except Exception as exc:
                print(f"Warning: skipping synthetic {key}: {exc}")

    print(f"Synthetic: {len(synthetic_pairs)} pairs")

    # 4. Merge, shuffle, and write gold
    gold_pairs = teacher_pairs + synthetic_pairs
    total = len(gold_pairs)
    print(f"Total pairs: {total} (teacher={len(teacher_pairs)}, synthetic={len(synthetic_pairs)})")

    if total > 1:
        random.shuffle(gold_pairs)

    if min_threshold and total < min_threshold:
        raise RuntimeError(f"Only {total} pairs found, below threshold {min_threshold}")

    out_parts = output_s3_path.replace("s3://", "").split("/", 1)
    out_bucket, out_key = out_parts[0], out_parts[1]
    body = "\n".join(json.dumps(p) for p in gold_pairs)

    s3.put_object(Bucket=out_bucket, Key=out_key, Body=body.encode())
    print(f"Uploaded {total} gold pairs to {output_s3_path}")

    # 5. Advance cursor
    if max_ts > last_ts:
        cursor_body = json.dumps({"last_processed_timestamp": max_ts})
        s3.put_object(
            Bucket=teacher_bucket,
            Key=cursor_key,
            Body=cursor_body.encode(),
            ContentType="application/json",
        )
        print(f"Cursor advanced to {max_ts}")
    else:
        print("Cursor unchanged (no new teacher interactions).")

    return output_s3_path
