"""
KFP Component 0 -- Resolve Version

Scans MinIO for existing student-1b-vN/ prefixes and returns the next
version string plus pre-built S3 paths for downstream components.
"""

from typing import NamedTuple
from kfp import dsl

VersionOutputs = NamedTuple(
    "VersionOutputs",
    [
        ("version", str),
        ("gold_data_path", str),
        ("model_output_path", str),
    ],
)


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["boto3"],
)
def resolve_version(
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    model_bucket: str,
    model_prefix: str,
    gold_bucket: str,
    explicit_version: str = "",
) -> NamedTuple("VersionOutputs", [("version", str), ("gold_data_path", str), ("model_output_path", str)]):
    """Find the latest student-1b-vN/ in MinIO and return vN+1 with paths."""
    import re
    from collections import namedtuple
    import boto3

    print("=" * 60)
    print("RESOLVE VERSION STEP")
    print("=" * 60)
    print(f"  Model bucket: {model_bucket}")
    print(f"  Model prefix: {model_prefix}")
    print(f"  Gold bucket:  {gold_bucket}")
    print(f"  Explicit ver: {explicit_version or '(auto)'}")
    print("=" * 60)

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )

    if explicit_version:
        version = explicit_version
        print(f"Using explicit version: {version}")
    else:
        paginator = s3.get_paginator("list_objects_v2")
        version_numbers = []

        for page in paginator.paginate(Bucket=model_bucket, Prefix=model_prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                folder = cp["Prefix"]
                match = re.search(r"-v(\d+)/?$", folder)
                if match:
                    version_numbers.append(int(match.group(1)))

        if version_numbers:
            latest = max(version_numbers)
            version = f"v{latest + 1}"
            print(f"Found versions: {sorted(version_numbers)}. Latest: v{latest}. Next: {version}")
        else:
            version = "v1"
            print(f"No existing versions found under {model_bucket}/{model_prefix}. Starting at {version}")

    gold_data_path = f"s3://{gold_bucket}/gold/train-{version}.jsonl"
    model_output_path = f"s3://{model_bucket}/{model_prefix}{version}/"

    print(f"Version: {version}")
    print(f"Gold data path: {gold_data_path}")
    print(f"Model output path: {model_output_path}")

    Outputs = namedtuple("VersionOutputs", ["version", "gold_data_path", "model_output_path"])
    return Outputs(version=version, gold_data_path=gold_data_path, model_output_path=model_output_path)
