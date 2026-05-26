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
        ("dpo_model_output_path", str),
        ("grpo_model_output_path", str),
        ("prev_model_path", str),
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
    hf_base_model_id: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    explicit_version: str = "",
) -> NamedTuple(
    "VersionOutputs",
    [
        ("version", str),
        ("gold_data_path", str),
        ("model_output_path", str),
        ("dpo_model_output_path", str),
        ("grpo_model_output_path", str),
        ("prev_model_path", str),
    ],
):
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

    paginator = s3.get_paginator("list_objects_v2")
    version_numbers = []

    for page in paginator.paginate(Bucket=model_bucket, Prefix=model_prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            folder = cp["Prefix"]
            match = re.search(r"-v(\d+)/?$", folder)
            if match:
                version_numbers.append(int(match.group(1)))

    if explicit_version:
        version = explicit_version
        print(f"Using explicit version: {version}")
    elif version_numbers:
        latest = max(version_numbers)
        version = f"v{latest + 1}"
        print(f"Found versions: {sorted(version_numbers)}. Latest: v{latest}. Next: {version}")
    else:
        version = "v1"
        print(f"No existing versions found under {model_bucket}/{model_prefix}. Starting at {version}")

    if version_numbers:
        prev_version = max(version_numbers)
        sft_path = f"s3://{model_bucket}/{model_prefix}v{prev_version}/"
        dpo_path = f"s3://{model_bucket}/{model_prefix}v{prev_version}-dpo/"
        gate_marker_key = f"{model_prefix}v{prev_version}-dpo/.gate-passed"

        # Prefer the DPO model if it passed the quality gate (marker file exists)
        use_dpo = False
        try:
            s3.head_object(Bucket=model_bucket, Key=gate_marker_key)
            use_dpo = True
            print(f"Found .gate-passed marker at {gate_marker_key}")
        except s3.exceptions.NoSuchKey:
            print(f"No .gate-passed marker at {gate_marker_key} -- using SFT model")
        except Exception as e:
            err_code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if err_code in ("404", "NoSuchKey"):
                print(f"No .gate-passed marker at {gate_marker_key} -- using SFT model")
            else:
                print(f"WARNING: S3 error checking gate marker ({e}) -- falling back to SFT model")

        if use_dpo:
            prev_model_path = dpo_path
            print(f"Previous model: {prev_model_path} (DPO-improved, gate passed)")
        else:
            prev_model_path = sft_path
            print(f"Previous model: {prev_model_path} (SFT checkpoint)")
    else:
        prev_model_path = hf_base_model_id
        print(f"No previous model found -- SFT will start from HuggingFace base: {hf_base_model_id}")

    gold_data_path = f"s3://{gold_bucket}/gold/train-{version}.jsonl"
    model_output_path = f"s3://{model_bucket}/{model_prefix}{version}/"
    dpo_model_output_path = f"s3://{model_bucket}/{model_prefix}{version}-dpo/"
    grpo_model_output_path = f"s3://{model_bucket}/{model_prefix}{version}-grpo/"

    print(f"Version: {version}")
    print(f"Gold data path: {gold_data_path}")
    print(f"Model output path (SFT): {model_output_path}")
    print(f"Model output path (DPO): {dpo_model_output_path}")
    print(f"Model output path (GRPO): {grpo_model_output_path}")
    print(f"Prev model path: {prev_model_path or '(none)'}")

    Outputs = namedtuple(
        "VersionOutputs",
        [
            "version", "gold_data_path", "model_output_path",
            "dpo_model_output_path", "grpo_model_output_path", "prev_model_path",
        ],
    )
    return Outputs(
        version=version,
        gold_data_path=gold_data_path,
        model_output_path=model_output_path,
        dpo_model_output_path=dpo_model_output_path,
        grpo_model_output_path=grpo_model_output_path,
        prev_model_path=prev_model_path,
    )
