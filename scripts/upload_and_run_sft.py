#!/usr/bin/env python3
"""
Step 1.7 — Upload Kubeflow data to MinIO and compile/submit the SFT pipeline.

Three modes:
    python scripts/upload_and_run_sft.py upload     # upload data to MinIO
    python scripts/upload_and_run_sft.py compile     # compile pipeline YAML
    python scripts/upload_and_run_sft.py run         # upload + compile + submit
    python scripts/upload_and_run_sft.py all         # alias for run

Prerequisites:
    - MinIO reachable (oc port-forward svc/minio 9000:9000 -n sridharproject)
    - For 'run': KFP endpoint reachable (oc port-forward svc/ds-pipeline-dspa 3000:8888 -n sridharproject)

Environment variables (or .env):
    S3_ENDPOINT    — default http://127.0.0.1:9000
    S3_ACCESS_KEY  — default minioadmin
    S3_SECRET_KEY  — default minioadmin123
    GROQ_API_KEY   — needed for pipeline eval step
    KFP_ENDPOINT   — default http://127.0.0.1:3000
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://127.0.0.1:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"))
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123"))
BUCKET = os.getenv("SYNTHETIC_BUCKET", "mlflow-artifacts")
PREFIX = "synthetic/kubeflow/"
KFP_ENDPOINT = os.getenv("KFP_ENDPOINT", "https://127.0.0.1:3000")
TEACHER_API_URL = os.getenv("TEACHER_API_URL", "http://ollama.sridharproject.svc.cluster.local:11434")
TEACHER_MODEL = os.getenv("TEACHER_MODEL", "llama3.1:8b-instruct-q4_K_M")
TEACHER_API_KEY = os.getenv("TEACHER_API_KEY", "")


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        verify=False,
    )


def upload_data(data_path: str):
    """Upload filtered JSONL to MinIO under synthetic/kubeflow/."""
    path = Path(data_path)
    if not path.exists():
        sys.exit(f"ERROR: Data file not found: {data_path}")

    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(records)} records from {data_path}")

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_id = uuid.uuid4().hex[:12]
    key = f"{PREFIX}date={date_str}/filtered_{run_id}.jsonl"
    body = "\n".join(json.dumps(r) for r in records)

    s3 = get_s3_client()

    try:
        s3.head_bucket(Bucket=BUCKET)
    except Exception:
        print(f"Creating bucket: {BUCKET}")
        s3.create_bucket(Bucket=BUCKET)

    s3.put_object(Bucket=BUCKET, Key=key, Body=body.encode())
    print(f"Uploaded {len(records)} records → s3://{BUCKET}/{key}")

    existing = s3.list_objects_v2(Bucket=BUCKET, Prefix="synthetic/")
    total_keys = existing.get("KeyCount", 0)
    print(f"Total objects under synthetic/: {total_keys}")

    return f"s3://{BUCKET}/{key}"


def verify_upload():
    """List what's in MinIO under synthetic/ to confirm data is there."""
    s3 = get_s3_client()
    print(f"\nMinIO contents under s3://{BUCKET}/synthetic/:")
    paginator = s3.get_paginator("list_objects_v2")
    total = 0
    for page in paginator.paginate(Bucket=BUCKET, Prefix="synthetic/"):
        for obj in page.get("Contents", []):
            size_kb = obj["Size"] / 1024
            print(f"  {obj['Key']:70s} {size_kb:>8.1f} KB")
            total += 1
    print(f"  Total: {total} objects")
    return total


def compile_pipeline():
    """Compile the KFP pipeline to YAML."""
    pipeline_dir = Path(__file__).resolve().parent.parent / "pipeline"
    sys.path.insert(0, str(pipeline_dir))
    os.chdir(pipeline_dir)

    from pipeline import distillation_pipeline
    from kfp import compiler

    output_path = pipeline_dir / "distillation_flywheel.yaml"
    compiler.Compiler().compile(
        pipeline_func=distillation_pipeline,
        package_path=str(output_path),
    )
    print(f"\nCompiled pipeline → {output_path}")
    return str(output_path)


def submit_pipeline(yaml_path: str):
    """Submit the compiled pipeline to KFP."""
    try:
        from kfp import client as kfp_client
    except ImportError:
        sys.exit("ERROR: kfp package not installed. Run: pip install kfp")

    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    print(f"\nConnecting to KFP at {KFP_ENDPOINT}...")
    import subprocess
    token = subprocess.check_output(["oc", "whoami", "-t"]).decode().strip()
    c = kfp_client.Client(host=KFP_ENDPOINT, existing_token=token, ssl_ca_cert=False, verify_ssl=False)

    params = {
        "s3_access_key": S3_ACCESS_KEY,
        "s3_secret_key": S3_SECRET_KEY,
        "teacher_api_url": TEACHER_API_URL,
        "teacher_model": TEACHER_MODEL,
        "teacher_api_key": TEACHER_API_KEY,
        "num_epochs": 3,
        "min_gold_threshold": 0,
    }

    run = c.create_run_from_pipeline_package(
        yaml_path,
        arguments=params,
        run_name=f"sft-kubeflow-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        enable_caching=False,
    )
    print(f"Pipeline run submitted: {run.run_id}")
    print(f"Monitor at: {KFP_ENDPOINT}/#/runs/details/{run.run_id}")
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description="Step 1.7 — Upload data & run SFT pipeline")
    parser.add_argument(
        "action",
        choices=["upload", "verify", "compile", "submit", "run", "all"],
        help="upload: push data to MinIO | verify: list MinIO contents | "
             "compile: build pipeline YAML | submit: submit to KFP | "
             "run/all: upload + compile + submit",
    )
    parser.add_argument("--data", default="data/kubeflow_filtered.jsonl",
                        help="Path to filtered JSONL file")
    parser.add_argument("--yaml", default="pipeline/distillation_flywheel.yaml",
                        help="Pipeline YAML path (for submit)")
    args = parser.parse_args()

    if args.action == "upload":
        upload_data(args.data)
        verify_upload()

    elif args.action == "verify":
        verify_upload()

    elif args.action == "compile":
        compile_pipeline()

    elif args.action == "submit":
        submit_pipeline(args.yaml)

    elif args.action in ("run", "all"):
        upload_data(args.data)
        verify_upload()
        yaml_path = compile_pipeline()
        submit_pipeline(yaml_path)


if __name__ == "__main__":
    main()
