"""
KFP Pipeline -- Code Review SLM (Phase 3)

Distills a code review SLM from a teacher LLM using SFT + DPO.
Completely separate from the Phase 2 Kubeflow Q&A pipeline.

Steps:
  0. Resolve version (auto-increment code-review-1.5b-vN)
  1. Extract gold data (reads pre-built training JSONL from MinIO)
  2. SFT fine-tune (QLoRA on Qwen2.5-Coder-1.5B-Instruct)
  3. Deploy SFT model via KServe (temporary, needed for DPO preference extraction)
  4. Extract DPO preference pairs (teacher vs deployed SFT student)
  5. DPO fine-tune (refine SFT model with preferences)
  6. Deploy final DPO model via KServe (overwrites SFT)
  7. Evaluate final model

Compile:
    cd pipeline && python code_review_pipeline.py

Upload via RHOAI Dashboard or:
    from kfp import client
    c = client.Client(host="https://ds-pipeline-dspa-sridharproject.apps.<cluster>/")
    c.upload_pipeline("code_review_pipeline.yaml", pipeline_name="code-review-slm")
"""

import json
import os
from pathlib import Path

from kfp import dsl, compiler
from components.resolve_version import resolve_version
from components.finetune import finetune
from components.extract_preferences import extract_preferences
from components.collect_human_feedback import collect_human_feedback
from components.merge_preferences import merge_preferences
from components.dpo_finetune import dpo_finetune
from components.deploy_model import deploy_model
from components.evaluate import evaluate


def _load_config():
    """Load distill.config.yaml and resolve {{namespace}} templates."""
    try:
        import yaml
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml", "-q"])
        import yaml

    config_path = Path(__file__).resolve().parent.parent / "distill.config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ns = cfg["cluster"]["namespace"]
    def _resolve(val):
        if isinstance(val, str):
            return val.replace("{{namespace}}", ns)
        return val

    for section in cfg.values():
        if isinstance(section, dict):
            for k, v in section.items():
                section[k] = _resolve(v)
    return cfg


_CFG = _load_config()

# -- Cluster config (from distill.config.yaml) --------------------------------
NAMESPACE = _CFG["cluster"]["namespace"]
S3_ENDPOINT = _CFG["cluster"]["s3_endpoint"]
MLFLOW_URI = _CFG["cluster"]["mlflow_uri"]

# -- Student / Teacher --------------------------------------------------------
ISVC_NAME = _CFG["student"]["isvc_name"]
BASE_MODEL_ID = _CFG["student"]["base_model_id"]

TEACHER_BUCKET = _CFG["cluster"]["data_bucket"]
SYNTHETIC_BUCKET = _CFG["cluster"]["data_bucket"]
SYNTHETIC_PREFIX = _CFG["domain"]["training_data_prefix"]
QUESTION_BANK_S3 = _CFG["domain"]["question_bank_s3"]

# -- Test questions (loaded from domain config) --------------------------------
_tq_file = Path(__file__).resolve().parent.parent / _CFG["domain"]["test_questions_file"]
if _tq_file.exists():
    with open(_tq_file) as f:
        _tq_data = json.load(f)
    TEST_QUESTIONS = [item["question"] for item in _tq_data]
else:
    raise FileNotFoundError(f"Test questions file not found: {_tq_file}")


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["boto3"],
)
def extract_code_review_gold(
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    synthetic_bucket: str,
    synthetic_prefix: str,
    output_s3_path: str,
) -> str:
    """Read pre-built code review training JSONL from MinIO.

    Unlike the generic extract_gold, this component reads JSONL files
    that already have a `text` field in ChatML format, so no instruction/output
    parsing is needed.
    """
    import json
    import random
    import boto3

    print("=" * 60)
    print("EXTRACT CODE REVIEW GOLD DATA STEP")
    print("=" * 60)
    print(f"  Source:  s3://{synthetic_bucket}/{synthetic_prefix}")
    print(f"  Output:  {output_s3_path}")
    print("=" * 60)

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )

    records = []
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=synthetic_bucket, Prefix=synthetic_prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.endswith(".jsonl"):
                continue
            try:
                body = s3.get_object(Bucket=synthetic_bucket, Key=key)["Body"].read().decode("utf-8")
                for line in body.strip().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if not record.get("text"):
                        continue
                    records.append(record)
            except Exception as exc:
                print(f"Warning: skipping {key}: {exc}")

    print(f"Loaded {len(records)} training records from s3://{synthetic_bucket}/{synthetic_prefix}")

    if len(records) > 1:
        random.shuffle(records)

    out_parts = output_s3_path.replace("s3://", "").split("/", 1)
    out_bucket, out_key = out_parts[0], out_parts[1]
    body = "\n".join(json.dumps(r) for r in records)
    s3.put_object(Bucket=out_bucket, Key=out_key, Body=body.encode())
    print(f"Uploaded {len(records)} gold records to {output_s3_path}")

    return output_s3_path


@dsl.pipeline(
    name="code-review-slm",
    description="Code Review SLM: SFT + DPO distillation on Go/Python/K8s code diffs using Qwen2.5-Coder-1.5B.",
)
def code_review_pipeline(
    model_version: str = "",
    s3_access_key: str = _CFG["cluster"]["s3_access_key"],
    s3_secret_key: str = _CFG["cluster"]["s3_secret_key"],
    teacher_api_url: str = _CFG["teacher"]["api_url"],
    teacher_model: str = _CFG["teacher"]["model"],
    teacher_api_key: str = _CFG["teacher"].get("api_key", ""),
    num_epochs: int = _CFG["training"]["sft_epochs"],
    dpo_epochs: int = _CFG["training"]["dpo_epochs"],
    dpo_beta: float = _CFG["training"]["dpo_beta"],
    min_dpo_pairs: int = _CFG["training"]["min_dpo_pairs"],
    max_supplement_questions: int = _CFG["training"]["max_supplement_questions"],
):
    # Step 0 -- Resolve version (auto-increment code-review-1.5b-vN)
    version_task = resolve_version(
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        model_bucket=_CFG["cluster"]["model_bucket"],
        model_prefix=_CFG["student"]["model_prefix"],
        gold_bucket=TEACHER_BUCKET,
        hf_base_model_id=BASE_MODEL_ID,
        explicit_version=model_version,
    )
    version_task.set_caching_options(False)

    # Step 1 -- Extract gold data (reads pre-built ChatML JSONL from MinIO)
    extract_task = extract_code_review_gold(
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        synthetic_bucket=SYNTHETIC_BUCKET,
        synthetic_prefix=SYNTHETIC_PREFIX,
        output_s3_path=version_task.outputs["gold_data_path"],
    )
    extract_task.set_caching_options(False)

    # Step 2 -- SFT fine-tune with QLoRA (GPU)
    # Iterative training (Option B): resolve_version returns the previous run's
    # S3 model path if one exists, otherwise returns the HF base model ID.
    sft_task = finetune(
        gold_data_path=extract_task.output,
        model_output_s3_path=version_task.outputs["model_output_path"],
        base_model_id=version_task.outputs["prev_model_path"],
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        num_epochs=num_epochs,
    )
    sft_task.set_caching_options(False)

    # Step 3 -- Deploy SFT model via KServe (needed for DPO preference extraction)
    deploy_sft_task = deploy_model(
        model_s3_path=sft_task.output,
        isvc_name=ISVC_NAME,
        namespace=NAMESPACE,
    )
    deploy_sft_task.set_caching_options(False)

    # Step 4a -- Collect human feedback DPO pairs from MinIO (runs early,
    #            parallel with SFT+deploy since it only reads from S3)
    human_fb_task = collect_human_feedback(
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
    )
    human_fb_task.after(extract_task)
    human_fb_task.set_caching_options(False)

    # Step 4b -- Extract DPO preference pairs (teacher vs deployed SFT on diff-bank)
    pref_task = extract_preferences(
        student_url=f"http://{ISVC_NAME}-predictor.{NAMESPACE}.svc.cluster.local:8080",
        teacher_api_url=teacher_api_url,
        teacher_model=teacher_model,
        teacher_api_key=teacher_api_key,
        question_bank_s3_path=QUESTION_BANK_S3,
        mlflow_tracking_uri=MLFLOW_URI,
        sft_run_name_prefix="pipeline-eval-",
        model_version=version_task.outputs["version"],
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        max_supplement_questions=max_supplement_questions,
    )
    pref_task.after(deploy_sft_task)
    pref_task.set_caching_options(False)

    # Step 4c -- Merge all preference sources (static bank + live + human + SFT regularization)
    merge_task = merge_preferences(
        pipeline_pref_path=pref_task.output,
        human_feedback_path=human_fb_task.output,
        gold_data_path=extract_task.output,
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
    )
    merge_task.set_caching_options(False)

    # Step 5 -- DPO fine-tune (refine SFT model with merged preference data)
    dpo_task = dpo_finetune(
        sft_model_s3_path=sft_task.output,
        pref_data_s3_path=merge_task.output,
        model_version=version_task.outputs["version"],
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        num_epochs=dpo_epochs,
        dpo_beta=dpo_beta,
        min_pairs=min_dpo_pairs,
    )
    dpo_task.set_caching_options(False)

    # Step 6 -- Deploy final DPO model via KServe
    deploy_dpo_task = deploy_model(
        model_s3_path=dpo_task.output,
        isvc_name=ISVC_NAME,
        namespace=NAMESPACE,
    )
    deploy_dpo_task.set_caching_options(False)

    # Step 7 -- Evaluate final model
    eval_task = evaluate(
        student_url=deploy_dpo_task.output,
        teacher_api_url=teacher_api_url,
        teacher_model=teacher_model,
        teacher_api_key=teacher_api_key,
        test_questions=TEST_QUESTIONS,
        mlflow_tracking_uri=MLFLOW_URI,
        model_version=version_task.outputs["version"],
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        grading_prompt=_CFG["domain"]["grading_prompt"],
    )
    eval_task.set_caching_options(False)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=code_review_pipeline,
        package_path="code_review_pipeline.yaml",
    )
    print("Compiled -> code_review_pipeline.yaml")
