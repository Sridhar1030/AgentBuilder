#!/usr/bin/env python3
"""
Step 1.4 — Kubeflow Question Bank

Generates standalone Q&A pairs organised by Kubeflow sub-project and topic.
Unlike Step 1.3 (grounded on chunks), these questions let the teacher answer
from parametric knowledge — covering breadth across the Kubeflow ecosystem.

This question list serves double duty:
  • SFT  — the Q&A pairs are uploaded to synthetic/kubeflow-qbank/ as training data
  • DPO  — the raw question list is reused by extract_preferences in Phase 2

Output: data/kubeflow_qbank.jsonl  (gold schema: {instruction, output, text})
        data/kubeflow_questions.json  (raw question list for DPO reuse)

Usage:
    python scripts/kubeflow_question_bank.py [--target-pairs 300] [--upload]

Environment variables:
    GROQ_API_KEY   — required
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

DEFAULT_S3_ENDPOINT = "http://minio.sridharproject.svc.cluster.local:9000"
DEFAULT_BUCKET = "mlflow-artifacts"
UPLOAD_PREFIX = "synthetic/kubeflow-qbank/"

# ---------------------------------------------------------------------------
# Comprehensive topic → seed questions bank
# Each topic has 8-20 seed questions the teacher will answer parametrically.
# ---------------------------------------------------------------------------

TOPIC_QUESTIONS: dict[str, list[str]] = {
    # ── Kubeflow Pipelines (KFP) ──────────────────────────────────────
    "KFP SDK & Components": [
        "What is a KFP component and how do you define one using the @component decorator?",
        "How do you pass artifacts between KFP components?",
        "What is the difference between lightweight Python components and containerized components in KFP?",
        "How do you specify pip packages for a KFP component?",
        "Explain the role of InputPath and OutputPath in KFP components.",
        "How do you use kfp.dsl.pipeline to define a pipeline?",
        "What is a KFP ContainerOp and when would you use it?",
        "How do you compile a KFP pipeline to YAML?",
        "What data types can KFP component parameters accept?",
        "How do you set caching options for a KFP task?",
        "What is kfp.dsl.Condition and how is it used for conditional execution?",
        "How do you implement a loop (ParallelFor) in a KFP pipeline?",
    ],
    "KFP Pipeline Execution & Management": [
        "How do you submit a KFP pipeline run via the Python SDK?",
        "What is a KFP experiment and how does it relate to pipeline runs?",
        "How do you pass runtime parameters to a KFP pipeline?",
        "What is a recurring run (schedule) in KFP and how do you create one?",
        "How do you access pipeline run logs and artifacts?",
        "What is the difference between KFP v1 and KFP v2 SDK?",
        "How does KFP handle pipeline versioning?",
        "What backends can KFP use for execution (Argo Workflows, Tekton)?",
        "How do you configure resource limits (CPU, memory, GPU) for a KFP task?",
        "What is the KFP REST API and how do you authenticate with it?",
    ],
    "KFP Infrastructure & Architecture": [
        "What are the main components of a KFP deployment (API server, UI, scheduler)?",
        "How does KFP store pipeline artifacts and metadata?",
        "What is ML Metadata (MLMD) and how does KFP use it?",
        "How do you configure KFP to use an external MySQL database?",
        "What is the role of the KFP persistence agent?",
        "How do you deploy KFP in a namespace-scoped configuration?",
        "What is the KFP cache server and how does it determine cache hits?",
    ],

    # ── Training Operator / Trainer ───────────────────────────────────
    "Training Operator Overview": [
        "What is the Kubeflow Training Operator and what ML frameworks does it support?",
        "Explain the difference between PyTorchJob, TFJob, and MPIJob.",
        "What is the TrainJob CRD introduced in Kubeflow Trainer v2?",
        "How does TrainJob differ from the legacy PyTorchJob CRD?",
        "What is a ClusterTrainingRuntime and how is it used with TrainJob?",
        "How do you install the Kubeflow Training Operator on a Kubernetes cluster?",
        "What Kubernetes RBAC permissions does the Training Operator need?",
        "How does the Training Operator handle distributed training across multiple nodes?",
        "What is the MLPolicy in the context of Kubeflow Trainer?",
        "How do you monitor the status and logs of a training job?",
    ],
    "TrainJob API & Configuration": [
        "What fields does the TrainJob spec contain (trainer, runtimeRef, etc.)?",
        "How do you specify the number of training nodes in a TrainJob?",
        "How do you mount a PVC for dataset storage in a TrainJob?",
        "What is the role of the runtimeRef field in a TrainJob?",
        "How do you pass environment variables to a training container?",
        "How do you configure GPU resources for a TrainJob?",
        "What happens when a training pod fails — how does the operator handle retries?",
        "How do you use init containers with TrainJob for model or data initialization?",
        "What is the relationship between TrainJob, JobSet, and the Kubernetes Job API?",
        "How do you migrate from PyTorchJob to TrainJob?",
    ],
    "Distributed Training Patterns": [
        "How does PyTorch DDP (DistributedDataParallel) work with the Training Operator?",
        "What is the difference between data parallelism and model parallelism?",
        "How do you configure NCCL environment variables for multi-GPU training?",
        "What role does the master/worker topology play in PyTorchJob?",
        "How do you enable elastic training with the Training Operator?",
        "What is DeepSpeed and how do you use it with Kubeflow training jobs?",
        "How do you run FSDP (Fully Sharded Data Parallel) training on Kubeflow?",
        "What network requirements exist for distributed training on Kubernetes?",
    ],

    # ── KServe ────────────────────────────────────────────────────────
    "KServe Model Serving Basics": [
        "What is KServe and how does it integrate with Kubeflow?",
        "What is an InferenceService and what fields does its spec contain?",
        "How do you deploy a simple sklearn model using KServe?",
        "What serving runtimes does KServe support out of the box?",
        "What is the V2 inference protocol and how does it differ from V1?",
        "How does KServe handle model storage (S3, GCS, PVC)?",
        "What is a StorageContainer in KServe?",
        "How do you configure resource requests and limits for an InferenceService?",
        "What is the difference between Serverless and RawDeployment mode in KServe?",
        "How do you check the status and readiness of a deployed model?",
    ],
    "KServe Advanced Features": [
        "How does KServe handle canary deployments for ML models?",
        "What is traffic splitting in KServe and how do you configure it?",
        "How do you set up autoscaling for a KServe InferenceService?",
        "What is a Transformer in KServe and when would you use one?",
        "How do you implement pre/post-processing with a custom Transformer?",
        "What is an InferenceGraph and how does it enable model ensembles?",
        "How does KServe integrate with ModelMesh for multi-model serving?",
        "What is the Explainer component in KServe?",
        "How do you configure GPU sharing for KServe inference pods?",
        "How do you serve LLMs with vLLM through KServe?",
    ],

    # ── Katib ─────────────────────────────────────────────────────────
    "Katib Hyperparameter Tuning": [
        "What is Katib and what problems does it solve?",
        "What search algorithms does Katib support (Random, Grid, Bayesian, TPE, CMA-ES)?",
        "How do you define an Experiment CR in Katib?",
        "What is the difference between an Experiment, Suggestion, and Trial in Katib?",
        "How do you specify the objective metric and optimization direction?",
        "How do you define the search space (parameters) for a Katib experiment?",
        "What is early stopping in Katib and how do you configure it?",
        "How does the Hyperband algorithm work in Katib?",
        "What metrics collectors does Katib support?",
        "How do you integrate Katib with the Training Operator?",
        "What is the maxTrialCount and parallelTrialCount in a Katib experiment?",
        "How do you resume a failed Katib experiment?",
    ],
    "Katib Neural Architecture Search": [
        "What NAS algorithms does Katib support (ENAS, DARTS)?",
        "How does ENAS (Efficient Neural Architecture Search) work in Katib?",
        "How do you configure a NAS experiment in Katib?",
        "What is the difference between hyperparameter tuning and NAS?",
    ],

    # ── Notebooks ─────────────────────────────────────────────────────
    "Kubeflow Notebooks": [
        "What is Kubeflow Notebooks and what IDEs does it support?",
        "How do you create a Jupyter notebook server in Kubeflow?",
        "What container images are available for Kubeflow Notebooks?",
        "How do you attach GPUs to a Kubeflow Notebook?",
        "How do you mount persistent storage (PVC) in a Kubeflow Notebook?",
        "What is the Notebook Controller and how does it manage notebook servers?",
        "How do you configure idle notebook culling to save resources?",
        "How do Kubeflow Notebooks integrate with Kubeflow Pipelines?",
    ],

    # ── Model Registry ────────────────────────────────────────────────
    "Kubeflow Model Registry": [
        "What is the Kubeflow Model Registry and why is it useful?",
        "How do you register a model version in the Kubeflow Model Registry?",
        "What metadata does the Model Registry store for each model?",
        "How does the Model Registry integrate with KServe for deployment?",
        "What is the relationship between RegisteredModel, ModelVersion, and ModelArtifact?",
        "How do you query the Model Registry via its REST API?",
        "How does the Model Registry handle model lineage and provenance?",
    ],

    # ── Multi-Tenancy & Platform ──────────────────────────────────────
    "Kubeflow Multi-Tenancy & Profiles": [
        "How does Kubeflow implement multi-tenancy?",
        "What is a Kubeflow Profile and what resources does it manage?",
        "How do you create a new namespace/profile in Kubeflow?",
        "How does RBAC work across Kubeflow profiles?",
        "How do you share resources between Kubeflow profiles?",
        "What is Istio's role in Kubeflow multi-tenancy?",
    ],
    "Kubeflow Installation & Architecture": [
        "What are the core components of a Kubeflow deployment?",
        "How do you install Kubeflow using manifests?",
        "What is the Kubeflow Central Dashboard and what does it provide?",
        "How does Kubeflow integrate with Istio for networking?",
        "What are the hardware requirements for a Kubeflow cluster?",
        "How do you upgrade Kubeflow from one version to another?",
        "What distributions of Kubeflow exist (standalone, on AWS, GCP, Azure)?",
    ],

    # ── Fine-Tuning & LLMOps on Kubeflow ─────────────────────────────
    "LLM Fine-Tuning with Kubeflow": [
        "How do you fine-tune an LLM using Kubeflow Training Operator?",
        "What is QLoRA and how do you run a QLoRA fine-tuning job on Kubeflow?",
        "How do you set up knowledge distillation with Kubeflow?",
        "What is SFT (Supervised Fine-Tuning) and how does it relate to Kubeflow training?",
        "How do you use PEFT (Parameter-Efficient Fine-Tuning) with Kubeflow?",
        "What is DPO (Direct Preference Optimization) and how can it run on Kubeflow?",
        "How do you serve a fine-tuned model from S3 storage using KServe?",
        "How do you track fine-tuning experiments with MLflow in a Kubeflow pipeline?",
        "What GPU types and quantities are recommended for LLM fine-tuning on Kubeflow?",
        "How do you implement a training flywheel (continuous SFT + evaluation) with KFP?",
    ],

    # ── Integrations & Ecosystem ──────────────────────────────────────
    "Kubeflow Ecosystem Integrations": [
        "How does Kubeflow integrate with MLflow for experiment tracking?",
        "How do you use MinIO as the artifact store in a Kubeflow pipeline?",
        "What is the Data Science Pipelines Application (DSPA) in OpenShift AI?",
        "How do you use Kubeflow on OpenShift AI (RHOAI)?",
        "How does Kubeflow Pipelines compare to Airflow for ML workflows?",
        "How do you configure S3-compatible storage for Kubeflow components?",
        "What is vLLM and how does it integrate with KServe for LLM inference?",
        "How do you use Prometheus and Grafana for monitoring Kubeflow workloads?",
    ],
}

# Flatten for DPO reuse
ALL_QUESTIONS = [q for questions in TOPIC_QUESTIONS.values() for q in questions]

SYSTEM_PROMPT = """\
You are an expert Kubeflow instructor. Answer the following question about
Kubeflow accurately and concisely in 2-5 sentences. Be specific, mention
API names, CRD fields, CLI commands, or config snippets where appropriate.
Do not hedge — give a direct, confident, instructional answer.
"""


def call_groq_single(api_key: str, model: str, question: str) -> str:
    """Get teacher answer for a single question."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "max_tokens": 512,
        "temperature": 0.5,
    }
    for attempt in range(6):
        resp = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 429:
            wait = min(2 ** attempt * 5, 120)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break
    else:
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


BATCH_SYSTEM = """\
You are an expert Kubeflow instructor creating training data.
You will receive {n} questions. For each, produce a JSON object with
"question" and "answer" fields. Answers must be 2-5 sentences, accurate,
and instructional. Mention API names, CRD fields, CLI commands, or YAML
snippets where appropriate.

Return ONLY a JSON array — no markdown fences, no extra text:
[{{"question":"...","answer":"..."}}, ...]
"""


def call_groq_batch(api_key: str, model: str, questions: list[str]) -> list[dict]:
    """Get teacher answers for a batch of questions in one call."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    q_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": BATCH_SYSTEM.format(n=len(questions))},
            {"role": "user", "content": q_block},
        ],
        "max_tokens": 4096,
        "temperature": 0.5,
    }
    for attempt in range(6):
        resp = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=90)
        if resp.status_code == 429:
            wait = min(2 ** attempt * 5, 120)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break
    else:
        resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]

    return json.loads(raw.strip())


def to_gold(question: str, answer: str) -> dict:
    return {
        "instruction": question,
        "output": answer,
        "text": f"### Instruction:\n{question}\n\n### Response:\n{answer}",
    }


def upload_to_s3(records, bucket, prefix, endpoint, access_key, secret_key):
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_id = uuid.uuid4().hex[:12]
    key = f"{prefix}date={date_str}/run_{run_id}.jsonl"
    body = "\n".join(json.dumps(r) for r in records)

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        verify=False,
    )
    s3.put_object(Bucket=bucket, Key=key, Body=body.encode())
    print(f"Uploaded {len(records)} records → s3://{bucket}/{key}")


def main():
    parser = argparse.ArgumentParser(description="Generate Kubeflow question bank Q&A")
    parser.add_argument("--output", default="data/kubeflow_qbank.jsonl")
    parser.add_argument("--questions-json", default="data/kubeflow_questions.json",
                        help="Save raw question list for DPO reuse")
    parser.add_argument("--target-pairs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Questions per Groq call")
    parser.add_argument("--groq-model", default="llama-3.3-70b-versatile")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--bucket", default=os.getenv("SYNTHETIC_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--s3-endpoint", default=os.getenv("S3_ENDPOINT", DEFAULT_S3_ENDPOINT))
    parser.add_argument("--s3-access-key", default=os.getenv("S3_ACCESS_KEY", "minioadmin"))
    parser.add_argument("--s3-secret-key", default=os.getenv("S3_SECRET_KEY", "minioadmin123"))
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit("ERROR: GROQ_API_KEY environment variable is required")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.questions_json, "w") as f:
        json.dump({"topics": TOPIC_QUESTIONS, "all_questions": ALL_QUESTIONS}, f, indent=2)
    print(f"Saved {len(ALL_QUESTIONS)} raw questions → {args.questions_json}")

    gold_records: list[dict] = []
    errors = 0

    questions_to_process = ALL_QUESTIONS[:args.target_pairs]
    print(f"Processing {len(questions_to_process)} questions in batches of {args.batch_size}...")

    with open(args.output, "w") as fout:
        for batch_start in range(0, len(questions_to_process), args.batch_size):
            batch = questions_to_process[batch_start:batch_start + args.batch_size]

            try:
                results = call_groq_batch(api_key, args.groq_model, batch)
                for item in results:
                    q = item.get("question", "").strip()
                    a = item.get("answer", "").strip()
                    if q and a and len(a) > 30:
                        rec = to_gold(q, a)
                        gold_records.append(rec)
                        fout.write(json.dumps(rec) + "\n")
            except Exception as exc:
                errors += 1
                print(f"  Batch {batch_start//args.batch_size + 1} FAILED: {exc}")
                for q in batch:
                    try:
                        a = call_groq_single(api_key, args.groq_model, q)
                        if a and len(a) > 30:
                            rec = to_gold(q, a)
                            gold_records.append(rec)
                            fout.write(json.dumps(rec) + "\n")
                    except Exception as inner:
                        errors += 1
                        print(f"    Single Q failed: {inner}")
                    time.sleep(args.delay)

            done = min(batch_start + args.batch_size, len(questions_to_process))
            print(f"  [{done}/{len(questions_to_process)}] {len(gold_records)} pairs generated")
            time.sleep(args.delay)

    print(f"\nGenerated {len(gold_records)} question bank Q&A pairs ({errors} failures)")
    print(f"Output: {args.output}")

    if args.upload:
        print("Uploading to MinIO...")
        upload_to_s3(
            gold_records, args.bucket, UPLOAD_PREFIX,
            args.s3_endpoint, args.s3_access_key, args.s3_secret_key,
        )


if __name__ == "__main__":
    main()
