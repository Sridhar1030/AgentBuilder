# 70B → 1B Knowledge Distillation on Red Hat OpenShift AI

An automated distillation flywheel that transfers knowledge from a 70B Teacher LLM into a 1B Student SLM, fully running on OpenShift AI with GPU training, model serving, experiment tracking, and a Kubernetes operator for one-click pipeline execution.

---

## How it works

```
  70B Teacher (Groq)                                    1B Student (KServe)
  ┌──────────────┐       gold data        ┌──────────┐       ┌──────────────┐
  │  Llama 3.3   │ ───────────────────►   │  QLoRA   │ ───►  │  Llama 3.2   │
  │  70B params  │   high-quality Q&A     │  SFT     │       │  1B params   │
  │  (answers)   │   logged in MLflow     │  (GPU)   │       │  (serving)   │
  └──────┬───────┘                        └──────────┘       └──────┬───────┘
         │                                                          │
         │  grades the student                                      │  answers users
         └──────────────────────────────────────────────────────────┘
                              feedback loop
```

1. Users chat with models through a **Gradio web app**
2. The **70B Teacher** (via Groq API) produces high-quality answers — these are logged as training data in **MLflow**
3. The **1B Student** (served on KServe) answers questions and gets graded by the Teacher in real-time
4. An automated **5-step pipeline** extracts gold data, fine-tunes the Student with QLoRA on GPU, deploys the improved model, and evaluates it
5. A **Kubernetes operator** triggers the entire pipeline with a single `oc apply` command

Each cycle, the Student gets smarter — closing the gap with the Teacher at a fraction of the serving cost.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           OpenShift AI Cluster                                  │
│                                                                                 │
│  ┌─────────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌─────────────┐ │
│  │   KServe    │  │  MLflow  │  │  MinIO   │  │    DSP     │  │  Operator   │ │
│  │  Student 1B │  │  Traces  │  │  Models  │  │  Pipeline  │  │  Controller │ │
│  │  (vLLM)     │  │  & Eval  │  │  & Data  │  │  Server    │  │             │ │
│  └──────┬──────┘  └────┬─────┘  └────┬─────┘  └─────┬──────┘  └──────┬──────┘ │
│         │              │             │               │                │         │
└─────────┼──────────────┼─────────────┼───────────────┼────────────────┼─────────┘
          │              │             │               │                │
          ▼              ▼             ▼               ▼                ▼
     serves model   tracks all    stores models   runs 5-step      triggers
     via REST API   interactions  & gold data     distillation     pipeline via
                                                  flywheel         DistillationJob CR
```

---

## Distillation Pipeline (5 steps)

```
Resolve Version → Extract Gold Data → Fine-Tune (GPU) → Deploy Model → Evaluate
     (CPU)             (CPU)            (QLoRA/T4)         (CPU)         (CPU)
```

| Step | What it does |
|------|-------------|
| **Resolve Version** | Scans MinIO for existing models, auto-increments (v6 → v7 → v8) |
| **Extract Gold Data** | Pulls high-scoring Teacher traces from MLflow into training JSONL |
| **Fine-Tune** | QLoRA SFT on a single T4 GPU via Kubeflow Training Operator |
| **Deploy Model** | Hot-swaps the live KServe InferenceService with the new model |
| **Evaluate** | Teacher grades the new Student on test questions, reports scores |

---

## Tech Stack

| Component | Role |
|-----------|------|
| **Red Hat OpenShift AI 3.2** | ML platform (KServe, Pipelines, Training Operator) |
| **KServe + vLLM** | Serves the 1B Student model as a REST API |
| **Groq API** | Hosts the 70B Teacher (Llama 3.3 70B Versatile) |
| **MLflow** | Experiment tracking — logs every chat interaction and Teacher grade |
| **MinIO** | On-cluster S3-compatible storage for models and training data |
| **Data Science Pipelines (KFP v2)** | Orchestrates the 5-step distillation pipeline |
| **Kubeflow Training Operator** | Manages GPU training jobs (PyTorchJob) |
| **QLoRA + SFTTrainer** | Memory-efficient fine-tuning (4-bit quantization, single GPU) |
| **Gradio** | Chat UI for interacting with Teacher and Student |
| **Custom Kubernetes Operator** | Triggers pipeline runs via DistillationJob CRD |

---

## Repository Structure

```
AgentBuilder/
├── app.py                          Gradio chat UI (Teacher + Student + grading)
├── gold_extractor.py               Extracts teacher traces from MLflow into JSONL
├── finetune.py                     QLoRA fine-tuning script (runs on-cluster)
├── datagen.py                      Generates Teacher Q&A pairs via Groq
├── pipeline/
│   ├── pipeline.py                 KFP pipeline definition (5-step DAG)
│   ├── distillation_flywheel.yaml  Compiled pipeline YAML (uploaded to DSP)
│   └── components/                 Individual pipeline step implementations
│       ├── resolve_version.py
│       ├── extract_gold.py
│       ├── finetune.py
│       ├── deploy_model.py
│       └── evaluate.py
├── distillation-operator/          Kubernetes operator (Go)
│   ├── api/v1alpha1/               CRD type definitions (DistillationJob)
│   ├── internal/controller/        Reconcile loop (state machine)
│   ├── internal/dsp/               DSP REST API client
│   ├── config/                     CRD, RBAC, Deployment manifests
│   └── Dockerfile
├── rhoai/                          OpenShift manifests (DSPA, ISVC, MinIO, etc.)
├── docs/                           Project documentation
├── sprint.md                       Sprint tracker with all tasks and issues
└── requirements.txt                Python dependencies
```

---

## Quick Start

**Run the Gradio app:**

```bash
source venv311/bin/activate
oc port-forward pod/<student-pod> 8080:8080 -n sridharproject &
oc port-forward svc/minio 9000:9000 -n sridharproject &
AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin123 \
  MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
  STUDENT_ENDPOINT=http://localhost:8080/v1 \
  MLFLOW_TRACKING_URI=https://mlflow-sridharproject.apps.<cluster>/
  MLFLOW_TRACKING_INSECURE_TLS=true \
  python app.py
```

**Trigger a distillation run (via operator):**

```bash
oc apply -f distillation-operator/config/samples/distillation_v1alpha1_distillationjob.yaml
oc get distillationjob test-run -n sridharproject -w
# Pending → Submitting → Running → Succeeded
```

---

## Key URLs (cluster-specific)

| Service | URL |
|---------|-----|
| Gradio App | `http://127.0.0.1:7860` (local) |
| MLflow UI | `https://mlflow-sridharproject.apps.<cluster>` |
| MinIO Console | `https://minio-console-sridharproject.apps.<cluster>` |
| DSP Dashboard | OpenShift AI Console → Data Science Pipelines → Runs |
