# LLM-to-SLM Knowledge Distillation on Red Hat OpenShift AI

An automated distillation flywheel that transfers knowledge from an 8B Teacher LLM into a 1B Student SLM using SFT + DPO, fully running on OpenShift AI with GPU training, model serving, experiment tracking, and a Kubernetes operator for one-click pipeline execution.

**Result:** Student score improved from **5.4/10 → 8.13/10** (Teacher baseline: 9.0/10).

---

## How it works

```
  8B Teacher (Ollama, in-cluster)                       1B Student (KServe + vLLM)
  ┌──────────────┐       gold data        ┌──────────┐       ┌──────────────┐
  │  Llama 3.1   │ ───────────────────►   │  QLoRA   │ ───►  │  Llama 3.2   │
  │  8B params   │   827 Kubeflow Q&A     │  SFT     │       │  1B params   │
  │  (teacher)   │                        └────┬─────┘       │  (serving)   │
  └──────┬───────┘                             │              └──────┬───────┘
         │                              ┌──────▼─────┐              │
         │  preference pairs            │    DPO     │              │
         └─────────────────────────────►│  alignment │──────────────┘
                                        └────────────┘
                              feedback loop
```

1. Users chat with models through a **Gradio web app**
2. The **8B Teacher** (Ollama, in-cluster) produces high-quality answers — logged as training data
3. The **1B Student** (served on KServe + vLLM) answers questions and gets graded by the Teacher in real-time
4. An automated **7-step pipeline** extracts gold data, fine-tunes with SFT, builds preference pairs, refines with DPO, deploys, and evaluates
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

## Distillation Pipeline (7 steps — SFT + DPO)

```
Resolve Version → Extract Gold → SFT Fine-Tune → Extract Preferences → DPO Fine-Tune → Deploy → Evaluate
     (CPU)           (CPU)        (QLoRA/T4)          (CPU)              (DPO/T4)       (CPU)     (CPU)
```

| Step | What it does |
|------|-------------|
| **Resolve Version** | Scans MinIO for existing models, auto-increments (v30 → v31 → v32) |
| **Extract Gold Data** | Merges Teacher interactions + 827 synthetic Kubeflow Q&A pairs into training JSONL |
| **SFT Fine-Tune** | QLoRA SFT on a single T4 GPU via Kubeflow TrainJob CRD |
| **Extract Preferences** | Student & Teacher answer same questions; where Teacher wins → preference pairs |
| **DPO Fine-Tune** | Refines SFT model using DPOTrainer on preference pairs |
| **Deploy Model** | Hot-swaps the live KServe InferenceService with the DPO-aligned model |
| **Evaluate** | Teacher grades the Student on 15 domain questions, logs scores to MLflow |

---

## Tech Stack

| Component | Role |
|-----------|------|
| **Red Hat OpenShift AI 3.2** | ML platform (KServe, Pipelines, Training Operator) |
| **KServe + vLLM** | Serves the 1B Student model (Llama 3.2 1B) as a REST API |
| **Ollama (in-cluster)** | Hosts the 8B Teacher model (Llama 3.1 8B Instruct) — no external API deps |
| **MLflow** | Experiment tracking — logs every chat interaction and Teacher grade |
| **MinIO** | On-cluster S3-compatible storage for models, training data, and preference pairs |
| **Data Science Pipelines (KFP v2)** | Orchestrates the 7-step SFT+DPO distillation pipeline |
| **Kubeflow Training Operator v2** | Manages GPU training jobs via TrainJob CRD (`trainer.kubeflow.org/v1alpha1`) |
| **QLoRA + SFTTrainer** | Memory-efficient SFT (4-bit quantization, single T4 GPU) |
| **DPOTrainer (trl)** | Direct Preference Optimization — aligns model on Teacher vs Student pairs |
| **Gradio** | Chat UI for interacting with Teacher and Student |
| **Custom Kubernetes Operator** | Triggers pipeline runs via DistillationJob CRD |
| **Hardware** | AWS g4dn.12xlarge — 4× NVIDIA Tesla T4 (15 GB VRAM each) |

---

## Repository Structure

```
AgentBuilder/
├── app.py                          Gradio chat UI (Teacher + Student + grading)
├── gold_extractor.py               Extracts teacher traces from MLflow into JSONL
├── datagen.py                      Generates Teacher Q&A pairs
├── pipeline/
│   ├── pipeline.py                 KFP pipeline definition (7-step SFT+DPO DAG)
│   ├── distillation_flywheel.yaml  Compiled pipeline YAML (uploaded to DSP)
│   ├── training/
│   │   ├── finetune_job.py         Training script (SFT + DPO modes)
│   │   └── Dockerfile              Training container image
│   └── components/                 Individual pipeline step implementations
│       ├── resolve_version.py
│       ├── extract_gold.py
│       ├── finetune.py             Submits SFT TrainJob
│       ├── extract_preferences.py  Builds DPO preference pairs (Teacher vs Student)
│       ├── dpo_finetune.py         Submits DPO TrainJob
│       ├── deploy_model.py
│       └── evaluate.py
├── scripts/
│   ├── upload_and_run_sft.py       Compiles, registers, and submits pipeline
│   ├── generate_synthetic_gold.py  Generates Kubeflow Q&A pairs
│   └── generate_preferences_local.py
├── data/
│   ├── kubeflow_filtered.jsonl     827 Kubeflow Q&A pairs (SFT data)
│   └── kubeflow_questions.json     139 questions across 16 topics (DPO source)
├── distillation-operator/          Kubernetes operator (Go)
│   ├── api/v1alpha1/               CRD type definitions (DistillationJob)
│   ├── internal/controller/        Reconcile loop (state machine)
│   ├── internal/dsp/               DSP REST API client
│   ├── config/                     CRD, RBAC, Deployment manifests
│   └── Dockerfile
├── rhoai/                          OpenShift manifests (DSPA, ISVC, MinIO, etc.)
├── docs/                           Project documentation
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
