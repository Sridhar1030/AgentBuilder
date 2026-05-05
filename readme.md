# Agentic Continual Learning on Red Hat OpenShift AI

A unified continual learning system that improves a Code Review SLM autonomously -- optimizing at both the skill/prompt level and the model/weight level simultaneously, running entirely on OpenShift AI.

Uses [agent-eval-harness](https://github.com/opendatahub-io/agent-eval-harness) for structured evaluation and a distillation pipeline for model weight optimization, combining skill-level and model-level optimization in nested feedback loops.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                  Kubeflow Pipeline: Agentic Continual Learning            │
│                                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────────────────┐   │
│  │ eval-analyze │──>│ eval-dataset │──>│     distill-pipeline        │   │
│  │              │   │              │   │  ┌─────────────────────┐    │   │
│  │ Read MLflow  │   │ Generate new │   │  │ resolve-version     │    │   │
│  │ history,     │   │ training     │   │  │ extract-gold        │    │   │
│  │ find model   │   │ data for     │   │  │ SFT finetune (GPU)  │    │   │
│  │ weaknesses   │   │ weak areas   │   │  │ deploy-sft          │    │   │
│  └──────────────┘   └──────────────┘   │  │ extract-preferences │    │   │
│                                        │  │ DPO finetune (GPU)  │    │   │
│                                        │  └─────────────────────┘    │   │
│                                        └──────────────┬──────────────┘   │
│                                                       │                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────▼──────────────┐   │
│  │eval-optimize │<──│ quality-gate │<──│     deploy-candidate        │   │
│  │              │   │              │   │                             │   │
│  │ Analyze      │   │ Pass: keep   │   │  eval-run (harness judges) │   │
│  │ failures,    │   │ Fail: roll   │   │  Score with eval.yaml      │   │
│  │ recommend    │   │ back to SFT  │   │  judges + thresholds       │   │
│  │ adjustments  │   │              │   │                             │   │
│  └──────────────┘   └──────────────┘   └─────────────────────────────┘   │
│                                                                          │
│  MLflow: shared data plane (traces, metrics, recommendations)            │
└────────────────────────────────────────────────────────────────────────────┘
```

The distill-pipeline renders as a **single collapsible node** in the KFP UI -- click it to expand and see the inner training steps.

---

## Components

| # | Component | Source | What it does |
|---|-----------|--------|-------------|
| 1 | **eval-analyze** | `pipeline/components/eval_analyze.py` | Reads MLflow evaluation history, identifies score trends and weak categories |
| 2 | **eval-dataset** | `pipeline/components/eval_dataset.py` | Generates targeted synthetic training data for weak categories using the teacher model |
| 3 | **distill-pipeline** | Sub-pipeline in `pipeline/unified_pipeline.py` | SFT + DPO training loop (resolve version, extract gold, SFT, deploy, extract preferences, DPO) |
| 4 | **deploy-candidate** | `pipeline/components/deploy_model.py` | Deploys the DPO model to KServe for evaluation |
| 5 | **eval-run** | `pipeline/components/evaluate.py` | Scores the model using structured judges from `eval/eval.yaml` via agent-eval-harness `EvalConfig` |
| 6 | **quality-gate** | `pipeline/components/quality_gate.py` | Pass/fail check against eval.yaml thresholds; triggers rollback on failure |
| 7 | **eval-optimize** | `pipeline/components/eval_optimize.py` | Analyzes judge failures, generates training adjustment recommendations to MLflow |

---

## Evaluation: agent-eval-harness Integration

Judges are defined in [`eval/eval.yaml`](eval/eval.yaml) using the [agent-eval-harness](https://github.com/opendatahub-io/agent-eval-harness) format:

| Judge | Type | What it checks |
|-------|------|---------------|
| **correctness** | Inline check | Does the review catch real bugs? Does it avoid hallucinating issues on clean code? |
| **conciseness** | Inline check | Is the review under 200 words (PR-comment length)? |
| **review_quality** | LLM judge | Is the review actionable, specific, and relevant? (1-5 score via teacher) |
| **format_check** | Inline check | Does the review avoid boilerplate filler phrases? |

Thresholds (from `eval.yaml`):
- correctness: min 70% pass rate
- conciseness: min 80% pass rate
- review_quality: min 3.5/5 mean
- format_check: min 70% pass rate

If any threshold fails, the quality gate rolls back to the SFT model automatically.

---

## Models

| Role | Model | Size | Serving |
|------|-------|------|---------|
| **Teacher** | `qwen2.5-coder:32b-instruct-q4_K_M` | 32B (4-bit) | Ollama, in-cluster |
| **Student** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1.5B | KServe + vLLM |

The student specializes in code review for Go, Python, and Kubernetes diffs.

---

## Repository Structure

```
AgentBuilder/
├── readme.md                         This file
├── distill.config.yaml               Single config driving the entire pipeline
│
├── eval/
│   ├── eval.yaml                     agent-eval-harness judge definitions + thresholds
│   ├── generate_cases.py             Generates harness case directories from test_questions.json
│   ├── dataset/cases/                Harness-native case directories (input.yaml, annotations.yaml)
│   └── prompts/                      External LLM judge prompt templates
│
├── pipeline/
│   ├── unified_pipeline.py           Unified Agentic Continual Learning pipeline (main)
│   ├── code_review_pipeline.py       Inner pipeline only (legacy, kept for reference)
│   ├── outer_pipeline.py             Outer pipeline only (legacy, kept for reference)
│   ├── components/
│   │   ├── eval_analyze.py           /eval-analyze from agent-eval-harness
│   │   ├── eval_dataset.py           /eval-dataset -- synthetic data generation
│   │   ├── evaluate.py               /eval-run -- harness judge execution
│   │   ├── quality_gate.py           Eval-gated deployment decision
│   │   ├── eval_optimize.py          /eval-optimize -- failure analysis + recommendations
│   │   ├── resolve_version.py        Auto-increment model version in MinIO
│   │   ├── finetune.py               SFT fine-tune (QLoRA, TrainJob CRD)
│   │   ├── extract_preferences.py    DPO preference pair extraction (teacher vs student)
│   │   ├── collect_human_feedback.py Human feedback DPO pairs from MinIO
│   │   ├── merge_preferences.py      Merge all preference sources
│   │   ├── dpo_finetune.py           DPO fine-tune (TrainJob CRD)
│   │   └── deploy_model.py           KServe InferenceService deployment
│   ├── domain/
│   │   └── test_questions.json       15 curated test cases with annotations
│   ├── training/
│   │   └── finetune_job.py           Training script (SFT + DPO modes)
│   └── scripts/
│       ├── baseline_eval.py          Standalone baseline evaluation script
│       └── generate_preference_bank.py
│
├── agent-eval-harness/               Forked harness with OpenAI-compatible runner
│   └── agent_eval/agent/openai_compatible.py
│
├── distillation-operator/            Kubernetes operator (Go) for one-click runs
├── rhoai/                            OpenShift manifests (DSPA, ISVC, MinIO, Ollama)
└── .cursor/skills/distill/           Cursor skill for managing the pipeline
```

---

## Quick Start

### Prerequisites

- OpenShift AI cluster with Data Science Pipelines, KServe, and Training Operator
- GPU nodes (T4 or better) for SFT and DPO training
- Ollama deployed in-cluster with the teacher model pulled

### Configure

Edit `distill.config.yaml` with your cluster details:

```yaml
cluster:
  namespace: "your-namespace"
  s3_endpoint: "http://minio.your-namespace.svc.cluster.local:9000"
  mlflow_uri: "http://mlflow.your-namespace.svc.cluster.local:5000"

teacher:
  api_url: "http://ollama.your-namespace.svc.cluster.local:11434"
  model: "qwen2.5-coder:32b-instruct-q4_K_M"
```

### Compile and Upload

```bash
cd pipeline
python3 unified_pipeline.py          # Compiles -> unified_pipeline.yaml

# Upload via DSPA REST API
TOKEN=$(oc whoami -t)
oc port-forward svc/ds-pipeline-dspa 3991:8443 -n your-namespace &

curl -sk "https://localhost:3991/apis/v2beta1/pipelines/upload" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "uploadfile=@unified_pipeline.yaml"
```

### Trigger a Run

```bash
PIPELINE_ID="<from upload response>"

curl -sk "https://localhost:3991/apis/v2beta1/runs" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{
    \"display_name\": \"run-1\",
    \"pipeline_version_reference\": {\"pipeline_id\": \"${PIPELINE_ID}\"},
    \"runtime_config\": {}
  }"
```

Or use the RHOAI Dashboard: Data Science Pipelines > Create Run.

### Monitor

```bash
# MLflow UI
oc port-forward svc/mlflow 5000:5000 -n your-namespace &
open http://localhost:5000   # Experiment: CodeReview-Eval-Hub

# Pipeline logs
oc get pods -n your-namespace | grep agentic
oc logs <pod-name> -c main -n your-namespace
```

---

## Tech Stack

| Component | Role |
|-----------|------|
| **Red Hat OpenShift AI** | ML platform (KServe, Pipelines, Training Operator) |
| **KServe + vLLM** | Serves the 1.5B student model |
| **Ollama** | Hosts the 32B teacher model in-cluster |
| **MLflow** | Experiment tracking, shared data plane between loops |
| **MinIO** | S3-compatible storage for models, training data, preferences |
| **Data Science Pipelines (KFP v2)** | Orchestrates the unified pipeline |
| **Kubeflow Training Operator v2** | GPU training jobs via TrainJob CRD |
| **agent-eval-harness** | Structured judge framework (EvalConfig, eval.yaml) |
| **QLoRA + SFTTrainer** | Memory-efficient SFT (4-bit quantization) |
| **DPOTrainer (trl)** | Direct Preference Optimization |

---

## Design Decisions

**Why one unified pipeline instead of two?**
Each agent-eval-harness skill maps to a pipeline component. The distill-pipeline is a nested sub-pipeline that collapses into one node in the UI. This enables scheduled, automated execution of the full improvement loop.

**Why embed eval.yaml content as a parameter?**
KFP containers don't have filesystem access to the project repo. The eval.yaml content is read at compile time and passed as a string parameter to the evaluate component, which writes it to a temp file and parses it with `EvalConfig.from_yaml()`.

**Why the quality gate + rollback?**
Every stage transition is eval-gated. If the DPO model scores worse than thresholds, the pipeline automatically redeploys the SFT model. This prevents bad models from going live.

---

## References

- [agent-eval-harness](https://github.com/opendatahub-io/agent-eval-harness)
- [Red Hat OpenShift AI](https://www.redhat.com/en/technologies/cloud-computing/openshift/openshift-ai)
