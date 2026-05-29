# Agentic Continual Learning on Red Hat OpenShift AI

I built an **agentic continual learning pipeline** that autonomously improves a 1.5B-parameter language model through progressive training stages — SFT, DPO, and GRPO — running entirely on Red Hat OpenShift AI (RHOAI). Code review is the demonstration domain, but **the pipeline itself is the product**: a self-improving ML system with eval gates, safety fallbacks, and cycle-to-cycle knowledge retention.

The system distills a 32B teacher (Qwen2.5-Coder via Ollama) into a 1.5B student (KServe + vLLM), orchestrates the full training loop via Kubeflow Pipelines, and closes the loop with structured evaluation, human feedback ingestion, and canary deployment.

---

## What Makes This Different

| Capability | What it means in practice |
|------------|---------------------------|
| **Agentic** | The pipeline reads its own eval history, generates targeted training data for weak categories, and triggers the next training cycle without manual intervention |
| **Progressive training** | SFT → DPO → GRPO with RL rewards at each stage, not a single fine-tune pass |
| **Eval-gated promotion** | Every stage is scored by structured judges; only models that pass thresholds get `.gate-passed` markers in S3 |
| **Safety nets** | DPO regression detection falls back to SFT before GRPO; quality gates prevent bad models from going live |
| **Continual learning** | AB_1 → AB_2 cycles warm-start from the best prior checkpoint (GRPO > DPO > SFT), proving knowledge retention |
| **Platform-native** | Kubeflow Training Operator v2 (`TrainJob` CRD), KFP v2, KServe, MLflow — all first-class RHOAI components |

---

## Architecture

The outer pipeline (`agent_builder_final_pipeline`) orchestrates analysis, data generation, human feedback collection, and the inner distillation loop. The inner pipeline renders as a **single collapsible node** in the KFP UI — click to expand the full SFT → DPO → GRPO sequence.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Outer Pipeline: agent_builder_final_pipeline                  │
│                                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐                    │
│  │ eval-analyze │──>│ eval-dataset │   │ collect-human-   │                    │
│  │              │   │              │   │ feedback         │                    │
│  │ Read MLflow  │   │ Generate     │   │ VS Code ext →    │                    │
│  │ history,     │   │ targeted     │   │ DPO preference   │                    │
│  │ find weak    │   │ training     │   │ pairs in MinIO   │                    │
│  │ categories   │   │ data         │   └────────┬─────────┘                    │
│  └──────────────┘   └──────────────┘            │                              │
│         │                   │                   │                              │
│         └───────────────────┴───────────────────┘                              │
│                                 │                                               │
│                                 v                                               │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                    Inner Pipeline: distill-pipeline                       │  │
│  │                                                                           │  │
│  │  resolve-version ──> extract-code-review-gold ──> finetune (SFT)         │  │
│  │       │                                              │                    │  │
│  │       │  Pick best prior model                       v                    │  │
│  │       │  (GRPO > DPO > SFT via .gate-passed)    deploy-model             │  │
│  │       │                                              │                    │  │
│  │       │                                              v                    │  │
│  │       │                                         evaluate (eval-sft)       │  │
│  │       │                                              │                    │  │
│  │       │                                              v                    │  │
│  │       │                                    extract-preferences             │  │
│  │       │                                              │                    │  │
│  │       │                                              v                    │  │
│  │       │                                    merge-preferences             │  │
│  │       │                                    (static bank + human feedback) │  │
│  │       │                                              │                    │  │
│  │       │                                              v                    │  │
│  │       │                                         dpo-finetune             │  │
│  │       │                                              │                    │  │
│  │       │                                              v                    │  │
│  │       │                                         evaluate (eval-dpo)       │  │
│  │       │                                              │                    │  │
│  │       │                                              v                    │  │
│  │       │                              pick-training-checkpoint             │  │
│  │       │                              (DPO regression → fallback to SFT)  │  │
│  │       │                                              │                    │  │
│  │       │                                              v                    │  │
│  │       │                                         grpo-finetune             │  │
│  │       │                                              │                    │  │
│  │       │                                              v                    │  │
│  │       │                                         deploy-model             │  │
│  │       │                                              │                    │  │
│  │       │                                              v                    │  │
│  │       │                                         evaluate (eval-grpo)       │  │
│  │       │                              Write .gate-passed marker on pass    │  │
│  │       └──────────────────────────────────────────────┘                    │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                 │                                               │
│                                 v                                               │
│                          traffic-shift                                          │
│                    (canary: teacher → student)                                  │
│                                                                                 │
│  Shared data plane: MLflow (metrics, traces) + MinIO (models, preferences)    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Training Progression

Each pipeline run (labeled `AB_N`) executes the full progressive training arc. Evaluation runs after every stage; only the final GRPO model writes a `.gate-passed` marker that `resolve_version` reads on the next cycle.

```
  Base Model (HF)          SFT                  DPO                  GRPO
  ───────────────    ───────────────    ───────────────    ───────────────
  Qwen2.5-Coder      QLoRA fine-tune    Preference         RL with reward
  1.5B Instruct      on gold data       optimization       signal from eval
                     │                  │                  │
                     v                  v                  v
                  eval-sft           eval-dpo           eval-grpo
                  judges             judges             judges + gate
                     │                  │                  │
                     │            pick_training_            │
                     │            checkpoint              │
                     │            (safety net)           │
                     └──────────────────┴──────────────────┘
                                        │
                                        v
                              .gate-passed in S3
                              (next cycle starts here)
```

All three training stages use **Kubeflow Training Operator v2** — the `TrainJob` CRD with the `torch-distributed` `ClusterTrainingRuntime`. This is the new Trainer v2 API, not legacy PyTorchJob.

| Stage | Trainer | Runtime | What it optimizes |
|-------|---------|---------|-------------------|
| **SFT** | `SFTTrainer` (TRL) | TrainJob v2, multi-node | Imitation learning on teacher-generated gold data |
| **DPO** | `DPOTrainer` (TRL) | TrainJob v2, single-node | Preference pairs from eval failures + static bank + human feedback |
| **GRPO** | `GRPOTrainer` (TRL) | TrainJob v2, single-node | Group-relative policy optimization with RL rewards |

---

## Results

### AB_1 — First cycle (cold start from HuggingFace base)

| Stage | Correctness | Conciseness | Format | Quality |
|-------|-------------|-------------|--------|---------|
| Base model | — | — | — | — |
| **SFT** | 73% | — | — | — |
| DPO | (evaluated) | — | — | — |
| **GRPO** | **73%** | **100%** | **100%** | Passed gate |

AB_1 established that the pipeline could take a base 1.5B model to production-quality code review output with perfect format and conciseness compliance.

### AB_2 — Continual learning (warm start from AB_1 GRPO)

| Observation | Evidence |
|-------------|----------|
| **Knowledge retention** | SFT loss curves start lower than AB_1 — the model retains prior cycle learning |
| **Autonomous safety** | Pipeline detected DPO regression and `pick_training_checkpoint` fell back to SFT for GRPO |
| **Cycle continuity** | `resolve_version` picked AB_1's GRPO checkpoint via `.gate-passed` marker |

The AB_1 → AB_2 progression demonstrates the core continual learning thesis: each cycle builds on the best promoted model, and safety mechanisms prevent regression from propagating downstream.

---

## Evaluation Framework

I integrated [agent-eval-harness](https://github.com/opendatahub-io/agent-eval-harness) for structured, multi-dimensional scoring. Judges and thresholds are defined in [`eval/eval.yaml`](eval/eval.yaml):

| Judge | Type | What it checks |
|-------|------|----------------|
| **correctness** | Inline check | Catches real bugs; avoids hallucinating issues on clean code |
| **conciseness** | Inline check | Review stays under 200 words (PR-comment length) |
| **review_quality** | LLM judge | Actionable, specific, relevant (1–5 via teacher) |
| **format_check** | Inline check | No boilerplate filler phrases |

Thresholds gate model promotion. MLflow run names follow a clear convention: `AB_1-sft-train`, `AB_1-sft-eval`, `AB_1-dpo-train`, `AB_1-grpo-eval`, etc.

---

## Models

| Role | Model | Size | Serving |
|------|-------|------|---------|
| **Teacher** | `qwen2.5-coder:32b-instruct-q4_K_M` | 32B (4-bit) | Ollama, in-cluster |
| **Student** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1.5B | KServe + vLLM |

The student specializes in code review for Go, Python, and Kubernetes diffs. The teacher generates gold training data, grades responses, and serves as the LLM judge for quality scoring.

---

## Key Design Decisions

**Why progressive SFT → DPO → GRPO instead of SFT alone?**
SFT teaches the model what good reviews look like. DPO aligns it with human and eval-derived preferences. GRPO adds RL-style optimization against reward signals from the eval harness. Each stage addresses a different failure mode; the pipeline runs all three with gates between them.

**Why `.gate-passed` markers in S3 instead of a database?**
Model artifacts already live in MinIO. Co-locating promotion markers with model weights keeps the system simple, auditable, and decoupled from MLflow availability. `resolve_version` scans for markers with priority GRPO > DPO > SFT.

**Why `pick_training_checkpoint` before GRPO?**
DPO can regress on correctness even when preference loss improves. Comparing SFT vs DPO eval results and falling back to SFT prevents GRPO from amplifying a bad checkpoint. This safety net fired autonomously in AB_2.

**Why TrainJob v2 / `torch-distributed`?**
Training Operator v2 is the current RHOAI direction — cleaner API, better runtime abstraction, and native support for distributed PyTorch. All three training components (`finetune.py`, `dpo_finetune.py`, `grpo_finetune.py`) submit `TrainJob` CRDs referencing the same `ClusterTrainingRuntime`.

**Why a single config file (`distill.config.yaml`)?**
The entire pipeline — cluster endpoints, teacher/student models, training hyperparameters, eval paths, canary settings — is driven by one YAML file. No code changes needed to switch domains, clusters, or model sizes.

**Why embed `eval.yaml` as a compile-time parameter?**
KFP containers don't have repo filesystem access. The eval config is read at pipeline compile time and passed as a string parameter; the evaluate component writes it to a temp file and parses it with `EvalConfig.from_yaml()`.

**Why canary traffic shifting?**
Progressive migration from teacher to student (configurable increment via Istio VirtualService) reduces risk when promoting a new model version to production inference.

---

## Tech Stack

| Component | Role |
|-----------|------|
| **Red Hat OpenShift AI** | ML platform — KServe, Data Science Pipelines, Training Operator |
| **Kubeflow Training Operator v2** | GPU training via `TrainJob` CRD + `torch-distributed` runtime |
| **Data Science Pipelines (KFP v2)** | Full orchestration with nested sub-pipelines |
| **KServe + vLLM** | High-throughput student model serving |
| **Ollama** | In-cluster 32B teacher model |
| **MLflow** | Experiment tracking, shared metrics data plane |
| **MinIO** | S3-compatible storage for models, training data, preferences |
| **agent-eval-harness** | Structured judge framework (`EvalConfig`, `eval.yaml`) |
| **QLoRA + TRL** | Memory-efficient SFT, DPO, and GRPO training |
| **Istio** | Canary traffic shifting between teacher and student |

---

## Repository Structure

```
AgentBuilder/
├── distill.config.yaml              # Single config file for entire pipeline
├── eval/
│   └── eval.yaml                    # Judge definitions + thresholds
├── pipeline/
│   ├── unified_pipeline.py          # Main pipeline definition
│   ├── components/                  # All KFP components
│   │   ├── finetune.py              # SFT via TrainJob v2
│   │   ├── dpo_finetune.py          # DPO via TrainJob v2
│   │   ├── grpo_finetune.py         # GRPO via TrainJob v2
│   │   ├── evaluate.py              # agent-eval-harness evaluation
│   │   ├── resolve_version.py       # Model version resolution
│   │   ├── pick_training_checkpoint.py  # DPO regression safety net
│   │   ├── extract_preferences.py     # DPO preference extraction
│   │   ├── merge_preferences.py       # Preference merging
│   │   ├── deploy_model.py            # KServe deployment
│   │   ├── eval_analyze.py            # MLflow analysis
│   │   ├── eval_dataset.py            # Targeted data generation
│   │   └── traffic_shift.py           # Canary deployment
│   ├── training/
│   │   └── finetune_job.py            # Training script (SFT/DPO/GRPO)
│   └── scripts/
│       ├── trigger_ab_run.py          # Pipeline trigger with preflight
│       ├── baseline_eval.py           # Standalone baseline eval
│       └── generate_preference_bank.py
├── rhoai/                             # OpenShift manifests
├── feedback-service/                  # Human feedback API
└── .cursor/skills/distill/            # Developer experience skill
```

---

## Developer Experience

I packaged pipeline operations into a Cursor agent skill (`/distill`) that provides one-command workflows:

| Command | Purpose |
|---------|---------|
| `/distill.prerequisites` | Bootstrap MinIO, MLflow, Ollama, KServe on a fresh cluster |
| `/distill.setup` | Validate config and infrastructure readiness |
| `/distill.run` | Compile, upload, and trigger a pipeline run |
| `/distill.status` | Monitor active runs and training jobs |
| `/distill.eval` | Run standalone evaluation against current model |
| `/distill.scores` | Pull MLflow metrics for a run label |

Everything reads from `distill.config.yaml` — the skill is a thin orchestration layer over the same pipeline components.

---

## References

- [agent-eval-harness](https://github.com/opendatahub-io/agent-eval-harness)
- [Red Hat OpenShift AI](https://www.redhat.com/en/technologies/cloud-computing/openshift/openshift-ai)
- [Kubeflow Trainer v2](https://www.kubeflow.org/docs/components/trainer/)
