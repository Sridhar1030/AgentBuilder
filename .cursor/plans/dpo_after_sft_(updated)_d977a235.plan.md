---
name: DPO After SFT (Updated)
overview: Add a DPO (Direct Preference Optimization) stage after SFT in the distillation pipeline. The evaluate step already produces both Teacher and Student answers for the same questions — this data becomes the preference source. Two new components (extract_preferences, dpo_finetune) slot between SFT Fine-Tune and Deploy.
todos:
  - id: extract-prefs-component
    content: "Build extract_preferences.py KFP component: fetch eval_results.json from MLflow, optionally query both models on extra questions, output preference JSONL to MinIO"
    status: pending
  - id: dpo-training-script
    content: Add DPO mode to finetune_job.py (TRAINING_MODE env var switches between SFT and DPO using DPOTrainer), or create separate dpo_job.py
    status: pending
  - id: dpo-finetune-component
    content: Build dpo_finetune.py KFP component (same TrainJob pattern as finetune.py, passes DPO-specific env vars)
    status: pending
  - id: rebuild-image
    content: Rebuild training container image with DPO support, push as v0.2.0
    status: pending
  - id: update-pipeline
    content: Update pipeline.py to insert extract_preferences and dpo_finetune steps between SFT and Deploy
    status: pending
  - id: test-e2e
    content: "Run full pipeline twice: first run = SFT only (no prior eval), second run = SFT + DPO. Compare scores."
    status: pending
isProject: false
---

# Adding DPO to the Distillation Flywheel (Updated)

## Why DPO After SFT?

SFT teaches the Student to imitate the Teacher's answers, but it has no concept of preference — it never sees bad answers. DPO fixes this by teaching the model: "given two responses, prefer this one over that one." This is the standard two-stage alignment approach (SFT then DPO), and `trl==0.17.0` (already in the training Dockerfile) supports `DPOTrainer` with QLoRA.

## Where Preference Data Comes From (Updated)

The old plan assumed preference data lived in MLflow traces. After recent changes, the data sources are:

- `**eval_results.json` artifact** — The [evaluate.py](pipeline/components/evaluate.py) component (lines 151-199) already queries both Student and Teacher on the same 5 questions, grades both, and saves everything to `eval_results.json` in MLflow. Each entry has `question`, `student_answer`, `student_score`, `teacher_answer`, `teacher_score`. This is ready-made preference data.
- **Dedicated preference collection** — For more than 5 pairs, a new script can query both models on a larger question set (e.g. 50-100 questions from the synthetic topic pool) and write preference JSONL directly to MinIO.

The evaluate artifact is the simplest starting point. For the POC, even 5 high-quality preference pairs can move the needle with DPO.

## Data Format Required by DPOTrainer

```json
{"prompt": "What is QLoRA?", "chosen": "QLoRA combines quantization with...", "rejected": "I don't have information on QLoRA."}
```

Three fields: `prompt`, `chosen`, `rejected`. One row per preference pair.

## Pipeline Changes

Current pipeline:

```
Resolve Version -> Extract Gold -> Fine-Tune (SFT) -> Deploy -> Evaluate
```

New pipeline:

```
Resolve Version -> Extract Gold -> Fine-Tune (SFT) -> Extract Preferences -> Fine-Tune (DPO) -> Deploy -> Evaluate
```

## New Components

### 1. `extract_preferences.py` — KFP Component

Pulls preference pairs and writes JSONL to MinIO. Two data sources:

**Source A (primary):** Download `eval_results.json` from the *previous* pipeline run's MLflow artifact. Filter to entries where `student_score < teacher_score` (i.e., the Student was worse). Map to DPO format:

- `prompt` = `question`
- `chosen` = `teacher_answer`
- `rejected` = `student_answer`

**Source B (supplement):** Query both Student and Teacher on a larger question set (passed as parameter or read from MinIO), grade both, produce more pairs. This is essentially a mini-evaluate that writes preference JSONL instead of metrics.

Parameters:

- `s3_endpoint`, `s3_access_key`, `s3_secret_key` — MinIO access
- `mlflow_tracking_uri` — to fetch eval_results.json from previous run
- `model_version` — to find the right MLflow run (`pipeline-eval-{version}`)
- `student_url` — for Source B (querying the deployed SFT model)
- `groq_api_key`, `groq_model` — for Source B (querying Teacher)
- `output_s3_path` — where to write the preference JSONL
- `extra_questions: list` — optional additional questions for Source B
- `min_pairs: int` — minimum preference pairs required (default: 5)

Output: S3 path to preference JSONL file.

The component needs `packages_to_install=["requests", "mlflow", "boto3"]` and the same SSL/TLS handling pattern from evaluate.py (lines 33-43).

### 2. `dpo_finetune.py` — Training Script (runs in TrainJob)

Based on existing [pipeline/training/finetune_job.py](pipeline/training/finetune_job.py), adapted for DPO:

- Reads preference JSONL from S3 (same S3 download pattern as finetune_job.py lines 60-65)
- Loads the **SFT-trained model** (output of the SFT step, not the base model) with QLoRA
- Uses `trl.DPOTrainer` + `DPOConfig` instead of `SFTTrainer` + `SFTConfig`
- Same LoRA config: `r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]`
- Key DPO params: `beta=0.1`, `learning_rate=5e-5` (lower than SFT's 2e-4), `num_train_epochs=1`, `per_device_train_batch_size=2`
- With PEFT, pass `ref_model=None` — trl uses the base (non-LoRA) weights as reference automatically
- Same merge + upload logic as finetune_job.py lines 124-169

New env vars (in addition to existing ones): `PREF_DATA_PATH`, `TRAINING_MODE=dpo`, `DPO_BETA=0.1`.

**Option:** Instead of a separate script, add a `TRAINING_MODE` env var to the existing `finetune_job.py` that switches between SFT and DPO mode. This avoids a second Docker image.

### 3. `dpo_finetune_component.py` — KFP Component (orchestrator)

Same pattern as [pipeline/components/finetune.py](pipeline/components/finetune.py) — creates a Kubeflow TrainJob, polls for completion. The only differences:

- Passes `PREF_DATA_PATH` instead of `GOLD_DATA_PATH`
- Passes `SFT_MODEL_PATH` (the S3 path to the SFT model from the previous step) as `BASE_MODEL_ID`
- Sets `TRAINING_MODE=dpo` and `DPO_BETA=0.1`
- Uses the same container image (`quay.io/rh-ee-srpillai/distillation-trainer:v0.2.0` after adding DPO support)

### 4. Pipeline Wiring in [pipeline/pipeline.py](pipeline/pipeline.py)

Insert two new steps between the current `finetune_task` (line 85) and `deploy_task` (line 99):

```python
# Step 2.5 -- Extract Preferences (from previous eval run)
pref_task = extract_preferences(
    s3_endpoint=S3_ENDPOINT,
    s3_access_key=s3_access_key,
    s3_secret_key=s3_secret_key,
    mlflow_tracking_uri=MLFLOW_URI,
    model_version=version_task.outputs["version"],
    student_url=...,  # SFT model just deployed? Or skip Source B initially
    groq_api_key=groq_api_key,
    groq_model=groq_model,
    output_s3_path=...,  # s3://mlflow-artifacts/preferences/pref-{version}.jsonl
)

# Step 2.75 -- DPO Fine-Tune
dpo_task = dpo_finetune(
    pref_data_path=pref_task.output,
    sft_model_path=finetune_task.output,  # SFT model as starting point
    model_output_s3_path=version_task.outputs["model_output_path"],
    s3_endpoint=S3_ENDPOINT,
    s3_access_key=s3_access_key,
    s3_secret_key=s3_secret_key,
)

# Step 3 -- Deploy now uses DPO output instead of SFT output
deploy_task = deploy_model(
    model_s3_path=dpo_task.output,  # changed from finetune_task.output
    ...
)
```

## Docker Image Update

The existing [Dockerfile](pipeline/training/Dockerfile) already has `trl==0.17.0` which includes `DPOTrainer`. Changes needed:

- Copy the updated `finetune_job.py` (with DPO mode) into the image
- Rebuild and push as `v0.2.0`
- Update the image tag in the KFP component

## Chicken-and-Egg: First Run

DPO needs preference data from a *previous* evaluate run. On the very first run, there is no previous eval. Handle this with a simple check:

- `extract_preferences.py` checks if a previous `pipeline-eval-`* run exists in MLflow
- If none found, output an empty preference file and log a warning
- `dpo_finetune` component skips training if the preference file has fewer than `min_pairs` entries, and passes through the SFT model path unchanged

This way the pipeline still works end-to-end on the first run (SFT only), and DPO kicks in starting from the second run.

## Key Considerations

- **Data volume**: Even 5-10 preference pairs help. The 5 eval questions produce up to 5 pairs per run. After 3 pipeline cycles, you accumulate 15 pairs.
- **Beta parameter**: Start with `beta=0.1`. Lower = more divergence from SFT, higher = closer to SFT.
- **Learning rate**: 5e-5 for DPO (vs 2e-4 for SFT) since DPO is a refinement step.
- **GPU memory**: DPO with QLoRA fits on a single T4. Use batch_size=2 (vs 4 for SFT) and gradient_accumulation_steps=4.
- **MinIO storage**: Each DPO model adds ~2.5GB. The old-model cleanup pattern (keep only last 2 versions) should be applied.

