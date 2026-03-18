---
name: DPO After SFT
overview: Add a DPO (Direct Preference Optimization) stage after the existing SFT step in the distillation pipeline, using the Teacher-as-Judge grading system to automatically generate preference pairs (chosen/rejected) for training.
todos:
  - id: collect-prefs
    content: Extend app.py or create a script to ensure both Teacher and Student answer the same questions, building up preference data in MLflow
    status: pending
  - id: extract-prefs-component
    content: Build extract_preferences.py KFP component that pulls Teacher-vs-Student pairs from MLflow and outputs preference JSONL
    status: pending
  - id: dpo-finetune-component
    content: Build dpo_finetune.py KFP component using DPOTrainer + QLoRA on the SFT-trained model
    status: pending
  - id: update-pipeline
    content: Update pipeline.py to add the two new steps between SFT Fine-Tune and Deploy
    status: pending
  - id: test-e2e
    content: Run full pipeline and compare SFT-only vs SFT+DPO evaluation scores
    status: pending
isProject: false
---

# Adding DPO to the Distillation Flywheel

## Why DPO After SFT?

The current pipeline uses **SFT only** — the Student learns to imitate the Teacher's answers. But SFT alone has a limitation: the Student learns what good answers look like, but not what bad answers look like. It has no concept of preference.

**DPO fixes this.** It teaches the model: "given two responses, prefer this one over that one." This is the standard two-stage approach used in production LLM alignment:

1. **SFT** — teaches the model to follow instructions and produce reasonable outputs
2. **DPO** — refines the model to prefer higher-quality responses over lower-quality ones

DPO is simpler than RLHF (no reward model, no PPO), and the `trl` library already supports it via `DPOTrainer` with QLoRA — the same stack the project already uses.

## The Key Insight: You Already Have Preference Data

The existing system already generates everything DPO needs:

- **The Student answers questions** — these are candidate responses (some good, some bad)
- **The Teacher grades each answer (1-10)** — this is the preference signal
- **The Teacher also answers the same questions** — these are the "gold" responses

This means preference pairs can be constructed automatically:

```
prompt:   "What is knowledge distillation?"
chosen:   Teacher's answer (score: 9/10)
rejected: Student's answer (score: 4/10)
```

Or even Student-vs-Student pairs across versions:

```
prompt:   "Explain LoRA"
chosen:   Student v8 answer (score: 7/10)
rejected: Student v6 answer (score: 3/10)
```

## Data Format Required by DPOTrainer

The `trl.DPOTrainer` expects this structure:

```json
{
  "prompt": "What is knowledge distillation?",
  "chosen": "Knowledge distillation is a technique where...",
  "rejected": "Knowledge distillation is when you take a model and..."
}
```

Three fields: `prompt`, `chosen`, `rejected`. One row per preference pair.

## How to Generate Preference Pairs

Three strategies, from simplest to most powerful:

### Strategy 1: Teacher-vs-Student (easiest, use first)

For each question where both the Teacher and Student answered:

- `prompt` = the question
- `chosen` = Teacher's response
- `rejected` = Student's response (especially when Teacher scored it low)

**Source:** MLflow traces already contain both Teacher and Student responses for the same questions, plus the Teacher's grade. The existing `extract_gold.py` component can be extended to also extract Student responses.

### Strategy 2: Student-vs-Student across versions

After multiple flywheel cycles, you have Student responses from v6, v7, v8, etc. If the Teacher graded them differently:

- `prompt` = the question
- `chosen` = Student v8 response (score: 8)
- `rejected` = Student v6 response (score: 4)

**Source:** Run evaluation on the same test questions for each model version, store results in MLflow.

### Strategy 3: Best-of-N sampling (most powerful, future)

For each prompt, generate N responses from the Student (with temperature sampling), have the Teacher grade all N, then:

- `chosen` = highest-scored response
- `rejected` = lowest-scored response

This produces the highest-quality on-policy preference data since both responses come from the same model.

## Pipeline Changes

The current pipeline is:

```
Resolve Version -> Extract Gold -> Fine-Tune (SFT) -> Deploy -> Evaluate
```

The new pipeline adds two steps:

```
Resolve Version -> Extract Gold -> Fine-Tune (SFT) -> Extract Preferences -> Fine-Tune (DPO) -> Deploy -> Evaluate
```

### New Component 1: `extract_preferences.py`

- Reads MLflow traces for the current Student version
- Finds questions where both Teacher and Student answered
- Pairs Teacher response (chosen) with Student response (rejected), filtered by Teacher score (only where Student scored below a threshold, e.g. < 7)
- Outputs JSONL with `prompt`, `chosen`, `rejected` fields
- Uploads to MinIO at `s3://mlflow-artifacts/preferences/pref-{version}.jsonl`

### New Component 2: `dpo_finetune.py`

Based on the existing [pipeline/components/finetune.py](pipeline/components/finetune.py), adapted for DPO:

- Loads the SFT-trained model (output of step 3) as the training model
- Loads the same model as `ref_model` (DPO needs a reference model to compute the preference loss)
- Uses `trl.DPOTrainer` with `DPOConfig` instead of `SFTTrainer`
- Same QLoRA setup: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`
- Same LoRA config: `LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])`
- Key DPO params: `beta=0.1` (controls divergence from reference model), `max_length=512`, `max_prompt_length=256`
- Merges LoRA adapters and uploads to MinIO (same as SFT step)

### DPO Training Code Sketch

```python
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

# Load SFT model (from previous step) with QLoRA
model = AutoModelForCausalLM.from_pretrained(sft_model_path, quantization_config=bnb_config)

dpo_config = DPOConfig(
    output_dir="/tmp/dpo-checkpoints",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,       # lower than SFT
    beta=0.1,                 # DPO temperature
    max_length=512,
    max_prompt_length=256,
    fp16=True,
    logging_steps=1,
)

peft_config = LoraConfig(r=16, lora_alpha=32, ...)

trainer = DPOTrainer(
    model=model,
    ref_model=None,           # when using peft, ref_model can be None (uses base model)
    args=dpo_config,
    train_dataset=pref_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
)
trainer.train()
```

## Minimum Viable Approach (Recommended Start)

1. **Collect preference data** by running the Gradio app — ask the same questions to both Teacher and Student, let the Teacher grade the Student
2. **Build `extract_preferences.py`** to pull Teacher-vs-Student pairs from MLflow where Student scored below 7
3. **Build `dpo_finetune.py`** using `DPOTrainer` with QLoRA on the SFT-trained model
4. **Add both as pipeline steps** between SFT and Deploy
5. **Evaluate** — compare Student-SFT-only vs Student-SFT+DPO scores

## Key Considerations

- **Data volume**: DPO needs fewer examples than SFT (50-200 preference pairs can make a difference), but the pairs must be high quality
- **Beta parameter**: Start with `beta=0.1` (default). Lower values (0.05) allow more divergence from the reference model; higher values (0.5) keep the model closer to SFT
- **Learning rate**: Use a lower LR for DPO than SFT (5e-5 vs 2e-4) since DPO is a refinement step
- **ref_model**: When using PEFT/LoRA with DPOTrainer, pass `ref_model=None` — trl automatically uses the base (non-LoRA) weights as the reference
- **GPU memory**: DPO with QLoRA fits on a single T4 (same as current SFT), but with smaller batch sizes (2 instead of 4)

