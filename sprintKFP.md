# KFP Integration Sprint

## Step 0 — Verify Data Science Pipelines is Enabled

```bash
# Check DSC component
oc get datasciencecluster default-dsc -o jsonpath='{.spec.components.datasciencepipelines}'

# Check pipeline pods
oc get pods -A | grep pipeline
```

If `datasciencepipelines` shows `managementState: Removed`, flip it to `Managed`.  
If already `Managed`, you still need a **DataSciencePipelinesApplication** (DSPA) CR in your namespace to get a pipeline server.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    KFP Pipeline: distillation-flywheel                  │
│                                                                         │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐        │
│   │ Extract  │───▶│ Fine-    │───▶│ Deploy   │───▶│ Evaluate │        │
│   │ Gold     │    │ Tune     │    │ Model    │    │          │        │
│   │ Data     │    │ (QLoRA)  │    │ (KServe) │    │          │        │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘        │
│     CPU pod        GPU pod         CPU pod         CPU pod             │
│     ~30s           ~2min           ~3min           ~1min               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4 Pipeline Components

### Component 1 — Extract Gold Data (lightweight, CPU)

- Reuses `gold_extractor.py` logic
- Pulls Teacher traces from MLflow
- Outputs `train.jsonl` to MinIO
- Base image: `python:3.11-slim` + `mlflow boto3`

### Component 2 — Fine-Tune (GPU)

- Reuses `finetune.py` logic
- Reads gold data from MinIO, runs QLoRA, uploads merged model
- Base image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` + `peft trl etc`
- Needs `nvidia.com/gpu: 1`

### Component 3 — Deploy Model (lightweight, CPU)

- Patches the KServe InferenceService `storageUri` to the new model path
- Waits for rollout to complete (pod becomes Ready)
- Base image: image with `kubernetes` Python client, or just use `oc` CLI image

### Component 4 — Evaluate (lightweight, CPU)

- Sends test questions to the new Student
- Has the 70B Teacher grade each response
- Outputs a score comparison (before vs after)
- Base image: `python:3.11-slim` + `requests groq`

---

## Files to Create

```
AgentBuilder/
├── pipeline/
│   ├── pipeline.py              # KFP pipeline definition (all 4 components)
│   ├── components/
│   │   ├── extract_gold.py      # Component 1 (wraps gold_extractor.py logic)
│   │   ├── finetune.py          # Component 2 (wraps finetune.py logic)
│   │   ├── deploy_model.py      # Component 3 (patches KServe ISVC)
│   │   └── evaluate.py          # Component 4 (Teacher grades Student)
│   └── rhoai/
│       └── 07-dspa.yaml         # DataSciencePipelinesApplication CR
```

---

## pipeline.py Sketch

```python
from kfp import dsl

@dsl.component(base_image="python:3.11-slim",
               packages_to_install=["mlflow", "boto3"])
def extract_gold_data(mlflow_uri: str, s3_endpoint: str,
                      output_path: str) -> str:
    # ... gold_extractor.py logic ...
    return output_path

@dsl.component(base_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
               packages_to_install=["peft", "trl", "bitsandbytes", ...])
def finetune(gold_data_path: str, model_output_path: str,
             s3_endpoint: str) -> str:
    # ... finetune.py logic ...
    return model_output_path

@dsl.component(base_image="python:3.11-slim",
               packages_to_install=["kubernetes"])
def deploy_model(model_s3_path: str, isvc_name: str,
                 namespace: str) -> str:
    # Patches InferenceService storageUri
    # Waits for Ready=True
    return student_url

@dsl.component(base_image="python:3.11-slim",
               packages_to_install=["requests"])
def evaluate(student_url: str, test_questions: list) -> dict:
    # Sends questions to Student
    # Has Teacher grade each
    # Returns {before_avg: 4.2, after_avg: 7.8}
    return scores

@dsl.pipeline(name="distillation-flywheel")
def distillation_pipeline(model_version: str = "v3"):
    gold = extract_gold_data(...)
    model = finetune(gold_data_path=gold.output, ...)
    deploy = deploy_model(model_s3_path=model.output, ...)
    scores = evaluate(student_url=deploy.output, ...)
```

---

## Key Decisions

| Decision | Options |
|---|---|
| **Component packaging** | Lightweight (`@dsl.component` with `packages_to_install`) vs Custom Docker images. Lightweight is faster to build, custom is faster to run. |
| **GPU for fine-tune** | The finetune component needs `resources` with `nvidia.com/gpu: 1`. KFP supports this via `set_gpu_limit(1)`. |
| **Pipeline trigger** | Manual (click Run in dashboard) vs Recurring (cron schedule) vs Event-driven (trigger when new traces appear in MLflow) |
| **Model versioning** | Pass `model_version` as a pipeline parameter (e.g. "v3", "v4") so each run produces a new version |
| **DSPA backend** | The pipeline server needs its own database. RHOAI DSPA can use the existing MinIO for artifact storage. |

---

## Implementation Order

- [ ] **1. Verify/enable DSP** — check if pipeline server exists, create DSPA if needed
- [ ] **2. Create `pipeline.py`** with all 4 components using `@dsl.component` (lightweight approach)
- [ ] **3. Handle the GPU component** — finetune step needs special resource config and `anyuid` SCC
- [ ] **4. Compile and upload** — `kfp.compiler.Compiler().compile(pipeline, 'pipeline.yaml')` then upload via RHOAI dashboard or `kfp.Client()`
- [ ] **5. Test run** — trigger from dashboard, watch the 4 steps execute in sequence
- [ ] **6. Update docs** — update `sprint.md` and `docs/architecture.md`

---

## Effort Estimate

| Task | Time |
|---|---|
| Verify/enable DSP + DSPA | 30 min |
| Write 4 components + pipeline.py | 1–2 hours |
| Debug GPU permissions, image pulls | 1 hour |
| Test end-to-end | 30 min |
| **Total** | **~3–4 hours** |

The bulk of the logic already exists in `gold_extractor.py` and `finetune.py` — it's mostly wrapping them into KFP components and wiring the pipeline together.
