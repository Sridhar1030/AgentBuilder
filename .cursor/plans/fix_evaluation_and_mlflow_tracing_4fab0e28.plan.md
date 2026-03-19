---
name: Fix Evaluation and MLflow Tracing
overview: "Fix three interconnected problems: the messy MLflow traces in app.py (history pollution, grade embedded in reply text, no chartable metrics), the KFP evaluate component that computes scores but never logs them anywhere, and the missing model version context throughout."
todos:
  - id: fix-app-tracing
    content: "app.py: remove history from chat() signature, add mlflow.start_run per student turn with log_metric for teacher_score, add MODEL_VERSION env var"
    status: pending
  - id: fix-evaluate-mlflow
    content: "evaluate.py: add mlflow_tracking_uri and model_version params, add Teacher baseline scoring, log full eval run to MLflow with student_avg_score, teacher_avg_score, score_gap metrics and eval_results.json artifact"
    status: pending
  - id: fix-pipeline-params
    content: "pipeline.py: pass mlflow_tracking_uri and model_version to evaluate task"
    status: pending
  - id: recompile-upload
    content: Recompile distillation_flywheel.yaml and upload new pipeline version to RHOAI
    status: pending
isProject: false
---

# Fix Evaluation and MLflow Tracing

## Current Problems

```mermaid
flowchart TD
    subgraph ui [Gradio UI - app.py]
        A["chat(message, history, model_choice)"]
        B["@mlflow.trace captures history list\n= pollutes every trace with old turns"]
        C["grade embedded in reply text\n= no clean separate field"]
        D["teacher_score as span attribute only\n= not chartable as metric"]
    end
    subgraph kfp [KFP Pipeline - evaluate.py]
        E["avg_score computed"]
        F["printed to stdout only\n= lost after pod terminates"]
        G["no teacher baseline\n= no gap measurement"]
        H["no model_version tag\n= can't track improvement over runs"]
    end
    A --> B
    A --> C
    A --> D
    E --> F
    E --> G
    E --> H
```



## Target State

```mermaid
flowchart LR
    subgraph ui [Gradio UI - app.py]
        U1["chat(message, model_choice)\nno history in signature"]
        U2["mlflow.start_run per student turn\nlog_metric teacher_score"]
        U3["span attrs: question, student_reply,\nteacher_score, teacher_reason, model_version"]
    end
    subgraph kfp [KFP Pipeline - evaluate.py]
        K1["mlflow.start_run pipeline-eval-vN"]
        K2["log_metric student_avg_score\nlog_metric teacher_avg_score"]
        K3["log_dict eval_results.json\nset_tag model_version=vN"]
    end
    subgraph mlflow [MLflow UI]
        M1["Metrics tab: chartable scores\nacross versions and time"]
        M2["Traces tab: clean per-turn detail"]
        M3["Artifacts: full eval JSON per version"]
    end
    ui --> mlflow
    kfp --> mlflow
```



## Changes

### 1. `[app.py](app.py)` — Fix UI tracing (3 changes)

**Problem 1: history pollutes traces.**
`chat()` accepts `history: list` but never uses it in the function body. MLflow captures all args, so every trace shows the full prior conversation.

- Remove `history` from `chat()` signature entirely
- Update the `respond()` wrapper to call `chat(message, model_choice)`

**Problem 2: grade has no separate MLflow field / not chartable.**
Currently `teacher_score` is only a trace span attribute — not a metric. Add `mlflow.start_run()` in the student branch to log it as a proper metric:

```python
with mlflow.start_run(run_name=f"student-turn-{int(time.time())}",
                       tags={"model_version": MODEL_VERSION, "turn_type": "student"}):
    mlflow.log_metric("teacher_score", score)
    mlflow.log_param("question", message[:500])
    mlflow.log_param("teacher_reason", reason)
```

**Problem 3: no model version.**
Add `MODEL_VERSION = os.getenv("MODEL_VERSION", "unknown")` and include it in both span attributes and run tags.

The grade suffix in Gradio chat display stays (`**Grade: 7/10`**) — that's useful UX. It just won't pollute the trace anymore since history is removed.

---

### 2. `[pipeline/components/evaluate.py](pipeline/components/evaluate.py)` — Log to MLflow

Add two new params: `mlflow_tracking_uri: str`, `model_version: str`.

Add `mlflow` to `packages_to_install`.

After computing student scores, also query Teacher on the same questions to get a baseline. Then open a proper MLflow run:

```python
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("Distillation-Eval-Hub")
with mlflow.start_run(run_name=f"pipeline-eval-{model_version}"):
    mlflow.set_tag("model_version", model_version)
    mlflow.set_tag("eval_type", "pipeline_benchmark")
    mlflow.log_metric("student_avg_score", student_avg)
    mlflow.log_metric("teacher_avg_score", teacher_avg)
    mlflow.log_metric("score_gap", teacher_avg - student_avg)
    for i, r in enumerate(results):
        mlflow.log_metric(f"q{i+1}_student_score", r["student_score"])
        mlflow.log_metric(f"q{i+1}_teacher_score", r["teacher_score"])
    mlflow.log_dict({"results": results}, "eval_results.json")
```

This gives you: a chart of `student_avg_score` vs `teacher_avg_score` across v1, v2, v3... in MLflow's Compare Runs view.

---

### 3. `[pipeline/pipeline.py](pipeline/pipeline.py)` — Pass new params to evaluate

`version_task.outputs["version"]` already exists (returns e.g. `"v11"`). Just pass it through:

```python
eval_task = evaluate(
    student_url=deploy_task.output,
    groq_api_key=groq_api_key,
    groq_model=groq_model,
    test_questions=TEST_QUESTIONS,
    mlflow_tracking_uri=MLFLOW_URI,          # new
    model_version=version_task.outputs["version"],  # new
)
```

## What You'll See in MLflow After This

- **Metrics tab (Runs view):** Chart showing `student_avg_score` and `teacher_avg_score` across pipeline runs, with `score_gap` shrinking as distillation improves
- **Traces tab:** Clean single-turn traces — `question`, `student_reply`, `teacher_score`, `teacher_reason`, `model_version` — no history pollution
- **Artifacts:** `eval_results.json` per pipeline run with full per-question breakdown
- **Tags:** Filter by `model_version=v11` or `eval_type=pipeline_benchmark`

## Files Changed

- `[app.py](app.py)`
- `[pipeline/components/evaluate.py](pipeline/components/evaluate.py)`
- `[pipeline/pipeline.py](pipeline/pipeline.py)`

