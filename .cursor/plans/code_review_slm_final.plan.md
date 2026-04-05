---
name: Code Review SLM
overview: Evolve the existing SFT+DPO distillation pipeline into a specialized Code Review SLM -- a small model that reviews code diffs and generates reviewer-quality comments (bugs, suggestions, severity, category) for Go/Python/Kubernetes/OpenShift code. Leverages the 167K+ ronantakizawa/github-codereview dataset as raw material, enriches it with Teacher-generated expert reviews, trains via the existing pipeline, and provides a simple API for integration into GitHub PR workflows and IDE chat.
todos:
  - id: download-dataset
    content: Download ronantakizawa/github-codereview (167K examples), filter for Go/Python/YAML/Dockerfile, quality_score >= 0.4, non-negative, comment_length > 50
    status: pending
  - id: teacher-enrich
    content: Build scripts/prepare_code_review_data.py -- batch-process filtered diffs through Ollama Teacher to produce structured reviews (issue, severity, category, suggestion)
    status: pending
  - id: format-upload
    content: Format enriched reviews as training JSONL matching existing pipeline format, include negative examples, upload to MinIO s3://mlflow-artifacts/synthetic/code-review/
    status: pending
  - id: eval-holdout
    content: Create 50-diff evaluation benchmark covering bug, security, performance, style, reliability categories across Go/Python/K8s
    status: pending
  - id: switch-base-model
    content: Update BASE_MODEL_ID to Qwen/Qwen2.5-Coder-1.5B-Instruct in pipeline.py, increase model_max_length to 1024 in finetune_job.py
    status: pending
  - id: update-eval
    content: Replace TEST_QUESTIONS in pipeline.py with 15 code review diffs, adapt evaluate.py scoring for review quality
    status: pending
  - id: first-training
    content: Run full 7-step SFT+DPO pipeline with code review data, measure baseline vs fine-tuned review quality
    status: pending
  - id: gradio-ui
    content: Adapt Gradio app for code review -- diff input, structured review output with syntax highlighting
    status: pending
  - id: review-api
    content: Build POST /v1/review API wrapper around KServe endpoint for GitHub/IDE integration
    status: pending
  - id: github-action
    content: Create GitHub Action that calls review API on PR open/update and posts inline review comments
    status: pending
isProject: false
---

# Phase 3: Code Review SLM

## What You Already Have vs. What Changes

Your existing 7-step pipeline does 90% of the work. The pivot is **domain** (knowledge Q&A to code review) and **data format** (question-answer to diff-review).


| Layer         | Phase 2 (current)                                     | Phase 3 (Code Review SLM)                                     |
| ------------- | ----------------------------------------------------- | ------------------------------------------------------------- |
| Base model    | `unsloth/Llama-3.2-1B-Instruct`                       | `Qwen/Qwen2.5-Coder-1.5B-Instruct`                            |
| Training data | 827 Kubeflow Q&A pairs                                | 15-20K code review pairs (diff -> review comment)             |
| Data source   | Synthetic generation from docs                        | `ronantakizawa/github-codereview` (167K) + Teacher enrichment |
| Input format  | `{"instruction": "What is KServe?", "output": "..."}` | `{"instruction": "<diff>", "output": "<structured review>"}`  |
| DPO signal    | Teacher scores higher on same question                | Teacher writes better review than Student on same diff        |
| Evaluation    | 15 knowledge questions                                | 50 held-out diffs graded for review quality                   |
| Serving       | KServe + vLLM (reuse)                                 | Same                                                          |
| Pipeline      | 7-step SFT+DPO (reuse)                                | Same 7 steps, new data + eval                                 |


---

## The Data Pipeline (the hard part)

### Available Datasets


| Dataset                                                                                            | Size                                        | What It Has                                                                                                  | How We Use It                                                                             |
| -------------------------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| [ronantakizawa/github-codereview](https://huggingface.co/datasets/ronantakizawa/github-codereview) | **167K+ triplets**, 37 languages, 725 repos | `before_code`, `reviewer_comment`, `after_code`, `diff_context`, `language`, `comment_type`, `quality_score` | **Primary source**. Filter for Go/Python/YAML, use as raw material for Teacher enrichment |
| [Tomo-Melb/CodeReviewQA](https://huggingface.co/datasets/Tomo-Melb/CodeReviewQA)                   | 900 curated examples, 9 languages           | Code refinement tasks, change localization, solution identification                                          | **Evaluation benchmark**. Use as held-out test set                                        |
| [CROP](https://github.com/crop-repo/crop)                                                          | Code Review Open Platform                   | Structured code review data                                                                                  | **Supplementary** if more data is needed                                                  |


The `github-codereview` dataset is the goldmine. It has exactly the triplet structure you need: `(before_code, reviewer_comment, after_code)` plus metadata like `comment_type` (bug, security, performance, style, suggestion) and `quality_score`.

---

## Step 1: Data Preparation Script

New script: `scripts/prepare_code_review_data.py`

### 1a. Filter the raw dataset

From the 167K examples, filter to your target domain:

```python
from datasets import load_dataset

ds = load_dataset("ronantakizawa/github-codereview", split="train")

domain_reviews = ds.filter(lambda x:
    x["language"] in ["Go", "Python", "YAML", "Dockerfile", "Shell", "HCL"]
    and not x["is_negative"]       # positive examples only for now
    and x["quality_score"] >= 0.4  # decent quality
    and x["comment_length"] > 50   # substantive comments, not "LGTM"
)
```

Expected yield: ~20-30K examples from the Go/Python/YAML subset.

### 1b. Teacher Enrichment (the key step)

The raw human reviewer comments are noisy -- short, informal, sometimes just "nit". The Teacher model (Ollama 8B or a code-specialized 8B) rewrites each into a **structured expert review**.

For each raw example, send to Teacher:

```text
You are a senior code reviewer. Given the following code diff and the original reviewer's comment, write a structured review.

Diff:
{diff_context}

Before:
{before_code}

Original reviewer comment: {reviewer_comment}

Produce a JSON review with these fields:
- issue: Clear description of the problem
- why_it_matters: Why this matters (reliability, security, performance, maintainability)
- suggestion: Concrete code fix or improvement
- severity: critical / high / medium / low / nitpick
- category: bug / security / performance / reliability / style / refactor / maintainability
```

This transforms noisy human comments into clean, structured training data -- the same Teacher enrichment pattern from Phase 2 but for code reviews.

### 1c. Format for SFT Training

Each enriched example becomes a training record in the same JSONL format the existing pipeline expects:

```json
{
  "instruction": "Review the following code diff and identify any issues:\n\nFile: pkg/controller/reconciler.go\nLanguage: Go\n\n

```diff\n- if err != nil {\n-     return err\n- }\n+ if err != nil {\n+     log.Error(err)\n+ }\n

```",
  "output": "**Issue:** Error is logged but not returned to the caller.\n**Why it matters:** The caller will assume success and proceed with invalid state, which can cause cascading failures.\n**Suggestion:** Return the error after logging: `log.Error(err); return err`\n**Severity:** high\n**Category:** reliability",
  "text": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nReview the following code diff...<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n**Issue:** Error is logged but not returned..."
}
```

### 1d. Add Negative Examples

The dataset includes 51K examples where reviewers had **no comment** (clean code). Include a portion of these as negative training examples:

```json
{
  "instruction": "Review the following code diff and identify any issues:\n\n

```diff\n...\n

```",
  "output": "No issues found. The code change is clean and follows best practices."
}
```

This teaches the SLM **when not to flag** -- critical for a useful reviewer that doesn't cry wolf.

### 1e. Target Data Volume


| Category                          | Count       | Source                               |
| --------------------------------- | ----------- | ------------------------------------ |
| Go reviews (Teacher-enriched)     | ~8,000      | github-codereview filtered + Teacher |
| Python reviews (Teacher-enriched) | ~6,000      | github-codereview filtered + Teacher |
| YAML/Dockerfile/K8s reviews       | ~2,000      | github-codereview filtered + Teacher |
| Negative examples (no issues)     | ~4,000      | github-codereview negatives          |
| **Total**                         | **~20,000** |                                      |


Upload to MinIO at `s3://mlflow-artifacts/synthetic/code-review/` -- the existing pipeline picks up anything under `synthetic/`.

---

## Step 2: Switch Base Model

Update [pipeline/pipeline.py](pipeline/pipeline.py) line 34:

```python
BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
```

**Why Qwen2.5-Coder-1.5B-Instruct:**

- Trained on 5.5T code tokens (vs general text for Llama-3.2-1B)
- Native understanding of diffs, code structure, multiple languages
- 1.5B params -- still fits on a single T4 with QLoRA (same as today)
- Apache 2.0 license
- ChatML chat template (works with the existing SFTTrainer)

The only code change in `finetune_job.py` is increasing `model_max_length` from 512 to **1024** -- code diffs need more context than Q&A:

```python
tokenizer.model_max_length = 1024  # was 512
```

And in the DPO config, increase `max_length` from 256 to 512:

```python
max_length=512,       # was 256
max_prompt_length=256, # was 128
```

---

## Step 3: Teacher Model (Optional Upgrade)

The current `llama3.1:8b-instruct-q4_K_M` in Ollama works but is a general model. For code review distillation, consider pulling a code-specialized teacher:

```bash
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
```

This stays on the same Ollama deployment, same cluster resources. Update the pipeline parameter:

```python
teacher_model: str = "qwen2.5-coder:7b-instruct-q4_K_M"
```

---

## Step 4: Update Evaluation

Replace the 15 Kubeflow knowledge questions in [pipeline/pipeline.py](pipeline/pipeline.py) (lines 43-65) with **15 code review tasks**.

Each task is a real diff that the Student must review. The Teacher grades the Student's review on a 1-10 scale for: correctness, completeness, actionability, and false positive avoidance.

```python
TEST_QUESTIONS = [
    # Bug detection
    """Review this Go code diff:
```diff
- if err != nil { return ctrl.Result{}, err }
+ if err != nil { log.Error(err) }
```""",
    # Security
    """Review this Kubernetes YAML diff:
```diff
+ securityContext:
+   privileged: true
+   runAsUser: 0
```""",
    # Resource management
    """Review this Deployment YAML:
```diff
+ containers:
+ - name: app
+   image: myapp:latest
+   # no resources specified
```""",
    # Error handling
    """Review this Python diff:
```diff
- except Exception as e:
-     logger.error(f"Failed: {e}")
-     raise
+ except Exception:
+     pass
```""",
    # ... 11 more covering: Go patterns, Python anti-patterns,
    #     Dockerfile issues, CI/CD misconfigs, RBAC problems,
    #     race conditions, resource leaks, API misuse
]
```

The evaluation component grades the Student review against the Teacher review using the same LLM-as-Judge scoring that already works.

---

## Step 5: Preference Extraction for DPO

Adapt [pipeline/components/extract_preferences.py](pipeline/components/extract_preferences.py) for code review:

**How it works today:** Both Teacher and Student answer the same knowledge question; Teacher grades both; preference pairs are created where Teacher scores higher.

**How it works for code review:** Both Teacher and Student review the same code diff. The Teacher then grades both reviews on:

- Did it find the real issue?
- Is the explanation accurate?
- Is the suggestion actionable?
- Did it avoid false positives?

Preference pair:

```json
{
  "prompt": "Review this diff:\n

```diff\n- if err != nil { return err }\n+ if err != nil { log.Error(err) }\n

```",
  "chosen": "[Teacher's review: Issue: Error swallowed. Severity: high. Fix: return err after logging]",
  "rejected": "[Student's review: Code looks fine, logging was added]"
}
```

This teaches the SLM to prefer precise, actionable reviews over vague or wrong ones.

---

## Step 6: Chat UI for Code Review

Adapt the existing Gradio app (`app.py`) for code review interaction:

**New input modes:**

- Paste a code diff directly
- Paste a code file and ask "review this"
- Paste an error + code and ask "what's wrong?"

**New output format:** Structured review with syntax-highlighted suggestions:

```
Issue: Error is logged but not returned to the caller
Severity: high | Category: reliability

Why it matters:
The caller assumes success and continues with potentially invalid state.

Suggestion:
```go
if err != nil {
    log.Error(err, "reconciliation failed")
    return ctrl.Result{}, err
}
```

```

The Gradio app already supports markdown rendering, which handles code blocks natively.

---

## Step 7: API for GitHub/IDE Integration

Once the SLM is trained and serving on KServe, expose a simple review API:

```

POST /v1/review
{
  "diff": "...",
  "file_path": "pkg/controller/reconciler.go",
  "language": "go"
}

Response:
{
  "reviews": [
    {
      "issue": "Error swallowed silently",
      "severity": "high",
      "category": "reliability",
      "suggestion": "Return the error after logging",
      "line": 42
    }
  ]
}

```

This can be called from:
- A **GitHub Action** that runs on PR open/update and posts review comments
- A **CLI tool** that reviews staged changes before commit
- An **IDE extension** that reviews the current file on demand

These integrations come after the core SLM is working -- they are wrappers around the same API.

---

## Pipeline Execution: What Runs Unchanged

```

Resolve Version --> Extract Gold Data --> SFT Fine-Tune --> Extract Preferences --> DPO Fine-Tune --> Deploy --> Evaluate
       |                  |                    |                    |                    |              |           |
    same code       reads from new         same QLoRA          same pattern         same DPO       same KServe  new review
                    code-review data       on Qwen2.5-Coder    (review vs review)   training       hot-swap     benchmarks
                    in MinIO                                                                                     

```

The 7-step pipeline DAG in `pipeline/pipeline.py` stays structurally identical. Changes are config (base model ID, eval questions) and data (code review pairs instead of Q&A).

---

## Implementation Order

### Track A: Data (can start immediately)

1. **Download and filter** `ronantakizawa/github-codereview` for Go/Python/YAML (script)
2. **Teacher enrichment** -- batch process filtered diffs through Ollama to produce structured reviews
3. **Format as training JSONL** and upload to MinIO
4. **Split out 50 held-out diffs** as evaluation benchmark

### Track B: Pipeline Config (after data is ready)

5. **Switch base model** to `Qwen2.5-Coder-1.5B-Instruct` in pipeline config
6. **Increase context length** in `finetune_job.py` (512 -> 1024 for SFT, 256 -> 512 for DPO)
7. **Replace eval questions** with code review diffs
8. **Optionally upgrade Teacher** to `qwen2.5-coder:7b` in Ollama

### Track C: First Training Run

9. **Run the full 7-step pipeline** with code review data
10. **Compare scores** -- baseline Qwen2.5-Coder-1.5B vs fine-tuned on your review data
11. **Iterate** -- analyze where reviews are weak, generate more data for those categories

### Track D: Integration (after model proves useful)

12. **Adapt Gradio UI** for diff input and structured review output
13. **Build review API** endpoint
14. **GitHub Action** wrapper that calls the API on PR events
15. **CLI tool** for local pre-commit review
```

