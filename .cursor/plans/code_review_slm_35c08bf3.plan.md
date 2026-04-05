---
name: Code Review SLM
overview: Evolve the existing SFT+DPO distillation pipeline into a specialized Code Review SLM. Use the ronantakizawa/github-codereview dataset (167K triplets) enriched by the Teacher LLM to train a 1.5B code model that reviews diffs, flags issues, suggests fixes, and categorizes by severity -- all running on-premise on existing infrastructure. Start as a code reviewer, extend to agentic later.
todos:
  - id: download-dataset
    content: Download ronantakizawa/github-codereview (167K triplets), filter for Go/Python/TS/YAML/Dockerfile/Shell + quality >= 0.3, convert to chat JSONL, upload to MinIO
    status: pending
  - id: enrich-teacher
    content: Build scripts/enrich_reviews_with_teacher.py -- send top 10-15K diffs to Ollama Teacher, generate structured JSON reviews (issue, severity, category, suggestion)
    status: pending
  - id: test-base-model
    content: Deploy Qwen2.5-Coder-1.5B-Instruct on KServe+vLLM, verify it loads on T4, test baseline code review quality on sample diffs
    status: pending
  - id: update-pipeline
    content: Update BASE_MODEL_ID in pipeline.py, adapt extract_gold.py for code review data, adapt evaluate.py with 15 code review test diffs
    status: pending
  - id: eval-benchmarks
    content: Create data/code_review_eval.json with 15 hand-crafted test diffs covering bug/security/reliability/performance/style/K8s/clean-code
    status: pending
  - id: first-sft-run
    content: Run first SFT training cycle on code review data through the pipeline, measure Teacher-graded review quality
    status: pending
  - id: dpo-run
    content: Run SFT+DPO cycle -- extract preferences where Teacher review beats Student review on same diffs, fine-tune with DPO
    status: pending
  - id: gradio-ui
    content: Add 'Code Review' tab to Gradio app -- diff input, language selector, structured review output with Teacher grading
    status: pending
  - id: iterate-domain
    content: Add domain-specific diffs (Kubernetes YAML, OpenShift, CI/CD, Dockerfile) to training data, run additional flywheel cycles
    status: pending
isProject: false
---

# Phase 3: Code Review SLM

## Progression

```
Phase 1: SFT-only Q&A SLM (5.4/10)
Phase 2: SFT+DPO Kubeflow domain SLM (8.13/10)
Phase 3: Code Review SLM  <-- YOU ARE HERE
Phase 4: Agentic Code Review (GitHub PR integration, IDE plugin)  <-- future
```

The entire existing pipeline (7-step KFP, Ollama teacher, KServe+vLLM, MinIO, MLflow, TrainJob CRD) is reused. The changes are: **new base model, new training data, new eval benchmarks**.

---

## Step 1: Base Model -- Switch to Code-Specialized 1.5B

Current `unsloth/Llama-3.2-1B-Instruct` is a general text model. For code review, start with a model pre-trained on code.

**Primary choice: `Qwen/Qwen2.5-Coder-1.5B-Instruct`**

- 1.5B params, fits on single T4 with QLoRA (same as today)
- Trained on 5.5 trillion tokens of source code
- Covers Go, Python, TypeScript, Rust, Java, C++, YAML, and 90+ languages
- Native chat template (ChatML) with instruction following
- Apache 2.0 license

**What changes:**

- `BASE_MODEL_ID` in [pipeline/pipeline.py](pipeline/pipeline.py) line 34: `"unsloth/Llama-3.2-1B-Instruct"` becomes `"Qwen/Qwen2.5-Coder-1.5B-Instruct"`
- [pipeline/training/finetune_job.py](pipeline/training/finetune_job.py): Update chat template handling (Qwen uses ChatML `<|im_start|>` tokens instead of Llama's `<|begin_of_text|>`)
- The QLoRA config (r=16, 4-bit NF4) stays identical

**Optional Teacher upgrade:** Switch Ollama teacher from `llama3.1:8b-instruct` to `qwen2.5-coder:7b-instruct` for better code review quality in generated training data.

---

## Step 2: Training Data -- Three Sources

### Source A: github-codereview Dataset (primary, 167K examples)

The [ronantakizawa/github-codereview](https://huggingface.co/datasets/ronantakizawa/github-codereview) dataset is ideal:

- **167K positive triplets**: `before_code` + `reviewer_comment` + `after_code` from 725 top GitHub repos
- **51K negative examples**: Clean code labeled "No issues found" (teaches the model when NOT to comment)
- **37 languages** including Go, Python, TypeScript, Rust
- Pre-categorized: `comment_type` (suggestion, bug, refactor, style, security, performance, question, nitpick)
- Quality scored: `quality_score` (0.0-1.0)

**Filtering strategy for domain focus:**

```python
from datasets import load_dataset

ds = load_dataset("ronantakizawa/github-codereview", split="train")

# Filter for target languages and quality
domain_reviews = ds.filter(lambda x:
    x["language"] in ["Go", "Python", "TypeScript", "YAML", "Dockerfile", "Shell"] and
    x["quality_score"] >= 0.3
)
# Expected: ~40-60K examples after filtering
```

**Training format** -- convert each triplet into a chat conversation:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert code reviewer. Given a code diff, identify issues, explain why they matter, suggest fixes, and classify by severity (low/medium/high) and category (bug/security/reliability/performance/style/refactor)."
    },
    {
      "role": "user",
      "content": "Review the following code diff:\n\nFile: {file_path}\nLanguage: {language}\n\n

```diff\n{diff_context}\n

```\n\nCode before:\n

```{language}\n{before_code}\n

```"
    },
    {
      "role": "assistant",
      "content": "{structured_review}"
    }
  ]
}
```

For **negative examples** (clean code), the assistant response is:

```json
{"issues": [], "summary": "No issues found. The code looks clean."}
```

This teaches the model **when to stay silent** -- critical for a code reviewer that should not be noisy.

### Source B: Teacher-Enriched Reviews (distillation step)

Raw human review comments are often terse ("nit: rename this", "LGTM", "+1"). The Teacher LLM enriches them into structured, detailed reviews.

New script: `scripts/generate_code_review_gold.py`

For each training example from Source A, prompt the Teacher (Ollama 8B):

```
You are a senior code reviewer. You are given a code diff and a human reviewer's comment.
Expand the review into a structured analysis:

Code diff:
{diff_context}

Human reviewer said: "{reviewer_comment}"

Provide your review in this JSON format:
{
  "issues": [
    {
      "description": "Clear explanation of the problem",
      "why_it_matters": "Impact on reliability/security/performance",
      "suggestion": "Concrete fix or improvement",
      "severity": "low|medium|high",
      "category": "bug|security|reliability|performance|style|refactor|maintainability"
    }
  ],
  "summary": "One-line summary of the review"
}
```

This gives a **much richer supervision signal** than raw human comments. The SLM learns to produce structured, actionable reviews.

**Volume:** Process 10-15K high-quality examples through the Teacher (takes ~4-6 hours on Ollama 8B). Combined with the raw dataset, total training set is ~50K+ examples.

### Source C: CROP Dataset (supplementary, Gerrit-style reviews)

The [CROP](https://github.com/crop-repo/crop) dataset provides 50K code reviews from Eclipse and Couchbase projects (Java/C++ heavy). Useful as supplementary data for enterprise-style review patterns. Lower priority than Source A/B but adds diversity.

---

## Step 3: Pipeline Adaptation

The existing 7-step pipeline maps directly:

```
PHASE 2 (current)                    PHASE 3 (code review)
------------------------------       ------------------------------
1. Resolve Version                   1. Resolve Version  (unchanged)
2. Extract Gold Data (Q&A)           2. Extract Review Data (diffs + reviews)
3. SFT Fine-Tune (QLoRA)            3. SFT Fine-Tune (QLoRA, new base model)
4. Extract Preferences               4. Extract Preferences (Teacher vs Student reviews)
5. DPO Fine-Tune                     5. DPO Fine-Tune  (unchanged logic)
6. Deploy (KServe)                   6. Deploy (KServe)  (unchanged)
7. Evaluate (knowledge Q&A)          7. Evaluate (code review benchmarks)
```

### Changes to `pipeline/components/extract_gold.py`

Instead of reading teacher Q&A interactions from MinIO, read the pre-processed code review dataset:

- Download `github-codereview` filtered subset from MinIO (`s3://mlflow-artifacts/code-review-data/`)
- Merge with Teacher-enriched reviews
- Format as chat conversations in JSONL
- Upload combined training data

### Changes to `pipeline/training/finetune_job.py`

- Add `TRAINING_MODE=code_review` (or reuse `sft` mode since the format is the same chat JSONL)
- Update `BASE_MODEL_ID` to Qwen2.5-Coder-1.5B
- The Qwen tokenizer handles ChatML format natively -- the `tokenizer.apply_chat_template()` call works without changes
- QLoRA config stays the same (r=16, 4-bit NF4, target modules: q/k/v/o projections)

### Changes to `pipeline/components/extract_preferences.py`

Adapt for code review preference pairs:

- Give both Teacher and Student the **same code diff** to review
- Teacher grades both reviews on: accuracy, actionability, correct severity, correct category
- Where the Teacher's review is better, create a preference pair: `(diff, teacher_review, student_review)`
- This teaches the SLM to prefer **detailed, actionable reviews** over vague ones

### Changes to `pipeline/components/evaluate.py`

Replace the 15 Kubeflow knowledge questions with **15 code review test cases**:

```python
TEST_DIFFS = [
    # Bug detection
    {"diff": "- if err != nil { return }\n+ if err != nil { return err }",
     "language": "Go", "expected_category": "bug"},
    # Security
    {"diff": "- password := os.Getenv('DB_PASS')\n+ password := flag.String('password', '', '')",
     "language": "Go", "expected_category": "security"},
    # K8s best practices
    {"diff": "+ containers:\n+   - name: app\n+     image: myapp:latest",
     "language": "YAML", "expected_category": "reliability"},
    # Missing resource limits
    {"diff": "apiVersion: apps/v1\nkind: Deployment\nspec:\n  template:\n    spec:\n      containers:\n      - name: web\n        image: nginx",
     "language": "YAML", "expected_category": "reliability"},
    # Python error handling
    {"diff": "- except:\n-     pass\n+ except Exception as e:\n+     logger.error(f'Failed: {e}')",
     "language": "Python", "expected_category": "reliability"},
    # Clean code (should return "no issues")
    {"diff": "func healthCheck(w http.ResponseWriter, r *http.Request) {\n    w.WriteHeader(http.StatusOK)\n    w.Write([]byte(`ok`))\n}",
     "language": "Go", "expected_category": "none"},
    # ... 9 more covering: Dockerfile, CI/CD, OpenShift, concurrency, SQL injection, etc.
]
```

**Evaluation metrics:**

- **Review quality** (1-10, Teacher-graded): Does the review correctly identify the issue?
- **Category accuracy**: Did the SLM assign the right category?
- **Severity accuracy**: Did it get the severity right?
- **False positive rate**: Does it flag clean code as having issues?
- **Structured output compliance**: Does the JSON parse correctly?

---

## Step 4: Data Preparation Scripts

### New: `scripts/prepare_code_review_data.py`

```python
# 1. Download and filter github-codereview
# 2. Filter for target languages (Go, Python, TS, YAML, Dockerfile, Shell)
# 3. Filter quality_score >= 0.3
# 4. Convert to chat format (system + user + assistant messages)
# 5. Include negative examples (clean code -> "no issues found")
# 6. Upload to MinIO: s3://mlflow-artifacts/code-review-data/train.jsonl
```

### New: `scripts/enrich_reviews_with_teacher.py`

```python
# 1. Take top 10-15K examples by quality_score
# 2. For each: send diff + human comment to Ollama Teacher
# 3. Teacher generates structured JSON review
# 4. Validate JSON parses correctly
# 5. Save enriched dataset to MinIO
```

### Updated: `data/code_review_eval.json`

15 hand-crafted code review test cases with expected categories, severities, and reference reviews for evaluation.

---

## Step 5: Gradio UI -- Code Review Mode

Adapt the existing `app.py` Gradio interface:

**New "Code Review" tab:**

- Text area for pasting a code diff (or file upload)
- Language dropdown (Go / Python / YAML / Dockerfile / General)
- "Review" button
- Output panel showing:
  - Structured review (issues, severity, category, suggestions)
  - Teacher grade (1-10) of the review quality
  - Side-by-side: Student review vs Teacher review

**Keep existing "Chat" tab** for general coding questions.

This demonstrates the SLM can do **focused code review** better than a general-purpose model.

---

## Step 6: Serving -- Same Infrastructure

No infrastructure changes:

```
Engineer --> Gradio UI (paste diff) --> KServe (vLLM, Qwen2.5-Coder-1.5B) --> Structured Review
                                             |
                                        Ollama Teacher (grades review quality)
                                             |
                                        MLflow (logs quality scores)
```

The model is served via the same KServe InferenceService, same vLLM runtime, same MinIO model storage. Just a different model binary.

---

## Step 7: Future -- Agentic Extension (Phase 4 roadmap, not built now)

Once the Code Review SLM works well in the Gradio UI, the next phase adds:

1. **GitHub PR webhook** -- watches for new PRs, auto-extracts diff, sends to SLM, posts review comments back
2. **Static analysis fusion** -- runs `golangci-lint`, `pylint`, `kubeconform` alongside the SLM, combines results
3. **IDE plugin** -- VS Code / Cursor extension that sends diffs to the SLM API and shows inline review comments
4. **Multi-file context** -- feeds the SLM additional files from the repo for deeper understanding

This is Phase 4. Phase 3 focuses on getting the SLM to produce high-quality reviews first.

---

## Expected Results


| Metric                                | Baseline (Qwen2.5-Coder-1.5B, no fine-tuning) | After SFT (50K reviews) | After SFT+DPO |
| ------------------------------------- | --------------------------------------------- | ----------------------- | ------------- |
| Review quality (Teacher-graded, 1-10) | ~4-5                                          | ~6-7                    | ~7-8          |
| Category accuracy                     | ~40%                                          | ~70%                    | ~80%+         |
| False positive rate                   | ~30%                                          | ~15%                    | ~10%          |
| Structured JSON compliance            | ~50%                                          | ~90%                    | ~95%          |


The same flywheel effect from Phase 2 (5.4 to 8.13) is expected to apply here.

---

## Implementation Order

Sequential -- each step builds on the previous:

1. **Download and filter** `github-codereview` dataset, upload to MinIO
2. **Enrich** top 10-15K examples with Teacher LLM structured reviews
3. **Test base model** -- deploy Qwen2.5-Coder-1.5B on KServe, verify it loads and serves
4. **Update pipeline config** -- new `BASE_MODEL_ID`, new extract/eval components
5. **Run first SFT training cycle** on code review data
6. **Run SFT+DPO cycle** with preference extraction on code review tasks
7. **Evaluate** -- compare Student vs Teacher on 15 test diffs
8. **Build Gradio code review tab** for demo
9. **Iterate** -- add more domain-specific diffs (K8s, OpenShift, CI/CD) to training data

