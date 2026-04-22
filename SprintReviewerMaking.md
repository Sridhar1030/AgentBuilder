# Sprint: Code Review SLM — Phase 3

> **Goal:** Evolve the SFT+DPO distillation pipeline from Kubeflow Q&A into a Code Review SLM that reviews Go/Python/K8s diffs and produces structured reviewer-quality comments.
>
> **Scope:** Tracks A–C only. Track D (API, GitHub Action, Gradio UI) deferred until model benchmarks prove useful.
>
> **Data Strategy:** Dual-source — mine real PR reviews from Kubeflow/K8s repos (primary, domain-specific) + supplement with HuggingFace generic dataset (volume).
>
> **Branch:** `agentic-trainer-implementation`
>
> **Base Plan:** `.cursor/plans/code_review_slm_final.plan.md`
> **Q&A Decisions:** `.cursor/plans/QnA.txt`

### Progress Overview (Updated Apr 6, 2026)

| Track | Task | Status |
| ----- | ---- | ------ |
| A0 | GitHub Token & Dependencies | **DONE** |
| A1 | Mine PR Reviews (GitHub) | **DONE** — 200 examples from `kubeflow/trainer` (Go:181 Python:4 YAML:15) |
| A2 | Teacher Enrichment (GitHub) | **DONE** — 200/200 enriched locally via `qwen2.5-coder:7b` in 56 min |
| A3 | HF Supplement (Go/Python/YAML) | **DONE** — 7,996 examples (Go:2522 Python:5474), quality>=0.43, stratified |
| A4 | Teacher Enrichment (HF) | **SKIPPED** — raw HF comments already decent quality; DPO will refine |
| A5 | Merge & Format Training Data | **DONE** — 9,835 total (8,196 positive + 1,639 negative) in `data/code_review_train.jsonl` |
| A6 | Eval Diff Bank | **DONE** — 50 held-out diffs in `data/diff-bank.json` (30 domain + 20 generic) |
| A7 | 15 In-Pipeline TEST_QUESTIONS | **DONE** — embedded in pipeline.py |
| A8 | **Upload to MinIO** | **DONE** — 21.2 MB training + 98 KB eval in `s3://mlflow-artifacts/synthetic/code-review/` |
| B1 | Switch Base Model | **DONE** |
| B2 | Pipeline Config Constants | **DONE** |
| B3 | Replace TEST_QUESTIONS | **DONE** |
| B4 | Increase Context Lengths | **DONE** |
| B5 | Grading Prompt (evaluate.py) | **DONE** |
| B6 | Grading Prompt (extract_preferences.py) | **DONE** |
| B7 | MLflow Experiment Name | **DONE** |
| B8 | Training Image Rebuild | **DONE** — `v0.7.0` built (linux/amd64) + pushed to `quay.io/rh-ee-srpillai/distillation-trainer:v0.7.0` (adds `mlflow-skinny` for training metrics logging) |
| B8.5 | Update Image Tags | **DONE** — `finetune.py` + `dpo_finetune.py` → `v0.7.0` |
| B9 | Compile Pipeline | **DONE** — `code_review_pipeline.yaml` re-compiled with `v0.7.0` image |
| B10 | KServe ISVC for Code Review | **DONE** — `code-review-llm` InferenceService created in `sridharproject` |
| B11 | Add Structured Logging to All KFP Components | **DONE** — all 7 components have clear banner logs, timestamps, human-readable elapsed times |
| B12 | MLflow Training Metrics | **DONE** — `finetune_job.py` logs SFT/DPO training loss to `CodeReview-Training` experiment |
| C1 | Pre-Flight Checks | PENDING |
| C2 | Execute Pipeline Run | PENDING |
| C3 | Analyze Results | PENDING |

**Completed:** 24/27 &nbsp; | &nbsp; **Next:** C1 pre-flight checks → C2 pipeline run → C3 analyze results

### Data Summary (Final Numbers)

| File | Records | Source |
| ---- | ------- | ------ |
| `data/kubeflow_reviews.json` | 200 | GitHub mining from `kubeflow/trainer` — Go:181, Python:4, YAML:15 |
| `data/hf_supplement.json` | 7,996 | HuggingFace `ronantakizawa/github-codereview` — Go:2522, Python:5474 |
| `data/code_review_train.jsonl` | **9,835** | Merged: 8,196 positive + 1,639 negative, ChatML formatted |
| `data/diff-bank.json` | 50 | Held-out eval diffs (30 domain, 20 generic) — not used in training |

**Category breakdown of training data:**
- suggestion: 4,441 | bug: 1,194 | question: 914 | negative: 1,639 | refactor: 549
- performance: 422 | security: 392 | style: 117 | nitpick: 93 | other: 74

### What's Left (in order)

1. **C1 — Pre-flight checks** — verify teacher model, data in MinIO, image accessible, ISVCs healthy
2. **C2 — Upload `code_review_pipeline.yaml`** to RHOAI Dashboard and trigger a pipeline run
3. **C3 — Monitor & analyze** — watch each step, review MLflow `CodeReview-Eval-Hub` metrics

---

## Track A: Data Preparation

### A0. GitHub Token & Dependencies Setup

- **Status:** `[x]` DONE
- **Subtasks:**
  - [x] A0.1 — GitHub token available via `gh auth token`
  - [x] A0.2 — Dependencies installed: `requests`, `datasets`, `transformers`
  - [ ] A0.3 — Verify Ollama teacher endpoint is reachable (cluster or port-forward) — *deferred to A2*

---

### Stream 1: GitHub Mining Pipeline (Primary — Domain-Specific)

### A1. Mine PR Reviews from Target Repos

- **Status:** `[x]` DONE — full production run complete
- **Script:** `scripts/mine_github_reviews.py`
- **Output:** `data/kubeflow_reviews.json` — **200 examples** from `kubeflow/trainer`
- **Actual yield:** 135 merged PRs scanned → 442 raw comments → 200 accepted (Go:181, Python:4, YAML:15)
- **Quality score:** min=0.30, mean=0.63, max=0.90
- **Comment types:** question:87, other:69, suggestion:20, bug:7, security:6, style:4, reliability:3, performance:2, refactoring:2
- **Schema:** Exact 20-column match with `ronantakizawa/github-codereview`
- **Target Repos:**
  ```
  REPOS = [
      "kubeflow/trainer",           # 2.1K stars, Go-heavy K8s ML training operator
      "ray-project/kuberay",        # Go, K8s operator for Ray
      "kserve/kserve",              # Go/Python, model serving on K8s
      "openshift/machine-config-operator",  # Go, OpenShift operator
      "tektoncd/pipeline",          # Go, CI/CD on K8s
  ]
  ```
- **Subtasks:**
  - A1.1 — Write `get_merged_prs(repo, max_pages)` — paginate GitHub API `/repos/{repo}/pulls?state=closed`, filter `merged_at != null`
    - Fetch up to 500 merged PRs per repo (5 pages × 100 per page)
    - Rate limit: 5,000 requests/hour with token; add `time.sleep(0.75)` between calls
  - A1.2 — Write `get_review_comments(repo, pr_number)` — fetch `/repos/{repo}/pulls/{pr_number}/comments`
    - Each comment has: `body`, `path`, `diff_hunk`, `line`, `side`, `user.login`
  - A1.3 — Write `get_pr_diff_files(repo, pr_number)` — fetch `/repos/{repo}/pulls/{pr_number}/files`
    - For negative examples: files changed in PR but with zero review comments = clean code
  - A1.4 — Filter comments:
    - Minimum `len(body) >= 20` characters
    - Exclude bot usernames: `["dependabot", "codecov", "github-actions", "renovate", "copilot"]`
    - Exclude noise patterns (case-insensitive): `["lgtm", "looks good", "+1", "nit:", "thanks", "ack", "approved"]`
  - A1.5 — Filter files: keep only Go/Python/YAML
    ```python
    VALID_EXTENSIONS = [".go", ".py", ".yaml", ".yml"]
    ```
  - A1.6 — Build raw examples list:
    ```python
    {
        "repo": "kubeflow/trainer",
        "pr_number": 1234,
        "pr_title": "Fix reconciler error handling",
        "file_path": "pkg/controller/job_controller.go",
        "language": "Go",          # derived from extension
        "diff_hunk": "- if err != nil { return nil }\n+ if err != nil { log.Error(err) }",
        "human_comment": "The error is logged but never returned to the caller.",
        "reviewer_username": "gaocegege",
        "author_username": "johnDoe"
    }
    ```
  - A1.7 — Build negative examples from clean files (changed in PR, zero comments):
    - Extract first ~50 lines of the diff for each clean file
    - Label as negative: `human_comment = "No issues found."`
  - A1.8 — Save raw mined data to `data/github_mined_raw.jsonl`
  - A1.9 — Log stats: total PRs scanned, total comments found, post-filter count per repo, per language
- **Expected yield:**

  | Repo                              | Est. merged PRs | Est. useful comments      |
  | --------------------------------- | --------------- | ------------------------- |
  | kubeflow/trainer                  | ~500            | ~300–800                  |
  | ray-project/kuberay               | ~500            | ~200–500                  |
  | kserve/kserve                     | ~500            | ~300–700                  |
  | openshift/machine-config-operator | ~500            | ~200–500                  |
  | tektoncd/pipeline                 | ~500            | ~300–700                  |
  | **Total**                         | **~2,500 PRs**  | **~1,300–3,200 comments** |


### A2. Teacher Enrichment (GitHub-Mined Data)

- **Status:** `[x]` DONE — 200/200 enriched locally via `qwen2.5-coder:7b` in 56 minutes
- **Script:** `scripts/enrich_with_teacher.py` (new script with checkpoint/resume, parallel workers)
- **Teacher:** `qwen2.5-coder:7b-instruct-q4_K_M` via local Ollama (also pulled on cluster)
- **Output:** `data/kubeflow_reviews_enriched.json` — structured reviews (Issue/Why/Severity/Category/Suggestion)
- **Subtasks:**
  - A2.1 — Pull teacher model: done locally + on cluster
  - A2.2 — Enrichment function with prompt:
    ```
    You are a senior Kubernetes and Go code reviewer.
    Given this diff and the original reviewer's comment, create a structured review.

    File: {file_path}
    Language: {language}

    Diff:
    {diff_hunk}

    Original reviewer comment: {human_comment}

    Produce a JSON review with these fields:
    - issue: Clear description of the problem
    - why_it_matters: Why this matters (reliability, security, performance, maintainability)
    - suggestion: Concrete code fix or improvement
    - severity: critical / high / medium / low / nitpick
    - category: bug / security / performance / reliability / style / refactor / kubernetes
    ```
    - Note: `kubernetes` category is new — for K8s-specific issues (RBAC, resource limits, probes, etc.)
  - A2.3 — Implement 8-worker parallelism (ThreadPoolExecutor)
  - A2.4 — Checkpoint/resume: write incrementally to `data/github_mined_enriched.jsonl`, track by `repo+pr_number+file_path+line`
  - A2.5 — Run enrichment (~1–2 hours for ~2K examples with 8 workers)
  - A2.6 — Validate: spot-check 20–30 enriched examples for quality

---

### Stream 2: HuggingFace Supplement (Volume — Generic)

### A3. Download & Filter HuggingFace Dataset

- **Status:** `[x]` DONE — full production run complete
- **Script:** `scripts/prepare_hf_supplement.py`
- **Run:** `python3 scripts/prepare_hf_supplement.py --max-examples 8000 -o data/hf_supplement.json`
- **Source:** `[ronantakizawa/github-codereview](https://huggingface.co/datasets/ronantakizawa/github-codereview)` — 334,323 rows
- **Actual yield:** 334,323 total → 70,257 passed filters → **7,996 stratified sample**
- **Filter stats:** filtered_language:211,944, filtered_quality:26,962, filtered_negative:23,509, filtered_short:1,651
- **By language:** Go:2,522 | Python:5,474 (no YAML passed quality threshold — YAML comes from GitHub mining)
- **By comment_type:** suggestion:4,421, bug:1,187, question:827, refactor:549, performance:420, security:386, style:113, nitpick:93
- **Quality score:** min=0.43, avg=0.70, max=1.00
- **Subtasks:**
  - A3.1 — Download and filter:
    - `language in ["Go", "Python", "YAML"]`
    - `not is_negative` (positive examples only for supplement)
    - `quality_score >= 0.4`
    - `comment_length > 50`
  - A3.2 — Exclude repos already mined in Stream 1 (by `repo_name` field) to avoid duplicates
  - A3.3 — Stratified sampling (target 8,000) by `comment_type`, prioritizing `security`, `performance`, `bug`
  - A3.4 — Save to `data/hf_supplement.json`
  - A3.5 — Log stats: counts per language, per comment_type

### A4. Teacher Enrichment (HF Supplement)

- **Status:** `[~]` SKIPPED — raw HF comments are already decent quality from real reviewers
- **Rationale:** At ~24s/example, enriching 7,996 HF examples would take ~53 hours. The raw comments are
  already substantive (min 50 chars, quality >= 0.43). The pipeline's DPO step will further refine output format.
  Teacher enrichment was applied to the 200 domain-specific GitHub-mined examples where it matters most.

---

### Merge & Format

### A5. Merge Both Streams into Training JSONL

- **Status:** `[x]` DONE — full production run complete
- **Script:** `scripts/format_training_data.py`
- **Run:** `python3 scripts/format_training_data.py` (needs both `data/kubeflow_reviews.json` + `data/hf_supplement.json`)
- **Output:** `data/code_review_train.jsonl` — **9,835 training examples** (8,196 positive + 1,639 negative)
- **By source:** hf_supplement:7,996, hf_supplement_negative:1,588, github_mined:200, github_mined_negative:51
- **By language:** Go:3,223, Python:6,592, YAML:20
- **Also builds:** `data/diff-bank.json` (50 held-out eval diffs: 30 domain + 20 generic) and supports `--upload` to MinIO
- **Subtasks:**
  - A5.1 — Load Qwen2.5-Coder-1.5B-Instruct tokenizer
  - A5.2 — Format each enriched example (both streams) into training JSONL:
    - `instruction`: `"Review the following code diff and identify any issues:\n\nFile: {file_path}\nLanguage: {language}\n\n```diff\n{diff_hunk}\n```"`
    - `output`: Structured review from Teacher enrichment (Issue / Why it matters / Suggestion / Severity / Category)
    - `text`: ChatML via `tokenizer.apply_chat_template()` → `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>`
  - A5.3 — Add negative examples (~2,000–3,000 total):
    - ~1,000 from GitHub mining (clean PR files with no review comments)
    - ~1,500–2,000 from HF dataset (`is_negative=True`, `language in ["Go", "Python", "YAML"]`)
    - Output: `"No issues found. The code change is clean and follows best practices."`
  - A5.4 — Tag each record with `source` field for analysis: `"github_mined"` or `"hf_supplement"`
  - A5.5 — Shuffle and merge into single `data/code_review_train.jsonl`
  - A5.6 — Upload to MinIO: `s3://mlflow-artifacts/synthetic/code-review/`
  - A5.7 — Verify upload: confirm file size and line count via boto3

### A6. Create Evaluation Diff Bank

- **Status:** `[x]` DONE — built into `scripts/format_training_data.py`
- **Output:** `diff-bank.json` → `s3://mlflow-artifacts/synthetic/code-review/diff-bank.json`
- **Subtasks:**
  - A6.1 — Select 50 held-out diffs (NOT used in training):
    - ~30 from GitHub-mined data (highest quality K8s/Go/Python examples)
    - ~20 from HF supplement (diversity in categories)
    - Cover: bug, security, performance, style, refactor, reliability, kubernetes
  - A6.2 — Format as `{"all_questions": ["Review the following code diff...", ...]}`
  - A6.3 — Upload to MinIO

### A7. Curate 15 In-Pipeline TEST_QUESTIONS

- **Status:** `[x]` DONE — 15 real code review diffs from `kubeflow/trainer` embedded in `pipeline.py`
- **Target:** `pipeline/pipeline.py` lines 43–65
- **Subtasks:**
  - A7.1 — Select 15 from the 50 held-out diffs (curated subset):
    - 3 bug detection (Go error handling, Python exception, nil pointer)
    - 2 security (K8s privileged container, hardcoded secret)
    - 2 performance (missing resource limits, inefficient loop)
    - 2 kubernetes-specific (missing probes, RBAC over-permission, no resource requests)
    - 2 reliability (error swallowing, missing nil check)
    - 2 style/refactor (naming, dead code)
    - 2 clean code (negative — should return "No issues found")
  - A7.2 — Format each as a multi-line Python string for `TEST_QUESTIONS` list

### A8. Upload Data to MinIO

- **Status:** `[x]` DONE — uploaded Apr 5, 2026
- **Target:** `s3://mlflow-artifacts/synthetic/code-review/`
- **Files uploaded:**
  - `code_review_train.jsonl` — **21,203,627 bytes** (9,835 training examples)
  - `diff-bank.json` — **98,692 bytes** (50 eval diffs)
- **Verified:** Listed objects in MinIO via boto3, sizes and timestamps confirmed

**Actual Data Volume (Production Runs):**


| Source                           | Role                    | Actual Count |
| -------------------------------- | ----------------------- | ------------ |
| GitHub-mined (kubeflow/trainer)  | Primary domain data     | 200          |
| HF supplement Go/Python          | Volume + diversity      | 7,996        |
| Negative examples (both sources) | False positive training | 1,639        |
| **Total training data**          |                         | **9,835**    |
| Held-out eval diffs              | Evaluation              | 50           |
| In-pipeline TEST_QUESTIONS       | Quick regression        | 15           |


**New Files Summary:**


| File                               | Purpose                                                  |
| ---------------------------------- | -------------------------------------------------------- |
| `scripts/mine_github_reviews.py`   | Fetch PRs, extract comments, filter, enrich, save        |
| `scripts/prepare_hf_supplement.py` | Download HF dataset, filter, enrich, save                |
| `scripts/format_training_data.py`  | Merge both streams, format ChatML JSONL, upload to MinIO |


---

## Track B: Pipeline Configuration

### B1. Switch Base Model

- **Status:** `[x]` DONE
- **File:** `pipeline/pipeline.py` line 34
- **Subtasks:**
  - B1.1 — Change `BASE_MODEL_ID`
    - Old: `"unsloth/Llama-3.2-1B-Instruct"`
    - New: `"Qwen/Qwen2.5-Coder-1.5B-Instruct"`

### B2. Update Pipeline Config Constants

- **Status:** `[x]` DONE
- **File:** `pipeline/pipeline.py` lines 37–41, 77, 92
- **Subtasks:**
  - B2.1 — Change `TEACHER_PREFIX` (line 37)
    - Old: `"teacher-interactions/"`
    - New: `"code-review-interactions/"`
  - B2.2 — Change `SYNTHETIC_PREFIX` (line 39)
    - Old: `"synthetic/"`
    - New: `"synthetic/code-review/"`
  - B2.3 — Change `CURSOR_KEY` (line 40)
    - Old: `"teacher-interactions/.cursor.json"`
    - New: `"code-review-interactions/.cursor.json"`
  - B2.4 — Change `QUESTION_BANK_S3` (line 41)
    - Old: `"s3://mlflow-artifacts/synthetic/kubeflow-qbank/questions.json"`
    - New: `"s3://mlflow-artifacts/synthetic/code-review/diff-bank.json"`
  - B2.5 — Change `teacher_model` default (line 77)
    - Old: `"llama3.1:8b-instruct-q4_K_M"`
    - New: `"qwen2.5-coder:7b-instruct-q4_K_M"`
  - B2.6 — Change `model_prefix` in `resolve_version()` call (line 92)
    - Old: `"student-1b-"`
    - New: `"code-review-1.5b-"`

### B3. Replace TEST_QUESTIONS

- **Status:** `[x]` DONE
- **File:** `pipeline/pipeline.py` lines 43–65
- **Subtasks:**
  - B3.1 — Replace the 15 Kubeflow Q&A strings with the 15 code review diffs curated in A7
  - B3.2 — Verify the list compiles (no unterminated strings, proper escaping of backticks/diffs)

### B4. Increase Context Lengths in finetune_job.py

- **Status:** `[x]` DONE
- **File:** `pipeline/training/finetune_job.py`
- **Subtasks:**
  - B4.1 — SFT `model_max_length` (line 99): `512` → `1024`
  - B4.2 — DPO `model_max_length` (line 270): `512` → `1024`
  - B4.3 — DPO `max_length` in `DPOConfig` (line 300): `256` → `512`
  - B4.4 — DPO `max_prompt_length` in `DPOConfig` (line 301): `128` → `256`

### B5. Update Grading Prompt in evaluate.py

- **Status:** `[x]` DONE
- **File:** `pipeline/components/evaluate.py` lines 61–65
- **Subtasks:**
  - B5.1 — Replace `GRADING_PROMPT` with code review criteria:
    - Issue identification (did it find the real problem, not a hallucinated one?)
    - Technical accuracy (is the explanation correct?)
    - Actionability (is the suggestion concrete and implementable?)
    - Severity accuracy (is the severity rating appropriate?)
    - False positive avoidance (did it avoid flagging non-issues?)
    - Keep output format: `{"score": <number>, "reason": "<brief reason>"}`

### B6. Update Grading Prompt in extract_preferences.py

- **Status:** `[x]` DONE
- **File:** `pipeline/components/extract_preferences.py` lines 122–126
- **Subtasks:**
  - B6.1 — Replace `GRADING_PROMPT` with same code review grading criteria as B5
  - B6.2 — Keep output format: `{"score": <number>, "reason": "<brief reason>"}`

### B7. Update MLflow Experiment Name

- **Status:** `[x]` DONE
- **Files:** `evaluate.py` line 188, `extract_preferences.py` line 148
- **Subtasks:**
  - B7.1 — Change experiment in `evaluate.py` (line 188): `"Distillation-Eval-Hub"` → `"CodeReview-Eval-Hub"`
  - B7.2 — Change experiment in `extract_preferences.py` (line 148): `"Distillation-Eval-Hub"` → `"CodeReview-Eval-Hub"`

### B8. Training Image Rebuild (Conditional)

- **Status:** `[x]` DONE — `v0.6.0` built and pushed Apr 5, 2026
- **File:** `pipeline/training/Dockerfile`
- **Image:** `quay.io/rh-ee-srpillai/distillation-trainer:v0.6.0` (was `v0.5.2`)
- **Subtasks:**
  - B8.1 — Verify locally: `python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct')"`
  - B8.2 — If tiktoken needed: add `tiktoken` to Dockerfile pip install
  - B8.3 — Build: `docker build -t quay.io/rh-ee-srpillai/distillation-trainer:v0.6.0 .`
  - B8.4 — Push: `docker push quay.io/rh-ee-srpillai/distillation-trainer:v0.6.0`
  - B8.5 — Update image tag in `pipeline/components/finetune.py` and `pipeline/components/dpo_finetune.py`

### B9. Compile Pipeline

- **Status:** `[x]` DONE — compiled Apr 5, 2026
- **File:** `pipeline/code_review_pipeline.py` → `pipeline/code_review_pipeline.yaml`
- **Note:** Phase 3 uses a **separate pipeline file** (`code_review_pipeline.py`) to keep Phase 2 (`pipeline.py`) intact for demos
- **Subtasks:**
  - B9.1 — Compile: `cd pipeline && python3 code_review_pipeline.py` → `code_review_pipeline.yaml`
  - B9.2 — Upload to RHOAI Dashboard or via KFP client

### B10. Create KServe ISVC for Code Review

- **Status:** `[x]` DONE — created Apr 5, 2026
- **Resource:** `InferenceService/code-review-llm` in namespace `sridharproject`
- **Note:** Separate from `student-llm` (Phase 2 demo). Pipeline will patch `storageUri` after training.
- **Spec:** vLLM runtime, 1x GPU, 16Gi memory, `student-model-sa` service account

---

## Track C: First Training Run & Benchmarking

### C1. Pre-Flight Checks

- **Status:** `[ ]` NOT STARTED
- **Subtasks:**
  - C1.1 — Confirm `qwen2.5-coder:7b-instruct-q4_K_M` is pulled and responding on Ollama
  - C1.2 — Confirm training data uploaded to `s3://mlflow-artifacts/synthetic/code-review/`
  - C1.3 — Confirm diff-bank.json uploaded to `s3://mlflow-artifacts/synthetic/code-review/diff-bank.json`
  - C1.4 — Confirm training image is accessible (v0.5.2 or v0.6.0)
  - C1.5 — Confirm KServe ISVC `code-review-llm` exists (Phase 3) and `student-llm` is untouched (Phase 2)

### C2. Execute Pipeline Run

- **Status:** `[ ]` NOT STARTED
- **Subtasks:**
  - C2.1 — Trigger pipeline via RHOAI Dashboard or KFP client
  - C2.2 — Monitor Step 0 (Resolve Version) — expect `v1` under `code-review-1.5b-` prefix
  - C2.3 — Monitor Step 1 (Extract Gold) — expect ~8–14K training pairs from synthetic/code-review/
  - C2.4 — Monitor Step 2 (SFT) — QLoRA on Qwen2.5-Coder-1.5B-Instruct, ~3 epochs
  - C2.5 — Monitor Step 3 (Extract Preferences) — Source A likely empty (first run), Source B queries 50 diffs from diff-bank
  - C2.6 — Monitor Step 4 (DPO) — depends on preference pair count; may passthrough SFT if < `min_dpo_pairs`
  - C2.7 — Monitor Step 5 (Deploy) — patches `code-review-llm` ISVC with new model S3 path
  - C2.8 — Monitor Step 6 (Evaluate) — runs 15 code review diffs, grades with new criteria

### C3. Analyze Results

- **Status:** `[ ]` NOT STARTED
- **Subtasks:**
  - C3.1 — Review MLflow experiment `CodeReview-Eval-Hub` for first run metrics
  - C3.2 — Record baseline scores: `student_avg_score`, `teacher_avg_score`, `score_gap`
  - C3.3 — Break down per-question scores by category (bug, security, performance, kubernetes, style, reliability)
  - C3.4 — Identify weak categories (any category averaging below 6/10)
  - C3.5 — Compare performance on GitHub-mined vs HF-sourced eval diffs (is domain data paying off?)

### C4. Iteration Round 1 (if needed)

- **Status:** `[ ]` NOT STARTED
- **Trigger:** Any category averaging below 6/10 after C3
- **Subtasks:**
  - C4.1 — For weak categories: mine more PRs from additional repos or deeper pagination
  - C4.2 — Or pull more targeted examples from HF dataset for weak `comment_type`
  - C4.3 — Run additional Teacher enrichment
  - C4.4 — Append to training JSONL, re-upload to MinIO
  - C4.5 — Re-run pipeline (Steps 1–6)
  - C4.6 — Compare scores: Run 1 vs Run 2 in MLflow

### C5. Iteration Round 2 (if needed)

- **Status:** `[ ]` NOT STARTED
- **Trigger:** Still below 6/10 after C4
- **Subtasks:**
  - C5.1 — Same process as C4 with further data augmentation
  - C5.2 — Consider expanding repo list (e.g., `kubernetes/kubernetes`, `argoproj/argo-workflows`)
  - C5.3 — Final benchmark comparison across all runs

---

## Track D: Integration (DEFERRED — not in this sprint)

> Included here for completeness. Do NOT start until Tracks A–C deliver a model scoring 6+/10 across all categories.

### D1. Adapt Gradio UI — `[ ]` DEFERRED

### D2. Build FastAPI Review API — `[ ]` DEFERRED

### D3. GitHub Action for PR Review — `[ ]` DEFERRED

### D4. CLI Pre-Commit Reviewer — `[ ]` DEFERRED

---

## Quick Reference: All File Changes


| #    | File                                         | Lines   | Change                                                      |
| ---- | -------------------------------------------- | ------- | ----------------------------------------------------------- |
| B1   | `pipeline/pipeline.py`                       | 34      | `BASE_MODEL_ID` → `"Qwen/Qwen2.5-Coder-1.5B-Instruct"`      |
| B2.1 | `pipeline/pipeline.py`                       | 37      | `TEACHER_PREFIX` → `"code-review-interactions/"`            |
| B2.2 | `pipeline/pipeline.py`                       | 39      | `SYNTHETIC_PREFIX` → `"synthetic/code-review/"`             |
| B2.3 | `pipeline/pipeline.py`                       | 40      | `CURSOR_KEY` → `"code-review-interactions/.cursor.json"`    |
| B2.4 | `pipeline/pipeline.py`                       | 41      | `QUESTION_BANK_S3` → `"s3://...code-review/diff-bank.json"` |
| B2.5 | `pipeline/pipeline.py`                       | 77      | `teacher_model` → `"qwen2.5-coder:7b-instruct-q4_K_M"`      |
| B2.6 | `pipeline/pipeline.py`                       | 92      | `model_prefix` → `"code-review-1.5b-"`                      |
| B3   | `pipeline/pipeline.py`                       | 43–65   | Replace `TEST_QUESTIONS` with 15 code review diffs          |
| B4.1 | `pipeline/training/finetune_job.py`          | 99      | `model_max_length` → `1024`                                 |
| B4.2 | `pipeline/training/finetune_job.py`          | 270     | `model_max_length` → `1024`                                 |
| B4.3 | `pipeline/training/finetune_job.py`          | 300     | `max_length` → `512`                                        |
| B4.4 | `pipeline/training/finetune_job.py`          | 301     | `max_prompt_length` → `256`                                 |
| B5   | `pipeline/components/evaluate.py`            | 61–65   | Replace `GRADING_PROMPT` with code review criteria          |
| B6   | `pipeline/components/extract_preferences.py` | 122–126 | Replace `GRADING_PROMPT` with code review criteria          |
| B7.1 | `pipeline/components/evaluate.py`            | 188     | Experiment → `"CodeReview-Eval-Hub"`                        |
| B7.2 | `pipeline/components/extract_preferences.py` | 148     | Experiment → `"CodeReview-Eval-Hub"`                        |
| B8   | `pipeline/training/Dockerfile`               | 3–12    | Add `tiktoken` (conditional)                                |
| B8.5 | `pipeline/components/finetune.py`            | 64      | Image tag → `v0.6.0`                                       |
| B8.5 | `pipeline/components/dpo_finetune.py`        | 64      | Image tag → `v0.6.0`                                       |
| NEW  | `scripts/mine_github_reviews.py`             | —       | GitHub API mining, filtering, enrichment                    |
| NEW  | `scripts/prepare_hf_supplement.py`           | —       | HF dataset download, filter, enrich                         |
| NEW  | `scripts/format_training_data.py`            | —       | Merge streams, ChatML format, upload to MinIO               |
| NEW  | `scripts/enrich_with_teacher.py`             | —       | Teacher enrichment with Ollama (checkpoint/resume)          |
| NEW  | `pipeline/code_review_pipeline.py`           | —       | Phase 3 pipeline definition (separate from Phase 2)         |
| NEW  | `pipeline/code_review_pipeline.yaml`         | —       | Compiled Phase 3 pipeline YAML                              |
| NEW  | `pipeline/pipeline_phase2.py`                | —       | Backup of original Phase 2 pipeline                         |


---

## Definition of Done

- `scripts/mine_github_reviews.py` mines 1,000+ domain-specific review examples from 5 K8s repos
- `scripts/prepare_hf_supplement.py` produces 5,000–8,000 supplementary examples
- `scripts/format_training_data.py` merges both streams into ~8–14K training JSONL
- Training data uploaded to `s3://mlflow-artifacts/synthetic/code-review/`
- 50-diff `diff-bank.json` uploaded to MinIO (30 domain + 20 generic)
- All pipeline config changes applied (B1–B7)
- Training image verified (rebuilt if needed)
- Pipeline compiles cleanly
- At least one full pipeline run completes successfully
- MLflow `CodeReview-Eval-Hub` shows benchmark scores
- All categories average 6+/10 (after up to 2 iteration rounds)
- Baseline vs fine-tuned comparison documented
- Domain-specific performance validated (K8s/Go operator review quality)

