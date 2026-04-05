---
name: Kubeflow Domain Fine-Tuning
overview: Fine-tune the 1B student on Kubeflow ecosystem knowledge by collecting content from upstream projects, docs, release notes, and videos, then using the 70B teacher (Groq) to generate grounded Q&A training data from those sources. New scripts collect raw content, chunk it, generate context-grounded Q&A pairs, filter for quality, and upload to MinIO as synthetic gold data that the existing pipeline consumes.
todos:
  - id: source-collector
    content: Build scripts/kubeflow_sources.py — crawls Kubeflow docs site, GitHub READMEs, release notes, and YouTube transcripts into raw JSONL on disk
    status: pending
  - id: chunker
    content: Build scripts/chunk_sources.py — splits raw documents into ~800-token chunks with metadata (source_url, project, doc_type)
    status: pending
  - id: grounded-qagen
    content: Build scripts/generate_kubeflow_qa.py — feeds each chunk as context to Groq teacher, generates 3-5 Q&A pairs per chunk, writes gold JSONL to MinIO
    status: pending
  - id: quality-filter
    content: "Add quality filtering: dedup by question similarity, drop low-quality pairs, validate answer references the source content"
    status: pending
  - id: topic-question-bank
    content: Build scripts/kubeflow_question_bank.py — generates standalone questions per project/topic for the teacher to answer (no source context needed)
    status: pending
  - id: eval-questions
    content: Add Kubeflow-specific test questions to the pipeline evaluate step so we can measure domain knowledge improvement
    status: pending
  - id: run-and-iterate
    content: Run the full generation pipeline, review sample outputs, tune prompts, run SFT training, evaluate
    status: pending
isProject: false
---

# Fine-Tuning the Student on Kubeflow Ecosystem Knowledge

## The Problem

The 1B student model has no deep knowledge of the Kubeflow ecosystem — upstream projects, architecture, APIs, best practices, release changes, etc. We want it to become a Kubeflow domain expert.

## The Core Idea: Context-Grounded Teacher Generation

The teacher (70B via Groq) has general knowledge about Kubeflow, but it's not enough on its own — it may hallucinate details, miss recent changes, or lack depth on specific sub-projects. The fix:

1. **Collect raw content** from authoritative Kubeflow sources (docs, READMEs, release notes, video transcripts)
2. **Chunk** that content into digestible pieces (~800 tokens each)
3. **Feed each chunk as context to the teacher** and ask it to generate Q&A pairs *grounded in that specific content*
4. **The teacher reads, then teaches** — answers are faithful to the source material, not just parametric knowledge

This is essentially building a synthetic textbook from primary sources, authored by the teacher.

## Sources to Collect

### 1. Kubeflow Documentation (kubeflow.org)

The main docs site covers installation, components, tutorials, and guides.


| Section            | URL Pattern                                   | Priority |
| ------------------ | --------------------------------------------- | -------- |
| Getting Started    | `kubeflow.org/docs/started/`                  | High     |
| Kubeflow Pipelines | `kubeflow.org/docs/components/pipelines/`     | High     |
| Training Operator  | `kubeflow.org/docs/components/training/`      | High     |
| KServe             | `kserve.github.io/kserve/`                    | High     |
| Katib              | `kubeflow.org/docs/components/katib/`         | Medium   |
| Notebooks          | `kubeflow.org/docs/components/notebooks/`     | Medium   |
| Multi-Tenancy      | `kubeflow.org/docs/components/multi-tenancy/` | Medium   |


### 2. Upstream GitHub Repositories


| Repo                         | What to Extract                                                |
| ---------------------------- | -------------------------------------------------------------- |
| `kubeflow/kubeflow`          | Main README, architecture docs                                 |
| `kubeflow/pipelines`         | README, SDK docs, `CHANGELOG.md`, component authoring guides   |
| `kubeflow/training-operator` | README, CRD specs, PyTorchJob/TrainJob docs                    |
| `kubeflow/katib`             | README, algorithm docs, trial templates                        |
| `kubeflow/model-registry`    | README, API docs                                               |
| `kubeflow/manifests`         | Installation docs, overlay explanations                        |
| `kserve/kserve`              | README, InferenceService spec, model formats, transformer docs |
| `kubeflow/mpi-operator`      | README, MPIJob spec                                            |
| `kubeflow/notebooks`         | README, workspace docs                                         |


What to pull from each repo:

- `README.md` at root
- `docs/` directory (all `.md` files)
- `CHANGELOG.md` or `RELEASES.md`
- GitHub Releases (title + body for last ~10 releases)
- Key example YAMLs (CRD samples, pipeline examples)

### 3. Release Notes

- GitHub Releases API for each repo above
- Kubeflow blog posts about releases (`blog.kubeflow.org`)
- Focus on Kubeflow 1.8, 1.9, and latest

### 4. Video Transcripts

- KubeCon talks about Kubeflow (YouTube)
- Kubeflow community meeting recordings
- Red Hat / OpenShift AI demos involving Kubeflow components

YouTube transcripts can be fetched via `youtube-transcript-api` (Python library) given video IDs.

Key video topics to search for:

- "Kubeflow Pipelines v2"
- "KServe model serving"
- "Kubeflow Training Operator"
- "MLOps on Kubernetes with Kubeflow"
- "Katib hyperparameter tuning"

## Architecture: Three-Script Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    scripts/kubeflow_sources.py                      │
│                                                                     │
│  Crawl docs site ──┐                                                │
│  GitHub READMEs ───┤──→ raw_sources.jsonl (one doc per line)        │
│  Release notes ────┤    {url, project, doc_type, title, content}    │
│  YouTube transcripts┘                                               │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    scripts/chunk_sources.py                          │
│                                                                     │
│  Read raw_sources.jsonl                                             │
│  Split each doc into ~800-token chunks with overlap                 │
│  Write chunks.jsonl                                                 │
│    {chunk_id, source_url, project, doc_type, title, chunk_text}     │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                scripts/generate_kubeflow_qa.py                      │
│                                                                     │
│  For each chunk:                                                    │
│    → Send chunk as context to Groq teacher                          │
│    → Teacher generates 3-5 Q&A pairs grounded in the chunk          │
│    → Format as {instruction, output, text}                          │
│    → Upload to MinIO under synthetic-kubeflow/ prefix               │
└─────────────────────────────────────────────────────────────────────┘
```

### Script 1: `scripts/kubeflow_sources.py` — Source Collector

```python
# Pseudo-structure
KUBEFLOW_DOCS_SITEMAP = "https://www.kubeflow.org/sitemap.xml"
GITHUB_REPOS = [
    "kubeflow/kubeflow",
    "kubeflow/pipelines",
    "kubeflow/training-operator",
    "kubeflow/katib",
    "kubeflow/model-registry",
    "kubeflow/manifests",
    "kubeflow/notebooks",
    "kserve/kserve",
]
YOUTUBE_VIDEO_IDS = [...]  # curated list

# For docs: parse sitemap, fetch each page, extract text (strip HTML)
# For GitHub: use GitHub API to list .md files in docs/, fetch raw content
# For releases: GitHub Releases API, last 10 per repo
# For videos: youtube-transcript-api to get captions
```

Outputs `data/raw_sources.jsonl` — each line:

```json
{
  "url": "https://www.kubeflow.org/docs/components/pipelines/overview/",
  "project": "kubeflow-pipelines",
  "doc_type": "docs",
  "title": "Pipelines Overview",
  "content": "Kubeflow Pipelines is a platform for building and deploying..."
}
```

### Script 2: `scripts/chunk_sources.py` — Chunker

Splits each document into ~800-token overlapping chunks (200-token overlap for context continuity). Uses `tiktoken` for accurate token counting.

Outputs `data/chunks.jsonl`:

```json
{
  "chunk_id": "kubeflow-pipelines-docs-overview-001",
  "source_url": "https://...",
  "project": "kubeflow-pipelines",
  "doc_type": "docs",
  "title": "Pipelines Overview",
  "chunk_text": "Kubeflow Pipelines is a platform for building..."
}
```

### Script 3: `scripts/generate_kubeflow_qa.py` — Grounded Q&A Generator

The key script. For each chunk, asks the teacher to generate Q&A pairs that are grounded in the source content.

**System prompt:**

```
You are a training-data generator for a Kubeflow AI assistant.
Given a passage from official Kubeflow documentation, generate {n} diverse
question-answer pairs. Requirements:
- Questions should be things a Kubeflow user or operator would actually ask
- Answers must be grounded in the provided passage (don't invent facts)
- Mix question types: conceptual, how-to, troubleshooting, comparison
- Answers should be 2-5 sentences, accurate, and instructional
- Include relevant CLI commands, YAML snippets, or API details when applicable

Return ONLY a JSON array:
[{"question": "...", "answer": "..."}, ...]
```

**User prompt:**

```
Source: {project} ({doc_type})
Title: {title}

---
{chunk_text}
---

Generate {n} question-answer pairs from this passage.
```

Same upload pattern as `generate_synthetic_gold.py` — writes to MinIO under `synthetic-kubeflow/date=YYYY-MM-DD/`.

## Supplementary: Question Bank Script

`scripts/kubeflow_question_bank.py` generates **standalone questions** organized by Kubeflow sub-project and topic. These don't need source chunks — they're the kind of questions a user would ask, and the teacher answers from its parametric knowledge.

```python
KUBEFLOW_QUESTION_TOPICS = {
    "kubeflow-pipelines": [
        "KFP v2 SDK and component authoring",
        "Pipeline compilation and execution",
        "Artifact passing between components",
        "Pipeline parameters and conditions",
        "Caching and retry strategies",
        "KFP vs Argo Workflows differences",
    ],
    "training-operator": [
        "PyTorchJob spec and multi-node training",
        "TrainJob API and Kubeflow training SDK",
        "Gang scheduling with volcano or scheduler-plugins",
        "TFJob for TensorFlow distributed training",
        "Fine-tuning LLMs with Training Operator",
    ],
    "kserve": [
        "InferenceService spec and predictor/transformer/explainer",
        "Model formats: sklearn, xgboost, pytorch, tensorflow, vLLM",
        "Canary rollouts and traffic splitting",
        "Autoscaling and scale-to-zero",
        "Custom serving runtimes",
        "KServe with model registry",
    ],
    "katib": [
        "Experiment CRD and trial templates",
        "Search algorithms: random, grid, bayesian, TPE, ENAS",
        "Early stopping strategies",
        "Metrics collection: stdout, file, TensorFlow events",
        "Katib with Training Operator integration",
    ],
    "general-kubeflow": [
        "Kubeflow architecture and components overview",
        "Multi-tenancy with profiles and namespaces",
        "Kubeflow on different platforms (OpenShift, EKS, GKE)",
        "Kubeflow manifests and kustomize installation",
        "Kubeflow vs managed MLOps platforms",
        "Kubeflow roadmap and community governance",
    ],
}
```

For each topic, ask the teacher to generate 10 questions, then answer each one. This gives breadth where the doc-grounded approach gives depth.

## Integration with Existing Pipeline

The generated data lands in MinIO under `synthetic-kubeflow/` prefix. Two options to integrate:

**Option A (simplest):** Add a second synthetic prefix to the pipeline's extract_gold step.

In `pipeline/pipeline.py`, the extract step already reads `synthetic/`. We add `synthetic-kubeflow/` as an additional prefix. The `extract_gold_data` component gets a new parameter `extra_synthetic_prefixes: list[str]` and iterates over all of them.

**Option B (no pipeline changes):** Upload the Kubeflow Q&A data directly under the existing `synthetic/` prefix. The extract step already merges everything there. This works today with zero code changes.

**Recommended: Option B** for the first iteration. Just upload under `synthetic/` with a sub-prefix like `synthetic/kubeflow/`. The existing pipeline picks it up automatically.

## Evaluation

Add Kubeflow-specific questions to `TEST_QUESTIONS` in `pipeline/pipeline.py`:

```python
KUBEFLOW_TEST_QUESTIONS = [
    "What is a KFP component and how do you create one with the Python SDK?",
    "Explain the difference between PyTorchJob and TrainJob in the Training Operator.",
    "How does KServe handle canary deployments for ML models?",
    "What search algorithms does Katib support for hyperparameter tuning?",
    "How do you set up multi-tenancy in Kubeflow using profiles?",
]
```

Run the pipeline before and after adding the Kubeflow training data. The eval scores on these questions directly measure domain knowledge improvement.

## Expected Data Volume


| Source                     | Est. Documents | Est. Chunks       | Est. Q&A Pairs   |
| -------------------------- | -------------- | ----------------- | ---------------- |
| Kubeflow docs site         | ~150 pages     | ~600              | ~2,400           |
| GitHub READMEs + docs      | ~80 files      | ~320              | ~1,280           |
| Release notes              | ~60 releases   | ~120              | ~480             |
| YouTube transcripts        | ~20 videos     | ~200              | ~800             |
| Question bank (parametric) | —              | —                 | ~500             |
| **Total**                  | **~310 docs**  | **~1,240 chunks** | **~5,460 pairs** |


5,000+ pairs is substantial for QLoRA fine-tuning on a 1B model. Combined with the existing general-purpose synthetic data, this gives a strong Kubeflow-specialized assistant.

## Execution Order

1. Run `kubeflow_sources.py` to collect raw content → `data/raw_sources.jsonl`
2. Run `chunk_sources.py` to split into chunks → `data/chunks.jsonl`
3. Run `generate_kubeflow_qa.py` to generate grounded Q&A → MinIO `synthetic/kubeflow/`
4. Run `kubeflow_question_bank.py` for breadth Q&A → MinIO `synthetic/kubeflow-qbank/`
5. Run the existing distillation pipeline — extract step merges everything, SFT trains on the combined data
6. Evaluate with Kubeflow-specific test questions

## Dependencies

```
requests          # HTTP fetching for docs/GitHub
beautifulsoup4    # HTML parsing for docs site
tiktoken          # Token counting for chunking
youtube-transcript-api  # YouTube captions
```

All other deps (boto3, groq) are already in `requirements.txt`.

## Key Considerations

- **Rate limits**: Groq has rate limits. With ~1,240 chunks × 1 call each, use `--delay 1.5` and expect ~30 min for the full generation run.
- **Cost**: Groq API is free-tier friendly for this volume. If hitting limits, batch across multiple sessions.
- **Freshness**: Re-run the source collector periodically to pick up new releases and doc updates. Date-partition the output so old data isn't lost.
- **Quality over quantity**: The grounded approach prevents hallucination. The teacher can only teach what's in the chunk — it can't make up Kubeflow features that don't exist.
- **Deduplication**: Similar chunks produce similar Q&A. The quality filter step deduplicates by question similarity (simple: exact substring match; better: embedding cosine similarity).

