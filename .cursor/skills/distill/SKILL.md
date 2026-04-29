---
name: distill
description: >-
  Run, manage, and bootstrap an SFT+DPO distillation pipeline on OpenShift AI.
  Use when the user runs /distill, /distill.prerequisites, /distill.setup,
  /distill.run, /distill.status, /distill.eval, /distill.baseline,
  /distill.deploy, /distill.feedback, /distill.scores, or /distill.portforward.
---

# Distillation Pipeline Skill

This skill automates an iterative SFT + DPO distillation pipeline on OpenShift AI.
It is **config-driven** -- all cluster, domain, and training parameters are read from
[`distill.config.yaml`](distill.config.yaml) at the repo root. No code changes are
needed to switch domains, clusters, or models.

## Configuration

Before running any command, read [`distill.config.yaml`](distill.config.yaml) and
resolve all `{{namespace}}` template references using the `cluster.namespace` value.

Key config sections:
- `cluster` -- namespace, S3 endpoints, MLflow URI, bucket names
- `teacher` -- Ollama URL, model name, system prompt
- `student` -- HuggingFace base model, model prefix, KServe InferenceService name
- `domain` -- training data paths, grading prompt, test questions file
- `training` -- SFT epochs, DPO epochs, DPO beta, static bank sample size
- `images` -- container image references for trainer and feedback-api

---

## Commands

### `/distill.prerequisites`

**Full infrastructure bootstrap for a brand-new OpenShift cluster.**

Read `distill.config.yaml` to get `cluster.namespace`, `cluster.s3_access_key`,
`cluster.s3_secret_key`, and `teacher.model`.

Execute these steps IN ORDER. Each `oc apply` is idempotent so re-running is safe.
For every manifest, first replace `NAMESPACE_PLACEHOLDER` and `sridharproject` with
the namespace from config using `sed`.

```
NS=$(yq '.cluster.namespace' distill.config.yaml)
```

**Step 1 -- Create the namespace (if it does not exist)**

```bash
oc new-project "$NS" 2>/dev/null || oc project "$NS"
```

**Step 2 -- S3 Secret + ServiceAccount (KServe needs this)**

```bash
sed "s/sridharproject/$NS/g" rhoai/00-s3-secret.yaml | oc apply -f -
```

**Step 3 -- MinIO (S3-compatible object store)**

```bash
sed "s/sridharproject/$NS/g" rhoai/05-minio.yaml | oc apply -f -
oc rollout status deploy/minio -n "$NS" --timeout=120s
```

Wait for MinIO to be ready, then create the required buckets:

```bash
MINIO_POD=$(oc get pod -n "$NS" -l app=minio -o jsonpath='{.items[0].metadata.name}')
oc exec "$MINIO_POD" -n "$NS" -- bash -c '
  for b in mlflow-artifacts sridhar-models; do
    mkdir -p /data/$b
  done
'
```

**Step 4 -- MLflow (experiment tracking)**

```bash
sed "s/sridharproject/$NS/g" rhoai/06-mlflow.yaml | oc apply -f -
oc rollout status deploy/mlflow -n "$NS" --timeout=120s
```

**Step 5 -- Ollama (teacher LLM)**

```bash
sed "s/NAMESPACE_PLACEHOLDER/$NS/g" rhoai/04-ollama.yaml | oc apply -f -
oc rollout status deploy/ollama -n "$NS" --timeout=300s
```

Pull the teacher model (this takes 5-15 min depending on GPU node):

```bash
OLLAMA_POD=$(oc get pod -n "$NS" -l app=ollama -o jsonpath='{.items[0].metadata.name}')
TEACHER_MODEL=$(yq '.teacher.model' distill.config.yaml)
oc exec "$OLLAMA_POD" -n "$NS" -- ollama pull "$TEACHER_MODEL"
```

Verify it loaded:

```bash
oc exec "$OLLAMA_POD" -n "$NS" -- ollama list
```

**Step 6 -- KServe ServingRuntime + InferenceService**

```bash
sed "s/sridharproject/$NS/g" rhoai/01-serving-runtime-vllm.yaml | oc apply -f -
sed "s/sridharproject/$NS/g" rhoai/02-inference-service-student.yaml | oc apply -f -
```

Note: The InferenceService will fail until a model is actually uploaded to MinIO.
That happens during the first pipeline run. This is expected.

**Step 7 -- DSPA (DataSciencePipelinesApplication for Kubeflow Pipelines)**

```bash
sed "s/sridharproject/$NS/g" pipeline/rhoai/07-dspa.yaml | oc apply -f -
```

Wait for the pipeline server to come up (can take 2-3 min):

```bash
oc wait --for=condition=Available deploy/ds-pipeline-dspa -n "$NS" --timeout=300s 2>/dev/null || \
  echo "DSPA still starting -- check: oc get pods -n $NS -l app=ds-pipeline-dspa"
```

**Step 8 -- RBAC (let the pipeline SA create PyTorchJobs)**

```bash
sed "s/sridharproject/$NS/g" pipeline/training/rbac.yaml | oc apply -f -
```

Also grant the pipeline SA permissions to manage KServe and Deployments:

```bash
oc adm policy add-role-to-user admin system:serviceaccount:$NS:pipeline-runner-dspa -n "$NS"
```

**Step 9 -- Build trainer image in-cluster**

```bash
oc apply -f - <<EOF
apiVersion: image.openshift.io/v1
kind: ImageStream
metadata:
  name: distillation-trainer
  namespace: $NS
EOF

oc apply -f - <<EOF
apiVersion: build.openshift.io/v1
kind: BuildConfig
metadata:
  name: distillation-trainer
  namespace: $NS
spec:
  source:
    type: Binary
  strategy:
    type: Docker
    dockerStrategy:
      dockerfilePath: Dockerfile
  output:
    to:
      kind: ImageStreamTag
      name: distillation-trainer:latest
EOF

cd pipeline/training
oc start-build distillation-trainer --from-dir=. --follow -n "$NS"
cd ../..
```

**Step 10 -- Upload training data to MinIO**

If training data (JSONL files) exist locally in `pipeline/data/`, upload them:

```bash
if [ -d pipeline/data ]; then
  MINIO_POD=$(oc get pod -n "$NS" -l app=minio -o jsonpath='{.items[0].metadata.name}')
  for f in pipeline/data/*.jsonl; do
    KEY="synthetic/code-review/$(basename $f)"
    oc cp "$f" "$NS/$MINIO_POD:/data/mlflow-artifacts/$KEY"
    echo "Uploaded $f -> s3://mlflow-artifacts/$KEY"
  done
fi
```

Also upload the diff-bank if present:

```bash
if [ -f pipeline/scripts/diff-bank.json ]; then
  MINIO_POD=$(oc get pod -n "$NS" -l app=minio -o jsonpath='{.items[0].metadata.name}')
  oc cp pipeline/scripts/diff-bank.json "$NS/$MINIO_POD:/data/mlflow-artifacts/synthetic/code-review/diff-bank.json"
  echo "Uploaded diff-bank.json"
fi
```

**Step 11 -- Build and deploy feedback-api (optional, needed for human feedback)**

```bash
if [ -d feedback-service ]; then
  oc apply -f - <<EOF
apiVersion: image.openshift.io/v1
kind: ImageStream
metadata:
  name: feedback-api
  namespace: $NS
EOF

  oc apply -f - <<EOF
apiVersion: build.openshift.io/v1
kind: BuildConfig
metadata:
  name: feedback-api
  namespace: $NS
spec:
  source:
    type: Binary
  strategy:
    type: Docker
    dockerStrategy:
      dockerfilePath: Dockerfile
  output:
    to:
      kind: ImageStreamTag
      name: feedback-api:latest
EOF

  cd feedback-service
  oc start-build feedback-api --from-dir=. --follow -n "$NS"
  cd ..
  sed "s/sridharproject/$NS/g" feedback-service/k8s/deployment.yaml | oc apply -f -
fi
```

**Step 12 -- Compile and upload the pipeline**

```bash
cd pipeline
python3 code_review_pipeline.py
echo "Compiled -> pipeline/code_review_pipeline.yaml"
```

Upload via the RHOAI Dashboard UI, or via curl:

```bash
ROUTE=$(oc get route ds-pipeline-dspa -n "$NS" -o jsonpath='{.spec.host}' 2>/dev/null)
if [ -n "$ROUTE" ]; then
  TOKEN=$(oc whoami -t)
  curl -sSk -X POST "https://$ROUTE/apis/v2beta1/pipelines/upload?name=distill-pipeline" \
    -H "Authorization: Bearer $TOKEN" \
    -F "uploadfile=@code_review_pipeline.yaml"
  echo "Pipeline uploaded."
fi
cd ..
```

**Done.** Print a summary of all Routes for the user:

```bash
echo "=========================================="
echo "Infrastructure ready in namespace: $NS"
echo "=========================================="
echo "MinIO Console : https://$(oc get route minio-console -n $NS -o jsonpath='{.spec.host}')"
echo "MinIO API     : https://$(oc get route minio-api -n $NS -o jsonpath='{.spec.host}')"
echo "MLflow        : https://$(oc get route mlflow -n $NS -o jsonpath='{.spec.host}')"
echo "Ollama        : https://$(oc get route ollama -n $NS -o jsonpath='{.spec.host}')"
echo "KFP Dashboard : https://$(oc get route ds-pipeline-dspa -n $NS -o jsonpath='{.spec.host}' 2>/dev/null || echo 'via RHOAI dashboard')"
echo "=========================================="
```

---

### `/distill.setup`

**Validate that all infrastructure is healthy.**

Read `distill.config.yaml` for namespace and endpoints. Then run:

```bash
NS=$(yq '.cluster.namespace' distill.config.yaml)
echo "--- Checking namespace $NS ---"
oc get deploy minio mlflow ollama -n "$NS" -o wide
oc get isvc -n "$NS"
oc get dspa -n "$NS"

# Test MinIO connectivity
MINIO_POD=$(oc get pod -n "$NS" -l app=minio -o jsonpath='{.items[0].metadata.name}')
oc exec "$MINIO_POD" -n "$NS" -- ls /data/mlflow-artifacts/

# Test Ollama
OLLAMA_POD=$(oc get pod -n "$NS" -l app=ollama -o jsonpath='{.items[0].metadata.name}')
oc exec "$OLLAMA_POD" -n "$NS" -- ollama list

# Test MLflow
MLFLOW_POD=$(oc get pod -n "$NS" -l app=mlflow -o jsonpath='{.items[0].metadata.name}')
oc exec "$MLFLOW_POD" -n "$NS" -- curl -s http://localhost:5000/health
```

Print pass/fail for each service. If anything fails, suggest running `/distill.prerequisites`.

---

### `/distill.run`

**Trigger a new pipeline run.**

1. Read `distill.config.yaml` for namespace.
2. Get the KFP route: `oc get route ds-pipeline-dspa -n "$NS" -o jsonpath='{.spec.host}'`
   If no route, use the RHOAI dashboard route.
3. Get the pipeline ID by listing pipelines.
4. Get the latest pipeline version ID.
5. Create a run:

```bash
TOKEN=$(oc whoami -t)
ROUTE=$(oc get route ds-pipeline-dspa -n "$NS" -o jsonpath='{.spec.host}')

# List pipeline versions to get the latest
PIPELINE_ID=$(curl -sSk "https://$ROUTE/apis/v2beta1/pipelines" \
  -H "Authorization: Bearer $TOKEN" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for p in data.get('pipelines', []):
    print(p['pipeline_id'])
    break
")

VERSION_ID=$(curl -sSk "https://$ROUTE/apis/v2beta1/pipelines/$PIPELINE_ID/versions" \
  -H "Authorization: Bearer $TOKEN" | python3 -c "
import sys, json
data = json.load(sys.stdin)
versions = data.get('pipeline_versions', [])
if versions:
    print(versions[0]['pipeline_version_id'])
")

RUN_NAME="Run $(date +%Y%m%d-%H%M)"
curl -sSk -X POST "https://$ROUTE/apis/v2beta1/runs" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"display_name\": \"$RUN_NAME\",
    \"pipeline_version_reference\": {
      \"pipeline_id\": \"$PIPELINE_ID\",
      \"pipeline_version_id\": \"$VERSION_ID\"
    }
  }"

echo "Triggered: $RUN_NAME"
```

Ask the user for an optional run display name. If provided, use it instead.

---

### `/distill.status`

**Check the status of the most recent pipeline run.**

```bash
NS=$(yq '.cluster.namespace' distill.config.yaml)
TOKEN=$(oc whoami -t)
ROUTE=$(oc get route ds-pipeline-dspa -n "$NS" -o jsonpath='{.spec.host}')

curl -sSk "https://$ROUTE/apis/v2beta1/runs?page_size=1&sort_by=created_at%20desc" \
  -H "Authorization: Bearer $TOKEN" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for r in data.get('runs', []):
    print(f\"Run:    {r['display_name']}\")
    print(f\"State:  {r['state']}\")
    print(f\"Created: {r['created_at']}\")
    if r.get('finished_at'):
        print(f\"Finished: {r['finished_at']}\")
    if r.get('error'):
        print(f\"Error:  {r['error']['message']}\")
"
```

Also check GPU training pods:

```bash
oc get pods -n "$NS" -l training.kubeflow.org/job-name --sort-by=.metadata.creationTimestamp | tail -10
```

---

### `/distill.eval`

**Run evaluation on the currently deployed model.**

This triggers only the evaluate step -- useful after a manual deploy or to re-score.

1. Read `distill.config.yaml` for `domain.test_questions_file` and load the questions.
2. Read the `domain.grading_prompt`.
3. Get the current KServe student URL:
   ```bash
   NS=$(yq '.cluster.namespace' distill.config.yaml)
   ISVC=$(yq '.student.isvc_name' distill.config.yaml)
   STUDENT_URL="http://${ISVC}-predictor.${NS}.svc.cluster.local:8080"
   ```
4. Port-forward to the student and teacher, run the evaluate logic against them,
   and log results to MLflow.

Alternatively, point the user to the MLflow dashboard:

```bash
echo "MLflow: https://$(oc get route mlflow -n $NS -o jsonpath='{.spec.host}')"
```

---

### `/distill.baseline`

**Run baseline evaluation on the untuned base model.**

Execute the baseline eval Kubernetes Job:

```bash
NS=$(yq '.cluster.namespace' distill.config.yaml)

# Apply the ConfigMap with the script
oc create configmap baseline-eval-script \
  --from-file=baseline_eval.py=pipeline/scripts/baseline_eval.py \
  -n "$NS" --dry-run=client -o yaml | oc apply -f -

# Apply the Job
sed "s/sridharproject/$NS/g" pipeline/scripts/baseline_eval_job.yaml | oc apply -f -

echo "Baseline eval job started. Monitor with:"
echo "  oc logs -f job/baseline-eval -n $NS"
```

---

### `/distill.deploy`

**Manually deploy a specific model version to KServe.**

Ask the user which model version to deploy (e.g., `v4`).

```bash
NS=$(yq '.cluster.namespace' distill.config.yaml)
PREFIX=$(yq '.student.model_prefix' distill.config.yaml)
ISVC=$(yq '.student.isvc_name' distill.config.yaml)
MODEL_BUCKET=$(yq '.cluster.model_bucket' distill.config.yaml)
VERSION="$1"  # e.g. v4

STORAGE_URI="s3://${MODEL_BUCKET}/${PREFIX}${VERSION}/"

oc patch inferenceservice "$ISVC" -n "$NS" --type=merge -p \
  "{\"spec\":{\"predictor\":{\"model\":{\"storageUri\":\"$STORAGE_URI\"}}}}"

echo "Deployed $STORAGE_URI to $ISVC"
echo "Waiting for rollout..."
sleep 10
oc get inferenceservice "$ISVC" -n "$NS"
```

---

### `/distill.feedback`

**Check the state of the human feedback loop.**

```bash
NS=$(yq '.cluster.namespace' distill.config.yaml)
MINIO_POD=$(oc get pod -n "$NS" -l app=minio -o jsonpath='{.items[0].metadata.name}')

echo "=== Pending feedback (not yet consumed by pipeline) ==="
oc exec "$MINIO_POD" -n "$NS" -- bash -c \
  'ls /data/mlflow-artifacts/preferences/human-feedback/pending/ 2>/dev/null | wc -l'

echo "=== Used feedback (already consumed) ==="
oc exec "$MINIO_POD" -n "$NS" -- bash -c \
  'ls /data/mlflow-artifacts/preferences/human-feedback/used/ 2>/dev/null | wc -l'

echo "=== Static preference bank ==="
oc exec "$MINIO_POD" -n "$NS" -- bash -c \
  'wc -l < /data/mlflow-artifacts/preferences/static-bank/preference-bank.jsonl 2>/dev/null || echo "Not generated yet"'

echo ""
echo "Feedback API:"
oc get route feedback-api -n "$NS" -o jsonpath='https://{.spec.host}' 2>/dev/null || echo "Not deployed"
echo ""
```

---

### `/distill.scores`

**Show evaluation scores across all runs from MLflow.**

```bash
NS=$(yq '.cluster.namespace' distill.config.yaml)
MLFLOW_ROUTE=$(oc get route mlflow -n "$NS" -o jsonpath='{.spec.host}')
EXPERIMENT=$(yq '.cluster.mlflow_experiment' distill.config.yaml)

echo "MLflow UI: https://$MLFLOW_ROUTE"
echo ""

# Fetch experiment metrics via MLflow REST API
curl -sSk "https://$MLFLOW_ROUTE/api/2.0/mlflow/experiments/get-by-name?experiment_name=$EXPERIMENT" | \
  python3 -c "
import sys, json
data = json.load(sys.stdin)
exp_id = data['experiment']['experiment_id']
print(f'Experiment: {data[\"experiment\"][\"name\"]} (id={exp_id})')
print(f'Open: https://$MLFLOW_ROUTE/#/experiments/{exp_id}')
" 2>/dev/null || echo "Could not fetch experiment. Check MLflow route."
```

---

### `/distill.portforward`

**Set up port-forwards for local testing with the VS Code extension.**

```bash
NS=$(yq '.cluster.namespace' distill.config.yaml)
ISVC=$(yq '.student.isvc_name' distill.config.yaml)

# Find the KServe predictor pod
PREDICTOR_POD=$(oc get pod -n "$NS" -l serving.kserve.io/inferenceservice="$ISVC" \
  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$PREDICTOR_POD" ]; then
  echo "No predictor pod found for $ISVC. Is the model deployed?"
  echo "Run /distill.deploy first."
  exit 1
fi

echo "Port-forwarding $PREDICTOR_POD:8080 -> localhost:8080"
echo "Use Ctrl+C to stop."
oc port-forward "$PREDICTOR_POD" 8080:8080 -n "$NS"
```

For the feedback API as well (in a second terminal):

```bash
NS=$(yq '.cluster.namespace' distill.config.yaml)
FEEDBACK_POD=$(oc get pod -n "$NS" -l app=feedback-api -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -n "$FEEDBACK_POD" ]; then
  echo "Port-forwarding $FEEDBACK_POD:8000 -> localhost:8000"
  oc port-forward "$FEEDBACK_POD" 8000:8000 -n "$NS"
fi
```

---

## Adapting to a New Domain

To use this pipeline for a different use case (e.g., SQL query optimization, security
audit, documentation generation), change ONLY these items:

### 1. `distill.config.yaml`

| Field | What to change |
|---|---|
| `teacher.model` | A teacher model suited to your domain |
| `teacher.system_prompt` | Domain-specific instructions for the teacher |
| `student.base_model_id` | Pick a small model suited to your domain |
| `student.model_prefix` | e.g., `sql-optimizer-1.5b-` |
| `student.isvc_name` | e.g., `sql-optimizer-llm` |
| `domain.name` | e.g., `sql-optimization` |
| `domain.training_data_prefix` | Where your JSONL lives in MinIO |
| `domain.grading_prompt` | How the teacher judges quality |
| `domain.test_questions_file` | Path to your test questions JSON |
| `training.*` | Tune epochs, beta, etc. for your domain |

### 2. Training Data

Prepare a JSONL file with ChatML-formatted training examples and upload it to MinIO
under the `domain.training_data_prefix`. Each line must have a `text` field:

```json
{"text": "<|im_start|>system\nYou are a SQL optimizer...<|im_end|>\n<|im_start|>user\nOptimize this query: SELECT ...<|im_end|>\n<|im_start|>assistant\nUse an index on...<|im_end|>"}
```

### 3. Test Questions

Create a new JSON file at the path specified in `domain.test_questions_file`:

```json
[
  {"category": "performance", "question": "Optimize this SQL query: ..."},
  {"category": "security", "question": "Review this query for injection: ..."}
]
```

### 4. Diff Bank (for DPO)

Replace [`pipeline/scripts/diff-bank.json`](pipeline/scripts/diff-bank.json) with
domain-specific prompts used during DPO preference extraction.

### 5. No Code Changes Required

The pipeline components (`finetune.py`, `dpo_finetune.py`, `extract_preferences.py`,
`evaluate.py`) read all domain-specific values from pipeline parameters which flow
from `code_review_pipeline.py`. The pipeline reads from `distill.config.yaml`.

---

## File Reference

| File | Purpose |
|---|---|
| [`distill.config.yaml`](distill.config.yaml) | Single config file for all parameters |
| [`rhoai/00-s3-secret.yaml`](rhoai/00-s3-secret.yaml) | KServe S3 credentials |
| [`rhoai/01-serving-runtime-vllm.yaml`](rhoai/01-serving-runtime-vllm.yaml) | vLLM ServingRuntime |
| [`rhoai/02-inference-service-student.yaml`](rhoai/02-inference-service-student.yaml) | KServe InferenceService for student |
| [`rhoai/04-ollama.yaml`](rhoai/04-ollama.yaml) | Ollama teacher deployment |
| [`rhoai/05-minio.yaml`](rhoai/05-minio.yaml) | MinIO object storage |
| [`rhoai/06-mlflow.yaml`](rhoai/06-mlflow.yaml) | MLflow tracking server |
| [`pipeline/rhoai/07-dspa.yaml`](pipeline/rhoai/07-dspa.yaml) | DataSciencePipelinesApplication |
| [`pipeline/training/rbac.yaml`](pipeline/training/rbac.yaml) | RBAC for PyTorchJob creation |
| [`pipeline/training/Dockerfile`](pipeline/training/Dockerfile) | Trainer container image |
| [`pipeline/training/finetune_job.py`](pipeline/training/finetune_job.py) | SFT + DPO training script |
| [`pipeline/code_review_pipeline.py`](pipeline/code_review_pipeline.py) | KFP pipeline definition |
| [`pipeline/domain/test_questions.json`](pipeline/domain/test_questions.json) | Evaluation test questions |
| [`pipeline/scripts/diff-bank.json`](pipeline/scripts/diff-bank.json) | DPO question bank |
| [`pipeline/scripts/generate_preference_bank.py`](pipeline/scripts/generate_preference_bank.py) | Static DPO bank generator |
| [`pipeline/scripts/baseline_eval.py`](pipeline/scripts/baseline_eval.py) | Baseline evaluation script |
| [`feedback-service/`](feedback-service/) | Human feedback API service |
| [`extension/`](extension/) | VS Code extension for code review |
