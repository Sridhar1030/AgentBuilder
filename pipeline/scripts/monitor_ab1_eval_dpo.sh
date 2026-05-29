#!/usr/bin/env bash
# Poll AB_1 KFP run until eval-dpo (evaluate-2) completes, then print MLflow metrics.
set -euo pipefail

RUN_ID="${1:-bfafab0d-44ae-4e0a-8410-9c24a844516c}"
NAMESPACE="sridharproject"
INTERVAL="${INTERVAL:-300}"

TOKEN=$(oc whoami -t)
KFP_HOST=$(oc get route ds-pipeline-dspa -n "$NAMESPACE" -o jsonpath='{.spec.host}')
MLFLOW_HOST=$(oc get route mlflow -n "$NAMESPACE" -o jsonpath='{.spec.host}')

fetch_tasks() {
  curl -sk -H "Authorization: Bearer $TOKEN" \
    "https://${KFP_HOST}/apis/v2beta1/runs/${RUN_ID}" \
    | python3 -c "
import json, sys
r = json.load(sys.stdin)
print(r.get('state', 'UNKNOWN'))
for t in r.get('run_details', {}).get('task_details', []):
    name = t.get('display_name', '')
    if name in ('evaluate', 'evaluate-2', 'evaluate-3', 'finetune', 'dpo-finetune', 'extract-preferences', 'merge-preferences', 'deploy-model', 'deploy-model-2'):
        state = t.get('state', '?')
        start = (t.get('start_time') or '')[:19]
        end = (t.get('end_time') or '')[:19] if (t.get('end_time') or '').startswith('20') else '-'
        print(f'TASK|{name}|{state}|{start}|{end}')
"
}

fetch_mlflow_dpo() {
  curl -sk "https://${MLFLOW_HOST}/api/2.0/mlflow/runs/search" \
    -H "Content-Type: application/json" \
    -d '{"experiment_ids": ["4"], "filter": "tags.run_label = '\''AB_1'\'' and tags.stage = '\''dpo'\''", "order_by": ["start_time DESC"], "max_results": 1}' \
    | python3 -c "
import json, sys
data = json.load(sys.stdin)
runs = data.get('runs', [])
if not runs:
    print('MLFLOW|none')
    sys.exit(0)
r = runs[0]
info = r['info']
metrics = {m['key']: m['value'] for m in r.get('data', {}).get('metrics', [])}
tags = {t['key']: t['value'] for t in r.get('data', {}).get('tags', [])}
corr = metrics.get('judge_correctness_pass_rate', '?')
review = metrics.get('judge_review_quality_mean', '?')
comp = metrics.get('composite_score', '?')
gate = tags.get('quality_gate', '?')
print(f'MLFLOW|{info[\"run_id\"]}|{info.get(\"run_name\",\"\")}|correctness={corr}|review={review}|composite={comp}|gate={gate}')
"
}

echo "Monitoring AB_1 run ${RUN_ID} every ${INTERVAL}s for eval-dpo (evaluate-2)..."

while true; do
  OUT=$(fetch_tasks)
  RUN_STATE=$(echo "$OUT" | head -1)
  echo "--- $(date -u +%Y-%m-%dT%H:%M:%SZ) run=${RUN_STATE} ---"
  echo "$OUT" | grep '^TASK|' || true

  EVAL_DPO=$(echo "$OUT" | grep '^TASK|evaluate-2|' || true)
  if echo "$EVAL_DPO" | grep -q '|SUCCEEDED|'; then
    echo "AGENT_LOOP_WAKE_AB1_EVAL_DPO {\"prompt\":\"AB_1 eval-dpo completed — fetch MLflow metrics and compare vs SFT baseline 80%\",\"status\":\"succeeded\"}"
    fetch_mlflow_dpo
    exit 0
  fi
  if echo "$EVAL_DPO" | grep -qE '\|(FAILED|CANCELED|SKIPPED)\|'; then
    echo "AGENT_LOOP_WAKE_AB1_EVAL_DPO {\"prompt\":\"AB_1 eval-dpo FAILED — investigate logs\",\"status\":\"failed\"}"
    exit 1
  fi
  if [[ "$RUN_STATE" == "FAILED" ]] || [[ "$RUN_STATE" == "SUCCEEDED" ]]; then
    echo "AGENT_LOOP_WAKE_AB1_EVAL_DPO {\"prompt\":\"AB_1 run ended (${RUN_STATE}) before eval-dpo tracked — check pipeline\",\"status\":\"${RUN_STATE}\"}"
    exit 2
  fi

  sleep "$INTERVAL"
done
