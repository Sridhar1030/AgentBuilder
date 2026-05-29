#!/usr/bin/env bash
# Create/update ConfigMap for baseline harness eval job.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NS="${NAMESPACE:-sridharproject}"

oc create configmap baseline-eval-config \
  --from-file=baseline_eval.py="$ROOT/pipeline/scripts/baseline_eval.py" \
  --from-file=test_questions.json="$ROOT/pipeline/domain/test_questions.json" \
  --from-file=eval.yaml="$ROOT/eval/eval.yaml" \
  -n "$NS" \
  --dry-run=client -o yaml | oc apply -f -

echo "ConfigMap baseline-eval-config updated in $NS"
