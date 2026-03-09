#!/bin/bash
set -euo pipefail

# ── Distillation Flywheel — RHOAI Deployment Script ──────────────────────────
# Run from repo root:  bash rhoai/deploy.sh
#
# Prerequisites:
#   1. oc login <cluster-url> --token=<token>
#   2. Edit placeholder values in the YAML files:
#      - 00-s3-secret.yaml         → S3 credentials
#      - 03-gradio-deployment.yaml → Groq API key + image registry
#   3. Upload merged student model to S3:
#      aws s3 sync ./student_model_merged s3://sridhar-models/student-1b-merged/
#   4. Build & push the Gradio container image

NAMESPACE="sridharproject"
IMAGE_REGISTRY="${IMAGE_REGISTRY:-quay.io/sridhar}"

echo "============================================"
echo " Distillation Flywheel — RHOAI Deployment"
echo "============================================"
echo ""

# ── Step 0: Verify cluster connection ──────────────────────────────────────────
echo "[0/6] Verifying cluster connection..."
oc whoami || { echo "ERROR: Not logged in. Run: oc login <cluster> --token=<token>"; exit 1; }
oc project "$NAMESPACE" 2>/dev/null || oc new-project "$NAMESPACE"
echo ""

# ── Step 1: S3 credentials + ServiceAccount ───────────────────────────────────
echo "[1/6] Creating S3 secret and ServiceAccount..."
oc apply -f rhoai/00-s3-secret.yaml
echo ""

# ── Step 2: vLLM ServingRuntime ────────────────────────────────────────────────
echo "[2/6] Deploying vLLM ServingRuntime..."
oc apply -f rhoai/01-serving-runtime-vllm.yaml
echo ""

# ── Step 3: Student InferenceService ───────────────────────────────────────────
echo "[3/6] Creating Student InferenceService..."
oc apply -f rhoai/02-inference-service-student.yaml
echo ""
echo "Waiting for InferenceService to become ready..."
oc wait --for=condition=Ready inferenceservice/student-llm -n "$NAMESPACE" --timeout=300s || \
    echo "WARNING: InferenceService not ready yet. Check: oc get inferenceservice student-llm -n $NAMESPACE"
echo ""

# ── Step 4: Build & push Gradio image ─────────────────────────────────────────
echo "[4/6] Building and pushing Gradio UI image..."
IMAGE_TAG="${IMAGE_REGISTRY}/distillation-ui:latest"
if command -v podman &>/dev/null; then
    podman build -t "$IMAGE_TAG" .
    podman push "$IMAGE_TAG"
elif command -v docker &>/dev/null; then
    docker build -t "$IMAGE_TAG" .
    docker push "$IMAGE_TAG"
else
    echo "WARNING: No container runtime found (podman/docker). Build manually:"
    echo "  podman build -t $IMAGE_TAG . && podman push $IMAGE_TAG"
fi
echo ""

# ── Step 5: Deploy Gradio UI ──────────────────────────────────────────────────
echo "[5/6] Deploying Gradio UI + Route..."
oc apply -f rhoai/03-gradio-deployment.yaml
echo ""

# ── Step 6: Print access info ─────────────────────────────────────────────────
echo "[6/6] Deployment complete!"
echo ""
echo "── Access Points ──────────────────────────────────────────"
ROUTE_URL=$(oc get route distillation-ui -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null || echo "<pending>")
echo "  Gradio UI:    https://${ROUTE_URL}"
echo ""
echo "  Student LLM:  oc get inferenceservice student-llm -n $NAMESPACE"
echo ""
echo "── Next Steps ───────────────────────────────────────────"
echo "  1. Chat with the 70B Teacher to generate gold training data"
echo "  2. When enough data is collected, run the gold extractor:"
echo "     python gold_extractor.py --threshold 200 --output s3://sridhar-models/gold-data/gold_pairs.jsonl"
echo "  3. Kick off fine-tuning:"
echo "     oc apply -f rhoai/04-rayjob-finetune.yaml"
echo "  4. After training completes, roll the new model:"
echo "     oc patch inferenceservice student-llm --type merge -p"
echo "       '{\"spec\":{\"predictor\":{\"model\":{\"storageUri\":\"s3://sridhar-models/student-1b-merged-v2/\"}}}}'"
echo ""
