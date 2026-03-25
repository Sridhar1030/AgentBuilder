# Demo: Deploying a Fresh (Untrained) SLM

## Goal

Show the base Llama 3.2 1B model is "dumb" about Kubeflow, then run the pipeline to fine-tune it and show the improvement.

---

## How It Works

The KServe `InferenceService` (`student-llm`) points at an S3 path for its model weights. Right now it points to a fine-tuned model (e.g. `s3://sridhar-models/student-1b-v24-dpo/`). To reset it, you point it at the original base model instead.

---

## Step 1: Upload the Base Model to MinIO

The base model (`unsloth/Llama-3.2-1B-Instruct`) needs to be in MinIO so KServe can load it. This is a one-time setup.

```bash
# From your laptop (with MinIO port-forward active)
pip install huggingface_hub boto3

python3 -c "
from huggingface_hub import snapshot_download
import boto3, os

# Download base model from HuggingFace
local_dir = '/tmp/llama-3.2-1b-instruct'
snapshot_download('unsloth/Llama-3.2-1B-Instruct', local_dir=local_dir)

# Upload to MinIO
s3 = boto3.client('s3',
    endpoint_url='http://localhost:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin123')

bucket = 'sridhar-models'
prefix = 'student-1b-base/'

for root, dirs, files in os.walk(local_dir):
    for fname in files:
        local_path = os.path.join(root, fname)
        s3_key = prefix + os.path.relpath(local_path, local_dir)
        print(f'Uploading {fname}...')
        s3.upload_file(local_path, bucket, s3_key)

print('Done: s3://sridhar-models/student-1b-base/')
"
```

## Step 2: Point KServe at the Base Model

```bash
oc patch inferenceservice student-llm -n sridharproject --type merge -p '{
  "spec": {
    "predictor": {
      "model": {
        "storageUri": "s3://sridhar-models/student-1b-base/"
      }
    }
  }
}'
```

Wait for the rollout (~2-3 min):

```bash
oc get inferenceservice student-llm -n sridharproject -w
```

Once `READY = True`, the base model is serving.

## Step 3: Demo -- Ask the Base Model Kubeflow Questions

```bash
# Port-forward to the student model
oc port-forward svc/student-llm-predictor 8081:8080 -n sridharproject &

# Ask it something
curl -s http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/mnt/models",
    "messages": [{"role": "user", "content": "What is the difference between PyTorchJob and TrainJob in the Kubeflow Training Operator?"}],
    "max_tokens": 256, "temperature": 0.3
  }' | python3 -m json.tool | grep -A999 '"content"'
```

**Expected:** Vague, repetitive, or incorrect answer (the base model has no Kubeflow knowledge).

Good demo questions to show it's "dumb":
- "How does KServe handle canary deployments?"
- "What search algorithms does Katib support?"
- "How do you set up multi-tenancy in Kubeflow using profiles?"
- "Explain the difference between Serverless and RawDeployment mode in KServe."

## Step 4: Run the Fine-Tuning Pipeline

```bash
# Port-forward to KFP
oc port-forward svc/ds-pipeline-dspa 3000:8888 -n sridharproject &

# Submit the pipeline
cd ~/AgentBuilder
python3 scripts/upload_and_run_sft.py all
```

Or submit via the RHOAI Dashboard at:
`https://rhods-dashboard-redhat-ods-applications.apps.<cluster>`

The pipeline will:
1. Extract 827+ Kubeflow Q&A training pairs
2. SFT fine-tune the model (QLoRA, ~7 min)
3. Generate DPO preference data (~13 min)
4. DPO fine-tune on preferences (~7 min)
5. Deploy the fine-tuned model (replaces base model automatically)
6. Evaluate with 15 test questions

## Step 5: Demo -- Ask the Fine-Tuned Model the Same Questions

Once the pipeline completes, the `student-llm` InferenceService is already updated. Ask the same questions again:

```bash
curl -s http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/mnt/models",
    "messages": [{"role": "user", "content": "What is the difference between PyTorchJob and TrainJob in the Kubeflow Training Operator?"}],
    "max_tokens": 256, "temperature": 0.3
  }' | python3 -m json.tool | grep -A999 '"content"'
```

**Expected:** Detailed, accurate answer with specific Kubeflow knowledge.

## Step 6: Restore the Fine-Tuned Model (After Demo)

If you want to go back to the latest fine-tuned model:

```bash
# Point back to the DPO model
oc patch inferenceservice student-llm -n sridharproject --type merge -p '{
  "spec": {
    "predictor": {
      "model": {
        "storageUri": "s3://sridhar-models/student-1b-v24-dpo/"
      }
    }
  }
}'
```

Or to the SFT-only model:
```bash
oc patch inferenceservice student-llm -n sridharproject --type merge -p '{
  "spec": {
    "predictor": {
      "model": {
        "storageUri": "s3://sridhar-models/student-1b-v24/"
      }
    }
  }
}'
```

---

## Quick Reference

| Model | S3 Path | What It Is |
|-------|---------|------------|
| Base (dumb) | `s3://sridhar-models/student-1b-base/` | Original Llama 3.2 1B, no Kubeflow knowledge |
| SFT only | `s3://sridhar-models/student-1b-v24/` | Trained on 827 Kubeflow Q&A pairs |
| SFT + DPO | `s3://sridhar-models/student-1b-v24-dpo/` | SFT + preference alignment |

Switch between them with a single `oc patch` command. KServe handles the rollout automatically.
