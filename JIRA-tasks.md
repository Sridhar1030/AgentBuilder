# JIRA Tasks — Distillation Flywheel

**Epic:** RHOAIENG-51416 — PoC Agent distillation  
**Project:** Red Hat OpenShift AI Engineering

---

## Subtasks: RHOAI Infrastructure Setup

### DIST-1: Provision OpenShift cluster
- Provisioned OpenShift cluster for RHOAI workloads
- **Status:** Done

### DIST-2: Install RHOAI operator stack
- Installed RHOAI 3.2.0, OpenShift Serverless, Pipelines, Service Mesh 3, cert-manager, Authorino, Kueue
- Verified all operators in Succeeded state
- **Status:** Done

### DIST-3: Create DataScienceCluster
- Created `default-dsc` DataScienceCluster — Ready=True
- **Status:** Done

### DIST-4: Create project namespace
- Created `sridharproject` namespace for all distillation workloads
- **Status:** Done

### DIST-5: Connect to cluster via oc CLI
- Authenticated to OpenShift cluster using `oc login`
- Verified access to `sridharproject` namespace
- **Status:** Done

---

## Subtasks: Model Serving Infrastructure

### DIST-6: Deploy vLLM ServingRuntime
- Applied vLLM ServingRuntime to `sridharproject`
- Verified with `oc get servingruntime`
- **Status:** Done

### DIST-7: Deploy MinIO object storage on-cluster
- Deployed MinIO (API :9000, Console :9001) with 20Gi PVC
- Pod running: `minio-977c79dcb-fgfkv`
- **Status:** Done

### DIST-8: Merge student model LoRA adapters
- Merged `student_model/` LoRA adapters into `unsloth/Llama-3.2-1B-Instruct` base
- Output: `student_model_merged/` — 4.6G model.safetensors
- **Status:** Done

### DIST-9: Upload merged student model to MinIO
- Created `sridhar-models` bucket
- Uploaded all 7 model files to `sridhar-models/student-1b-merged/`
- Verified 4.6GiB model.safetensors present
- **Status:** Done
