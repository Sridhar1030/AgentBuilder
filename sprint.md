# Distillation Flywheel — Sprint Tracker

**Project:** 70B → 1B Knowledge Distillation on RHOAI  
**Timeline:** 6-month project  
**Cluster:** OpenShift + RHOAI 3.2.0  
**Namespace:** `sridharproject`

---

## Sprint 1: Get the Student Served (Target: 1–2 weeks)

**Goal:** The 1B Student model is live on KServe and the local `app.py` can send it a question and get a response back from the cluster.

**Definition of Done:** Run `python app.py` locally, select "1B Student", ask a question, and get a response served from the KServe pod on OpenShift.


| #    | Task                                   | Status | Notes                                                                                                 |
| ---- | -------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------- |
| 1.1  | Verify RHOAI operator stack is healthy | DONE   | RHOAI 3.2.0, Serverless, Pipelines, ServiceMesh, Kueue — all Succeeded. DataScienceCluster Ready=True |
| 1.2  | Set up MinIO object storage            | DONE   | MinIO deployed. Pod `minio-977c79dcb-fgfkv` Running. API on :9000, Console on :9001                   |
| 1.3  | Merge LoRA adapters into base model    | DONE   | `student_model_merged/` created — 4.6G model.safetensors + config + tokenizer                         |
| 1.4  | Upload merged model to Minio bucket    | DONE   | All 7 files in `sridhar-models/student-1b-merged/` — 4.6GiB model.safetensors confirmed               |
| 1.5  | Apply MinIO secret to cluster          | DONE   | Secret + ServiceAccount created                                                                       |
| 1.6  | Apply InferenceService                 | DONE   | Fixed ServingRuntime container name + image. Pod Running 1/1, READY=True                              |
| 1.7  | Wait for pod, debug if needed          | DONE   | Fixed: container name → kserve-container, image → docker.io/vllm/vllm-openai:v0.6.2                   |
| 1.8  | Port-forward and test from terminal    | DONE   | vLLM v0.7.3 fixed OOM. curl test successful — model generates Python code from cluster                |
| 1.9  | Point local app.py at cluster endpoint | DONE   | Installed Python 3.11 via brew, created `venv311`, updated app.py for Gradio 6.x compat               |
| 1.10 | Smoke test end-to-end                  | DONE   | Gradio UI → KServe Student → response + Teacher grading working. "What is 2+2?" → "4" (Grade: 10/10)  |


### Sprint 1 Complete

All 10 tasks done. Local Gradio app at `http://127.0.0.1:7860` sends questions to the 1B Student model served by KServe on the cluster via `oc port-forward`, receives generated responses, and the 70B Teacher grades them in real-time.

---

## Sprint 2: MLflow on-cluster

**Goal:** Move experiment tracking from local SQLite to a proper MLflow server on the cluster so traces persist and are accessible from anywhere.

**Definition of Done:** Traces from the Gradio app appear in the MLflow UI on the cluster.


| #   | Task                                    | Status | Notes                                                                                                      |
| --- | --------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------- |
| 2.1 | Create mlflow-artifacts bucket in MinIO | DONE   | `mc mb myminio/mlflow-artifacts` — separate bucket from model weights                                      |
| 2.2 | Deploy standalone MLflow server         | DONE   | Deployment + PVC + Service in sridharproject. RHOAI operator was namespace-scoped, used standalone instead |
| 2.3 | Expose MLflow Route                     | DONE   | `https://mlflow-sridharproject.apps.sridhartest-pool-7f6n4.aws.rh-ods.com` — edge TLS                      |
| 2.4 | Update app.py MLFLOW_TRACKING_URI       | DONE   | Default URI now points to cluster MLflow. Added `MLFLOW_TRACKING_INSECURE_TLS=true`                        |
| 2.5 | Smoke test: traces in cluster MLflow    | DONE   | 3 `chat_interaction` traces logged, artifacts in `s3://mlflow-artifacts/1/traces/`                         |


### Sprint 2 Complete

MLflow server running on-cluster with MinIO artifact storage. Traces from local Gradio app persist centrally. MLflow UI at `https://mlflow-sridharproject.apps.sridhartest-pool-7f6n4.aws.rh-ods.com`.

**Launch command** (needs both port-forwards running):

```
AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin123 MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 STUDENT_ENDPOINT=http://localhost:8080/v1 python app.py
```

---

## Sprint 3: Gold Data Pipeline

**Goal:** 70B Teacher interactions captured in MLflow are automatically extracted into a training-ready dataset.

**Definition of Done:** Training-ready JSONL file in MinIO, validated for SFT format.


| #   | Task                                 | Status | Notes                                                                         |
| --- | ------------------------------------ | ------ | ----------------------------------------------------------------------------- |
| 3.0 | Generate 70B Teacher interactions    | DONE   | 10 diverse Q&A pairs via Groq (Llama-3.3-70B) logged in MLflow                |
| 3.1 | Update gold_extractor.py for cluster | DONE   | Rewritten for MLflow v3 trace format (request/response columns)               |
| 3.2 | Run gold_extractor.py, extract pairs | DONE   | 10/15 traces identified as teacher pairs                                      |
| 3.3 | Upload gold dataset to MinIO         | DONE   | `s3://mlflow-artifacts/gold/train.jsonl` — 35KiB                              |
| 3.4 | Validate dataset format for SFT      | DONE   | All 10 examples valid. Keys: instruction, output, text. Avg 1690 chars/output |


### Sprint 3 Complete

Gold data pipeline working end-to-end: Teacher traces in MLflow → `gold_extractor.py` → validated JSONL → MinIO. Ready for fine-tuning in Sprint 4.

---

## Sprint 4: QLoRA Fine-Tuning on Cluster

**Goal:** Run QLoRA fine-tuning on-cluster using the gold dataset, upload improved model back to MinIO.

**Definition of Done:** Merged model files at `s3://sridhar-models/student-1b-v2/` ready for serving.


| #   | Task                                            | Status | Notes                                                                                                   |
| --- | ----------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------- |
| 4.1 | Verify KFT operator is installed                | DONE   | `kubeflow-training-operator` Running. CRD: `pytorchjobs.kubeflow.org`. DSC: `trainingoperator: Managed` |
| 4.2 | Update finetune.py with in-cluster MinIO config | DONE   | `get_s3_client()` with `MLFLOW_S3_ENDPOINT_URL`, QLoRA 4-bit + SFTTrainer, merge & upload to S3         |
| 4.3 | Write PyTorchJob manifest                       | DONE   | `rhoai/04-pytorchjob-finetune.yaml` — Master replica, `pytorch/pytorch:2.5.1-cuda12.4`, 1x T4 GPU       |
| 4.4 | Upload finetune.py via ConfigMap                | DONE   | ConfigMap `finetune-script` mounted at `/opt/scripts/finetune.py`                                       |
| 4.5 | Submit PyTorchJob and monitor training          | DONE   | 3 epochs, 37s. Loss: 0.96→0.91. LoRA merged. Model uploaded to MinIO. PyTorchJob state: Succeeded       |
| 4.6 | Verify new model files in MinIO                 | DONE   | 6 files at `s3://sridhar-models/student-1b-v2/` — model.safetensors (1.48GB), tokenizer, config (1.5GB) |


### Sprint 4 Complete

QLoRA fine-tuning ran on-cluster via **Kubeflow Training Operator** (`PyTorchJob`) on `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` with 1x T4 GPU. Trained 3 epochs on 10 gold samples (loss 0.96→0.91). Merged LoRA adapters and uploaded to `s3://sridhar-models/student-1b-v2/`. Ready for canary deployment in Sprint 5.

---

## Sprint 5: KFP Pipeline — Distillation Flywheel

**Goal:** Automate the full distillation loop (extract → fine-tune → deploy → evaluate) as a Kubeflow Pipeline on RHOAI Data Science Pipelines.

**Definition of Done:** Pipeline runs end-to-end — auto-versioned model is trained, deployed, and evaluated with Teacher-graded scores.


| #    | Task                                   | Status | Notes                                                                                                                              |
| ---- | -------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| 5.1  | Verify Data Science Pipelines operator | DONE   | `data-science-pipelines-operator-controller-manager` Running. DSC: `aipipelines: Managed`                                          |
| 5.2  | Create DSPA CR for pipeline server     | DONE   | `pipeline/rhoai/07-dspa.yaml` applied. 7 pods running: ds-pipeline, mariadb, metadata, persistence, workflow. Fixed apiVersion v1alpha1 → v1 |
| 5.3  | Write Extract Gold Data component      | DONE   | `pipeline/components/extract_gold.py` — pulls teacher traces from MLflow, uploads versioned JSONL to MinIO                         |
| 5.4  | Write Fine-Tune component (GPU)        | DONE   | `pipeline/components/finetune.py` — QLoRA SFT, `set_gpu_limit(1)`, 16–24Gi memory. Multiple fixes (see issues below)               |
| 5.5  | Write Deploy Model component           | DONE   | `pipeline/components/deploy_model.py` — patches ISVC storageUri, polls for Ready, waits for vLLM init                             |
| 5.6  | Write Evaluate component               | DONE   | `pipeline/components/evaluate.py` — queries Student, Teacher grades via Groq, prints per-question scores                           |
| 5.7  | Write pipeline.py and compile          | DONE   | `pipeline/pipeline.py` → `distillation_flywheel.yaml`. kfp 2.16.0                                                                 |
| 5.8  | Add auto-versioning                    | DONE   | `pipeline/components/resolve_version.py` — scans MinIO for `student-1b-vN/`, returns vN+1 + pre-built S3 paths via NamedTuple      |
| 5.9  | Upload pipeline to RHOAI dashboard     | DONE   | Uploaded via RHOAI Data Science Pipelines UI                                                                                       |
| 5.10 | Test end-to-end pipeline run           | DONE   | Full 5-step DAG succeeded. Model auto-versioned, trained, deployed, evaluated with Teacher scores                                  |


### Pipeline Architecture

```
Resolve Version (CPU) → Extract Gold Data (CPU) → Fine-Tune QLoRA (GPU) → Deploy Model (CPU) → Evaluate (CPU)
```

Pipeline server route: `https://ds-pipeline-dspa-sridharproject.apps.sridhartest-pool-7f6n4.aws.rh-ods.com`  
MinIO console: `https://minio-console-sridharproject.apps.sridhartest-pool-7f6n4.aws.rh-ods.com`


### Issues Encountered & Fixes

**1. DSPA apiVersion mismatch**  
`oc apply` rejected `v1alpha1`. The CRD on RHOAI 3.2.0 uses `datasciencepipelinesapplications.opendatahub.io/v1`. Fixed in `07-dspa.yaml`.

**2. bitsandbytes version too old**  
Finetune pod failed with `ImportError: Using bitsandbytes 4-bit quantization requires bitsandbytes>=0.46.1`. The component had `bitsandbytes==0.45.5` pinned. Updated to `>=0.46.1`.

**3. Multi-GPU device mismatch**  
Finetune pod failed with `RuntimeError: module must have its parameters and buffers on device cuda:0 but found one on cuda:3`. The `device_map="auto"` spread the model across all GPUs on the node even though `set_gpu_limit(1)` was set (Kubernetes limits are not visible to PyTorch). Fixed by setting `CUDA_VISIBLE_DEVICES=0` and `device_map={"": 0}`.

**4. ConcatPlaceholder not supported in KFP v2**  
Pipeline compilation failed when trying to use `ConcatPlaceholder` to build S3 paths from the version string. KFP v2 doesn't support `ConcatPlaceholder` as component input. Fixed by having `resolve_version` return a `NamedTuple` with pre-built `gold_data_path` and `model_output_path` strings.

**5. Tokenizer `tokenizer_class: TokenizersBackend` breaking vLLM**  
New student pods (v3, v4, v5) went into `CrashLoopBackOff` with `ValueError: Tokenizer class TokenizersBackend does not exist`. The `trl` SFTTrainer saved `tokenizer_config.json` with a `tokenizer_class` field that vLLM's older `transformers` couldn't parse. Fixed by saving a clean tokenizer from the base model and stripping `tokenizer_class` and `auto_map` fields from `tokenizer_config.json`.

**6. Quantized model weights saved instead of full-precision**  
Student pod (v5) crashed with `KeyError: 'layers.0.mlp.down_proj.weight.absmax'`. Calling `merge_and_unload()` on the 4-bit quantized model saved quantized weight artifacts. Fixed by saving the LoRA adapter separately, reloading the base model in `torch.float16`, applying the adapter via `PeftModel`, merging, then saving the full-precision merged model.

**7. Headless Service port routing (KServe RawDeployment)**  
Evaluate step got `404 Not Found` when hitting `http://student-llm-predictor...svc.cluster.local/v1/chat/completions`. KServe `RawDeployment` creates a headless Service — no kube-proxy port translation from 80 → 8080. Fixed by explicitly using port `:8080` in the student URL returned by `deploy_model`.

**8. Groq API key not passed to pipeline**  
Evaluate pod terminated with `exit status 143` (SIGTERM). The `groq_api_key` was in the local `.env` but not passed as a pipeline runtime parameter. Added a fail-fast `ValueError` check and user was instructed to pass the key when creating each run.

**9. MariaDB rejecting Unicode in pipeline spec**  
Pipeline upload failed with `Error 1366: Incorrect string value '\xE2\x95\x90\xE2\x95\x90...' for column PipelineSpec`. The `evaluate` component had Unicode box-drawing characters (`═══`, `★`) in print statements. MariaDB's `utf8` column (3-byte max) rejected 3-byte UTF-8 sequences. Replaced with ASCII equivalents.

**10. vLLM context window exceeded (max_tokens)**  
Evaluate step got `400 Bad Request` from the student. vLLM was configured with `max_model_len=512` (total context). The request sent `max_tokens: 512` for output alone, leaving zero room for prompt tokens. Fixed by reducing `max_tokens` to `256`.


### Cleanup Performed

Deleted stale/broken artifacts from MinIO after debugging:
- Models `student-1b-v2` through `student-1b-v5` (broken by tokenizer/quantization bugs)
- Old pre-versioning model `student-1b-merged/` (~4.9 GB)
- Stale gold data files (`train.jsonl`, `train-v3` through `train-v5`)
- Old `training-code/finetune.py` (superseded by pipeline component)

Also fixed MinIO console route (missing TLS termination → added `edge` TLS).

**Final clean state in MinIO:**
- `sridhar-models/student-1b-v6/`, `student-1b-v7/` — trained models
- `mlflow-artifacts/gold/train-v6.jsonl`, `train-v7.jsonl` — corresponding gold data


### Sprint 5 Complete

Full distillation flywheel automated as a 5-step KFP pipeline on RHOAI Data Science Pipelines. Pipeline auto-versions each run, extracts teacher traces, fine-tunes with QLoRA on GPU, hot-swaps the live model via KServe, and evaluates with Teacher-as-Judge scoring. 10 issues identified and resolved across bitsandbytes compatibility, multi-GPU scheduling, KFP DSL limitations, tokenizer/model serialization, Kubernetes networking, API key management, database encoding, and vLLM context limits.

---

## Sprint 6: Canary Deployment & Eval (Future)

**Goal:** Deploy the improved Student alongside the old one, compare traces against the LLM baseline, and roll forward when ready.


| #   | Task                               | Status      | Notes |
| --- | ---------------------------------- | ----------- | ----- |
| 6.1 | Set up offline evaluation baseline | NOT STARTED |       |
| 6.2 | Deploy canary InferenceService     | NOT STARTED |       |
| 6.3 | Traffic split: 80/20 old/new       | NOT STARTED |       |
| 6.4 | Compare SLM traces vs LLM baseline | NOT STARTED |       |
| 6.5 | Full rollover when metrics pass    | NOT STARTED |       |


---

## Cluster Info (Reference)


| Component              | Version     | Status                   |
| ---------------------- | ----------- | ------------------------ |
| RHOAI (rhods-operator) | 3.2.0       | Succeeded                |
| OpenShift Serverless   | 1.37.1      | Succeeded                |
| OpenShift Pipelines    | 1.21.0      | Succeeded                |
| Service Mesh 3         | 3.2.2       | Succeeded                |
| cert-manager           | 1.18.1      | Succeeded                |
| Authorino              | 1.3.0       | Succeeded                |
| Kueue                  | 1.2.0       | Succeeded                |
| DataScienceCluster     | default-dsc | Ready=True               |
| vLLM ServingRuntime    | deployed    | sridharproject namespace |


---

*Last updated: 2026-03-13 (Sprint 5 complete — distillation flywheel pipeline running end-to-end. Sprint 6 next.)*