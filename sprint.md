# Distillation Flywheel — Sprint Tracker

**Project:** 70B → 1B Knowledge Distillation on RHOAI  
**Timeline:** 6-month project  
**Cluster:** OpenShift + RHOAI 3.2.0  
**Namespace:** `sridharproject`

---

## Sprint 1: Get the Student Served (Target: 1–2 weeks)

**Goal:** The 1B Student model is live on KServe and the local `app.py` can send it a question and get a response back from the cluster.

**Definition of Done:** Run `python app.py` locally, select "1B Student", ask a question, and get a response served from the KServe pod on OpenShift.


| #    | Task                                   | Status      | Notes                                                                                                 |
| ---- | -------------------------------------- | ----------- | ----------------------------------------------------------------------------------------------------- |
| 1.1  | Verify RHOAI operator stack is healthy | DONE        | RHOAI 3.2.0, Serverless, Pipelines, ServiceMesh, Kueue — all Succeeded. DataScienceCluster Ready=True |
| 1.2  | Set up MinIO object storage            | DONE        | MinIO deployed. Pod `minio-977c79dcb-fgfkv` Running. API on :9000, Console on :9001                   |
| 1.3  | Merge LoRA adapters into base model    | DONE        | `student_model_merged/` created — 4.6G model.safetensors + config + tokenizer                         |
| 1.4  | Upload merged model to Minio bucket    | DONE        | All 7 files in `sridhar-models/student-1b-merged/` — 4.6GiB model.safetensors confirmed               |
| 1.5  | Apply MinIO secret to cluster          | DONE        | Secret + ServiceAccount created                                                                       |
| 1.6  | Apply InferenceService                 | DONE        | Fixed ServingRuntime container name + image. Pod Running 1/1, READY=True                              |
| 1.7  | Wait for pod, debug if needed          | DONE        | Fixed: container name → kserve-container, image → docker.io/vllm/vllm-openai:v0.6.2                  |
| 1.8  | Port-forward and test from terminal    | DONE        | vLLM v0.7.3 fixed OOM. curl test successful — model generates Python code from cluster                |
| 1.9  | Point local app.py at cluster endpoint | DONE        | Installed Python 3.11 via brew, created `venv311`, updated app.py for Gradio 6.x compat              |
| 1.10 | Smoke test end-to-end                  | DONE        | Gradio UI → KServe Student → response + Teacher grading working. "What is 2+2?" → "4" (Grade: 10/10) |


### Sprint 1 Complete

All 10 tasks done. Local Gradio app at `http://127.0.0.1:7860` sends questions to the 1B Student model served by KServe on the cluster via `oc port-forward`, receives generated responses, and the 70B Teacher grades them in real-time.

---

## Sprint 2: MLflow on-cluster

**Goal:** Move experiment tracking from local SQLite to a proper MLflow server on the cluster so traces persist and are accessible from anywhere.

**Definition of Done:** Traces from the Gradio app appear in the MLflow UI on the cluster.


| #   | Task                                     | Status | Notes                                                                                          |
| --- | ---------------------------------------- | ------ | ---------------------------------------------------------------------------------------------- |
| 2.1 | Create mlflow-artifacts bucket in MinIO  | DONE   | `mc mb myminio/mlflow-artifacts` — separate bucket from model weights                          |
| 2.2 | Deploy standalone MLflow server          | DONE   | Deployment + PVC + Service in sridharproject. RHOAI operator was namespace-scoped, used standalone instead |
| 2.3 | Expose MLflow Route                      | DONE   | `https://mlflow-sridharproject.apps.sridhartest-pool-7f6n4.aws.rh-ods.com` — edge TLS          |
| 2.4 | Update app.py MLFLOW_TRACKING_URI        | DONE   | Default URI now points to cluster MLflow. Added `MLFLOW_TRACKING_INSECURE_TLS=true`             |
| 2.5 | Smoke test: traces in cluster MLflow     | DONE   | 3 `chat_interaction` traces logged, artifacts in `s3://mlflow-artifacts/1/traces/`              |

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


| #   | Task                                    | Status | Notes                                                                       |
| --- | --------------------------------------- | ------ | --------------------------------------------------------------------------- |
| 3.0 | Generate 70B Teacher interactions       | DONE   | 10 diverse Q&A pairs via Groq (Llama-3.3-70B) logged in MLflow             |
| 3.1 | Update gold_extractor.py for cluster    | DONE   | Rewritten for MLflow v3 trace format (request/response columns)             |
| 3.2 | Run gold_extractor.py, extract pairs    | DONE   | 10/15 traces identified as teacher pairs                                    |
| 3.3 | Upload gold dataset to MinIO            | DONE   | `s3://mlflow-artifacts/gold/train.jsonl` — 35KiB                            |
| 3.4 | Validate dataset format for SFT         | DONE   | All 10 examples valid. Keys: instruction, output, text. Avg 1690 chars/output |

### Sprint 3 Complete

Gold data pipeline working end-to-end: Teacher traces in MLflow → `gold_extractor.py` → validated JSONL → MinIO. Ready for fine-tuning in Sprint 4.


---

## Sprint 4: Fine-Tuning on KubeRay (Future)

**Goal:** Run QLoRA fine-tuning as a RayJob on-cluster using the gold dataset.


| #   | Task                           | Status      | Notes |
| --- | ------------------------------ | ----------- | ----- |
| 4.1 | Verify Ray operator is enabled | NOT STARTED |       |
| 4.2 | Submit RayJob with finetune.py | NOT STARTED |       |
| 4.3 | Monitor training, pull metrics | NOT STARTED |       |
| 4.4 | Upload new merged model to S3  | NOT STARTED |       |


---

## Sprint 5: Canary Deployment & Eval (Future)

**Goal:** Deploy the improved Student alongside the old one, compare traces against the LLM baseline, and roll forward when ready.


| #   | Task                               | Status      | Notes |
| --- | ---------------------------------- | ----------- | ----- |
| 5.1 | Set up offline evaluation baseline | NOT STARTED |       |
| 5.2 | Deploy canary InferenceService     | NOT STARTED |       |
| 5.3 | Traffic split: 80/20 old/new       | NOT STARTED |       |
| 5.4 | Compare SLM traces vs LLM baseline | NOT STARTED |       |
| 5.5 | Full rollover when metrics pass    | NOT STARTED |       |


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

*Last updated: 2026-03-10 (Sprint 3 done)*


