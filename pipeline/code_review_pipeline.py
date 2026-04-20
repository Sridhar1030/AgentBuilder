"""
KFP Pipeline -- Code Review SLM (Phase 3)

Distills a code review SLM from a teacher LLM using SFT + DPO.
Completely separate from the Phase 2 Kubeflow Q&A pipeline.

Steps:
  0. Resolve version (auto-increment code-review-1.5b-vN)
  1. Extract gold data (reads pre-built training JSONL from MinIO)
  2. SFT fine-tune (QLoRA on Qwen2.5-Coder-1.5B-Instruct)
  3. Deploy SFT model via KServe (temporary, needed for DPO preference extraction)
  4. Extract DPO preference pairs (teacher vs deployed SFT student)
  5. DPO fine-tune (refine SFT model with preferences)
  6. Deploy final DPO model via KServe (overwrites SFT)
  7. Evaluate final model

Compile:
    cd pipeline && python code_review_pipeline.py

Upload via RHOAI Dashboard or:
    from kfp import client
    c = client.Client(host="https://ds-pipeline-dspa-sridharproject.apps.<cluster>/")
    c.upload_pipeline("code_review_pipeline.yaml", pipeline_name="code-review-slm")
"""

from kfp import dsl, compiler
from components.resolve_version import resolve_version
from components.finetune import finetune
from components.extract_preferences import extract_preferences
from components.collect_human_feedback import collect_human_feedback
from components.merge_preferences import merge_preferences
from components.dpo_finetune import dpo_finetune
from components.deploy_model import deploy_model
from components.evaluate import evaluate

# -- Cluster config -------------------------------------------------------
NAMESPACE = "sridharproject"
S3_ENDPOINT = "http://minio.sridharproject.svc.cluster.local:9000"
MLFLOW_URI = "http://mlflow.sridharproject.svc.cluster.local:5000"

# -- Phase 3 isolation ----------------------------------------------------
ISVC_NAME = "code-review-llm"
BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

TEACHER_BUCKET = "mlflow-artifacts"
TEACHER_PREFIX = "code-review-interactions/"
SYNTHETIC_BUCKET = "mlflow-artifacts"
SYNTHETIC_PREFIX = "synthetic/code-review/"
CURSOR_KEY = "code-review-interactions/.cursor.json"
QUESTION_BANK_S3 = "s3://mlflow-artifacts/synthetic/code-review/diff-bank.json"

# -- 15 curated test diffs from kubeflow/trainer for quick regression -----
TEST_QUESTIONS = [
    # Bug detection (3)
    "Review the following code diff and identify any issues:\n\nFile: pkg/controller/job_controller.go\nLanguage: Go\n\n```diff\n@@ -189,7 +189,7 @@\n func (p *Progress) buildProgressServerCaCrtConfigMap(ctx context.Context, trainJob *trainer.TrainJob) (*corev1ac.ConfigMapApplyConfiguration, error) {\n \tsecret := &corev1.Secret{}\n \tif err := p.client.Get(ctx, secretKey, secret); err == nil {\n \t\tif _, ok := secret.Data[\"ca.crt\"]; !ok {\n-\t\t\treturn nil, fmt.Errorf(\"ca.crt not found: %w\", err)\n+\t\t\treturn nil, fmt.Errorf(\"ca.crt not found in TLS secret\")\n \t\t}\n```",
    "Review the following code diff and identify any issues:\n\nFile: test/e2e/testdata/status_update.py\nLanguage: Python\n\n```diff\n@@ -0,0 +1,30 @@\n+import os, urllib.request\n+\n+token = open(os.environ[\"KUBEFLOW_TRAINER_SERVER_TOKEN\"]).read()\n+req = urllib.request.Request(url, method=\"POST\")\n+req.add_header(\"Authorization\", f\"Bearer {token}\")\n```",
    "Review the following code diff and identify any issues:\n\nFile: pkg/webhooks/trainjob_webhook.go\nLanguage: Go\n\n```diff\n@@ -50,6 +50,10 @@\n+func (d *TrainJobDefaulter) Default(ctx context.Context, trainJob *trainer.TrainJob) error {\n+\tnow := metav1.Now()\n+\tfor i := range trainJob.Spec.RuntimePatches {\n+\t\ttrainJob.Spec.RuntimePatches[i].Time = &now\n+\t}\n```",
    # Security (2)
    "Review the following code diff and identify any issues:\n\nFile: pkg/util/cert/cert.go\nLanguage: Go\n\n```diff\n@@ -69,6 +69,12 @@\n+func SetupTLSConfig(mgr ctrl.Manager) (*tls.Config, error) {\n+\tcertWatcher, err := certwatcher.New(certDir+\"/tls.crt\", certDir+\"/tls.key\")\n+\tif err != nil { return nil, err }\n+\treturn &tls.Config{\n+\t\tGetCertificate: certWatcher.GetCertificate,\n+\t}, nil\n+}\n```",
    "Review the following code diff and identify any issues:\n\nFile: manifests/overlays/manager/kustomization.yaml\nLanguage: YAML\n\n```diff\n@@ -3,4 +3,4 @@\n resources:\n   - ../../base/rbac\n   - ../../base/manager\n-  - ../../base/webhook\n+  # - ../../base/webhook\n```",
    # Performance (2)
    "Review the following code diff and identify any issues:\n\nFile: pkg/status/server.go\nLanguage: Go\n\n```diff\n@@ -100,6 +100,10 @@\n+func (s *Server) authorizeRequest(r *http.Request, namespace, name string) bool {\n+\tpod := &corev1.Pod{}\n+\tif err := s.client.Get(r.Context(), types.NamespacedName{Namespace: namespace, Name: podName}, pod); err != nil {\n+\t\treturn false\n+\t}\n```",
    "Review the following code diff and identify any issues:\n\nFile: pkg/runtime/framework/plugins/flux/flux.go\nLanguage: Go\n\n```diff\n@@ -402,8 +406,12 @@\n+\tvar tasks int32\n+\tnodes := *trainJob.Spec.Trainer.NumNodes\n+\tif trainJob.Spec.Trainer.NumProcPerNode != nil {\n+\t\ttasks = *trainJob.Spec.Trainer.NumProcPerNode\n+\t}\n+\tflags := fmt.Sprintf(\"-N %d -n %d\", nodes, tasks)\n+\trspec := fmt.Sprintf(\"--cores=0-%d\", tasks-1)\n```",
    # Kubernetes-specific (2)
    "Review the following code diff and identify any issues:\n\nFile: manifests/base/webhook/manifests.yaml\nLanguage: YAML\n\n```diff\n@@ -0,0 +1,15 @@\n+apiVersion: admissionregistration.k8s.io/v1\n+kind: MutatingWebhookConfiguration\n+metadata:\n+  name: mutating-webhook-configuration\n+webhooks:\n+- clientConfig:\n+    service:\n+      name: webhook-service\n+      namespace: system\n+      path: /mutate-trainer-kubeflow-org-v1alpha1-trainjob\n```",
    "Review the following code diff and identify any issues:\n\nFile: manifests/base/manager/controller_manager_config.yaml\nLanguage: YAML\n\n```diff\n@@ -5,4 +5,4 @@\n controller:\n   manager:\n-    leaderElect: true\n+    leaderElect: false\n   certManagement:\n-    enable: true\n+    enable: false\n```",
    # Reliability (2)
    "Review the following code diff and identify any issues:\n\nFile: pkg/statusserver/middleware.go\nLanguage: Go\n\n```diff\n@@ -0,0 +1,20 @@\n+func authenticationMiddleware(logger logr.Logger) Middleware {\n+\treturn func(next http.Handler) http.Handler {\n+\t\treturn http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {\n+\t\t\tauth := r.Header.Get(\"Authorization\")\n+\t\t\tif !strings.HasPrefix(auth, \"Bearer \") {\n+\t\t\t\thttp.Error(w, \"Invalid Authorization header format\", http.StatusUnauthorized)\n+\t\t\t\treturn\n+\t\t\t}\n+\t\t\ttoken := strings.TrimPrefix(auth, \"Bearer \")\n+\t\t\tif token == \"\" {\n+\t\t\t\thttp.Error(w, \"Empty bearer token\", http.StatusUnauthorized)\n+\t\t\t\treturn\n+\t\t\t}\n```",
    "Review the following code diff and identify any issues:\n\nFile: pkg/runtime/framework/plugins/plainml/plainml.go\nLanguage: Go\n\n```diff\n@@ -80,6 +80,10 @@\n+\tfor i, patch := range trainJob.Spec.RuntimePatches {\n+\t\tif i < len(old.Spec.RuntimePatches) && reflect.DeepEqual(patch, old.Spec.RuntimePatches[i]) {\n+\t\t\ttrainJob.Spec.RuntimePatches[i].Time = old.Spec.RuntimePatches[i].Time\n+\t\t} else {\n+\t\t\ttrainJob.Spec.RuntimePatches[i].Time = &now\n+\t\t}\n```",
    # Style / Refactor (2)
    "Review the following code diff and identify any issues:\n\nFile: pkg/constants/constants.go\nLanguage: Go\n\n```diff\n@@ -114,4 +114,7 @@\n \tFluxCurveVolumePath = \"/curve\"\n \n+\t// Ensure MPI has full memory of the host\n+\tFluxMemoryVolumeName = \"dshm\"\n```",
    "Review the following code diff and identify any issues:\n\nFile: pkg/runtime/core/trainingruntime.go\nLanguage: Go\n\n```diff\n@@ -50,6 +50,15 @@\n+func mergeRuntimePatches(jobSet *jobsetv1.JobSet, patches []trainer.RuntimePatch) {\n+\tfor _, rJobPatch := range patches {\n+\t\tfor i, rJob := range jobSet.Spec.ReplicatedJobs {\n+\t\t\tif rJob.Name == rJobPatch.TargetReplicatedJob {\n+\t\t\t\tapplyPodPatch(&jobSet.Spec.ReplicatedJobs[i].Template.Spec.Template, rJobPatch.Template.PodTemplatePatch)\n+\t\t\t}\n+\t\t}\n+\t}\n+}\n```",
    # Clean code / negative (2)
    "Review the following code diff and identify any issues:\n\nFile: pkg/features/features.go\nLanguage: Go\n\n```diff\n@@ -30,6 +30,13 @@\n+const (\n+\tTrainJobRuntimeStatus featuregate.Feature = \"TrainJobRuntimeStatus\"\n+)\n+\n+var defaultFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{\n+\tTrainJobRuntimeStatus: {Default: false, PreRelease: featuregate.Alpha},\n+}\n```",
    "Review the following code diff and identify any issues:\n\nFile: charts/kubeflow-trainer/values.yaml\nLanguage: YAML\n\n```diff\n@@ -138,6 +138,13 @@\n+    statusServer:\n+      # -- Port that the TrainJob status server serves on.\n+      port: 10443\n+      # -- QPS rate limit for the TrainJob Status Server api client\n+      qps: 5\n+      # -- Burst rate limit for the TrainJob Status Server api client\n+      burst: 10\n```",
]


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["boto3"],
)
def extract_code_review_gold(
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    synthetic_bucket: str,
    synthetic_prefix: str,
    output_s3_path: str,
) -> str:
    """Read pre-built code review training JSONL from MinIO.

    Unlike the generic extract_gold, this component reads JSONL files
    that already have a `text` field in ChatML format, so no instruction/output
    parsing is needed.
    """
    import json
    import random
    import boto3

    print("=" * 60)
    print("EXTRACT CODE REVIEW GOLD DATA STEP")
    print("=" * 60)
    print(f"  Source:  s3://{synthetic_bucket}/{synthetic_prefix}")
    print(f"  Output:  {output_s3_path}")
    print("=" * 60)

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )

    records = []
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=synthetic_bucket, Prefix=synthetic_prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.endswith(".jsonl"):
                continue
            try:
                body = s3.get_object(Bucket=synthetic_bucket, Key=key)["Body"].read().decode("utf-8")
                for line in body.strip().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if not record.get("text"):
                        continue
                    records.append(record)
            except Exception as exc:
                print(f"Warning: skipping {key}: {exc}")

    print(f"Loaded {len(records)} training records from s3://{synthetic_bucket}/{synthetic_prefix}")

    if len(records) > 1:
        random.shuffle(records)

    out_parts = output_s3_path.replace("s3://", "").split("/", 1)
    out_bucket, out_key = out_parts[0], out_parts[1]
    body = "\n".join(json.dumps(r) for r in records)
    s3.put_object(Bucket=out_bucket, Key=out_key, Body=body.encode())
    print(f"Uploaded {len(records)} gold records to {output_s3_path}")

    return output_s3_path


@dsl.pipeline(
    name="code-review-slm",
    description="Code Review SLM: SFT + DPO distillation on Go/Python/K8s code diffs using Qwen2.5-Coder-1.5B.",
)
def code_review_pipeline(
    model_version: str = "",
    s3_access_key: str = "minioadmin",
    s3_secret_key: str = "minioadmin123",
    teacher_api_url: str = "http://ollama.sridharproject.svc.cluster.local:11434",
    teacher_model: str = "qwen2.5-coder:32b-instruct-q4_K_M",
    teacher_api_key: str = "",
    num_epochs: int = 3,
    dpo_epochs: int = 1,
    dpo_beta: float = 0.3,
    min_dpo_pairs: int = 3,
    max_supplement_questions: int = 5,
):
    # Step 0 -- Resolve version (auto-increment code-review-1.5b-vN)
    version_task = resolve_version(
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        model_bucket="sridhar-models",
        model_prefix="code-review-1.5b-",
        gold_bucket=TEACHER_BUCKET,
        hf_base_model_id=BASE_MODEL_ID,
        explicit_version=model_version,
    )
    version_task.set_caching_options(False)

    # Step 1 -- Extract gold data (reads pre-built ChatML JSONL from MinIO)
    extract_task = extract_code_review_gold(
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        synthetic_bucket=SYNTHETIC_BUCKET,
        synthetic_prefix=SYNTHETIC_PREFIX,
        output_s3_path=version_task.outputs["gold_data_path"],
    )
    extract_task.set_caching_options(False)

    # Step 2 -- SFT fine-tune with QLoRA (GPU)
    # Iterative training (Option B): resolve_version returns the previous run's
    # S3 model path if one exists, otherwise returns the HF base model ID.
    sft_task = finetune(
        gold_data_path=extract_task.output,
        model_output_s3_path=version_task.outputs["model_output_path"],
        base_model_id=version_task.outputs["prev_model_path"],
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        num_epochs=num_epochs,
    )
    sft_task.set_caching_options(False)

    # Step 3 -- Deploy SFT model via KServe (needed for DPO preference extraction)
    deploy_sft_task = deploy_model(
        model_s3_path=sft_task.output,
        isvc_name=ISVC_NAME,
        namespace=NAMESPACE,
    )
    deploy_sft_task.set_caching_options(False)

    # Step 4a -- Collect human feedback DPO pairs from MinIO (runs early,
    #            parallel with SFT+deploy since it only reads from S3)
    human_fb_task = collect_human_feedback(
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
    )
    human_fb_task.after(extract_task)
    human_fb_task.set_caching_options(False)

    # Step 4b -- Extract DPO preference pairs (teacher vs deployed SFT on diff-bank)
    pref_task = extract_preferences(
        student_url=f"http://{ISVC_NAME}-predictor.{NAMESPACE}.svc.cluster.local:8080",
        teacher_api_url=teacher_api_url,
        teacher_model=teacher_model,
        teacher_api_key=teacher_api_key,
        question_bank_s3_path=QUESTION_BANK_S3,
        mlflow_tracking_uri=MLFLOW_URI,
        sft_run_name_prefix="pipeline-eval-",
        model_version=version_task.outputs["version"],
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        max_supplement_questions=max_supplement_questions,
    )
    pref_task.after(deploy_sft_task)
    pref_task.set_caching_options(False)

    # Step 4c -- Merge all preference sources (static bank + live + human + SFT regularization)
    merge_task = merge_preferences(
        pipeline_pref_path=pref_task.output,
        human_feedback_path=human_fb_task.output,
        gold_data_path=extract_task.output,
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
    )
    merge_task.set_caching_options(False)

    # Step 5 -- DPO fine-tune (refine SFT model with merged preference data)
    dpo_task = dpo_finetune(
        sft_model_s3_path=sft_task.output,
        pref_data_s3_path=merge_task.output,
        model_version=version_task.outputs["version"],
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        num_epochs=dpo_epochs,
        dpo_beta=dpo_beta,
        min_pairs=min_dpo_pairs,
    )
    dpo_task.set_caching_options(False)

    # Step 6 -- Deploy final DPO model via KServe
    deploy_dpo_task = deploy_model(
        model_s3_path=dpo_task.output,
        isvc_name=ISVC_NAME,
        namespace=NAMESPACE,
    )
    deploy_dpo_task.set_caching_options(False)

    # Step 7 -- Evaluate final model
    eval_task = evaluate(
        student_url=deploy_dpo_task.output,
        teacher_api_url=teacher_api_url,
        teacher_model=teacher_model,
        teacher_api_key=teacher_api_key,
        test_questions=TEST_QUESTIONS,
        mlflow_tracking_uri=MLFLOW_URI,
        model_version=version_task.outputs["version"],
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
    )
    eval_task.set_caching_options(False)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=code_review_pipeline,
        package_path="code_review_pipeline.yaml",
    )
    print("Compiled -> code_review_pipeline.yaml")
