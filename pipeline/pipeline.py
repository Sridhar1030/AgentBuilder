"""
KFP Pipeline -- Distillation Flywheel

Orchestrates the full loop:
  1. Extract gold data (teacher + synthetic)
  2. SFT fine-tune (QLoRA)
  3. Extract preference pairs (eval results + question bank)
  4. DPO fine-tune (on preference data, starting from SFT weights)
  5. Deploy via KServe
  6. Evaluate

Compile:
    python -m kfp.compiler.compiler pipeline.py distillation_flywheel.yaml

Upload via RHOAI Dashboard or:
    from kfp import client
    c = client.Client(host="https://ds-pipeline-dspa-sridharproject.apps.<cluster>/")
    c.upload_pipeline("distillation_flywheel.yaml", pipeline_name="distillation-flywheel")
"""

from kfp import dsl, compiler
from components.resolve_version import resolve_version
from components.extract_gold import extract_gold_data
from components.finetune import finetune
from components.extract_preferences import extract_preferences
from components.dpo_finetune import dpo_finetune
from components.deploy_model import deploy_model
from components.evaluate import evaluate

NAMESPACE = "sridharproject"
S3_ENDPOINT = "http://minio.sridharproject.svc.cluster.local:9000"
MLFLOW_URI = "http://mlflow.sridharproject.svc.cluster.local:5000"
ISVC_NAME = "student-llm"
BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

TEACHER_BUCKET = "mlflow-artifacts"
TEACHER_PREFIX = "code-review-interactions/"
SYNTHETIC_BUCKET = "mlflow-artifacts"
SYNTHETIC_PREFIX = "synthetic/code-review/"
CURSOR_KEY = "code-review-interactions/.cursor.json"
QUESTION_BANK_S3 = "s3://mlflow-artifacts/synthetic/code-review/diff-bank.json"

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
    # Clean code / negative examples (2)
    "Review the following code diff and identify any issues:\n\nFile: pkg/features/features.go\nLanguage: Go\n\n```diff\n@@ -30,6 +30,13 @@\n+const (\n+\tTrainJobRuntimeStatus featuregate.Feature = \"TrainJobRuntimeStatus\"\n+)\n+\n+var defaultFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{\n+\tTrainJobRuntimeStatus: {Default: false, PreRelease: featuregate.Alpha},\n+}\n```",
    "Review the following code diff and identify any issues:\n\nFile: charts/kubeflow-trainer/values.yaml\nLanguage: YAML\n\n```diff\n@@ -138,6 +138,13 @@\n+    statusServer:\n+      # -- Port that the TrainJob status server serves on.\n+      port: 10443\n+      # -- QPS rate limit for the TrainJob Status Server api client\n+      qps: 5\n+      # -- Burst rate limit for the TrainJob Status Server api client\n+      burst: 10\n```",
]


@dsl.pipeline(
    name="distillation-flywheel",
    description="SFT + DPO distillation flywheel: extract gold, SFT, extract preferences, DPO, deploy, evaluate.",
)
def distillation_pipeline(
    model_version: str = "",
    s3_access_key: str = "minioadmin",
    s3_secret_key: str = "minioadmin123",
    teacher_api_url: str = "http://ollama.sridharproject.svc.cluster.local:11434",
    teacher_model: str = "qwen2.5-coder:7b-instruct-q4_K_M",
    teacher_api_key: str = "",
    num_epochs: int = 3,
    dpo_epochs: int = 1,
    dpo_beta: float = 0.1,
    min_gold_threshold: int = 0,
    min_dpo_pairs: int = 5,
    max_supplement_questions: int = 50,
):
    # Step 0 -- Resolve version (auto-increment or use explicit)
    version_task = resolve_version(
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        model_bucket="sridhar-models",
        model_prefix="code-review-1.5b-",
        gold_bucket=TEACHER_BUCKET,
        explicit_version=model_version,
    )
    version_task.set_caching_options(False)

    # Step 1 -- Extract Gold Data (teacher from MinIO incremental + synthetic)
    extract_task = extract_gold_data(
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        teacher_bucket=TEACHER_BUCKET,
        teacher_prefix=TEACHER_PREFIX,
        synthetic_bucket=SYNTHETIC_BUCKET,
        synthetic_prefix=SYNTHETIC_PREFIX,
        cursor_key=CURSOR_KEY,
        output_s3_path=version_task.outputs["gold_data_path"],
        min_threshold=min_gold_threshold,
    )
    extract_task.set_caching_options(False)

    # Step 2 -- SFT Fine-Tune with QLoRA (GPU)
    sft_task = finetune(
        gold_data_path=extract_task.output,
        model_output_s3_path=version_task.outputs["model_output_path"],
        base_model_id=BASE_MODEL_ID,
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        num_epochs=num_epochs,
    )
    sft_task.set_caching_options(False)

    # Step 3 -- Extract Preference Pairs (eval results + question bank -> DPO data)
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
    pref_task.after(sft_task)
    pref_task.set_caching_options(False)

    # Step 4 -- DPO Fine-Tune (starts from SFT weights, uses preference data)
    dpo_task = dpo_finetune(
        sft_model_s3_path=sft_task.output,
        pref_data_s3_path=pref_task.output,
        model_version=version_task.outputs["version"],
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        num_epochs=dpo_epochs,
        dpo_beta=dpo_beta,
        min_pairs=min_dpo_pairs,
    )
    dpo_task.set_caching_options(False)

    # Step 5 -- Deploy Model (DPO model if it ran, otherwise SFT model)
    deploy_task = deploy_model(
        model_s3_path=dpo_task.output,
        isvc_name=ISVC_NAME,
        namespace=NAMESPACE,
    )
    deploy_task.set_caching_options(False)

    # Step 6 -- Evaluate
    eval_task = evaluate(
        student_url=deploy_task.output,
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
        pipeline_func=distillation_pipeline,
        package_path="distillation_flywheel.yaml",
    )
    print("Compiled -> distillation_flywheel.yaml")
