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
BASE_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"

TEACHER_BUCKET = "mlflow-artifacts"
TEACHER_PREFIX = "teacher-interactions/"
SYNTHETIC_BUCKET = "mlflow-artifacts"
SYNTHETIC_PREFIX = "synthetic/"
CURSOR_KEY = "teacher-interactions/.cursor.json"
QUESTION_BANK_S3 = "s3://mlflow-artifacts/synthetic/kubeflow-qbank/questions.json"

TEST_QUESTIONS = [
    # General ML / distillation
    "What is knowledge distillation and why is it useful?",
    "Explain the difference between LoRA and full fine-tuning.",
    "How does QLoRA reduce memory usage during training?",
    "What are the advantages of smaller language models in production?",
    "Describe the role of a teacher model in model compression.",
    # Kubeflow Pipelines
    "What is a KFP component and how do you create one with the Python SDK?",
    "How do you pass artifacts between components in a Kubeflow pipeline?",
    # Training Operator
    "Explain the difference between PyTorchJob and TrainJob in the Training Operator.",
    "How do you configure multi-node distributed training with the Training Operator?",
    # KServe
    "How does KServe handle canary deployments for ML models?",
    "What is the difference between Serverless and RawDeployment mode in KServe?",
    # Katib
    "What search algorithms does Katib support for hyperparameter tuning?",
    "How do you define the search space for a Katib experiment?",
    # Platform
    "How do you set up multi-tenancy in Kubeflow using profiles?",
    "What are the core components of a Kubeflow deployment?",
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
    teacher_model: str = "llama3.1:8b-instruct-q4_K_M",
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
        model_prefix="student-1b-",
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
