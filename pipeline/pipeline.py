"""
KFP Pipeline — Distillation Flywheel

Orchestrates the full loop: extract gold data from MLflow traces,
fine-tune the 1B student with QLoRA, deploy via KServe, and evaluate.

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
from components.deploy_model import deploy_model
from components.evaluate import evaluate

NAMESPACE = "sridharproject"
S3_ENDPOINT = "http://minio.sridharproject.svc.cluster.local:9000"
MLFLOW_URI = "http://mlflow.sridharproject.svc.cluster.local:5000"
ISVC_NAME = "student-llm"
BASE_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"

TEST_QUESTIONS = [
    "What is knowledge distillation and why is it useful?",
    "Explain the difference between LoRA and full fine-tuning.",
    "How does QLoRA reduce memory usage during training?",
    "What are the advantages of smaller language models in production?",
    "Describe the role of a teacher model in model compression.",
]


@dsl.pipeline(
    name="distillation-flywheel",
    description="Extract teacher traces, fine-tune student, deploy, and evaluate.",
)
def distillation_pipeline(
    model_version: str = "",
    s3_access_key: str = "minioadmin",
    s3_secret_key: str = "minioadmin123",
    groq_api_key: str = "",
    groq_model: str = "llama-3.3-70b-versatile",
    num_epochs: int = 3,
    min_gold_threshold: int = 0,
):
    # Step 0 — Resolve version (auto-increment or use explicit)
    version_task = resolve_version(
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        model_bucket="sridhar-models",
        model_prefix="student-1b-",
        gold_bucket="mlflow-artifacts",
        explicit_version=model_version,
    )
    version_task.set_caching_options(False)

    # Step 1 — Extract Gold Data
    extract_task = extract_gold_data(
        mlflow_tracking_uri=MLFLOW_URI,
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        experiment_name="Distillation-Eval-Hub",
        output_s3_path=version_task.outputs["gold_data_path"],
        min_threshold=min_gold_threshold,
    )
    extract_task.set_caching_options(False)

    # Step 2 — Fine-Tune with QLoRA (GPU)
    finetune_task = finetune(
        gold_data_path=extract_task.output,
        model_output_s3_path=version_task.outputs["model_output_path"],
        base_model_id=BASE_MODEL_ID,
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        num_epochs=num_epochs,
    )
    finetune_task.set_caching_options(False)
    # GPU/memory limits removed — resources are now set on the TrainJob pod spec,
    # not the KFP orchestration pod.

    # Step 3 — Deploy Model (patch ISVC)
    deploy_task = deploy_model(
        model_s3_path=finetune_task.output,
        isvc_name=ISVC_NAME,
        namespace=NAMESPACE,
    )
    deploy_task.set_caching_options(False)

    # Step 4 — Evaluate
    eval_task = evaluate(
        student_url=deploy_task.output,
        groq_api_key=groq_api_key,
        groq_model=groq_model,
        test_questions=TEST_QUESTIONS,
    )
    eval_task.set_caching_options(False)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=distillation_pipeline,
        package_path="distillation_flywheel.yaml",
    )
    print("Compiled -> distillation_flywheel.yaml")
