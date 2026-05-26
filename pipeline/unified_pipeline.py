"""
KFP Pipeline -- Agentic Continual Learning (Unified)

Single Kubeflow Pipeline implementing Antonin's full architecture.
The distill-pipeline is a nested sub-pipeline that renders as a
single collapsible node in the KFP UI -- click it to expand.

  eval-analyze --> eval-dataset --> [distill-pipeline] --> deploy-candidate --> eval-run --> eval-optimize
                                         |
                                    click to expand:
                                    resolve-version -> extract-gold -> SFT -> deploy-sft
                                    -> extract-prefs -> DPO -> GRPO

Compile:
    cd pipeline && python unified_pipeline.py
"""

import json
from pathlib import Path
from typing import NamedTuple

from kfp import dsl, compiler
from components.resolve_version import resolve_version
from components.finetune import finetune
from components.extract_preferences import extract_preferences
from components.collect_human_feedback import collect_human_feedback
from components.merge_preferences import merge_preferences
from components.dpo_finetune import dpo_finetune
from components.grpo_finetune import grpo_finetune
from components.deploy_model import deploy_model
from components.evaluate import evaluate
from components.quality_gate import quality_gate
from components.eval_optimize import eval_optimize
from components.eval_analyze import eval_analyze
from components.eval_dataset import eval_dataset
from components.traffic_shift import traffic_shift
from components.extract_gold import extract_code_review_gold
from config import load_config


_CFG = load_config()

NAMESPACE = _CFG["cluster"]["namespace"]
S3_ENDPOINT = _CFG["cluster"]["s3_endpoint"]
MLFLOW_URI = _CFG["cluster"]["mlflow_uri"]
ISVC_NAME = _CFG["student"]["isvc_name"]
BASE_MODEL_ID = _CFG["student"]["base_model_id"]
TEACHER_BUCKET = _CFG["cluster"]["data_bucket"]
SYNTHETIC_BUCKET = _CFG["cluster"]["data_bucket"]
SYNTHETIC_PREFIX = _CFG["domain"]["training_data_prefix"]
QUESTION_BANK_S3 = _CFG["domain"]["question_bank_s3"]

TEACHER_SYSTEM_PROMPT = _CFG["teacher"]["system_prompt"]

CANARY_ENABLED = _CFG.get("canary", {}).get("enabled", False)
CANARY_GATEWAY_URL = _CFG.get("canary", {}).get("gateway_url", "")
CANARY_VS_NAME = _CFG.get("canary", {}).get("virtualservice_name", "code-review-gateway")
CANARY_SHIFT_INCREMENT = _CFG.get("canary", {}).get("shift_increment", 10)

_tq_file = Path(__file__).resolve().parent.parent / _CFG["domain"]["test_questions_file"]
if _tq_file.exists():
    with open(_tq_file) as f:
        TEST_QUESTIONS = json.load(f)
else:
    raise FileNotFoundError(f"Test questions file not found: {_tq_file}")

_eval_yaml_file = Path(__file__).resolve().parent.parent / _CFG["domain"]["eval_yaml"]
if _eval_yaml_file.exists():
    with open(_eval_yaml_file) as f:
        EVAL_YAML_CONTENT = f.read()
else:
    EVAL_YAML_CONTENT = ""


# -- Helper: extract weak categories from eval-analyze output -----------------

@dsl.component(base_image="python:3.11-slim")
def _extract_weak_categories(analysis: dict) -> str:
    """Extract weak_categories list from eval-analyze output as JSON string."""
    import json
    cats = analysis.get("weak_categories", [])
    return json.dumps(cats)


# ==============================================================================
# SUB-PIPELINE: distill-pipeline (SFT + DPO)
#
# This renders as a SINGLE collapsible node in the KFP UI.
# Click it to see the inner training steps.
# ==============================================================================

DistillOutputs = NamedTuple("DistillOutputs", [
    ("grpo_model_path", str),
    ("grpo_output_s3_path", str),
    ("dpo_model_path", str),
    ("dpo_output_s3_path", str),
    ("sft_model_path", str),
    ("version", str),
])


@dsl.pipeline(
    name="distill-pipeline",
    description="Inner training loop: SFT + DPO distillation via Training Hub.",
)
def distill_pipeline(
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    model_bucket: str,
    model_prefix: str,
    gold_bucket: str,
    base_model_id: str,
    synthetic_bucket: str,
    synthetic_prefix: str,
    isvc_name: str,
    namespace: str,
    teacher_api_url: str,
    teacher_model: str,
    teacher_api_key: str,
    question_bank_s3: str,
    mlflow_tracking_uri: str,
    system_prompt: str = "",
    model_version: str = "",
    num_epochs: int = 3,
    dpo_epochs: int = 3,
    dpo_beta: float = 0.3,
    min_dpo_pairs: int = 10,
    max_supplement_questions: int = 5,
    grpo_data_s3_path: str = "",
    grpo_epochs: int = 1,
    grpo_learning_rate: float = 5e-7,
    grpo_num_generations: int = 4,
    grpo_beta: float = 0.0,
    grpo_min_prompts: int = 10,
    grpo_max_completion_length: int = 512,
    grpo_loss_type: str = "dapo",
    grpo_temperature: float = 0.7,
    grpo_use_vllm: bool = False,
    test_questions_json: str = "",
    weak_categories_json: str = "",
) -> DistillOutputs:
    # -- Resolve version (auto-increment code-review-1.5b-vN) --
    version_task = resolve_version(
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        model_bucket=model_bucket,
        model_prefix=model_prefix,
        gold_bucket=gold_bucket,
        hf_base_model_id=base_model_id,
        explicit_version=model_version,
    )
    version_task.set_caching_options(False)

    # -- Extract gold data --
    extract_task = extract_code_review_gold(
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        synthetic_bucket=synthetic_bucket,
        synthetic_prefix=synthetic_prefix,
        output_s3_path=version_task.outputs["gold_data_path"],
    )
    extract_task.set_caching_options(False)

    # -- SFT fine-tune with QLoRA (GPU) --
    sft_task = finetune(
        gold_data_path=extract_task.output,
        model_output_s3_path=version_task.outputs["model_output_path"],
        base_model_id=version_task.outputs["prev_model_path"],
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        num_epochs=num_epochs,
    )
    sft_task.set_caching_options(False)

    # -- Deploy SFT model (temporary, for preference extraction) --
    deploy_sft_task = deploy_model(
        model_s3_path=sft_task.output,
        isvc_name=isvc_name,
        namespace=namespace,
    )
    deploy_sft_task.set_caching_options(False)

    # -- Collect human feedback --
    human_fb_task = collect_human_feedback(
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
    )
    human_fb_task.after(extract_task)
    human_fb_task.set_caching_options(False)

    # -- Extract DPO preference pairs (teacher vs deployed SFT) --
    pref_task = extract_preferences(
        student_url=f"http://{isvc_name}-predictor.{namespace}.svc.cluster.local:8080",
        teacher_api_url=teacher_api_url,
        teacher_model=teacher_model,
        teacher_api_key=teacher_api_key,
        question_bank_s3_path=question_bank_s3,
        mlflow_tracking_uri=mlflow_tracking_uri,
        sft_run_name_prefix="pipeline-eval-",
        model_version=version_task.outputs["version"],
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        system_prompt=system_prompt,
        max_supplement_questions=max_supplement_questions,
    )
    pref_task.after(deploy_sft_task)
    pref_task.set_caching_options(False)

    # -- Merge all preference sources --
    merge_task = merge_preferences(
        pipeline_pref_path=pref_task.output,
        human_feedback_path=human_fb_task.output,
        gold_data_path=extract_task.output,
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
    )
    merge_task.set_caching_options(False)

    # -- DPO fine-tune (writes to separate vN-dpo/ prefix) --
    dpo_task = dpo_finetune(
        sft_model_s3_path=sft_task.output,
        pref_data_s3_path=merge_task.output,
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        dpo_output_s3_path=version_task.outputs["dpo_model_output_path"],
        num_epochs=dpo_epochs,
        dpo_beta=dpo_beta,
        min_pairs=min_dpo_pairs,
    )
    dpo_task.set_caching_options(False)

    # -- GRPO fine-tune (verifiable rewards; after DPO) --
    grpo_task = grpo_finetune(
        dpo_model_s3_path=dpo_task.output,
        grpo_data_s3_path=grpo_data_s3_path,
        model_version=version_task.outputs["version"],
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        grpo_output_s3_path=version_task.outputs["grpo_model_output_path"],
        system_prompt=system_prompt,
        test_questions_json=test_questions_json,
        num_epochs=grpo_epochs,
        learning_rate=grpo_learning_rate,
        num_generations=grpo_num_generations,
        grpo_beta=grpo_beta,
        temperature=grpo_temperature,
        min_prompts=grpo_min_prompts,
        max_completion_length=grpo_max_completion_length,
        loss_type=grpo_loss_type,
        weak_categories_json=weak_categories_json,
        use_vllm=grpo_use_vllm,
    )
    grpo_task.set_caching_options(False)

    return DistillOutputs(
        grpo_model_path=grpo_task.output,
        grpo_output_s3_path=version_task.outputs["grpo_model_output_path"],
        dpo_model_path=dpo_task.output,
        dpo_output_s3_path=version_task.outputs["dpo_model_output_path"],
        sft_model_path=sft_task.output,
        version=version_task.outputs["version"],
    )


# ==============================================================================
# MAIN PIPELINE: Agentic Continual Learning
#
# Top-level nodes visible in the KFP UI:
#   eval-analyze -> eval-dataset -> [distill-pipeline] -> deploy-candidate
#   -> eval-run -> quality-gate -> eval-optimize -> (rollback if fail)
# ==============================================================================

@dsl.pipeline(
    name="agentic-continual-learning",
    description=(
        "Unified Agentic Continual Learning pipeline: "
        "eval-analyze, eval-dataset, distill (SFT+DPO+GRPO), "
        "eval-run, eval-optimize, deploy-candidate."
    ),
)
def agentic_continual_learning_pipeline(
    model_version: str = "",
    s3_access_key: str = _CFG["cluster"]["s3_access_key"],
    s3_secret_key: str = _CFG["cluster"]["s3_secret_key"],
    teacher_api_url: str = _CFG["teacher"]["api_url"],
    teacher_model: str = _CFG["teacher"]["model"],
    teacher_api_key: str = _CFG["teacher"].get("api_key", ""),
    num_epochs: int = _CFG["training"]["sft_epochs"],
    dpo_epochs: int = _CFG["training"]["dpo_epochs"],
    dpo_beta: float = _CFG["training"]["dpo_beta"],
    min_dpo_pairs: int = _CFG["training"]["min_dpo_pairs"],
    max_supplement_questions: int = _CFG["training"]["max_supplement_questions"],
    max_new_sdg_examples: int = 10,
    grpo_epochs: int = _CFG.get("grpo", {}).get("num_epochs", 1),
    grpo_learning_rate: float = _CFG.get("grpo", {}).get("learning_rate", 5e-7),
    grpo_num_generations: int = _CFG.get("grpo", {}).get("num_generations", 4),
    grpo_beta: float = _CFG.get("grpo", {}).get("beta", 0.0),
    grpo_min_prompts: int = _CFG.get("grpo", {}).get("min_prompts", 10),
    grpo_max_completion_length: int = _CFG.get("grpo", {}).get("max_completion_length", 512),
    grpo_loss_type: str = _CFG.get("grpo", {}).get("loss_type", "dapo"),
    grpo_temperature: float = _CFG.get("grpo", {}).get("temperature", 0.7),
    grpo_use_vllm: bool = _CFG.get("grpo", {}).get("use_vllm", False),
):
    # =========================================================================
    # 1. EVAL-ANALYZE  (from agent-eval-harness)
    # =========================================================================
    analyze_task = eval_analyze(
        mlflow_tracking_uri=MLFLOW_URI,
        experiment_name="CodeReview-Eval-Hub",
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
    )
    analyze_task.set_caching_options(False)

    # =========================================================================
    # 2. EVAL-DATASET  (+ SDG Hub integration)
    # =========================================================================
    dataset_task = eval_dataset(
        analysis=analyze_task.output,
        teacher_api_url=teacher_api_url,
        teacher_model=teacher_model,
        teacher_api_key=teacher_api_key,
        teacher_system_prompt=_CFG["teacher"]["system_prompt"],
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        data_bucket=_CFG["cluster"]["data_bucket"],
        output_prefix=SYNTHETIC_PREFIX,
        model_version=model_version,
        max_new_examples=max_new_sdg_examples,
    )
    dataset_task.set_caching_options(False)

    # -- Extract weak categories from analysis for GRPO curriculum --
    extract_weak_cats = _extract_weak_categories(analysis=analyze_task.output)
    extract_weak_cats.set_caching_options(False)

    # =========================================================================
    # 3. DISTILL-PIPELINE  (SFT + DPO + GRPO -- collapsible sub-pipeline node)
    # =========================================================================
    distill_task = distill_pipeline(
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        model_bucket=_CFG["cluster"]["model_bucket"],
        model_prefix=_CFG["student"]["model_prefix"],
        gold_bucket=TEACHER_BUCKET,
        base_model_id=BASE_MODEL_ID,
        synthetic_bucket=SYNTHETIC_BUCKET,
        synthetic_prefix=SYNTHETIC_PREFIX,
        isvc_name=ISVC_NAME,
        namespace=NAMESPACE,
        teacher_api_url=teacher_api_url,
        teacher_model=teacher_model,
        teacher_api_key=teacher_api_key,
        question_bank_s3=QUESTION_BANK_S3,
        mlflow_tracking_uri=MLFLOW_URI,
        system_prompt=TEACHER_SYSTEM_PROMPT,
        model_version=model_version,
        num_epochs=num_epochs,
        dpo_epochs=dpo_epochs,
        dpo_beta=dpo_beta,
        min_dpo_pairs=min_dpo_pairs,
        max_supplement_questions=max_supplement_questions,
        grpo_data_s3_path=_CFG.get("grpo", {}).get(
            "diff_bank_path", QUESTION_BANK_S3
        ),
        grpo_epochs=grpo_epochs,
        grpo_learning_rate=grpo_learning_rate,
        grpo_num_generations=grpo_num_generations,
        grpo_beta=grpo_beta,
        grpo_min_prompts=grpo_min_prompts,
        grpo_max_completion_length=grpo_max_completion_length,
        grpo_loss_type=grpo_loss_type,
        grpo_temperature=grpo_temperature,
        grpo_use_vllm=grpo_use_vllm,
        test_questions_json=json.dumps(TEST_QUESTIONS),
        weak_categories_json=extract_weak_cats.output,
    )
    distill_task.after(dataset_task)

    # =========================================================================
    # 4. DEPLOY-CANDIDATE  (KServe -- staging for evaluation)
    # =========================================================================
    deploy_candidate_task = deploy_model(
        model_s3_path=distill_task.outputs["grpo_model_path"],
        isvc_name=ISVC_NAME,
        namespace=NAMESPACE,
    )
    deploy_candidate_task.set_caching_options(False)

    # =========================================================================
    # 5. EVAL-RUN  (via Eval Hub / agent-eval-harness judges)
    # =========================================================================
    eval_task = evaluate(
        student_url=deploy_candidate_task.output,
        teacher_api_url=teacher_api_url,
        teacher_model=teacher_model,
        teacher_api_key=teacher_api_key,
        test_questions=TEST_QUESTIONS,
        eval_yaml_content=EVAL_YAML_CONTENT,
        system_prompt=TEACHER_SYSTEM_PROMPT,
        mlflow_tracking_uri=MLFLOW_URI,
        model_version=distill_task.outputs["version"],
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
    )
    eval_task.set_caching_options(False)

    gate_task = quality_gate(eval_results=eval_task.output)
    gate_task.set_caching_options(False)

    # =========================================================================
    # 6. EVAL-OPTIMIZE  (grading criteria, system prompts, data mix)
    # =========================================================================
    optimize_task = eval_optimize(
        eval_results=eval_task.output,
        teacher_api_url=teacher_api_url,
        teacher_model=teacher_model,
        teacher_api_key=teacher_api_key,
        current_config=json.dumps(_CFG),
        dpo_model_s3_path=distill_task.outputs["grpo_model_path"],
        mlflow_tracking_uri=MLFLOW_URI,
        model_version=distill_task.outputs["version"],
        s3_endpoint=S3_ENDPOINT,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
    )
    optimize_task.after(gate_task)
    optimize_task.set_caching_options(False)

    # =========================================================================
    # DEPLOY DECISION: Rollback if quality gate fails
    # =========================================================================
    with dsl.If(gate_task.output == "fail", name="rollback-on-failure"):
        rollback_task = deploy_model(
            model_s3_path=distill_task.outputs["sft_model_path"],
            isvc_name=ISVC_NAME,
            namespace=NAMESPACE,
        )
        rollback_task.set_caching_options(False)

    # =========================================================================
    # 7. TRAFFIC SHIFT  (canary progressive migration)
    #    Only runs if canary.enabled=true in distill.config.yaml.
    #    Shifts traffic from teacher to student when eval score improves.
    # =========================================================================
    if CANARY_ENABLED:
        shift_task = traffic_shift(
            eval_results=eval_task.output,
            gateway_url=CANARY_GATEWAY_URL,
            namespace=NAMESPACE,
            virtualservice_name=CANARY_VS_NAME,
            shift_increment=CANARY_SHIFT_INCREMENT,
            mlflow_tracking_uri=MLFLOW_URI,
            model_version=distill_task.outputs["version"],
            s3_endpoint=S3_ENDPOINT,
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key,
        )
        shift_task.after(optimize_task)
        shift_task.set_caching_options(False)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=agentic_continual_learning_pipeline,
        package_path="unified_pipeline.yaml",
    )
    print("Compiled -> unified_pipeline.yaml")
