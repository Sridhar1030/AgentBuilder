"""
KFP Component -- Eval Optimize (agent-eval-harness /eval-optimize)

Analyzes evaluation results from the harness judges, identifies patterns
in failures, and generates recommendations for improving the next
training iteration.

This is the optimization feedback loop that Antonin's architecture calls
the "outer loop" -- it looks at judge failures and recommends adjustments
to training parameters, data mix, or system prompts.

Recommendations are logged to MLflow and returned as a dict that the
outer pipeline can feed into the next inner pipeline run.
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["requests", "mlflow", "boto3"],
)
def eval_optimize(
    eval_results: dict,
    teacher_api_url: str,
    teacher_model: str,
    teacher_api_key: str,
    current_config: str,
    mlflow_tracking_uri: str = "",
    model_version: str = "unknown",
    s3_endpoint: str = "",
    s3_access_key: str = "",
    s3_secret_key: str = "",
) -> dict:
    """Analyze judge failures and recommend training adjustments.

    Args:
        eval_results: Output from the evaluate component (judge scores,
            regressions, per-case results).
        current_config: JSON string of the current distill.config.yaml
            training parameters.

    Returns:
        Dict with recommendations for the next training iteration.
    """
    import json
    import os
    import sys
    import time
    import requests
    import mlflow

    sys.stdout.reconfigure(line_buffering=True)

    if mlflow_tracking_uri.startswith("https://"):
        os.environ.setdefault("MLFLOW_TRACKING_INSECURE_TLS", "true")
    if s3_endpoint:
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", s3_endpoint)
        if s3_endpoint.startswith("https://"):
            os.environ.setdefault("MLFLOW_S3_IGNORE_TLS", "true")
    if s3_access_key:
        os.environ.setdefault("AWS_ACCESS_KEY_ID", s3_access_key)
    if s3_secret_key:
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", s3_secret_key)

    api_url = teacher_api_url.rstrip("/")
    if not api_url.endswith("/v1/chat/completions"):
        api_url = api_url.rstrip("/") + "/v1/chat/completions"
    api_headers = {"Content-Type": "application/json"}
    if teacher_api_key:
        api_headers["Authorization"] = f"Bearer {teacher_api_key}"

    print("=" * 60)
    print("EVAL-OPTIMIZE (agent-eval-harness outer loop)")
    print("=" * 60)

    gate_result = eval_results.get("gate_result", "unknown")
    composite = eval_results.get("composite_score", 0.0)
    regressions = eval_results.get("regressions", [])
    judge_agg = eval_results.get("judge_aggregates", {})

    print(f"  Gate result:    {gate_result}")
    print(f"  Composite:      {composite:.4f}")
    print(f"  Regressions:    {len(regressions)}")

    config = {}
    if current_config:
        try:
            config = json.loads(current_config)
        except json.JSONDecodeError:
            print("  WARNING: could not parse current_config")

    # -- Analyze failure patterns ------------------------------------------

    per_case = eval_results.get("results", []) if "results" in eval_results else []

    failure_patterns = {
        "correctness_failures": [],
        "conciseness_failures": [],
        "quality_failures": [],
        "format_failures": [],
    }

    for case in per_case:
        judges = case.get("judges", {})
        case_id = case.get("case_id", "?")
        annotations = case.get("annotations", {})

        for jname, jresult in judges.items():
            v = jresult.get("value")
            if v is False or (isinstance(v, (int, float)) and v <= 2):
                key = f"{jname}_failures"
                if key in failure_patterns:
                    failure_patterns[key].append({
                        "case_id": case_id,
                        "category": annotations.get("category", "?"),
                        "has_bug": annotations.get("has_bug"),
                        "rationale": jresult.get("rationale", "")[:200],
                    })

    print("\n  Failure patterns:")
    for pattern, cases in failure_patterns.items():
        if cases:
            print(f"    {pattern}: {len(cases)} cases")
            categories = {}
            for c in cases:
                cat = c.get("category", "?")
                categories[cat] = categories.get(cat, 0) + 1
            print(f"      by category: {categories}")

    # -- Generate recommendations ------------------------------------------

    recommendations = {
        "training_adjustments": [],
        "data_adjustments": [],
        "system_prompt_adjustments": [],
        "priority": "low",
    }

    correctness_failures = failure_patterns["correctness_failures"]
    if correctness_failures:
        fail_categories = {}
        for f in correctness_failures:
            cat = f.get("category", "?")
            fail_categories[cat] = fail_categories.get(cat, 0) + 1

        worst_category = max(fail_categories, key=fail_categories.get)
        recommendations["training_adjustments"].append({
            "param": "training_data_mix",
            "action": "increase",
            "detail": f"Add more {worst_category} examples to training data. "
                      f"{fail_categories[worst_category]} correctness failures "
                      f"in this category.",
        })
        recommendations["priority"] = "high"

        bug_missed = sum(1 for f in correctness_failures
                         if f.get("has_bug") is True)
        false_positive = sum(1 for f in correctness_failures
                             if f.get("has_bug") is False)

        if bug_missed > false_positive:
            recommendations["system_prompt_adjustments"].append({
                "action": "encourage_issue_detection",
                "detail": f"Model missed {bug_missed} real bugs. "
                          "Consider adding 'Always look for potential issues' "
                          "to the system prompt during SFT.",
            })
        elif false_positive > 0:
            recommendations["system_prompt_adjustments"].append({
                "action": "reduce_hallucination",
                "detail": f"Model hallucinated {false_positive} issues on "
                          "clean code. Add 'Only flag real issues, say LGTM "
                          "if the code is clean' to the system prompt.",
            })

    conciseness_failures = failure_patterns["conciseness_failures"]
    if len(conciseness_failures) > len(per_case) * 0.3:
        recommendations["training_adjustments"].append({
            "param": "max_tokens",
            "action": "decrease",
            "detail": f"{len(conciseness_failures)}/{len(per_case)} reviews "
                      "were too verbose. Consider reducing max_tokens in "
                      "SFT training data or adding conciseness instruction.",
        })

    format_failures = failure_patterns["format_failures"]
    if len(format_failures) > len(per_case) * 0.2:
        recommendations["data_adjustments"].append({
            "action": "clean_training_data",
            "detail": f"{len(format_failures)} reviews contained boilerplate. "
                      "Remove filler phrases from SFT training examples.",
        })

    quality_failures = failure_patterns["quality_failures"]
    if quality_failures:
        quality_agg = judge_agg.get("review_quality", {})
        mean = quality_agg.get("mean", 0)
        if mean < 2.5:
            recommendations["training_adjustments"].append({
                "param": "dpo_beta",
                "action": "increase",
                "detail": f"Quality mean={mean:.2f}/5 is very low. "
                          "Increase DPO beta to strengthen preference signal.",
                "suggested_value": min(
                    config.get("training", {}).get("dpo_beta", 0.3) + 0.1,
                    0.5),
            })
            recommendations["priority"] = "high"
        elif mean < 3.5:
            recommendations["training_adjustments"].append({
                "param": "dpo_epochs",
                "action": "increase",
                "detail": f"Quality mean={mean:.2f}/5 is below threshold. "
                          "Consider adding an extra DPO epoch.",
                "suggested_value": config.get("training", {}).get(
                    "dpo_epochs", 1) + 1,
            })

    if not any(failure_patterns.values()):
        recommendations["priority"] = "none"
        recommendations["training_adjustments"].append({
            "param": "none",
            "action": "maintain",
            "detail": "All judges passed. No adjustments needed.",
        })

    # -- Use teacher to generate a natural-language summary ----------------

    summary_prompt = (
        "You are analyzing the evaluation results of a code review AI model.\n\n"
        f"Gate result: {gate_result}\n"
        f"Composite score: {composite:.4f}\n"
        f"Regressions: {json.dumps(regressions)}\n"
        f"Judge aggregates: {json.dumps(judge_agg)}\n"
        f"Failure patterns: {json.dumps({k: len(v) for k, v in failure_patterns.items()})}\n\n"
        "Write a 3-5 sentence analysis summary. What's working? What's not? "
        "What's the single most impactful change to make for the next iteration?"
    )

    try:
        resp = requests.post(
            api_url, headers=api_headers,
            json={"model": teacher_model,
                  "messages": [{"role": "user", "content": summary_prompt}],
                  "max_tokens": 300, "temperature": 0.3},
            timeout=120)
        resp.raise_for_status()
        analysis_summary = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        analysis_summary = f"(Could not generate summary: {e})"

    recommendations["analysis_summary"] = analysis_summary

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print(f"  Priority: {recommendations['priority']}")
    for adj in recommendations.get("training_adjustments", []):
        print(f"  Training: {adj['action']} {adj.get('param', '')} "
              f"-- {adj['detail'][:80]}")
    for adj in recommendations.get("data_adjustments", []):
        print(f"  Data: {adj['action']} -- {adj['detail'][:80]}")
    for adj in recommendations.get("system_prompt_adjustments", []):
        print(f"  Prompt: {adj['action']} -- {adj['detail'][:80]}")
    print(f"\n  Analysis: {analysis_summary[:200]}")
    print("=" * 60)

    # -- Log to MLflow -----------------------------------------------------

    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("CodeReview-Eval-Hub")
        with mlflow.start_run(run_name=f"eval-optimize-{model_version}"):
            mlflow.set_tag("model_version", model_version)
            mlflow.set_tag("eval_type", "optimization")
            mlflow.set_tag("priority", recommendations["priority"])

            mlflow.log_metric("num_correctness_failures",
                              len(correctness_failures))
            mlflow.log_metric("num_conciseness_failures",
                              len(conciseness_failures))
            mlflow.log_metric("num_quality_failures",
                              len(quality_failures))
            mlflow.log_metric("num_format_failures",
                              len(format_failures))

            mlflow.log_dict(recommendations, "recommendations.json")
            mlflow.log_text(analysis_summary, "analysis_summary.md")
        print(f"  Optimize results logged to MLflow")

    return recommendations
