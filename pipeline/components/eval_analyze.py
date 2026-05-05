"""
KFP Component -- Eval Analyze (agent-eval-harness /eval-analyze)

Reads the last N evaluation runs from MLflow, identifies trends
(improving/declining/stable) and weaknesses (judges below threshold,
consecutive gate failures).

Returns an analysis dict used by eval-dataset and eval-optimize.
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["requests", "mlflow", "boto3"],
)
def eval_analyze(
    mlflow_tracking_uri: str,
    experiment_name: str,
    s3_endpoint: str = "",
    s3_access_key: str = "",
    s3_secret_key: str = "",
    lookback_runs: int = 10,
) -> dict:
    """Analyze the model's performance history from MLflow.

    This is the /eval-analyze step from agent-eval-harness.
    """
    import json
    import os
    import sys
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

    print("=" * 60)
    print("EVAL-ANALYZE (agent-eval-harness)")
    print("=" * 60)

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if not exp:
            print(f"  Experiment '{experiment_name}' not found")
            return {"status": "no_data", "runs_analyzed": 0,
                    "trends": {}, "weaknesses": [], "weak_categories": []}

        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.eval_type = 'pipeline_benchmark'",
            order_by=["start_time DESC"],
            max_results=lookback_runs,
        )

        if runs.empty:
            print("  No benchmark runs found")
            return {"status": "no_data", "runs_analyzed": 0,
                    "trends": {}, "weaknesses": [], "weak_categories": []}

        print(f"  Found {len(runs)} evaluation runs")

        judge_metrics = [c for c in runs.columns if c.startswith("metrics.judge_")]
        trends = {}
        weaknesses = []

        for metric in judge_metrics:
            judge_name = metric.replace("metrics.judge_", "")
            values = runs[metric].dropna().tolist()
            if len(values) >= 2:
                latest, prev = values[0], values[1]
                trend = ("improving" if latest > prev else
                         "declining" if latest < prev else "stable")
                trends[judge_name] = {
                    "latest": round(latest, 4),
                    "previous": round(prev, 4),
                    "trend": trend,
                }
                if latest < 0.7:
                    weaknesses.append({
                        "judge": judge_name,
                        "score": round(latest, 4),
                        "trend": trend,
                    })
                print(f"    {judge_name}: {latest:.4f} ({trend})")
            elif len(values) == 1:
                trends[judge_name] = {
                    "latest": round(values[0], 4),
                    "trend": "first_run",
                }

        composite_col = runs.get("metrics.composite_score")
        if composite_col is not None:
            latest_c = composite_col.dropna().iloc[0] if not composite_col.dropna().empty else 0
            trends["composite_score"] = round(float(latest_c), 4)

        gate_values = [t for t in runs.get("tags.quality_gate", [])
                       if isinstance(t, str)]
        consecutive_fails = 0
        for g in gate_values:
            if g == "fail":
                consecutive_fails += 1
            else:
                break

        # Identify weak categories from per-question metrics
        weak_categories = []
        q_cols = [c for c in runs.columns
                  if c.startswith("metrics.q") and "correctness" in c]
        if q_cols and len(runs) > 0:
            latest_run = runs.iloc[0]
            for col in q_cols:
                val = latest_run.get(col)
                if val is not None and val == 0:
                    weak_categories.append(col.replace("metrics.", ""))

        analysis = {
            "status": "analyzed",
            "runs_analyzed": len(runs),
            "trends": trends,
            "weaknesses": weaknesses,
            "weak_categories": weak_categories,
            "consecutive_gate_fails": consecutive_fails,
            "latest_gate": gate_values[0] if gate_values else "unknown",
        }

        print(f"\n  Weaknesses: {len(weaknesses)}")
        print(f"  Weak categories: {weak_categories}")
        print(f"  Consecutive gate fails: {consecutive_fails}")
        print(f"  Latest gate: {analysis['latest_gate']}")
        print("=" * 60)

        return analysis

    except Exception as e:
        print(f"  Analysis error: {e}")
        return {"status": "error", "error": str(e),
                "runs_analyzed": 0, "trends": {}, "weaknesses": [],
                "weak_categories": []}
