"""
KFP Component -- Traffic Shift (Canary Progressive Migration)

Final step in the unified pipeline. Compares the current student
evaluation score against the last recorded score. If improved,
shifts traffic from teacher to student by shift_increment%.

Updates both the canary gateway (via HTTP) and the Istio
VirtualService (via Kubernetes API) to keep them in sync.
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["requests", "kubernetes==32.0.1", "mlflow", "boto3"],
)
def traffic_shift(
    eval_results: dict,
    gateway_url: str,
    namespace: str,
    virtualservice_name: str = "code-review-gateway",
    shift_increment: int = 10,
    mlflow_tracking_uri: str = "",
    model_version: str = "unknown",
    s3_endpoint: str = "",
    s3_access_key: str = "",
    s3_secret_key: str = "",
) -> dict:
    """Shift traffic from teacher to student if eval score improved.

    Returns dict with shift decision, old/new split, and scores.
    """
    import json
    import os
    import sys
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

    print("=" * 60)
    print("TRAFFIC SHIFT (Canary Progressive Migration)")
    print("=" * 60)

    current_score = eval_results.get("composite_score", 0.0)
    gate_result = eval_results.get("gate_result", "unknown")
    print(f"  Current eval score: {current_score:.4f}")
    print(f"  Gate result:        {gate_result}")

    gw = gateway_url.rstrip("/")

    # -- Read current state from gateway ------------------------------------
    try:
        resp = requests.get(f"{gw}/split", timeout=10)
        resp.raise_for_status()
        split_data = resp.json()
    except Exception as exc:
        print(f"  WARNING: Could not read gateway split: {exc}")
        split_data = {
            "teacher": 90,
            "student": 10,
            "last_student_score": 0.0,
        }

    old_teacher = split_data["teacher"]
    old_student = split_data["student"]
    last_score = split_data.get("last_student_score", 0.0)

    print(f"  Current split: teacher={old_teacher}% student={old_student}%")
    print(f"  Last score:    {last_score:.4f}")

    # -- Decide shift -------------------------------------------------------
    shifted = False
    reason = ""
    new_teacher = old_teacher
    new_student = old_student

    if gate_result == "fail":
        reason = f"Quality gate failed (score={current_score:.4f}). No shift."
        print(f"  Decision: {reason}")
    elif current_score > last_score:
        new_student = min(old_student + shift_increment, 100)
        new_teacher = 100 - new_student
        shifted = True
        reason = (
            f"Score improved {last_score:.4f} -> {current_score:.4f}. "
            f"Shifting +{shift_increment}% to student."
        )
        print(f"  Decision: {reason}")
        print(f"  New split: teacher={new_teacher}% student={new_student}%")
    else:
        reason = (
            f"No improvement ({current_score:.4f} <= {last_score:.4f}). "
            "Keeping current split."
        )
        print(f"  Decision: {reason}")

    # -- Apply shift via gateway (cascades to ConfigMap + VirtualService) ---
    if shifted:
        try:
            resp = requests.post(
                f"{gw}/split",
                json={
                    "teacher": new_teacher,
                    "student": new_student,
                    "last_student_score": current_score,
                    "reason": f"pipeline-{model_version}",
                },
                timeout=10,
            )
            resp.raise_for_status()
            print(f"  Gateway updated: {resp.json()}")
        except Exception as exc:
            print(f"  WARNING: Gateway update failed: {exc}")
            print("  Falling back to direct VirtualService + ConfigMap patch...")
            _fallback_patch(
                namespace, virtualservice_name, new_teacher, new_student,
                current_score,
            )
    else:
        if current_score > last_score:
            try:
                requests.post(
                    f"{gw}/split",
                    json={
                        "teacher": old_teacher,
                        "student": old_student,
                        "last_student_score": current_score,
                        "reason": f"score-update-{model_version}",
                    },
                    timeout=10,
                )
            except Exception:
                pass

    # -- Log to MLflow ------------------------------------------------------
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("CodeReview-Eval-Hub")
        with mlflow.start_run(run_name=f"traffic-shift-{model_version}"):
            mlflow.set_tag("model_version", model_version)
            mlflow.set_tag("eval_type", "traffic_shift")
            mlflow.set_tag("shifted", str(shifted))
            mlflow.log_metric("current_score", current_score)
            mlflow.log_metric("last_score", last_score)
            mlflow.log_metric("teacher_weight", new_teacher)
            mlflow.log_metric("student_weight", new_student)
            mlflow.log_metric("shift_applied", 1 if shifted else 0)
        print("  Traffic shift logged to MLflow")

    result = {
        "shifted": shifted,
        "reason": reason,
        "old_split": {"teacher": old_teacher, "student": old_student},
        "new_split": {"teacher": new_teacher, "student": new_student},
        "scores": {
            "current": current_score,
            "previous": last_score,
        },
    }

    print("\n" + "=" * 60)
    print(f"  Result: {'SHIFTED' if shifted else 'NO SHIFT'}")
    print(f"  Split:  teacher={new_teacher}% student={new_student}%")
    print("=" * 60)

    return result


def _fallback_patch(namespace, vs_name, teacher_w, student_w, score):
    """Direct K8s patch if the gateway is unreachable."""
    try:
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        v1 = client.CoreV1Api()
        v1.patch_namespaced_config_map(
            "traffic-split",
            namespace,
            client.V1ConfigMap(
                data={
                    "teacher_weight": str(teacher_w),
                    "student_weight": str(student_w),
                    "last_student_score": str(score),
                }
            ),
        )
        print("  ConfigMap patched directly")

        api = client.CustomObjectsApi()
        api.patch_namespaced_custom_object(
            group="networking.istio.io",
            version="v1",
            namespace=namespace,
            plural="virtualservices",
            name=vs_name,
            body={
                "spec": {
                    "http": [
                        {
                            "route": [
                                {
                                    "destination": {
                                        "host": f"ollama.{namespace}.svc.cluster.local",
                                        "port": {"number": 11434},
                                    },
                                    "weight": teacher_w,
                                },
                                {
                                    "destination": {
                                        "host": f"code-review-llm-predictor.{namespace}.svc.cluster.local",
                                        "port": {"number": 80},
                                    },
                                    "weight": student_w,
                                },
                            ]
                        }
                    ]
                }
            },
        )
        print("  VirtualService patched directly")
    except Exception as exc:
        print(f"  Fallback patch failed: {exc}")
