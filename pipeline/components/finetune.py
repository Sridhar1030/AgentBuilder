"""
KFP Component 2 — QLoRA Fine-Tune via Kubeflow TrainJob

Submits a Kubeflow TrainJob (trainer.kubeflow.org/v1alpha1) that runs
the training container, then polls until completion. The KFP pod itself
is lightweight (no GPU) — the training container gets the GPU.
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["kubernetes==31.0.0"],
)
def finetune(
    gold_data_path: str,
    model_output_s3_path: str,
    base_model_id: str,
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> str:
    """Submit a Kubeflow TrainJob for QLoRA training and wait for completion."""
    import time

    from kubernetes import client, config

    config.load_incluster_config()
    custom_api = client.CustomObjectsApi()

    namespace = "sridharproject"
    job_name = f"finetune-{int(time.time())}"
    image = "quay.io/rh-ee-srpillai/distillation-trainer:v0.1.0"

    trainjob = {
        "apiVersion": "trainer.kubeflow.org/v1alpha1",
        "kind": "TrainJob",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
        },
        "spec": {
            "runtimeRef": {
                "name": "torch-distributed",
            },
            "trainer": {
                "image": image,
                "env": [
                    {"name": "GOLD_DATA_PATH", "value": gold_data_path},
                    {"name": "MODEL_OUTPUT_S3_PATH", "value": model_output_s3_path},
                    {"name": "BASE_MODEL_ID", "value": base_model_id},
                    {"name": "NUM_EPOCHS", "value": str(num_epochs)},
                    {"name": "BATCH_SIZE", "value": str(batch_size)},
                    {"name": "LEARNING_RATE", "value": str(learning_rate)},
                    {"name": "LORA_R", "value": str(lora_r)},
                    {"name": "LORA_ALPHA", "value": str(lora_alpha)},
                    {"name": "S3_ENDPOINT", "value": s3_endpoint},
                    {"name": "S3_ACCESS_KEY", "value": s3_access_key},
                    {"name": "S3_SECRET_KEY", "value": s3_secret_key},
                    {"name": "CUDA_VISIBLE_DEVICES", "value": "0"},
                ],
                "numNodes": 1,
                "resourcesPerNode": {
                    "requests": {
                        "nvidia.com/gpu": "1",
                        "memory": "16Gi",
                        "cpu": "4",
                    },
                    "limits": {
                        "nvidia.com/gpu": "1",
                        "memory": "24Gi",
                        "cpu": "8",
                    },
                },
            },
        },
    }

    custom_api.create_namespaced_custom_object(
        group="trainer.kubeflow.org",
        version="v1alpha1",
        namespace=namespace,
        plural="trainjobs",
        body=trainjob,
    )
    print(f"Submitted TrainJob {job_name}")

    # Poll until Succeeded or Failed
    poll_interval = 30
    timeout = 7200  # 2 hours max
    elapsed = 0

    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval

        job = custom_api.get_namespaced_custom_object(
            group="trainer.kubeflow.org",
            version="v1alpha1",
            namespace=namespace,
            plural="trainjobs",
            name=job_name,
        )

        conditions = job.get("status", {}).get("conditions", [])
        for condition in conditions:
            ctype = condition.get("type")
            if (ctype == "Succeeded" or ctype == "Complete") and condition.get("status") == "True":
                print(f"TrainJob {job_name} succeeded after {elapsed}s")
                return model_output_s3_path
            if ctype == "Failed" and condition.get("status") == "True":
                msg = condition.get("message", "unknown error")
                # Try to get training pod logs so the real error is visible
                try:
                    core_api = client.CoreV1Api()
                    label = f"batch.kubernetes.io/job-name={job_name}-node-0"
                    pods = core_api.list_namespaced_pod(
                        namespace=namespace,
                        label_selector=label,
                    )
                    if pods.items:
                        pod_name = pods.items[0].metadata.name
                        logs = core_api.read_namespaced_pod_log(
                            name=pod_name,
                            namespace=namespace,
                            container="node",
                        )
                        msg = f"{msg}\n\n--- Training pod {pod_name} logs (last 100 lines) ---\n" + "\n".join(logs.splitlines()[-100:])
                except Exception as e:
                    msg = f"{msg} (could not fetch pod logs: {e})"
                raise RuntimeError(f"TrainJob {job_name} failed: {msg}")

        phase = job.get("status", {}).get("phase", "Unknown")
        print(f"TrainJob {job_name} still running (phase={phase}, elapsed={elapsed}s)")

    raise TimeoutError(f"TrainJob {job_name} did not complete within {timeout}s")
