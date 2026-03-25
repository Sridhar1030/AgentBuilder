"""
KFP Component -- DPO Fine-Tune via Kubeflow TrainJob

Same TrainJob pattern as finetune.py, but passes TRAINING_MODE=dpo,
PREF_DATA_PATH, and DPO_BETA. Uses the SFT model S3 path as BASE_MODEL_ID
so DPO starts from the SFT-trained weights.

If fewer than min_pairs preference entries exist, skips DPO and passes
through the SFT model path unchanged.
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["kubernetes==31.0.0", "boto3"],
)
def dpo_finetune(
    sft_model_s3_path: str,
    pref_data_s3_path: str,
    model_version: str,
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    num_epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 5e-5,
    lora_r: int = 16,
    lora_alpha: int = 32,
    dpo_beta: float = 0.1,
    min_pairs: int = 5,
) -> str:
    """Submit a Kubeflow TrainJob for DPO training on preference data."""
    import time

    import boto3
    from kubernetes import client, config

    model_output_s3_path = f"s3://sridhar-models/student-1b-{model_version}-dpo/"

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )

    parts = pref_data_s3_path.replace("s3://", "").split("/", 1)
    obj = s3.get_object(Bucket=parts[0], Key=parts[1])
    lines = [l for l in obj["Body"].read().decode().strip().split("\n") if l.strip()]
    num_pref_pairs = len(lines)

    print(f"Preference data has {num_pref_pairs} pairs (min_pairs={min_pairs})")
    if num_pref_pairs < min_pairs:
        print(f"Fewer than {min_pairs} preference pairs -- skipping DPO, passing through SFT model")
        return sft_model_s3_path

    config.load_incluster_config()
    custom_api = client.CustomObjectsApi()

    namespace = "sridharproject"
    job_name = f"dpo-{int(time.time())}"
    image = "quay.io/rh-ee-srpillai/distillation-trainer:v0.5.2"

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
                    {"name": "TRAINING_MODE", "value": "dpo"},
                    {"name": "PREF_DATA_PATH", "value": pref_data_s3_path},
                    {"name": "BASE_MODEL_ID", "value": sft_model_s3_path},
                    {"name": "MODEL_OUTPUT_S3_PATH", "value": model_output_s3_path},
                    {"name": "NUM_EPOCHS", "value": str(num_epochs)},
                    {"name": "BATCH_SIZE", "value": str(batch_size)},
                    {"name": "LEARNING_RATE", "value": str(learning_rate)},
                    {"name": "LORA_R", "value": str(lora_r)},
                    {"name": "LORA_ALPHA", "value": str(lora_alpha)},
                    {"name": "DPO_BETA", "value": str(dpo_beta)},
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
    print(f"Submitted DPO TrainJob {job_name}")

    poll_interval = 30
    timeout = 7200
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
                print(f"DPO TrainJob {job_name} succeeded after {elapsed}s")
                return model_output_s3_path
            if ctype == "Failed" and condition.get("status") == "True":
                msg = condition.get("message", "unknown error")
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
                        msg = f"{msg}\n\n--- DPO pod {pod_name} logs (last 100 lines) ---\n" + "\n".join(logs.splitlines()[-100:])
                except Exception as e:
                    msg = f"{msg} (could not fetch pod logs: {e})"
                raise RuntimeError(f"DPO TrainJob {job_name} failed: {msg}")

        phase = job.get("status", {}).get("phase", "Unknown")
        print(f"DPO TrainJob {job_name} still running (phase={phase}, elapsed={elapsed}s)")

    raise TimeoutError(f"DPO TrainJob {job_name} did not complete within {timeout}s")
