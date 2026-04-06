"""
KFP Component -- DPO Fine-Tune via PyTorchJob

Same PyTorchJob pattern as finetune.py but passes TRAINING_MODE=dpo.
Uses torchrun for multi-GPU DDP training.

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
    """Create a PyTorchJob for multi-GPU DPO training."""
    import time

    import boto3
    from kubernetes import client, config

    print("=" * 60)
    print("DPO FINE-TUNE STEP (PyTorchJob, multi-GPU)")
    print("=" * 60)
    print(f"  SFT model:    {sft_model_s3_path}")
    print(f"  Pref data:    {pref_data_s3_path}")
    print(f"  Version:      {model_version}")
    print(f"  Epochs:       {num_epochs}")
    print(f"  Batch size:   {batch_size}")
    print(f"  LR:           {learning_rate}")
    print(f"  LoRA r/alpha: {lora_r}/{lora_alpha}")
    print(f"  DPO beta:     {dpo_beta}")
    print(f"  Min pairs:    {min_pairs}")
    print("=" * 60)

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

    print(f"[{time.strftime('%H:%M:%S')}] Preference data has {num_pref_pairs} pairs (min required: {min_pairs})")
    if num_pref_pairs < min_pairs:
        print(f"[{time.strftime('%H:%M:%S')}] SKIPPING DPO: only {num_pref_pairs} pairs (< {min_pairs}). Returning SFT model.")
        return sft_model_s3_path
    print(f"[{time.strftime('%H:%M:%S')}] Sufficient pairs. Proceeding with DPO training.")

    config.load_incluster_config()
    custom_api = client.CustomObjectsApi()
    core_api = client.CoreV1Api()

    namespace = "sridharproject"
    job_name = f"dpo-{int(time.time())}"
    image = "quay.io/rh-ee-srpillai/distillation-trainer:v0.9.0"
    num_gpus = 4

    env_list = [
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
    ]

    pytorchjob = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {"name": job_name, "namespace": namespace},
        "spec": {
            "pytorchReplicaSpecs": {
                "Master": {
                    "replicas": 1,
                    "template": {
                        "spec": {
                            "containers": [{
                                "name": "pytorch",
                                "image": image,
                                "command": ["torchrun"],
                                "args": [
                                    f"--nproc_per_node={num_gpus}",
                                    "--master_addr=localhost",
                                    "--master_port=29500",
                                    "/opt/scripts/finetune_job.py",
                                ],
                                "env": env_list,
                                "resources": {
                                    "requests": {
                                        "nvidia.com/gpu": str(num_gpus),
                                        "memory": "48Gi",
                                        "cpu": "8",
                                    },
                                    "limits": {
                                        "nvidia.com/gpu": str(num_gpus),
                                        "memory": "64Gi",
                                        "cpu": "16",
                                    },
                                },
                                "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
                            }],
                            "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}}],
                            "restartPolicy": "Never",
                        },
                    },
                },
            },
        },
    }

    custom_api.create_namespaced_custom_object(
        group="kubeflow.org",
        version="v1",
        namespace=namespace,
        plural="pytorchjobs",
        body=pytorchjob,
    )

    poll_interval = 30
    timeout = 36000
    elapsed = 0

    print(f"[{time.strftime('%H:%M:%S')}] Submitted DPO PyTorchJob {job_name} ({num_gpus} GPUs)")
    print(f"[{time.strftime('%H:%M:%S')}] Output will go to: {model_output_s3_path}")
    print(f"[{time.strftime('%H:%M:%S')}] Timeout set to {timeout}s ({timeout/3600:.1f}h)")

    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval

        job = custom_api.get_namespaced_custom_object(
            group="kubeflow.org",
            version="v1",
            namespace=namespace,
            plural="pytorchjobs",
            name=job_name,
        )

        conditions = job.get("status", {}).get("conditions", [])
        for c in conditions:
            ctype = c.get("type")
            if ctype == "Succeeded" and c.get("status") == "True":
                hrs, rem = divmod(elapsed, 3600)
                mins = rem // 60
                print(f"[{time.strftime('%H:%M:%S')}] DPO PyTorchJob {job_name} succeeded ({hrs}h{mins}m)")
                return model_output_s3_path
            if ctype == "Failed" and c.get("status") == "True":
                msg = c.get("message", "unknown error")
                try:
                    pods = core_api.list_namespaced_pod(
                        namespace=namespace,
                        label_selector=f"training.kubeflow.org/job-name={job_name}",
                    )
                    if pods.items:
                        pod_name = pods.items[0].metadata.name
                        logs = core_api.read_namespaced_pod_log(
                            name=pod_name, namespace=namespace, tail_lines=100,
                        )
                        msg = f"{msg}\n\n--- Pod {pod_name} logs ---\n{logs}"
                except Exception as e:
                    msg = f"{msg} (could not fetch pod logs: {e})"
                raise RuntimeError(f"DPO PyTorchJob {job_name} failed: {msg}")

        hrs, rem = divmod(elapsed, 3600)
        mins = rem // 60
        print(f"[{time.strftime('%H:%M:%S')}] DPO PyTorchJob {job_name} running (elapsed={hrs}h{mins}m)")

    raise TimeoutError(f"DPO PyTorchJob {job_name} did not complete within {timeout}s")
