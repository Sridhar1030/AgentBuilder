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
    min_pairs: int = 3,
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

    model_output_s3_path = sft_model_s3_path

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
    apps_api = client.AppsV1Api()
    custom_api = client.CustomObjectsApi()
    core_api = client.CoreV1Api()

    namespace = "sridharproject"
    job_name = f"dpo-{int(time.time())}"
    image = "image-registry.openshift-image-registry.svc:5000/sridharproject/distillation-trainer:v1.1.5"

    # --- GPU evacuation and restore helpers ---
    isvc_deployment = "code-review-llm-predictor"

    def _scale_down_kserve():
        try:
            dep = apps_api.read_namespaced_deployment(isvc_deployment, namespace)
            if dep.spec.replicas and dep.spec.replicas > 0:
                print(f"[{time.strftime('%H:%M:%S')}] Scaling down {isvc_deployment} to free GPU node...")
                apps_api.patch_namespaced_deployment_scale(
                    isvc_deployment, namespace, {"spec": {"replicas": 0}},
                )
                for _ in range(60):
                    time.sleep(5)
                    pods = core_api.list_namespaced_pod(
                        namespace, label_selector=f"app=isvc.{isvc_deployment}",
                    )
                    if not pods.items:
                        break
                print(f"[{time.strftime('%H:%M:%S')}] {isvc_deployment} scaled to 0 -- GPU freed")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] KServe scale-down skipped: {e}")

    def _restore_kserve():
        """Scale KServe back up and wait until the pod is Ready."""
        try:
            dep = apps_api.read_namespaced_deployment(isvc_deployment, namespace)
            if (dep.spec.replicas or 0) < 1:
                print(f"[{time.strftime('%H:%M:%S')}] Restoring {isvc_deployment} to 1 replica...")
                apps_api.patch_namespaced_deployment_scale(
                    isvc_deployment, namespace, {"spec": {"replicas": 1}},
                )
            for attempt in range(60):
                time.sleep(10)
                pods = core_api.list_namespaced_pod(
                    namespace, label_selector=f"app=isvc.{isvc_deployment}",
                )
                for pod in pods.items:
                    if all(
                        cs.ready
                        for cs in (pod.status.container_statuses or [])
                    ):
                        print(f"[{time.strftime('%H:%M:%S')}] {isvc_deployment} is Ready (waited {(attempt+1)*10}s)")
                        return
                if attempt % 6 == 5:
                    print(f"[{time.strftime('%H:%M:%S')}] Still waiting for {isvc_deployment} to be Ready... ({(attempt+1)*10}s)")
            print(f"[{time.strftime('%H:%M:%S')}] WARNING: {isvc_deployment} not Ready after 600s, proceeding anyway")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] KServe restore failed: {e}")

    _scale_down_kserve()

    # --- Discover GPU topology from the cluster ---
    available_gpu_nodes = 0
    gpus_per_node_list = []
    blocked_taints = {"node.kubernetes.io/unreachable", "node.kubernetes.io/not-ready"}
    try:
        nodes = core_api.list_node(label_selector="nvidia.com/gpu.present=true")
        all_pods = core_api.list_pod_for_all_namespaces(field_selector="status.phase=Running").items
        gpu_used_per_node = {}
        for pod in all_pods:
            node_name = pod.spec.node_name
            if not node_name:
                continue
            for c in (pod.spec.containers or []):
                req = (c.resources.requests or {}) if c.resources else {}
                gpu_req = int(req.get("nvidia.com/gpu", 0))
                if gpu_req:
                    gpu_used_per_node[node_name] = gpu_used_per_node.get(node_name, 0) + gpu_req

        for node in nodes.items:
            ready = any(
                c.type == "Ready" and c.status == "True"
                for c in (node.status.conditions or [])
            )
            tainted = any(
                t.effect in ("NoSchedule", "NoExecute") and t.key in blocked_taints
                for t in (node.spec.taints or [])
            )
            if ready and not tainted:
                gpu_cap = int(node.status.allocatable.get("nvidia.com/gpu", "0"))
                gpu_used = gpu_used_per_node.get(node.metadata.name, 0)
                gpu_free = gpu_cap - gpu_used
                print(f"[{time.strftime('%H:%M:%S')}]   Node {node.metadata.name}: {gpu_cap} total, {gpu_used} used, {gpu_free} free")
                if gpu_free >= 4:
                    available_gpu_nodes += 1
                    gpus_per_node_list.append(gpu_free)
                else:
                    print(f"[{time.strftime('%H:%M:%S')}]   -> Skipping (need 4 free GPUs, only {gpu_free})")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Could not query nodes ({e}), falling back to 2 nodes x 4 GPUs")
        available_gpu_nodes = 2
        gpus_per_node_list = [4, 4]

    num_gpus_per_node = min(gpus_per_node_list) if gpus_per_node_list else 4
    print(f"[{time.strftime('%H:%M:%S')}] GPU topology: {available_gpu_nodes} usable nodes x {num_gpus_per_node} GPUs/node")

    # --- Cap by dataset size: need >= 6 samples per GPU for meaningful DPO ---
    min_samples_per_gpu = 6
    max_data_nodes = max(1, num_pref_pairs // (min_samples_per_gpu * num_gpus_per_node))

    num_total_nodes = min(max_data_nodes, available_gpu_nodes)
    num_workers = max(0, num_total_nodes - 1)
    total_gpus = num_total_nodes * num_gpus_per_node
    print(f"[{time.strftime('%H:%M:%S')}] Auto-scaled DPO: {num_pref_pairs} pairs -> {num_total_nodes} nodes ({num_workers} workers) x {num_gpus_per_node} GPUs = {total_gpus} GPUs")

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

    container_spec = {
        "name": "pytorch",
        "image": image,
        "env": env_list,
        "resources": {
            "requests": {
                "nvidia.com/gpu": str(num_gpus_per_node),
                "memory": "48Gi",
                "cpu": "8",
            },
            "limits": {
                "nvidia.com/gpu": str(num_gpus_per_node),
                "memory": "64Gi",
                "cpu": "16",
            },
        },
        "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
    }

    pod_spec = {
        "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}}],
    }

    node_selector = {"nvidia.com/gpu.present": "true"}

    torchrun_args = [
        f"--nproc_per_node={num_gpus_per_node}",
        "--master_port=29500",
        "/opt/scripts/finetune_job.py",
    ]

    pytorchjob = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {"name": job_name, "namespace": namespace},
        "spec": {
            "pytorchReplicaSpecs": {
                "Master": {
                    "replicas": 1,
                    "restartPolicy": "Never",
                    "template": {
                        "spec": {
                            **pod_spec,
                            "nodeSelector": node_selector,
                            "containers": [{
                                **container_spec,
                                "command": ["torchrun"],
                                "args": torchrun_args,
                            }],
                        },
                    },
                },
                "Worker": {
                    "replicas": num_workers,
                    "restartPolicy": "Never",
                    "template": {
                        "spec": {
                            **pod_spec,
                            "nodeSelector": node_selector,
                            "containers": [{
                                **container_spec,
                                "command": ["torchrun"],
                                "args": torchrun_args,
                            }],
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

    total_gpus = num_total_nodes * num_gpus_per_node
    print(f"[{time.strftime('%H:%M:%S')}] Submitted DPO PyTorchJob {job_name} ({num_total_nodes} nodes x {num_gpus_per_node} GPUs = {total_gpus} total)")
    print(f"[{time.strftime('%H:%M:%S')}] Output will go to: {model_output_s3_path}")
    print(f"[{time.strftime('%H:%M:%S')}] Timeout set to {timeout}s ({timeout/3600:.1f}h)")

    def _model_s3_timestamp():
        """Return LastModified of model.safetensors, or None if missing."""
        try:
            bucket = model_output_s3_path.replace("s3://", "").split("/", 1)[0]
            prefix = model_output_s3_path.replace("s3://", "").split("/", 1)[1]
            resp = s3.head_object(Bucket=bucket, Key=prefix + "model.safetensors")
            return resp["LastModified"]
        except Exception:
            return None

    model_ts_before = _model_s3_timestamp()
    print(f"[{time.strftime('%H:%M:%S')}] Model timestamp before DPO: {model_ts_before}")

    def _model_is_new() -> bool:
        ts = _model_s3_timestamp()
        if ts is None:
            return False
        if model_ts_before is None:
            return True
        return ts > model_ts_before

    result = None
    try:
        job_failed = False
        fail_msg = ""

        while elapsed < timeout:
            time.sleep(poll_interval)
            elapsed += poll_interval

            if _model_is_new():
                hrs, rem = divmod(elapsed, 3600)
                mins = rem // 60
                status = "Failed" if job_failed else "Running/Succeeded"
                print(f"[{time.strftime('%H:%M:%S')}] DPO model updated in S3 (job status={status}, elapsed={hrs}h{mins}m). Success.")
                try:
                    custom_api.delete_namespaced_custom_object(
                        group="kubeflow.org", version="v1", namespace=namespace,
                        plural="pytorchjobs", name=job_name,
                    )
                except Exception:
                    pass
                result = model_output_s3_path
                return result

            if job_failed:
                print(f"[{time.strftime('%H:%M:%S')}] Waiting for model upload... (elapsed={elapsed}s)")
                continue

            try:
                job = custom_api.get_namespaced_custom_object(
                    group="kubeflow.org", version="v1", namespace=namespace,
                    plural="pytorchjobs", name=job_name,
                )
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Could not fetch job status ({e}), will retry...")
                continue

            conditions = job.get("status", {}).get("conditions", [])
            for c in conditions:
                ctype = c.get("type")
                if ctype == "Succeeded" and c.get("status") == "True":
                    hrs, rem = divmod(elapsed, 3600)
                    mins = rem // 60
                    print(f"[{time.strftime('%H:%M:%S')}] DPO PyTorchJob {job_name} succeeded ({hrs}h{mins}m)")
                    result = model_output_s3_path
                    return result
                if ctype == "Failed" and c.get("status") == "True":
                    job_failed = True
                    fail_msg = c.get("message", "unknown error")
                    print(f"[{time.strftime('%H:%M:%S')}] DPO job {job_name} status=Failed. Continuing to poll S3 for model upload...")
                    break

            if not job_failed:
                hrs, rem = divmod(elapsed, 3600)
                mins = rem // 60
                print(f"[{time.strftime('%H:%M:%S')}] DPO PyTorchJob {job_name} running (elapsed={hrs}h{mins}m)")

        if job_failed:
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
                    fail_msg = f"{fail_msg}\n\n--- Pod {pod_name} logs ---\n{logs}"
            except Exception as e:
                fail_msg = f"{fail_msg} (could not fetch pod logs: {e})"
            raise RuntimeError(f"DPO PyTorchJob {job_name} failed and model never appeared in S3: {fail_msg}")

        raise TimeoutError(f"DPO PyTorchJob {job_name} did not complete within {timeout}s")
    finally:
        print(f"[{time.strftime('%H:%M:%S')}] DPO step done. Restoring KServe student model...")
        _restore_kserve()
