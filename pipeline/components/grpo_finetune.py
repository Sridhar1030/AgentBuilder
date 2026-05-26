"""
KFP Component -- GRPO Fine-Tune via Kubeflow TrainJob (Trainer v2)

Creates a trainer.kubeflow.org/v1alpha1 TrainJob for single-node multi-GPU GRPO
training with verifiable reward functions (correctness, format, conciseness).

If fewer than min_prompts training prompts exist, skips GRPO and passes
through the DPO (or SFT) model path unchanged.
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["kubernetes==31.0.0", "boto3"],
)
def grpo_finetune(
    dpo_model_s3_path: str,
    grpo_data_s3_path: str,
    model_version: str,
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    grpo_output_s3_path: str = "",
    system_prompt: str = "",
    test_questions_json: str = "",
    num_epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 5e-7,
    lora_r: int = 16,
    lora_alpha: int = 32,
    num_generations: int = 4,
    grpo_beta: float = 0.0,
    temperature: float = 0.7,
    max_completion_length: int = 512,
    loss_type: str = "dapo",
    min_prompts: int = 10,
    weak_categories_json: str = "",
    use_vllm: bool = False,
) -> str:
    """Create a TrainJob for single-node multi-GPU GRPO training."""
    import json
    import time

    import boto3
    from kubernetes import client, config

    TRAINJOB_GROUP = "trainer.kubeflow.org"
    TRAINJOB_VERSION = "v1alpha1"
    TRAINJOB_PLURAL = "trainjobs"

    print(f"--- GRPO FINE-TUNE STEP (TrainJob v2, single-node multi-GPU) ---")
    print(f"  Base model:     {dpo_model_s3_path}")
    print(f"  GRPO data:    {grpo_data_s3_path}")
    print(f"  Version:      {model_version}")
    print(f"  Epochs:       {num_epochs}")
    print(f"  Generations:  {num_generations}")
    print(f"  LR:           {learning_rate}")
    print(f"  Beta:         {grpo_beta}")
    print(f"  Temperature:  {temperature}")
    print(f"  Loss type:    {loss_type}")
    print(f"  Min prompts:  {min_prompts}")
    print(f"  Use vLLM:     {use_vllm}")
    print("=" * 60)

    model_output_s3_path = grpo_output_s3_path if grpo_output_s3_path else dpo_model_s3_path

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )

    def _load_s3_json(s3_path: str) -> dict:
        parts = s3_path.replace("s3://", "").split("/", 1)
        obj = s3.get_object(Bucket=parts[0], Key=parts[1])
        return json.loads(obj["Body"].read().decode())

    def _build_grpo_records() -> list[dict]:
        records = []
        seen = set()

        try:
            bank = _load_s3_json(grpo_data_s3_path)
            questions = bank.get("all_questions", [])
            if not questions and "topics" in bank:
                for topic_qs in bank["topics"].values():
                    questions.extend(topic_qs)
            for q in questions:
                if isinstance(q, dict):
                    text = q.get("question", q.get("prompt", ""))
                    bug = q.get("has_bug", True)
                    issues = q.get("expected_issues", [])
                    cat = q.get("category", "diff_bank")
                else:
                    text = str(q)
                    bug = True
                    issues = []
                    cat = "diff_bank"
                key = text.strip()[:200]
                if not text or key in seen:
                    continue
                seen.add(key)
                records.append({
                    "prompt": text,
                    "has_bug": bug,
                    "expected_issues": issues,
                    "category": cat,
                })
        except Exception as e:
            print(f"  Warning: could not load diff-bank from {grpo_data_s3_path}: {e}")

        if test_questions_json:
            try:
                test_cases = json.loads(test_questions_json)
                for entry in test_cases:
                    q = entry.get("question", "")
                    key = q.strip()[:200]
                    if not q or key in seen:
                        continue
                    seen.add(key)
                    records.append({
                        "prompt": q,
                        "has_bug": entry.get("has_bug", True),
                        "expected_issues": entry.get("expected_issues", []),
                        "category": entry.get("category", "eval"),
                    })
            except Exception as e:
                print(f"  Warning: could not parse test_questions_json: {e}")

        return records

    records = _build_grpo_records()
    num_prompts = len(records)
    print(f"[{time.strftime('%H:%M:%S')}] GRPO dataset has {num_prompts} prompts (min required: {min_prompts})")

    if num_prompts < min_prompts:
        print(
            f"[{time.strftime('%H:%M:%S')}] SKIPPING GRPO: only {num_prompts} prompts "
            f"(< {min_prompts}). Returning DPO model."
        )
        return dpo_model_s3_path

    # Upload merged GRPO JSONL for the training job
    grpo_jsonl_key = f"grpo/train-{model_version}-{int(time.time())}.jsonl"
    grpo_jsonl_path = grpo_data_s3_path.replace("s3://", "").split("/", 1)
    grpo_bucket = grpo_jsonl_path[0]
    body = "\n".join(json.dumps(r) for r in records)
    s3.put_object(Bucket=grpo_bucket, Key=grpo_jsonl_key, Body=body.encode())
    grpo_dataset_s3 = f"s3://{grpo_bucket}/{grpo_jsonl_key}"
    print(f"[{time.strftime('%H:%M:%S')}] Uploaded GRPO dataset to {grpo_dataset_s3}")

    config.load_incluster_config()
    apps_api = client.AppsV1Api()
    custom_api = client.CustomObjectsApi()
    core_api = client.CoreV1Api()

    namespace = "sridharproject"
    job_name = f"grpo-{int(time.time())}"
    image = "image-registry.openshift-image-registry.svc:5000/sridharproject/distillation-trainer:v1.3.5"

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
                    if all(cs.ready for cs in (pod.status.container_statuses or [])):
                        print(
                            f"[{time.strftime('%H:%M:%S')}] {isvc_deployment} is Ready "
                            f"(waited {(attempt + 1) * 10}s)"
                        )
                        return
            print(f"[{time.strftime('%H:%M:%S')}] WARNING: {isvc_deployment} not Ready after 600s")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] KServe restore failed: {e}")

    _scale_down_kserve()

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
                if gpu_free >= 1:
                    available_gpu_nodes += 1
                    gpus_per_node_list.append(gpu_free)
                else:
                    print(f"[{time.strftime('%H:%M:%S')}]   -> Skipping (0 free GPUs)")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Could not query nodes ({e}), falling back to 1 node x 4 GPUs")
        available_gpu_nodes = 1
        gpus_per_node_list = [4]

    num_gpus_per_node = min(gpus_per_node_list) if gpus_per_node_list else 4
    safe_gpu_nodes = 1
    total_gpus = safe_gpu_nodes * num_gpus_per_node
    print(f"[{time.strftime('%H:%M:%S')}] GRPO topology: {safe_gpu_nodes} node x {num_gpus_per_node} GPUs = {total_gpus} total (single-node, QLoRA)")

    env_list = [
        {"name": "TRAINING_MODE", "value": "grpo"},
        {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"},
        {"name": "GRPO_DATA_PATH", "value": grpo_dataset_s3},
        {"name": "BASE_MODEL_ID", "value": dpo_model_s3_path},
        {"name": "MODEL_OUTPUT_S3_PATH", "value": model_output_s3_path},
        {"name": "NUM_EPOCHS", "value": str(num_epochs)},
        {"name": "BATCH_SIZE", "value": str(batch_size)},
        {"name": "LEARNING_RATE", "value": str(learning_rate)},
        {"name": "LORA_R", "value": str(lora_r)},
        {"name": "LORA_ALPHA", "value": str(lora_alpha)},
        {"name": "NUM_GENERATIONS", "value": str(num_generations)},
        {"name": "GRPO_BETA", "value": str(grpo_beta)},
        {"name": "MAX_COMPLETION_LENGTH", "value": str(max_completion_length)},
        {"name": "GRPO_LOSS_TYPE", "value": loss_type},
        {"name": "GRPO_TEMPERATURE", "value": str(temperature)},
        {"name": "GRPO_SYSTEM_PROMPT", "value": system_prompt},
        {"name": "GRPO_WEAK_CATEGORIES", "value": weak_categories_json},
        {"name": "USE_VLLM", "value": "1" if use_vllm else "0"},
        {"name": "S3_ENDPOINT", "value": s3_endpoint},
        {"name": "S3_ACCESS_KEY", "value": s3_access_key},
        {"name": "S3_SECRET_KEY", "value": s3_secret_key},
    ]

    trainjob = {
        "apiVersion": f"{TRAINJOB_GROUP}/{TRAINJOB_VERSION}",
        "kind": "TrainJob",
        "metadata": {"name": job_name, "namespace": namespace},
        "spec": {
            "runtimeRef": {"name": "torch-distributed"},
            "trainer": {
                "image": image,
                "numNodes": safe_gpu_nodes,
                "env": env_list,
                "resourcesPerNode": {
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
            },
            "podTemplateOverrides": [{
                "targetJobs": [{"name": "node"}],
                "spec": {
                    "nodeSelector": {"nvidia.com/gpu.present": "true"},
                    "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}}],
                    "containers": [{"name": "node", "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}]}],
                },
            }],
        },
    }

    custom_api.create_namespaced_custom_object(
        group=TRAINJOB_GROUP,
        version=TRAINJOB_VERSION,
        namespace=namespace,
        plural=TRAINJOB_PLURAL,
        body=trainjob,
    )

    poll_interval = 30
    timeout = 36000
    elapsed = 0

    print(f"[{time.strftime('%H:%M:%S')}] Submitted GRPO TrainJob {job_name}")
    print(f"[{time.strftime('%H:%M:%S')}] Output will go to: {model_output_s3_path}")

    def _model_s3_timestamp():
        try:
            bucket = model_output_s3_path.replace("s3://", "").split("/", 1)[0]
            prefix = model_output_s3_path.replace("s3://", "").split("/", 1)[1]
            resp = s3.head_object(Bucket=bucket, Key=prefix + "model.safetensors")
            return resp["LastModified"]
        except Exception:
            return None

    model_ts_before = _model_s3_timestamp()

    def _model_is_new() -> bool:
        ts = _model_s3_timestamp()
        if ts is None:
            return False
        if model_ts_before is None:
            return True
        return ts > model_ts_before

    try:
        job_failed = False
        fail_msg = ""

        while elapsed < timeout:
            time.sleep(poll_interval)
            elapsed += poll_interval

            if _model_is_new():
                try:
                    custom_api.delete_namespaced_custom_object(
                        group=TRAINJOB_GROUP, version=TRAINJOB_VERSION,
                        namespace=namespace, plural=TRAINJOB_PLURAL, name=job_name,
                    )
                except Exception:
                    pass
                return model_output_s3_path

            if job_failed:
                continue

            try:
                job = custom_api.get_namespaced_custom_object(
                    group=TRAINJOB_GROUP, version=TRAINJOB_VERSION,
                    namespace=namespace, plural=TRAINJOB_PLURAL, name=job_name,
                )
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Could not fetch job status ({e})")
                continue

            conditions = job.get("status", {}).get("conditions", [])
            for c in conditions:
                ctype = c.get("type")
                if ctype == "Complete" and c.get("status") == "True":
                    return model_output_s3_path
                if ctype == "Failed" and c.get("status") == "True":
                    job_failed = True
                    fail_msg = c.get("message", "unknown error")
                    print(f"[{time.strftime('%H:%M:%S')}] GRPO job failed, polling S3...")
                    break

            hrs, rem = divmod(elapsed, 3600)
            mins = rem // 60
            print(f"[{time.strftime('%H:%M:%S')}] GRPO TrainJob running (elapsed={hrs}h{mins}m)")

        if job_failed:
            raise RuntimeError(f"GRPO TrainJob {job_name} failed: {fail_msg}")
        raise TimeoutError(f"GRPO TrainJob {job_name} did not complete within {timeout}s")
    finally:
        print(f"[{time.strftime('%H:%M:%S')}] GRPO step done. Restoring KServe...")
        _restore_kserve()
