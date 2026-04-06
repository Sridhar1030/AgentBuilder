"""
KFP Component 3 -- Deploy Model

Patches the KServe InferenceService storageUri to point to the newly
fine-tuned model, then waits for the rollout to become Ready.
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["kubernetes==32.0.1"],
)
def deploy_model(
    model_s3_path: str,
    isvc_name: str,
    namespace: str,
) -> str:
    """Patch InferenceService storageUri and wait for Ready."""
    import time
    from kubernetes import client, config

    print("=" * 60)
    print("DEPLOY MODEL STEP")
    print("=" * 60)
    print(f"  ISVC name:    {isvc_name}")
    print(f"  Namespace:    {namespace}")
    print(f"  Model path:   {model_s3_path}")
    print("=" * 60)

    config.load_incluster_config()

    custom_api = client.CustomObjectsApi()
    group = "serving.kserve.io"
    version = "v1beta1"
    plural = "inferenceservices"

    storage_uri = model_s3_path
    if not storage_uri.startswith("s3://"):
        storage_uri = f"s3://{storage_uri}"
    if not storage_uri.endswith("/"):
        storage_uri += "/"

    patch_body = {
        "spec": {
            "predictor": {
                "model": {
                    "storageUri": storage_uri,
                }
            }
        }
    }

    print(f"[{time.strftime('%H:%M:%S')}] Patching {isvc_name} storageUri -> {storage_uri}")
    custom_api.patch_namespaced_custom_object(
        group=group,
        version=version,
        namespace=namespace,
        plural=plural,
        name=isvc_name,
        body=patch_body,
    )

    print(f"[{time.strftime('%H:%M:%S')}] Patch applied. Waiting 30s for rollout to begin...")
    time.sleep(30)

    # Wait for the ISVC to become Ready with the NEW storageUri (up to 10 min)
    timeout = 600
    poll_interval = 15
    elapsed = 0

    while elapsed < timeout:
        isvc = custom_api.get_namespaced_custom_object(
            group=group,
            version=version,
            namespace=namespace,
            plural=plural,
            name=isvc_name,
        )

        # Confirm the storageUri was actually updated
        current_uri = (
            isvc.get("spec", {})
            .get("predictor", {})
            .get("model", {})
            .get("storageUri", "")
        )
        conditions = isvc.get("status", {}).get("conditions", [])
        ready = any(
            c.get("type") == "Ready" and c.get("status") == "True"
            for c in conditions
        )

        if ready and current_uri == storage_uri:
            url = isvc.get("status", {}).get("url", "")
            print(f"[{time.strftime('%H:%M:%S')}] ISVC {isvc_name} is Ready: {url}")
            svc_url = f"http://{isvc_name}-predictor.{namespace}.svc.cluster.local:8080"
            print(f"[{time.strftime('%H:%M:%S')}] Internal URL: {svc_url}")
            print(f"[{time.strftime('%H:%M:%S')}] Waiting 60s for vLLM to fully load model into GPU...")
            time.sleep(60)
            print(f"[{time.strftime('%H:%M:%S')}] Deploy complete.")
            return svc_url

        status_msg = f"ready={ready}, storageUri matches={current_uri == storage_uri}"
        print(f"[{time.strftime('%H:%M:%S')}] Waiting for {isvc_name}... ({elapsed}s/{timeout}s) [{status_msg}]")
        time.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(
        f"InferenceService {isvc_name} did not become Ready within {timeout}s"
    )
