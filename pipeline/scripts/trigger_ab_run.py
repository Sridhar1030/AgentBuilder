#!/usr/bin/env python3
"""Upload AgentBuilderPipeline_Final and trigger a named blog-metrics run."""
import json
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PIPELINE_YAML = Path(__file__).resolve().parents[2] / "AgentBuilderPipeline_Final.yaml"
NAMESPACE = "sridharproject"
PIPELINE_NAME = "AgentBuilderPipeline_Final"


def preflight(namespace: str) -> None:
    """Fail fast if cluster access or shared services look broken."""
    subprocess.check_call(["oc", "whoami"], stdout=subprocess.DEVNULL)
    for svc in ("ollama", "mlflow", "minio"):
        subprocess.run(
            ["oc", "get", "svc", svc, "-n", namespace],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    print("Preflight OK: oc logged in, core services present")


def _api(base: str, token: str, path: str, *, method: str = "GET", data: bytes | None = None,
         content_type: str | None = None) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(f"{base}{path}", data=data, headers=headers, method=method)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, context=ctx) as resp:
        raw = resp.read().decode()
        return json.loads(raw) if raw else {}


def find_pipeline_id(base: str, token: str) -> str:
    pl = _api(base, token, "/apis/v2beta1/pipelines?page_size=200")
    matches = []
    for p in pl.get("pipelines", []):
        dn = p.get("display_name", "")
        name = p.get("name", "")
        if PIPELINE_NAME in dn or PIPELINE_NAME in name:
            matches.append(p["pipeline_id"])
    if not matches:
        raise SystemExit(f"Pipeline {PIPELINE_NAME} not found in KFP")
    return matches[0]


def upload_pipeline_version(base: str, token: str, pipeline_id: str) -> str:
    version_name = f"{PIPELINE_NAME}-{int(time.time())}"
    boundary = "----boundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="uploadfile"; '
        f'filename="{version_name}.yaml"\r\n'
        f"Content-Type: application/x-yaml\r\n\r\n"
        f"{PIPELINE_YAML.read_text()}\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    url = (
        f"{base}/apis/v2beta1/pipelines/upload_version"
        f"?pipelineid={pipeline_id}&name={version_name}"
    )
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, context=ctx) as resp:
            upload = json.loads(resp.read().decode(), strict=False)
    except urllib.error.HTTPError as e:
        raise SystemExit(e.read().decode()) from e
    vid = upload.get("pipeline_version_id")
    print(f"Uploaded pipeline version: {version_name} ({vid})")
    return vid


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: trigger_ab_run.py <run_label>  (e.g. AB_1)")
    run_label = sys.argv[1]
    preflight(NAMESPACE)

    token = subprocess.check_output(["oc", "whoami", "-t"], text=True).strip()
    route = subprocess.check_output(
        ["oc", "get", "route", "ds-pipeline-dspa", "-n", NAMESPACE,
         "-o", "jsonpath={.spec.host}"],
        text=True,
    ).strip()
    base = f"https://{route}"

    pipeline_id = find_pipeline_id(base, token)
    version_id = upload_pipeline_version(base, token, pipeline_id)

    body = json.dumps({
        "display_name": run_label,
        "pipeline_version_reference": {
            "pipeline_id": pipeline_id,
            "pipeline_version_id": version_id,
        },
        "runtime_config": {
            "parameters": {
                "run_label": run_label,
            },
        },
    }).encode()
    run = _api(
        base,
        token,
        "/apis/v2beta1/runs",
        method="POST",
        data=body,
        content_type="application/json",
    )
    print(
        f"Run: {run.get('display_name')} id={run.get('run_id')} "
        f"state={run.get('state')} run_label={run_label}"
    )


if __name__ == "__main__":
    main()
