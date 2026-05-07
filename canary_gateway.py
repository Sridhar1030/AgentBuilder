"""
Canary Gateway -- Lightweight FastAPI proxy for progressive traffic splitting.

Routes /v1/chat/completions between Teacher (Ollama 32B) and Student (KServe 1.5B)
based on configurable weights. Adds model attribution to every response.

Syncs split ratio to:
  1. In-memory state (instant)
  2. Kubernetes ConfigMap (persistent across restarts)
  3. Istio VirtualService (production-level routing)

Usage:
    uvicorn canary_gateway:app --host 0.0.0.0 --port 8090

Env vars:
    TEACHER_URL        (default: http://ollama:11434)
    STUDENT_URL        (default: http://code-review-llm-predictor:80)
    TEACHER_MODEL      (default: qwen2.5-coder:32b-instruct-q4_K_M)
    NAMESPACE          (default: sridharproject)
    CONFIGMAP_NAME     (default: traffic-split)
    VIRTUALSERVICE_NAME (default: code-review-gateway)
"""

import json
import logging
import os
import random
import time
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

app = FastAPI(title="Canary Gateway", version="1.0.0")
logger = logging.getLogger("canary_gateway")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TEACHER_URL = os.getenv("TEACHER_URL", "http://ollama:11434").rstrip("/")
STUDENT_URL = os.getenv("STUDENT_URL", "http://code-review-llm-predictor:80").rstrip("/")
TEACHER_MODEL = os.getenv("TEACHER_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M")
NAMESPACE = os.getenv("NAMESPACE", "sridharproject")
CONFIGMAP_NAME = os.getenv("CONFIGMAP_NAME", "traffic-split")
VIRTUALSERVICE_NAME = os.getenv("VIRTUALSERVICE_NAME", "code-review-gateway")

_state = {
    "teacher_weight": 90,
    "student_weight": 10,
    "last_student_score": 0.0,
    "shift_history": [],
    "request_count": {"teacher": 0, "student": 0},
}


def _load_state_from_configmap():
    """Try to load persisted state from the K8s ConfigMap on startup."""
    try:
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        v1 = client.CoreV1Api()
        cm = v1.read_namespaced_config_map(CONFIGMAP_NAME, NAMESPACE)
        data = cm.data or {}

        _state["teacher_weight"] = int(data.get("teacher_weight", "90"))
        _state["student_weight"] = int(data.get("student_weight", "10"))
        _state["last_student_score"] = float(data.get("last_student_score", "0.0"))
        try:
            _state["shift_history"] = json.loads(data.get("shift_history", "[]"))
        except json.JSONDecodeError:
            _state["shift_history"] = []

        logger.info(
            "Loaded state from ConfigMap: teacher=%d%% student=%d%%",
            _state["teacher_weight"],
            _state["student_weight"],
        )
    except Exception as exc:
        logger.warning("Could not load ConfigMap (using defaults): %s", exc)


def _save_state_to_configmap():
    """Persist current split to K8s ConfigMap."""
    try:
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        v1 = client.CoreV1Api()
        body = client.V1ConfigMap(
            metadata=client.V1ObjectMeta(name=CONFIGMAP_NAME),
            data={
                "teacher_weight": str(_state["teacher_weight"]),
                "student_weight": str(_state["student_weight"]),
                "last_student_score": str(_state["last_student_score"]),
                "shift_history": json.dumps(_state["shift_history"][-50:]),
            },
        )
        v1.patch_namespaced_config_map(CONFIGMAP_NAME, NAMESPACE, body)
        logger.info("ConfigMap updated")
    except Exception as exc:
        logger.warning("Could not update ConfigMap: %s", exc)


def _patch_virtualservice():
    """Sync weights to the Istio VirtualService."""
    try:
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        api = client.CustomObjectsApi()
        patch = {
            "spec": {
                "http": [
                    {
                        "route": [
                            {
                                "destination": {
                                    "host": f"ollama.{NAMESPACE}.svc.cluster.local",
                                    "port": {"number": 11434},
                                },
                                "weight": _state["teacher_weight"],
                            },
                            {
                                "destination": {
                                    "host": f"code-review-llm-predictor.{NAMESPACE}.svc.cluster.local",
                                    "port": {"number": 80},
                                },
                                "weight": _state["student_weight"],
                            },
                        ]
                    }
                ]
            }
        }
        api.patch_namespaced_custom_object(
            group="networking.istio.io",
            version="v1",
            namespace=NAMESPACE,
            plural="virtualservices",
            name=VIRTUALSERVICE_NAME,
            body=patch,
        )
        logger.info(
            "VirtualService %s patched: teacher=%d%% student=%d%%",
            VIRTUALSERVICE_NAME,
            _state["teacher_weight"],
            _state["student_weight"],
        )
    except Exception as exc:
        logger.warning("Could not patch VirtualService: %s", exc)


def _sync_all():
    """Push current weights to ConfigMap + VirtualService."""
    _save_state_to_configmap()
    _patch_virtualservice()


def _pick_backend() -> str:
    """Weighted random selection: returns 'teacher' or 'student'."""
    return random.choices(
        ["teacher", "student"],
        weights=[_state["teacher_weight"], _state["student_weight"]],
        k=1,
    )[0]


@app.on_event("startup")
async def startup():
    _load_state_from_configmap()


@app.get("/health")
async def health():
    return {"status": "ok", "teacher_url": TEACHER_URL, "student_url": STUDENT_URL}


@app.get("/split")
async def get_split():
    return {
        "teacher": _state["teacher_weight"],
        "student": _state["student_weight"],
        "last_student_score": _state["last_student_score"],
        "request_count": _state["request_count"],
        "shift_history": _state["shift_history"][-10:],
    }


@app.post("/split")
async def set_split(request: Request):
    body = await request.json()
    teacher = int(body.get("teacher", _state["teacher_weight"]))
    student = int(body.get("student", _state["student_weight"]))

    if teacher + student != 100:
        return JSONResponse(
            status_code=400,
            content={"error": f"Weights must sum to 100, got {teacher + student}"},
        )

    old_teacher = _state["teacher_weight"]
    old_student = _state["student_weight"]
    _state["teacher_weight"] = teacher
    _state["student_weight"] = student

    if "last_student_score" in body:
        _state["last_student_score"] = float(body["last_student_score"])

    _state["shift_history"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "from": {"teacher": old_teacher, "student": old_student},
        "to": {"teacher": teacher, "student": student},
        "reason": body.get("reason", "manual"),
    })

    _sync_all()

    return {
        "teacher": teacher,
        "student": student,
        "previous": {"teacher": old_teacher, "student": old_student},
        "synced": ["configmap", "virtualservice"],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    backend = _pick_backend()
    _state["request_count"][backend] += 1

    if backend == "teacher":
        url = f"{TEACHER_URL}/v1/chat/completions"
        body["model"] = TEACHER_MODEL
        label = f"teacher-32b ({_state['teacher_weight']}% traffic)"
    else:
        url = f"{STUDENT_URL}/v1/chat/completions"
        body.pop("model", None)
        label = f"student-1.5b ({_state['student_weight']}% traffic)"

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            result = resp.json()
    except Exception as exc:
        logger.error("Backend %s failed: %s", backend, exc)
        return JSONResponse(
            status_code=502,
            content={"error": f"Backend {backend} failed: {str(exc)}"},
            headers={"X-Served-By": backend},
        )

    latency_ms = (time.monotonic() - start) * 1000
    result["model"] = label
    if "usage" not in result:
        result["usage"] = {}
    result["usage"]["_served_by"] = backend
    result["usage"]["_latency_ms"] = round(latency_ms, 1)

    return Response(
        content=json.dumps(result),
        media_type="application/json",
        headers={
            "X-Served-By": backend,
            "X-Split": f"teacher={_state['teacher_weight']},student={_state['student_weight']}",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8090)
