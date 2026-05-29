#!/usr/bin/env python3
"""
Baseline Evaluation -- Untuned Qwen2.5-Coder-1.5B-Instruct

Loads the raw base model from HuggingFace, runs the same 15 curated test
diffs used by the pipeline's evaluate step with agent-eval-harness judges
(correctness, review_quality, conciseness, format_check), logs to MLflow
experiment AgentBuilder-Final (run AB_baseline, stage=baseline), and saves
scores JSON to MinIO for comparison with pipeline eval runs.

Runs as a Kubernetes Job on a single GPU.
"""

import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import boto3
import mlflow
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)

# ── Config (override via env vars for domain portability) ─────────────────
MODEL_ID = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
TEACHER_URL = os.environ.get("TEACHER_URL", "http://ollama.sridharproject.svc.cluster.local:11434/v1/chat/completions")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M")
MLFLOW_URI = os.environ.get("MLFLOW_URI", "http://mlflow.sridharproject.svc.cluster.local:5000")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://minio.sridharproject.svc.cluster.local:9000")
S3_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
S3_SECRET = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin123")
BASELINE_BUCKET = os.environ.get("BASELINE_BUCKET", "mlflow-artifacts")
BASELINE_S3_KEY = os.environ.get("BASELINE_S3_KEY", "baseline/scores.json")
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "sridhar-models")
BASE_MODEL_S3_PREFIX = os.environ.get("BASE_MODEL_S3_PREFIX", "code-review-1.5b-base/")

MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "AgentBuilder-Final")
EVAL_YAML_PATH = os.environ.get("EVAL_YAML_PATH", "/opt/config/eval.yaml")
JUDGE_WEIGHTS = {"correctness": 0.4, "conciseness": 0.2, "review_quality": 0.3, "format_check": 0.1}


def _ensure_harness():
    """Install agent-eval-harness if not present (baseline job image may lack it)."""
    try:
        import agent_eval  # noqa: F401
    except ImportError:
        print("Installing agent-eval-harness...")
        subprocess.check_call(
            [
                sys.executable, "-m", "pip", "install", "-q",
                "https://github.com/opendatahub-io/agent-eval-harness/archive/refs/heads/main.tar.gz",
            ],
        )


def load_test_entries():
    """Load structured test cases (same schema as pipeline evaluate)."""
    tq_file = os.environ.get("TEST_QUESTIONS_FILE", "")
    if tq_file and os.path.exists(tq_file):
        with open(tq_file) as f:
            data = json.load(f)
        entries = []
        for item in data:
            if isinstance(item, dict):
                entries.append(item)
            else:
                entries.append({"question": str(item), "has_bug": True, "expected_issues": []})
        print(f"Loaded {len(entries)} test entries from {tq_file}")
        return entries
    # Fallback: legacy string-only list below
    return None


# ── Test questions: load from file if TEST_QUESTIONS_FILE env is set ──────
_tq_entries = load_test_entries()
if _tq_entries is not None:
    TEST_ENTRIES = _tq_entries
    TEST_QUESTIONS = [e["question"] for e in TEST_ENTRIES]
else:
    TEST_ENTRIES = None
    TEST_QUESTIONS = [
    "Review the following code diff and identify any issues:\n\nFile: pkg/controller/job_controller.go\nLanguage: Go\n\n```diff\n@@ -189,7 +189,7 @@\n func (p *Progress) buildProgressServerCaCrtConfigMap(ctx context.Context, trainJob *trainer.TrainJob) (*corev1ac.ConfigMapApplyConfiguration, error) {\n \tsecret := &corev1.Secret{}\n \tif err := p.client.Get(ctx, secretKey, secret); err == nil {\n \t\tif _, ok := secret.Data[\"ca.crt\"]; !ok {\n-\t\t\treturn nil, fmt.Errorf(\"ca.crt not found: %w\", err)\n+\t\t\treturn nil, fmt.Errorf(\"ca.crt not found in TLS secret\")\n \t\t}\n```",
    "Review the following code diff and identify any issues:\n\nFile: test/e2e/testdata/status_update.py\nLanguage: Python\n\n```diff\n@@ -0,0 +1,30 @@\n+import os, urllib.request\n+\n+token = open(os.environ[\"KUBEFLOW_TRAINER_SERVER_TOKEN\"]).read()\n+req = urllib.request.Request(url, method=\"POST\")\n+req.add_header(\"Authorization\", f\"Bearer {token}\")\n```",
    "Review the following code diff and identify any issues:\n\nFile: pkg/webhooks/trainjob_webhook.go\nLanguage: Go\n\n```diff\n@@ -50,6 +50,10 @@\n+func (d *TrainJobDefaulter) Default(ctx context.Context, trainJob *trainer.TrainJob) error {\n+\tnow := metav1.Now()\n+\tfor i := range trainJob.Spec.RuntimePatches {\n+\t\ttrainJob.Spec.RuntimePatches[i].Time = &now\n+\t}\n```",
    "Review the following code diff and identify any issues:\n\nFile: pkg/util/cert/cert.go\nLanguage: Go\n\n```diff\n@@ -69,6 +69,12 @@\n+func SetupTLSConfig(mgr ctrl.Manager) (*tls.Config, error) {\n+\tcertWatcher, err := certwatcher.New(certDir+\"/tls.crt\", certDir+\"/tls.key\")\n+\tif err != nil { return nil, err }\n+\treturn &tls.Config{\n+\t\tGetCertificate: certWatcher.GetCertificate,\n+\t}, nil\n+}\n```",
    "Review the following code diff and identify any issues:\n\nFile: manifests/overlays/manager/kustomization.yaml\nLanguage: YAML\n\n```diff\n@@ -3,4 +3,4 @@\n resources:\n   - ../../base/rbac\n   - ../../base/manager\n-  - ../../base/webhook\n+  # - ../../base/webhook\n```",
    "Review the following code diff and identify any issues:\n\nFile: pkg/status/server.go\nLanguage: Go\n\n```diff\n@@ -100,6 +100,10 @@\n+func (s *Server) authorizeRequest(r *http.Request, namespace, name string) bool {\n+\tpod := &corev1.Pod{}\n+\tif err := s.client.Get(r.Context(), types.NamespacedName{Namespace: namespace, Name: podName}, pod); err != nil {\n+\t\treturn false\n+\t}\n```",
    "Review the following code diff and identify any issues:\n\nFile: pkg/runtime/framework/plugins/flux/flux.go\nLanguage: Go\n\n```diff\n@@ -402,8 +406,12 @@\n+\tvar tasks int32\n+\tnodes := *trainJob.Spec.Trainer.NumNodes\n+\tif trainJob.Spec.Trainer.NumProcPerNode != nil {\n+\t\ttasks = *trainJob.Spec.Trainer.NumProcPerNode\n+\t}\n+\tflags := fmt.Sprintf(\"-N %d -n %d\", nodes, tasks)\n+\trspec := fmt.Sprintf(\"--cores=0-%d\", tasks-1)\n```",
    "Review the following code diff and identify any issues:\n\nFile: manifests/base/webhook/manifests.yaml\nLanguage: YAML\n\n```diff\n@@ -0,0 +1,15 @@\n+apiVersion: admissionregistration.k8s.io/v1\n+kind: MutatingWebhookConfiguration\n+metadata:\n+  name: mutating-webhook-configuration\n+webhooks:\n+- clientConfig:\n+    service:\n+      name: webhook-service\n+      namespace: system\n+      path: /mutate-trainer-kubeflow-org-v1alpha1-trainjob\n```",
    "Review the following code diff and identify any issues:\n\nFile: manifests/base/manager/controller_manager_config.yaml\nLanguage: YAML\n\n```diff\n@@ -5,4 +5,4 @@\n controller:\n   manager:\n-    leaderElect: true\n+    leaderElect: false\n   certManagement:\n-    enable: true\n+    enable: false\n```",
    "Review the following code diff and identify any issues:\n\nFile: pkg/statusserver/middleware.go\nLanguage: Go\n\n```diff\n@@ -0,0 +1,20 @@\n+func authenticationMiddleware(logger logr.Logger) Middleware {\n+\treturn func(next http.Handler) http.Handler {\n+\t\treturn http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {\n+\t\t\tauth := r.Header.Get(\"Authorization\")\n+\t\t\tif !strings.HasPrefix(auth, \"Bearer \") {\n+\t\t\t\thttp.Error(w, \"Invalid Authorization header format\", http.StatusUnauthorized)\n+\t\t\t\treturn\n+\t\t\t}\n+\t\t\ttoken := strings.TrimPrefix(auth, \"Bearer \")\n+\t\t\tif token == \"\" {\n+\t\t\t\thttp.Error(w, \"Empty bearer token\", http.StatusUnauthorized)\n+\t\t\t\treturn\n+\t\t\t}\n```",
    "Review the following code diff and identify any issues:\n\nFile: pkg/runtime/framework/plugins/plainml/plainml.go\nLanguage: Go\n\n```diff\n@@ -80,6 +80,10 @@\n+\tfor i, patch := range trainJob.Spec.RuntimePatches {\n+\t\tif i < len(old.Spec.RuntimePatches) && reflect.DeepEqual(patch, old.Spec.RuntimePatches[i]) {\n+\t\t\ttrainJob.Spec.RuntimePatches[i].Time = old.Spec.RuntimePatches[i].Time\n+\t\t} else {\n+\t\t\ttrainJob.Spec.RuntimePatches[i].Time = &now\n+\t\t}\n```",
    "Review the following code diff and identify any issues:\n\nFile: pkg/constants/constants.go\nLanguage: Go\n\n```diff\n@@ -114,4 +114,7 @@\n \tFluxCurveVolumePath = \"/curve\"\n \n+\t// Ensure MPI has full memory of the host\n+\tFluxMemoryVolumeName = \"dshm\"\n```",
    "Review the following code diff and identify any issues:\n\nFile: pkg/runtime/core/trainingruntime.go\nLanguage: Go\n\n```diff\n@@ -50,6 +50,15 @@\n+func mergeRuntimePatches(jobSet *jobsetv1.JobSet, patches []trainer.RuntimePatch) {\n+\tfor _, rJobPatch := range patches {\n+\t\tfor i, rJob := range jobSet.Spec.ReplicatedJobs {\n+\t\t\tif rJob.Name == rJobPatch.TargetReplicatedJob {\n+\t\t\t\tapplyPodPatch(&jobSet.Spec.ReplicatedJobs[i].Template.Spec.Template, rJobPatch.Template.PodTemplatePatch)\n+\t\t\t}\n+\t\t}\n+\t}\n+}\n```",
    "Review the following code diff and identify any issues:\n\nFile: pkg/features/features.go\nLanguage: Go\n\n```diff\n@@ -30,6 +30,13 @@\n+const (\n+\tTrainJobRuntimeStatus featuregate.Feature = \"TrainJobRuntimeStatus\"\n+)\n+\n+var defaultFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{\n+\tTrainJobRuntimeStatus: {Default: false, PreRelease: featuregate.Alpha},\n+}\n```",
    "Review the following code diff and identify any issues:\n\nFile: charts/kubeflow-trainer/values.yaml\nLanguage: YAML\n\n```diff\n@@ -138,6 +138,13 @@\n+    statusServer:\n+      # -- Port that the TrainJob status server serves on.\n+      port: 10443\n+      # -- QPS rate limit for the TrainJob Status Server api client\n+      qps: 5\n+      # -- Burst rate limit for the TrainJob Status Server api client\n+      burst: 10\n```",
    ]

GRADING_PROMPT = os.environ.get("GRADING_PROMPT", "") or (
    "You are grading an AI-generated code review comment as it would appear in a "
    "GitHub Pull Request. Rate it 1-10 based on how useful it would be to a developer"
    "reading their PR:\n be harsh and critical. If you think the code is bad, give it a 1."
    "- Correct identification: Does it spot the actual issue in the diff (not a hallucinated one)?\n"
    "- Conciseness: Is it brief and to-the-point, like a real PR comment? "
    "Verbose essays are BAD -- reviewers want short, clear feedback.\n"
    "- Actionability: Does it tell the developer exactly what to fix, "
    "ideally with a code suggestion?\n"
    "- False positive avoidance: If the code is fine, does it correctly say so "
    "instead of inventing problems?\n"
    "- Relevance: Does it focus on the changed lines, not lecture about unrelated topics?\n"
    "A score of 1 means hallucinated issues or completely irrelevant rambling. "
    "A score of 5 means verbose but technically correct. "
    "A score of 10 means a perfect, concise, actionable PR comment a senior engineer would leave.\n"
    'Respond with ONLY a JSON object: {"score": <number>, "reason": "<brief reason>"}'
)


def ts():
    return time.strftime("%H:%M:%S")


def teacher_call(messages, max_tokens=512, temperature=0.7):
    for attempt in range(8):
        try:
            resp = requests.post(
                TEACHER_URL,
                json={
                    "model": TEACHER_MODEL,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=600,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = min(2**attempt * 5, 120)
                print(f"  [{ts()}] Teacher {resp.status_code}, retry in {wait}s ({attempt+1}/8)")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except (requests.ConnectionError, requests.Timeout) as e:
            wait = min(2**attempt * 5, 120)
            print(f"  [{ts()}] Teacher conn error, retry in {wait}s: {e}")
            time.sleep(wait)
    raise RuntimeError("Teacher unreachable after 8 retries")


def teacher_grade(question, answer):
    raw = teacher_call(
        [
            {"role": "system", "content": GRADING_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nStudent Response: {answer}"},
        ],
        max_tokens=200,
        temperature=0.0,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]*\"score\"\s*:\s*\d+[^{}]*\}", raw)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {"score": 0, "reason": f"Parse fail: {raw[:200]}"}


def query_base_model(model, tokenizer, question):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run_harness_benchmark(model, tokenizer, entries):
    """Run same judges as pipeline evaluate.py for apples-to-apples metrics."""
    _ensure_harness()
    from agent_eval.config import EvalConfig

    eval_path = EVAL_YAML_PATH
    if not os.path.exists(eval_path):
        repo_eval = Path(__file__).resolve().parents[2] / "eval" / "eval.yaml"
        eval_path = str(repo_eval) if repo_eval.exists() else eval_path
    config = EvalConfig.from_yaml(eval_path)
    print(f"  Loaded {len(config.judges)} harness judges from {eval_path}")

    def _make_inline_check(jc):
        source = jc.check
        wrapped = f"def _check(outputs):\n{textwrap.indent(source, '    ')}"
        code = compile(wrapped, f"<check:{jc.name}>", "exec")
        ns = {"__builtins__": __builtins__}
        exec(code, ns)
        return ns["_check"]

    def _make_llm_judge(jc):
        prompt_template = jc.prompt or ""

        def scorer(outputs=None):
            outputs = outputs or {}
            review = outputs.get("artifacts_content", "")
            question = outputs.get("question", "")
            annotations = outputs.get("annotations", {})
            if not review:
                return 1, "No review output found"
            prompt = prompt_template.replace(
                "{{ outputs }}",
                f"## Student Review\n\n{review}\n\n## Code Diff\n\n{question}",
            )
            prompt = prompt.replace(
                "{{ annotations }}",
                f"## Annotations\n\n{json.dumps(annotations, indent=2)}",
            )
            raw = teacher_call(
                [
                    {"role": "system", "content": "You are a code review quality judge. Return only JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.0,
            )
            try:
                parsed = json.loads(raw)
                score = parsed.get("score", 3)
            except json.JSONDecodeError:
                m = re.search(r'"score"\s*:\s*(\d+)', raw)
                score = int(m.group(1)) if m else 3
                parsed = {"rationale": raw[:200]}
            score = max(1, min(5, score))
            return score, parsed.get("rationale", raw[:200])

        return scorer

    judges = []
    for jc in config.judges:
        if jc.check:
            judges.append((jc.name, _make_inline_check(jc)))
        elif jc.prompt or jc.prompt_file:
            judges.append((jc.name, _make_llm_judge(jc)))

    judge_aggregates = {name: {"values": []} for name, _ in judges}
    harness_results = []

    for i, entry in enumerate(entries):
        q = entry.get("question", entry) if isinstance(entry, dict) else str(entry)
        print(f"--- Harness Q{i+1}/{len(entries)} ---")
        answer = query_base_model(model, tokenizer, q)
        annotations = {
            "has_bug": entry.get("has_bug", True) if isinstance(entry, dict) else True,
            "expected_issues": entry.get("expected_issues", []) if isinstance(entry, dict) else [],
            "expected_behavior": entry.get("expected_behavior", "") if isinstance(entry, dict) else "",
            "category": entry.get("category", "unknown") if isinstance(entry, dict) else "unknown",
        }
        outputs = {
            "artifacts_content": answer,
            "annotations": annotations,
            "question": q,
        }
        case_judges = {}
        for judge_name, scorer_fn in judges:
            try:
                result = scorer_fn(outputs)
                value = result[0] if isinstance(result, tuple) else result
                case_judges[judge_name] = value
                judge_aggregates[judge_name]["values"].append(value)
                if isinstance(value, bool):
                    print(f"  {judge_name}: {'PASS' if value else 'FAIL'}")
                else:
                    print(f"  {judge_name}: {value}/5")
            except Exception as e:
                print(f"  {judge_name}: ERROR -- {e}")
        harness_results.append({"question": q, "answer": answer, "judges": case_judges})
        time.sleep(1)

    agg_summary = {}
    for name, data in judge_aggregates.items():
        values = [v for v in data["values"] if v is not None]
        if not values:
            agg_summary[name] = {"mean": None, "pass_rate": None}
            continue
        if all(isinstance(v, bool) for v in values):
            pr = sum(values) / len(values)
            agg_summary[name] = {"pass_rate": round(pr, 4), "mean": round(pr, 4)}
        else:
            mean = sum(values) / len(values)
            agg_summary[name] = {"mean": round(mean, 4), "pass_rate": None}

    composite_parts = []
    for name, weight in JUDGE_WEIGHTS.items():
        agg = agg_summary.get(name, {})
        if agg.get("pass_rate") is not None:
            composite_parts.append(agg["pass_rate"] * weight)
        elif agg.get("mean") is not None:
            composite_parts.append((agg["mean"] / 5.0) * weight)
    composite = round(sum(composite_parts), 4) if composite_parts else 0.0
    print(f"\n  Harness composite_score: {composite}")
    for name, agg in agg_summary.items():
        if agg.get("pass_rate") is not None:
            print(f"  {name}: pass_rate={agg['pass_rate']:.1%}")
        elif agg.get("mean") is not None:
            print(f"  {name}: mean={agg['mean']:.2f}/5")
    return agg_summary, composite, harness_results


def upload_model_to_s3(local_path, s3):
    """Upload the downloaded base model files to MinIO for future KServe use."""
    import glob as globmod

    print(f"\n[{ts()}] Uploading base model to s3://{MODEL_BUCKET}/{BASE_MODEL_S3_PREFIX} ...")
    count = 0
    for fpath in globmod.glob(os.path.join(local_path, "**", "*"), recursive=True):
        if not os.path.isfile(fpath):
            continue
        key = BASE_MODEL_S3_PREFIX + os.path.relpath(fpath, local_path)
        s3.upload_file(fpath, MODEL_BUCKET, key)
        count += 1
    print(f"[{ts()}] Uploaded {count} files to s3://{MODEL_BUCKET}/{BASE_MODEL_S3_PREFIX}")


def main():
    print("=" * 70)
    print("  BASELINE EVALUATION — Untuned Qwen2.5-Coder-1.5B-Instruct")
    print("=" * 70)
    print(f"  Model:    {MODEL_ID}")
    print(f"  Teacher:  {TEACHER_MODEL}")
    print(f"  MLflow:   {MLFLOW_URI}")
    print(f"  Questions: {len(TEST_QUESTIONS)}")
    print(f"  GPU:      {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)

    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_KEY,
        aws_secret_access_key=S3_SECRET,
    )

    # ── Load raw base model ──────────────────────────────────────────────
    print(f"\n[{ts()}] Loading base model from HuggingFace: {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    print(f"[{ts()}] Model loaded on {model.device}")

    model_cache_dir = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "hub",
    )
    local_model_path = None
    for d in os.listdir(model_cache_dir):
        if "Qwen2.5-Coder-1.5B-Instruct" in d:
            candidate = os.path.join(model_cache_dir, d, "snapshots")
            if os.path.isdir(candidate):
                snaps = os.listdir(candidate)
                if snaps:
                    local_model_path = os.path.join(candidate, snaps[0])
                    break
    # Check if base model already exists in S3, skip upload if so
    try:
        s3.head_object(Bucket=MODEL_BUCKET, Key=BASE_MODEL_S3_PREFIX + "config.json")
        print(f"[{ts()}] Base model already in S3, skipping upload")
    except Exception:
        if local_model_path:
            upload_model_to_s3(local_model_path, s3)
        else:
            print(f"[{ts()}] Could not find local model cache, skipping S3 upload")

    # ── Harness eval (same judges as pipeline — blog metrics) ─────────────
    entries = TEST_ENTRIES if TEST_ENTRIES else [
        {"question": q, "has_bug": True, "expected_issues": []} for q in TEST_QUESTIONS
    ]
    print(f"\n[{ts()}] Running harness baseline on {len(entries)} questions...\n")
    agg_summary, composite_score, harness_results = run_harness_benchmark(
        model, tokenizer, entries
    )

    # ── Legacy teacher 1-10 grading (reference only) ─────────────────────
    print(f"\n[{ts()}] Running teacher 1-10 reference grading...\n")
    results = []
    for i, q in enumerate(TEST_QUESTIONS):
        q_short = q[:80].replace("\n", " ")
        print(f"--- Question {i+1}/{len(TEST_QUESTIONS)} ---")
        print(f"Q: {q_short}...")

        answer = query_base_model(model, tokenizer, q)
        print(f"Base model response ({len(answer)} chars):\n{answer[:300]}\n")

        grade = teacher_grade(q, answer)
        print(f"Teacher Grade: {grade['score']}/10  Reason: {grade.get('reason', 'N/A')}\n")

        results.append({
            "question": q,
            "student_answer": answer,
            "student_score": grade["score"],
            "reason": grade.get("reason", ""),
        })

    scores = [r["student_score"] for r in results if isinstance(r.get("student_score"), (int, float))]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # ── Teacher self-grade (same as pipeline evaluate) ───────────────────
    print("\n" + "=" * 70)
    print("TEACHER BASELINE (teacher answers + self-grades)")
    print("=" * 70)
    for i, r in enumerate(results):
        teacher_answer = teacher_call(
            [
                {"role": "system", "content": (
                    "You are a senior code reviewer specializing in Go, Python, and Kubernetes. "
                    "Review the given code diff and identify any issues related to bugs, security, "
                    "performance, reliability, style, or Kubernetes best practices. "
                    "If the code is clean, say so. "
                    "Be concise -- 2-4 sentences max, like a real GitHub PR comment."
                )},
                {"role": "user", "content": r["question"]},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        teacher_grade_result = teacher_grade(r["question"], teacher_answer)
        r["teacher_answer"] = teacher_answer
        r["teacher_score"] = teacher_grade_result["score"]
        print(f"  Q{i+1}: teacher={teacher_grade_result['score']}/10")

    teacher_scores = [r["teacher_score"] for r in results if isinstance(r.get("teacher_score"), (int, float))]
    teacher_avg = sum(teacher_scores) / len(teacher_scores) if teacher_scores else 0.0

    # ── Print summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BASELINE RESULTS")
    print("=" * 70)
    print(f"  {'Q#':<4} {'Base Model':>12} {'Teacher':>10} {'Gap':>8}")
    print(f"  {'─'*4} {'─'*12} {'─'*10} {'─'*8}")
    for i, r in enumerate(results):
        gap = r.get("teacher_score", 0) - r["student_score"]
        print(f"  Q{i+1:<3} {r['student_score']:>10}/10 {r.get('teacher_score', 0):>8}/10 {gap:>+7.1f}")
    print(f"  {'─'*4} {'─'*12} {'─'*10} {'─'*8}")
    print(f"  {'AVG':<4} {avg_score:>10.2f}/10 {teacher_avg:>8.2f}/10 {teacher_avg - avg_score:>+7.2f}")
    print("=" * 70)

    # ── Log to MLflow ────────────────────────────────────────────────────
    print(f"\n[{ts()}] Logging to MLflow ({MLFLOW_URI}) ...")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = S3_ENDPOINT
    os.environ["AWS_ACCESS_KEY_ID"] = S3_KEY
    os.environ["AWS_SECRET_ACCESS_KEY"] = S3_SECRET
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("AgentBuilder-Final")
    with mlflow.start_run(run_name="AB_baseline"):
        mlflow.set_tag("model_version", "v0-baseline")
        mlflow.set_tag("eval_type", "harness_benchmark")
        mlflow.set_tag("eval_framework", "agent-eval-harness-native")
        mlflow.set_tag("stage", "baseline")
        mlflow.set_tag("model_id", MODEL_ID)
        mlflow.log_metric("composite_score", composite_score)
        for name, agg in agg_summary.items():
            if agg.get("pass_rate") is not None:
                mlflow.log_metric(f"judge_{name}_pass_rate", agg["pass_rate"])
            if agg.get("mean") is not None:
                mlflow.log_metric(f"judge_{name}_mean", agg["mean"])
        mlflow.log_metric("student_avg_score", round(avg_score, 4))
        mlflow.log_metric("teacher_avg_score", round(teacher_avg, 4))
        mlflow.log_metric("score_gap", round(teacher_avg - avg_score, 4))
        for i, r in enumerate(results):
            mlflow.log_metric(f"q{i+1}_student_score", r["student_score"])
            mlflow.log_metric(f"q{i+1}_teacher_score", r.get("teacher_score", 0))
        mlflow.log_dict({
            "harness_results": harness_results,
            "judge_aggregates": agg_summary,
            "composite_score": composite_score,
            "teacher_grading_results": results,
        }, "eval_results.json")
    print(f"[{ts()}] MLflow AB_baseline run logged to {MLFLOW_EXPERIMENT}.")

    # ── Save baseline scores to MinIO ────────────────────────────────────
    baseline_data = {
        "model_id": MODEL_ID,
        "eval_type": "harness_benchmark",
        "timestamp": int(time.time()),
        "num_questions": len(entries),
        "composite_score": composite_score,
        "judge_aggregates": agg_summary,
        "baseline_avg_score": round(avg_score, 4),
        "teacher_avg_score": round(teacher_avg, 4),
        "per_question": [
            {
                "q_index": i + 1,
                "student_score": r["student_score"],
                "teacher_score": r.get("teacher_score", 0),
            }
            for i, r in enumerate(results)
        ],
    }
    s3.put_object(
        Bucket=BASELINE_BUCKET,
        Key=BASELINE_S3_KEY,
        Body=json.dumps(baseline_data, indent=2).encode(),
    )
    print(f"[{ts()}] Baseline scores saved to s3://{BASELINE_BUCKET}/{BASELINE_S3_KEY}")
    print(f"\n[{ts()}] BASELINE EVALUATION COMPLETE.")


if __name__ == "__main__":
    main()