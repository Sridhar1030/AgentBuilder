"""
KFP Component -- Evaluate (agent-eval-harness native)

Evaluates the student model using judges loaded from eval.yaml via the
actual agent-eval-harness EvalConfig.  The eval.yaml is the SINGLE source
of truth for judge definitions, thresholds, and scoring logic.

Judge types supported:
  - Inline check: Python snippet from eval.yaml's `check` field
  - LLM judge: prompt from eval.yaml, executed via teacher API
  - External code: module+function (not used in code-review domain yet)

Teacher API: any OpenAI-compatible /v1/chat/completions endpoint.
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "requests", "mlflow", "boto3",
        "https://github.com/opendatahub-io/agent-eval-harness/archive/refs/heads/main.tar.gz",
    ],
)
def evaluate(
    student_url: str,
    teacher_api_url: str,
    teacher_model: str,
    teacher_api_key: str,
    test_questions: list,
    eval_yaml_content: str,
    system_prompt: str = "",
    mlflow_tracking_uri: str = "",
    model_version: str = "unknown",
    s3_endpoint: str = "",
    s3_access_key: str = "",
    s3_secret_key: str = "",
    model_bucket: str = "sridhar-models",
    model_prefix: str = "code-review-1.5b-",
    judge_weights: str = "",
    mlflow_experiment: str = "AgentBuilder-Final",
    stage: str = "",
    run_label: str = "",
) -> dict:
    """Evaluate student model using harness judges loaded from eval.yaml.

    Args:
        test_questions: List of dicts with keys: question, has_bug,
            expected_behavior, expected_issues, category, case_id.
        eval_yaml_content: The full contents of eval.yaml as a string.
            Passed as a parameter so it's available inside the KFP container
            without filesystem access to the project repo.
        judge_weights: JSON string of {judge_name: weight} for composite
            score.  Defaults to correctness=0.4, quality=0.3, conciseness=0.2,
            format=0.1.
    """
    import json
    import os
    import re
    import sys
    import textwrap
    import time
    import tempfile
    from pathlib import Path
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

    api_url = teacher_api_url.rstrip("/")
    if not api_url.endswith("/v1/chat/completions"):
        api_url = api_url.rstrip("/") + "/v1/chat/completions"
    api_headers = {"Content-Type": "application/json"}
    if teacher_api_key:
        api_headers["Authorization"] = f"Bearer {teacher_api_key}"

    DEFAULT_WEIGHTS = {
        "correctness": 0.4, "conciseness": 0.2,
        "review_quality": 0.3, "format_check": 0.1,
    }
    weights = DEFAULT_WEIGHTS
    if judge_weights:
        try:
            weights = json.loads(judge_weights)
        except json.JSONDecodeError:
            print("  WARNING: invalid judge_weights JSON, using defaults")

    # Normalize test_questions
    entries = []
    for item in test_questions:
        if isinstance(item, str):
            entries.append({"question": item, "has_bug": True,
                            "expected_issues": [], "expected_behavior": "",
                            "category": "unknown", "case_id": ""})
        elif isinstance(item, dict):
            entries.append(item)
        else:
            entries.append({"question": str(item), "has_bug": True,
                            "expected_issues": [], "expected_behavior": "",
                            "category": "unknown", "case_id": ""})

    print("=" * 60)
    print("EVALUATE (agent-eval-harness native)")
    print("=" * 60)
    print(f"  Student URL:   {student_url}")
    print(f"  Teacher API:   {api_url}")
    print(f"  Teacher model: {teacher_model}")
    print(f"  Model version: {model_version}")
    print(f"  Questions:     {len(entries)}")
    print(f"  MLflow URI:    {mlflow_tracking_uri or '(not set)'}")
    print("=" * 60)

    # -- Write eval.yaml to temp dir and load via harness EvalConfig --------

    tmpdir = tempfile.mkdtemp(prefix="eval-harness-")
    eval_yaml_path = Path(tmpdir) / "eval.yaml"
    eval_yaml_path.write_text(eval_yaml_content)
    os.chdir(tmpdir)

    from agent_eval.config import EvalConfig
    config = EvalConfig.from_yaml(str(eval_yaml_path))
    print(f"  Loaded {len(config.judges)} judges from eval.yaml via EvalConfig")
    for jc in config.judges:
        jtype = "check" if jc.check else ("llm" if (jc.prompt or jc.prompt_file) else "code")
        print(f"    {jc.name}: type={jtype}")
    print(f"  Thresholds: {config.thresholds}")

    # -- Build judge functions from harness config -------------------------

    def _make_inline_check(jc):
        """Compile inline check from eval.yaml (harness pattern)."""
        source = jc.check
        wrapped = f"def _check(outputs):\n{textwrap.indent(source, '    ')}"
        code = compile(wrapped, f"<check:{jc.name}>", "exec")
        ns = {"__builtins__": __builtins__}
        exec(code, ns)
        return ns["_check"]

    def _make_llm_judge(jc):
        """Create LLM judge that calls our teacher API."""
        prompt_template = jc.prompt or ""
        if jc.prompt_file:
            pf = Path(tmpdir) / jc.prompt_file
            if pf.exists():
                prompt_template = pf.read_text()

        def scorer(outputs=None):
            outputs = outputs or {}
            review = outputs.get("artifacts_content", "")
            question = outputs.get("question", "")
            annotations = outputs.get("annotations", {})
            if not review:
                return 1, "No review output found"

            prompt = prompt_template
            prompt = prompt.replace("{{ outputs }}",
                f"## Student Review\n\n{review}\n\n## Code Diff\n\n{question}")
            prompt = prompt.replace("{{ annotations }}",
                f"## Annotations\n\n{json.dumps(annotations, indent=2)}")

            raw = _teacher_call([
                {"role": "system",
                 "content": "You are a code review quality judge. Return only JSON."},
                {"role": "user", "content": prompt},
            ], max_tokens=200, temperature=0.0)

            try:
                parsed = json.loads(raw)
                score = parsed.get("score", 3)
            except json.JSONDecodeError:
                m = re.search(r'"score"\s*:\s*(\d+)', raw)
                if m:
                    score = int(m.group(1))
                else:
                    nums = re.findall(r'\b([1-5])\b', raw)
                    score = int(nums[-1]) if nums else 3
                parsed = {"rationale": raw[:200]}

            score = max(1, min(5, score))
            return score, parsed.get("rationale", raw[:200])
        return scorer

    judges = []
    for jc in config.judges:
        if jc.check:
            scorer = _make_inline_check(jc)
            judges.append((jc.name, scorer, jc.condition))
        elif jc.prompt or jc.prompt_file:
            scorer = _make_llm_judge(jc)
            judges.append((jc.name, scorer, jc.condition))
        elif jc.module and jc.function:
            print(f"  Skipping external code judge {jc.name} (not supported in KFP)")
        else:
            print(f"  Warning: judge '{jc.name}' has no check/prompt/module")

    thresholds = config.thresholds
    print(f"\n  Loaded {len(judges)} executable judges")

    # -- Helper functions --------------------------------------------------

    def wait_for_student_ready(max_wait: int = 600, label: str = "startup") -> None:
        """Block until the student model responds to a health check."""
        print(f"  Checking student readiness ({label}) at {student_url}...")
        start = time.time()
        while time.time() - start < max_wait:
            try:
                resp = requests.get(f"{student_url}/v1/models", timeout=15)
                if resp.ok:
                    elapsed = int(time.time() - start)
                    print(f"  Student is ready ({elapsed}s)")
                    return
                print(f"  Student returned HTTP {resp.status_code}, retrying...")
            except requests.RequestException as exc:
                elapsed = int(time.time() - start)
                print(f"  Student not reachable yet ({elapsed}s): {exc}")
            time.sleep(15)
        raise RuntimeError(
            f"Student not ready after {max_wait}s at {student_url}"
        )

    student_messages_prefix = []
    if system_prompt:
        student_messages_prefix = [{"role": "system", "content": system_prompt}]
        print(f"  System prompt: {system_prompt[:80]}...")
    else:
        print("  WARNING: No system prompt set -- student may underperform")

    def query_student(question: str, max_retries: int = 10) -> str:
        messages = student_messages_prefix + [{"role": "user", "content": question}]
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{student_url}/v1/chat/completions",
                    json={"model": "/mnt/models",
                          "messages": messages,
                          "max_tokens": 512, "temperature": 0.3},
                    timeout=120)
                if resp.status_code in (400, 404, 502, 503):
                    wait = min(20 * (attempt + 1), 120)
                    print(f"  [{attempt+1}/{max_retries}] HTTP {resp.status_code}, "
                          f"retry in {wait}s")
                    if attempt >= 2:
                        wait_for_student_ready(max_wait=180, label="mid-eval recovery")
                    else:
                        time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except requests.RequestException as e:
                wait = min(20 * (attempt + 1), 120)
                print(f"  [{attempt+1}/{max_retries}] {e}, retry in {wait}s")
                if attempt >= 2:
                    wait_for_student_ready(max_wait=180, label="mid-eval recovery")
                else:
                    time.sleep(wait)
        raise RuntimeError(f"Student unreachable after {max_retries} retries")

    def _teacher_call(messages: list, max_tokens: int = 512,
                      temperature: float = 0.7) -> str:
        last_error = None
        for attempt in range(8):
            try:
                resp = requests.post(
                    api_url, headers=api_headers,
                    json={"model": teacher_model, "messages": messages,
                          "max_tokens": max_tokens, "temperature": temperature},
                    timeout=600)
                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = min(2 ** attempt * 5, 120)
                    last_error = f"HTTP {resp.status_code}"
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except (requests.ConnectionError, requests.Timeout) as e:
                wait = min(2 ** attempt * 5, 120)
                last_error = str(e)
                time.sleep(wait)
        raise RuntimeError(f"Teacher unreachable after 8 retries: {last_error}")

    # -- Run evaluation ----------------------------------------------------

    wait_for_student_ready(max_wait=600, label="pre-eval")

    print("\n" + "=" * 60)
    print("RUNNING HARNESS JUDGES")
    print("=" * 60)

    results = []
    judge_aggregates = {name: {"values": []} for name, _, _ in judges}

    for i, entry in enumerate(entries):
        q = entry["question"]
        case_id = entry.get("case_id", f"q{i+1}")
        print(f"\n--- {case_id} ({i+1}/{len(entries)}) ---")
        print(f"Q: {q[:120]}...")

        student_answer = query_student(q)
        preview = (f"{student_answer[:300]}..."
                   if len(student_answer) > 300 else student_answer)
        print(f"\nStudent:\n{preview}\n")

        annotations = {
            "has_bug": entry.get("has_bug", True),
            "expected_issues": entry.get("expected_issues", []),
            "expected_behavior": entry.get("expected_behavior", ""),
            "category": entry.get("category", "unknown"),
        }
        outputs = {
            "artifacts_content": student_answer,
            "annotations": annotations,
            "question": q,
        }

        case_judge_results = {}
        for judge_name, scorer_fn, condition in judges:
            if condition:
                try:
                    skip = not eval(condition, {"__builtins__": {}},
                                    {"annotations": annotations,
                                     "outputs": outputs})
                    if skip:
                        case_judge_results[judge_name] = {
                            "value": None,
                            "rationale": f"Skipped: condition '{condition}'",
                        }
                        print(f"  {judge_name}: SKIPPED")
                        continue
                except Exception as e:
                    print(f"  {judge_name}: condition error: {e}")

            try:
                result = scorer_fn(outputs)
                if isinstance(result, tuple) and len(result) == 2:
                    value, rationale = result
                elif hasattr(result, "value"):
                    value = result.value
                    rationale = getattr(result, "rationale", "")
                else:
                    value = result
                    rationale = ""

                case_judge_results[judge_name] = {
                    "value": value, "rationale": rationale,
                }
                judge_aggregates[judge_name]["values"].append(value)
                if isinstance(value, bool):
                    status = "PASS" if value else "FAIL"
                else:
                    status = f"{value}/5"
                print(f"  {judge_name}: {status} -- {rationale[:80]}")
            except Exception as e:
                case_judge_results[judge_name] = {
                    "value": None, "error": str(e),
                }
                print(f"  {judge_name}: ERROR -- {e}")

        results.append({
            "case_id": case_id, "question": q,
            "student_answer": student_answer,
            "judges": case_judge_results, "annotations": annotations,
        })
        time.sleep(2)

    # -- Compute aggregates ------------------------------------------------

    print("\n" + "=" * 60)
    print("JUDGE AGGREGATES")
    print("=" * 60)

    agg_summary = {}
    for name in judge_aggregates:
        values = [v for v in judge_aggregates[name]["values"] if v is not None]
        if not values:
            agg_summary[name] = {"mean": None, "pass_rate": None}
            continue
        if all(isinstance(v, bool) for v in values):
            pr = sum(values) / len(values)
            agg_summary[name] = {"pass_rate": round(pr, 4),
                                 "mean": round(pr, 4)}
            print(f"  {name}: pass_rate={pr:.1%} ({sum(values)}/{len(values)})")
        elif all(isinstance(v, (int, float)) for v in values):
            mean = sum(values) / len(values)
            agg_summary[name] = {"mean": round(mean, 4), "pass_rate": None}
            print(f"  {name}: mean={mean:.2f}/5")
        else:
            agg_summary[name] = {"mean": None, "pass_rate": None}

    # -- Threshold checks (from eval.yaml) ---------------------------------

    print("\n" + "=" * 60)
    print("THRESHOLD CHECKS (from eval.yaml)")
    print("=" * 60)

    regressions = []
    all_pass = True
    for judge_name, threshold in thresholds.items():
        current = agg_summary.get(judge_name, {})
        if isinstance(threshold, dict):
            if "min_pass_rate" in threshold:
                rate = current.get("pass_rate")
                if rate is not None and rate < threshold["min_pass_rate"]:
                    regressions.append({"judge": judge_name, "metric": "pass_rate",
                        "threshold": threshold["min_pass_rate"], "actual": rate})
                    all_pass = False
                    print(f"  FAIL: {judge_name} pass_rate={rate:.1%} < "
                          f"{threshold['min_pass_rate']:.1%}")
                elif rate is not None:
                    print(f"  PASS: {judge_name} pass_rate={rate:.1%} >= "
                          f"{threshold['min_pass_rate']:.1%}")
            if "min_mean" in threshold:
                mean = current.get("mean")
                if mean is not None and mean < threshold["min_mean"]:
                    regressions.append({"judge": judge_name, "metric": "mean",
                        "threshold": threshold["min_mean"], "actual": mean})
                    all_pass = False
                    print(f"  FAIL: {judge_name} mean={mean:.2f} < "
                          f"{threshold['min_mean']:.2f}")
                elif mean is not None:
                    print(f"  PASS: {judge_name} mean={mean:.2f} >= "
                          f"{threshold['min_mean']:.2f}")

    gate_result = "pass" if all_pass else "fail"
    print(f"\n  QUALITY GATE: {gate_result.upper()}")
    print("=" * 60)

    # -- Baseline comparison -----------------------------------------------

    import boto3
    baseline_avg = None
    try:
        s3c = boto3.client("s3", endpoint_url=s3_endpoint,
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key)
        obj = s3c.get_object(Bucket="mlflow-artifacts",
                             Key="baseline/scores.json")
        baseline = json.loads(obj["Body"].read().decode())
        baseline_avg = baseline["baseline_avg_score"]
        print(f"  Baseline loaded: avg={baseline_avg}/10")
    except Exception as e:
        print(f"  Baseline not found ({e}) -- skipping comparison")

    # -- Log to MLflow -----------------------------------------------------

    composite = 0.0
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment)
        label = run_label or model_version
        run_name = (
            f"{label}-{stage}-eval" if stage else f"pipeline-eval-{label}"
        )
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("model_version", model_version)
            if run_label:
                mlflow.set_tag("run_label", run_label)
            mlflow.set_tag("eval_type", "pipeline_benchmark")
            if stage:
                mlflow.set_tag("stage", stage)
            mlflow.set_tag("eval_framework", "agent-eval-harness-native")
            mlflow.set_tag("quality_gate", gate_result)

            for name, agg in agg_summary.items():
                if agg.get("pass_rate") is not None:
                    mlflow.log_metric(f"judge_{name}_pass_rate",
                                     agg["pass_rate"])
                if agg.get("mean") is not None:
                    mlflow.log_metric(f"judge_{name}_mean", agg["mean"])

            for i, r in enumerate(results):
                for jname, jresult in r.get("judges", {}).items():
                    v = jresult.get("value")
                    if isinstance(v, bool):
                        mlflow.log_metric(f"q{i+1}_{jname}", 1 if v else 0)
                    elif isinstance(v, (int, float)):
                        mlflow.log_metric(f"q{i+1}_{jname}", v)

            composite_parts = []
            for name, weight in weights.items():
                agg = agg_summary.get(name, {})
                if agg.get("pass_rate") is not None:
                    composite_parts.append(agg["pass_rate"] * weight)
                elif agg.get("mean") is not None:
                    composite_parts.append((agg["mean"] / 5.0) * weight)
            composite = sum(composite_parts) if composite_parts else 0.0
            mlflow.log_metric("composite_score", round(composite, 4))

            if baseline_avg is not None:
                mlflow.log_metric("baseline_avg_score",
                                  round(baseline_avg, 4))

            mlflow.log_dict({
                "model_version": model_version,
                "judge_aggregates": agg_summary,
                "thresholds": dict(thresholds),
                "weights": weights,
                "regressions": regressions,
                "gate_result": gate_result,
                "composite_score": round(composite, 4),
                "results": results,
            }, "eval_results.json")
        print(f"MLflow run logged to {mlflow_tracking_uri}")
    else:
        print("mlflow_tracking_uri not set -- skipping MLflow logging")

    # -- Write .gate-passed marker to S3 if quality gate passed (GRPO only) --
    if gate_result == "pass" and stage == "grpo" and s3_endpoint and model_version:
        try:
            s3c = boto3.client("s3", endpoint_url=s3_endpoint,
                aws_access_key_id=s3_access_key,
                aws_secret_access_key=s3_secret_key)
            stage_suffix = f"-{stage}" if stage and stage != "sft" else ""
            marker_key = f"{model_prefix}{model_version}{stage_suffix}/.gate-passed"
            s3c.put_object(
                Bucket=model_bucket,
                Key=marker_key,
                Body=json.dumps({
                    "gate_result": gate_result,
                    "composite_score": round(composite, 4),
                    "stage": stage,
                    "run_label": run_label,
                }).encode(),
            )
            print(f"  Wrote .gate-passed marker to s3://{model_bucket}/{marker_key}")
        except Exception as e:
            raise RuntimeError(f"Failed to write .gate-passed marker: {e}") from e

    return {
        "num_questions": len(results),
        "gate_result": gate_result,
        "composite_score": round(composite, 4),
        "judge_aggregates": agg_summary,
        "regressions": regressions,
        "baseline_avg_score": (round(baseline_avg, 2)
                               if baseline_avg is not None else None),
    }
