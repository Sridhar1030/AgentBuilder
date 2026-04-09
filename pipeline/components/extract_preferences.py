"""
KFP Component -- Extract Preference Pairs for DPO

Two data sources:
  Source A: eval_results.json from the previous SFT run (MLflow artifact).
            Entries where student_score < teacher_score become preference pairs.
  Source B: Query the Kubeflow question bank against both Student and Teacher,
            grade both, and produce preference pairs where teacher wins.

Teacher API: any OpenAI-compatible /v1/chat/completions endpoint (Ollama, Groq, vLLM, etc.)

Output: preference JSONL to MinIO at preferences/pref-{version}.jsonl
Format: {"prompt": "...", "chosen": "...", "rejected": "..."}
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["requests", "mlflow", "boto3"],
)
def extract_preferences(
    student_url: str,
    teacher_api_url: str,
    teacher_model: str,
    teacher_api_key: str,
    question_bank_s3_path: str,
    mlflow_tracking_uri: str,
    sft_run_name_prefix: str,
    model_version: str,
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    pref_output_bucket: str = "mlflow-artifacts",
    min_score_gap: int = 1,
    max_supplement_questions: int = 50,
) -> str:
    """Build DPO preference pairs from eval results + supplementary question bank queries."""
    import json
    import os
    import sys
    import time
    import uuid
    from concurrent.futures import ThreadPoolExecutor

    import boto3
    import mlflow
    import requests

    sys.stdout.reconfigure(line_buffering=True)

    print("=" * 60)
    print("EXTRACT PREFERENCE PAIRS STEP")
    print("=" * 60)
    print(f"  Student URL:   {student_url}")
    print(f"  Teacher API:   {teacher_api_url}")
    print(f"  Teacher model: {teacher_model}")
    print(f"  Question bank: {question_bank_s3_path}")
    print(f"  Version:       {model_version}")
    print(f"  Min score gap: {min_score_gap}")
    print(f"  Max supplement: {max_supplement_questions}")
    print("=" * 60)

    if mlflow_tracking_uri.startswith("https://"):
        os.environ.setdefault("MLFLOW_TRACKING_INSECURE_TLS", "true")
    if s3_endpoint:
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", s3_endpoint)
        os.environ.setdefault("AWS_ACCESS_KEY_ID", s3_access_key)
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", s3_secret_key)

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )

    api_url = teacher_api_url.rstrip("/")
    if not api_url.endswith("/v1/chat/completions"):
        api_url = api_url.rstrip("/") + "/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    if teacher_api_key:
        headers["Authorization"] = f"Bearer {teacher_api_key}"

    def teacher_llm_call(messages: list, max_tokens: int = 512, temperature: float = 0.7) -> str:
        last_error = None
        for attempt in range(8):
            try:
                resp = requests.post(
                    api_url,
                    headers=headers,
                    json={
                        "model": teacher_model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    timeout=600,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = min(2 ** attempt * 5, 120)
                    print(f"  [{resp.status_code}] Server error, waiting {wait}s (attempt {attempt+1}/8)")
                    last_error = f"HTTP {resp.status_code}"
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except (requests.ConnectionError, requests.Timeout) as e:
                wait = min(2 ** attempt * 5, 120)
                print(f"  [Connection error] {e}, waiting {wait}s (attempt {attempt+1}/8)")
                last_error = str(e)
                time.sleep(wait)
        print(f"  [Teacher] All 8 retries exhausted. Last error: {last_error}")
        return ""

    def wait_for_student_ready(max_wait: int = 300) -> bool:
        """Block until the student model responds to a health check."""
        print(f"  Checking student readiness at {student_url}...")
        start = time.time()
        while time.time() - start < max_wait:
            try:
                r = requests.get(f"{student_url}/v1/models", timeout=10)
                if r.ok:
                    print(f"  Student is ready ({int(time.time()-start)}s)")
                    return True
                print(f"  Student returned {r.status_code}, retrying...")
            except (requests.ConnectionError, requests.Timeout):
                print(f"  Student not reachable yet ({int(time.time()-start)}s)...")
            time.sleep(15)
        print(f"  WARNING: Student not ready after {max_wait}s")
        return False

    def query_student(question: str) -> str:
        for attempt in range(8):
            try:
                resp = requests.post(
                    f"{student_url}/v1/chat/completions",
                    json={
                        "model": "/mnt/models",
                        "messages": [{"role": "user", "content": question}],
                        "max_tokens": 256,
                        "temperature": 0.3,
                    },
                    timeout=180,
                )
                if resp.status_code in (400, 404, 503):
                    wait = min(20 * (attempt + 1), 120)
                    print(f"    [Student {resp.status_code}] Retry {attempt+1}/8 in {wait}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except (requests.ConnectionError, requests.Timeout) as e:
                wait = min(20 * (attempt + 1), 120)
                print(f"    [Student conn error] {e}, retry {attempt+1}/8 in {wait}s")
                time.sleep(wait)
        print("    [Student] All retries exhausted, returning empty")
        return ""

    def query_teacher(question: str) -> str:
        return teacher_llm_call([
            {"role": "system", "content": "You are a helpful and concise assistant."},
            {"role": "user", "content": question},
        ], max_tokens=512, temperature=0.7)

    GRADING_PROMPT = (
        "You are an expert code review grader. Rate the following AI-generated code review "
        "on a scale of 1 to 10 based on these criteria:\n"
        "- Issue identification: Did it find the real problem (not a hallucinated one)?\n"
        "- Technical accuracy: Is the explanation correct?\n"
        "- Actionability: Is the suggestion concrete and implementable?\n"
        "- Severity accuracy: Is the severity rating appropriate?\n"
        "- False positive avoidance: Did it avoid flagging non-issues?\n"
        "A score of 1 means completely wrong or hallucinated issues. "
        "A score of 10 means a perfect, reviewer-quality response.\n"
        'Respond with ONLY a JSON object: {"score": <number>, "reason": "<brief reason>"}'
    )

    def grade(question: str, answer: str) -> int:
        raw = teacher_llm_call([
            {"role": "system", "content": GRADING_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nResponse: {answer}"},
        ], max_tokens=200, temperature=0.0)
        try:
            return int(json.loads(raw)["score"])
        except (json.JSONDecodeError, KeyError, ValueError):
            return 5

    preference_pairs = []

    # =====================================================================
    # Source A: Previous eval_results.json from MLflow
    # =====================================================================
    print("=" * 60)
    print("SOURCE A: Previous eval results from MLflow")
    print("=" * 60)
    try:
        print("  Connecting to MLflow...")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        experiment = mlflow.get_experiment_by_name("CodeReview-Eval-Hub")
        if experiment:
            print(f"  Found experiment: {experiment.experiment_id}")
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.eval_type = 'pipeline_benchmark'",
                order_by=["start_time DESC"],
                max_results=1,
            )
            if not runs.empty:
                run_id = runs.iloc[0]["run_id"]
                print(f"  Found eval run: {run_id}")
                print(f"  Downloading eval_results.json artifact...")
                artifact_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path="eval_results.json"
                )
                print(f"  Downloaded artifact to: {artifact_path}")
                with open(artifact_path) as f:
                    eval_data = json.load(f)

                results = eval_data.get("results", eval_data) if isinstance(eval_data, dict) else eval_data
                if isinstance(results, dict):
                    results = results.get("results", [])
                print(f"  Loaded {len(results)} results from eval_results.json")

                for i, r in enumerate(results):
                    student_score = r.get("student_score", 10)
                    teacher_score = r.get("teacher_score", 0)
                    gap = teacher_score - student_score

                    if gap >= min_score_gap and r.get("teacher_answer") and r.get("student_answer"):
                        preference_pairs.append({
                            "prompt": r["question"],
                            "chosen": r["teacher_answer"],
                            "rejected": r["student_answer"],
                            "source": "eval_results",
                            "score_gap": gap,
                        })
                    if (i + 1) % 25 == 0:
                        print(f"    Scanned {i+1}/{len(results)} results, {len(preference_pairs)} pairs so far")
                print(f"  Source A: {len(preference_pairs)} pairs from eval results (gap >= {min_score_gap})")
            else:
                print("  No previous eval runs found in MLflow")
        else:
            print("  Experiment 'CodeReview-Eval-Hub' not found")
    except Exception as exc:
        import traceback
        print(f"  Source A failed: {exc}")
        traceback.print_exc()

    # =====================================================================
    # Source B: Supplement from question bank
    # =====================================================================
    print("\n" + "=" * 60)
    print("SOURCE B: Question bank supplement")
    print("=" * 60)
    wait_for_student_ready(max_wait=600)

    try:
        print(f"  Downloading question bank from {question_bank_s3_path}...")
        parts = question_bank_s3_path.replace("s3://", "").split("/", 1)
        qb_bucket, qb_key = parts[0], parts[1]
        obj = s3.get_object(Bucket=qb_bucket, Key=qb_key)
        qb_data = json.loads(obj["Body"].read().decode())
        print(f"  Question bank downloaded, keys: {list(qb_data.keys())}")

        all_questions = qb_data.get("all_questions", [])
        if not all_questions and "topics" in qb_data:
            for topic_qs in qb_data["topics"].values():
                all_questions.extend(topic_qs)
        print(f"  Total questions in bank: {len(all_questions)}")

        existing_prompts = {p["prompt"].lower().strip() for p in preference_pairs}
        supplement_qs = [q for q in all_questions if q.lower().strip() not in existing_prompts]
        supplement_qs = supplement_qs[:max_supplement_questions]
        print(f"  After dedup, will query {len(supplement_qs)} supplement questions")

        print(f"  Querying {len(supplement_qs)} questions against Student + Teacher (parallel)...")
        source_b_count = 0
        executor = ThreadPoolExecutor(max_workers=2)
        for i, q in enumerate(supplement_qs):
            q_preview = q[:80].replace('\n', ' ')
            print(f"    [{i+1}/{len(supplement_qs)}] Querying student + teacher in parallel... ({q_preview}...)")

            fut_student = executor.submit(query_student, q)
            fut_teacher = executor.submit(query_teacher, q)
            student_answer = fut_student.result()
            teacher_answer = fut_teacher.result()

            if not student_answer:
                print(f"    [{i+1}] Student returned empty, skipping")
                continue
            if not teacher_answer:
                print(f"    [{i+1}] Teacher returned empty, skipping")
                continue
            print(f"    [{i+1}] Student={len(student_answer)}ch Teacher={len(teacher_answer)}ch, grading both in parallel...")

            fut_gs = executor.submit(grade, q, student_answer)
            fut_gt = executor.submit(grade, q, teacher_answer)
            student_score = fut_gs.result()
            teacher_score = fut_gt.result()

            gap = teacher_score - student_score
            print(f"    [{i+1}] Student={student_score} Teacher={teacher_score} Gap={gap} {'-> PAIR' if gap >= min_score_gap else '-> skip'}")

            if gap >= min_score_gap:
                preference_pairs.append({
                    "prompt": q,
                    "chosen": teacher_answer,
                    "rejected": student_answer,
                    "source": "question_bank",
                    "score_gap": gap,
                })
                source_b_count += 1

            if (i + 1) % 10 == 0:
                print(f"  -- Progress: [{i+1}/{len(supplement_qs)}] {source_b_count} pairs collected --")
        executor.shutdown(wait=False)

        print(f"  Source B: {source_b_count} pairs from question bank")
    except Exception as exc:
        import traceback
        print(f"  Source B failed: {exc}")
        traceback.print_exc()

    # =====================================================================
    # Write preferences to MinIO
    # =====================================================================
    print(f"\n{'=' * 60}")
    print(f"PREFERENCE EXTRACTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total preference pairs: {len(preference_pairs)}")
    by_source = {}
    for p in preference_pairs:
        src = p.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
    for src, cnt in by_source.items():
        print(f"    {src}: {cnt} pairs")

    run_id = uuid.uuid4().hex[:12]
    pref_key = f"preferences/pref-{model_version}-{run_id}.jsonl"
    body = "\n".join(json.dumps(p) for p in preference_pairs)

    print(f"  Uploading to s3://{pref_output_bucket}/{pref_key} ({len(body)} bytes)...")
    s3.put_object(Bucket=pref_output_bucket, Key=pref_key, Body=body.encode())
    pref_s3_path = f"s3://{pref_output_bucket}/{pref_key}"
    print(f"  Uploaded -> {pref_s3_path}")

    return pref_s3_path
