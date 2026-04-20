"""
KFP Component 4 -- Evaluate

Sends test questions to the Student model one by one, has the Teacher LLM
grade each response, logs metrics to MLflow, and prints clear per-question results.

Teacher API: any OpenAI-compatible /v1/chat/completions endpoint (Ollama, Groq, vLLM, etc.)
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["requests", "mlflow", "boto3"],
)
def evaluate(
    student_url: str,
    teacher_api_url: str,
    teacher_model: str,
    teacher_api_key: str,
    test_questions: list,
    mlflow_tracking_uri: str = "",
    model_version: str = "unknown",
    s3_endpoint: str = "",
    s3_access_key: str = "",
    s3_secret_key: str = "",
) -> dict:
    """Send test questions to Student, have Teacher grade responses, log to MLflow."""
    import json
    import os
    import sys
    import time
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

    print("=" * 60)
    print("EVALUATE MODEL STEP")
    print("=" * 60)
    print(f"  Student URL:   {student_url}")
    print(f"  Teacher API:   {api_url}")
    print(f"  Teacher model: {teacher_model}")
    print(f"  Model version: {model_version}")
    print(f"  Questions:     {len(test_questions)}")
    print(f"  MLflow URI:    {mlflow_tracking_uri or '(not set)'}")
    print("=" * 60)

    GRADING_PROMPT = (
        "You are grading an AI-generated code review comment. Rate it 1-10.Be on the rewarding side of the scale.\n"
        "Scoring guide:\n"
        "- 8-10: Correctly identifies the main issue(s) in the diff. Bonus if concise.\n"
        "- 6-7: Identifies the issue but is verbose, vague, or missing minor details.\n"
        "- 4-5: Partially correct -- mentions something relevant but misses the key issue.\n"
        "- 2-3: Mostly wrong, hallucinated issues, or says code is fine when it has real bugs.\n"
        "- 1: Completely irrelevant or nonsensical.\n"
        "If the code is genuinely clean, saying 'LGTM' or 'no issues' is correct and scores 8+.\n"
        "Correctness matters most. A correct but verbose answer is better than a wrong concise one.\n"
        'Respond with ONLY a JSON object: {"score": <number>, "reason": "<brief reason>"}'
    )

    def query_student(question: str, max_retries: int = 10) -> str:
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{student_url}/v1/chat/completions",
                    json={
                        "model": "/mnt/models",
                        "messages": [{"role": "user", "content": question}],
                        "max_tokens": 256,
                        "temperature": 0.3,
                    },
                    timeout=120,
                )
                if resp.status_code in (400, 404, 503):
                    wait = min(20 * (attempt + 1), 120)
                    print(f"  [{attempt+1}/{max_retries}] HTTP {resp.status_code}, vLLM not ready. Retry in {wait}s")
                    time.sleep(wait)
                    continue
                if not resp.ok:
                    print(f"  HTTP {resp.status_code} from student: {resp.text[:500]}")
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except (requests.ConnectionError, requests.Timeout) as e:
                wait = min(20 * (attempt + 1), 120)
                print(f"  [{attempt+1}/{max_retries}] Connection error, retry in {wait}s: {e}")
                time.sleep(wait)
        raise RuntimeError(f"Student unreachable after {max_retries} retries")

    def _teacher_call(messages: list, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Call teacher LLM API with exponential backoff on 429/5xx errors."""
        last_error = None
        for attempt in range(8):
            try:
                resp = requests.post(
                    api_url,
                    headers=api_headers,
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
        raise RuntimeError(f"Teacher unreachable after 8 retries. Last error: {last_error}")

    TEACHER_SYSTEM = (
        "You are a senior code reviewer specializing in Go, Python, and Kubernetes. "
        "Review the given code diff and identify any issues related to bugs, security, "
        "performance, reliability, style, or Kubernetes best practices. "
        "If the code is clean, say so. "
        "Be concise -- 2-4 sentences max, like a real GitHub PR comment."
    )

    def query_teacher(question: str) -> str:
        return _teacher_call([
            {"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": question},
        ], max_tokens=300, temperature=0.3)

    def teacher_grade(question: str, answer: str) -> dict:
        raw = _teacher_call([
            {"role": "system", "content": GRADING_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nStudent Response: {answer}"},
        ], max_tokens=200, temperature=0.0)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            import re
            m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
            m2 = re.search(r'\{[^{}]*"score"\s*:\s*\d+[^{}]*\}', raw)
            if m2:
                try:
                    return json.loads(m2.group(0))
                except json.JSONDecodeError:
                    pass
            return {"score": 0, "reason": f"Failed to parse: {raw}"}

    print("\n" + "=" * 60)
    results = []
    for i, q in enumerate(test_questions):
        print(f"\n--- Question {i+1}/{len(test_questions)} ---")
        print(f"Q: {q}")

        student_answer = query_student(q)
        print(f"\nStudent:\n{student_answer}\n")

        grade = teacher_grade(q, student_answer)
        print(f"Teacher Grade: {grade['score']}/10")
        print(f"Reason: {grade.get('reason', 'N/A')}")

        results.append({
            "question": q,
            "student_answer": student_answer,
            "student_score": grade["score"],
            "reason": grade.get("reason", ""),
        })
        time.sleep(3)

    student_scores = [r["student_score"] for r in results if isinstance(r.get("student_score"), (int, float))]
    student_avg = sum(student_scores) / len(student_scores) if student_scores else 0.0

    # Teacher baseline -- same questions, Teacher answers, then self-grade
    print("\n" + "=" * 60)
    print("TEACHER BASELINE")
    print("=" * 60)
    for i, r in enumerate(results):
        q = r["question"]
        teacher_answer = query_teacher(q)
        teacher_grade_result = teacher_grade(q, teacher_answer)
        r["teacher_answer"] = teacher_answer
        r["teacher_score"] = teacher_grade_result["score"]
        print(f"  Q{i+1}: {teacher_grade_result['score']}/10")
        time.sleep(3)

    teacher_scores = [r["teacher_score"] for r in results if isinstance(r.get("teacher_score"), (int, float))]
    teacher_avg = sum(teacher_scores) / len(teacher_scores) if teacher_scores else 0.0

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for i, r in enumerate(results):
        print(f"  Q{i+1}: student={r['student_score']}/10  teacher={r['teacher_score']}/10 -- {r.get('reason', '')}")
    print(f"\n  Student Avg: {student_avg:.2f}/10")
    print(f"  Teacher Avg: {teacher_avg:.2f}/10")
    print(f"  Score Gap:   {teacher_avg - student_avg:.2f}")
    print("=" * 60)

    # -- Load baseline scores from MinIO for comparison -----------------
    import boto3

    baseline_avg = None
    baseline_per_q = {}
    try:
        s3c = boto3.client(
            "s3",
            endpoint_url=s3_endpoint,
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
        )
        obj = s3c.get_object(Bucket="mlflow-artifacts", Key="baseline/scores.json")
        baseline = json.loads(obj["Body"].read().decode())
        baseline_avg = baseline["baseline_avg_score"]
        for pq in baseline.get("per_question", []):
            baseline_per_q[pq["q_index"]] = pq["student_score"]
        print(f"\n  Baseline loaded: avg={baseline_avg}/10 ({len(baseline_per_q)} questions)")
    except Exception as e:
        print(f"\n  Baseline not found ({e}) -- skipping comparison")

    # -- Comparison table -------------------------------------------------
    if baseline_avg is not None:
        improvement = student_avg - baseline_avg
        improvement_pct = (improvement / baseline_avg * 100) if baseline_avg > 0 else 0.0
        print("\n" + "=" * 60)
        print("BASELINE vs TRAINED COMPARISON")
        print("=" * 60)
        print(f"  {'Q#':<4} {'Baseline':>10} {'Trained':>10} {'Teacher':>10} {'Delta':>8}")
        print(f"  {'----':4} {'----------':10} {'----------':10} {'----------':10} {'--------':8}")
        for i, r in enumerate(results):
            b_score = baseline_per_q.get(i + 1, "-")
            delta = ""
            if isinstance(b_score, (int, float)):
                delta = f"{r['student_score'] - b_score:+.1f}"
            print(f"  Q{i+1:<3} {str(b_score):>8}/10 {r['student_score']:>8}/10 {r['teacher_score']:>8}/10 {delta:>8}")
        print(f"  {'----':4} {'----------':10} {'----------':10} {'----------':10} {'--------':8}")
        print(f"  {'AVG':<4} {baseline_avg:>8.2f}/10 {student_avg:>8.2f}/10 {teacher_avg:>8.2f}/10 {improvement:>+7.2f}")
        print(f"\n  Improvement over baseline: {improvement:+.2f} ({improvement_pct:+.1f}%)")
        print("=" * 60)
    else:
        improvement = None
        improvement_pct = None

    # Log to MLflow
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("CodeReview-Eval-Hub")
        with mlflow.start_run(run_name=f"pipeline-eval-{model_version}"):
            mlflow.set_tag("model_version", model_version)
            mlflow.set_tag("eval_type", "pipeline_benchmark")
            mlflow.log_metric("student_avg_score", round(student_avg, 4))
            mlflow.log_metric("teacher_avg_score", round(teacher_avg, 4))
            mlflow.log_metric("score_gap", round(teacher_avg - student_avg, 4))
            if baseline_avg is not None:
                mlflow.log_metric("baseline_avg_score", round(baseline_avg, 4))
                mlflow.log_metric("improvement_over_baseline", round(improvement, 4))
                mlflow.log_metric("improvement_pct", round(improvement_pct, 2))
            for i, r in enumerate(results):
                mlflow.log_metric(f"q{i+1}_student_score", r["student_score"])
                mlflow.log_metric(f"q{i+1}_teacher_score", r["teacher_score"])
            mlflow.log_dict({"results": results}, "eval_results.json")
        print(f"MLflow run logged to {mlflow_tracking_uri}")
    else:
        print("mlflow_tracking_uri not set -- skipping MLflow logging")

    return {
        "num_questions": len(results),
        "student_avg_score": round(student_avg, 2),
        "teacher_avg_score": round(teacher_avg, 2),
        "score_gap": round(teacher_avg - student_avg, 2),
        "baseline_avg_score": round(baseline_avg, 2) if baseline_avg is not None else None,
        "improvement_over_baseline": round(improvement, 2) if improvement is not None else None,
        "improvement_pct": round(improvement_pct, 1) if improvement_pct is not None else None,
        "results": results,
    }
