"""
KFP Component 4 -- Evaluate

Sends test questions to the Student model one by one, has the 70B Teacher
grade each response, logs metrics to MLflow, and prints clear per-question results.
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["requests", "mlflow", "boto3"],
)
def evaluate(
    student_url: str,
    groq_api_key: str,
    groq_model: str,
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
    import time
    import requests
    import mlflow

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

    if not groq_api_key:
        raise ValueError(
            "groq_api_key is empty. Pass your Groq API key as a pipeline parameter."
        )

    print(f"Student URL: {student_url}")
    print(f"Groq model: {groq_model}")
    print(f"Test questions: {len(test_questions)}")
    print(f"Groq API key: set ({len(groq_api_key)} chars)")

    GRADING_PROMPT = (
        "You are an expert grader. Rate the following AI response to the given question "
        "on a scale of 1 to 10, where 1 is completely wrong and 10 is perfect. "
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
                    timeout=90,
                )
                if resp.status_code in (404, 503):
                    wait = min(15 * (attempt + 1), 60)
                    print(f"  [{attempt+1}/{max_retries}] HTTP {resp.status_code}, vLLM not ready. Retry in {wait}s")
                    time.sleep(wait)
                    continue
                if not resp.ok:
                    print(f"  HTTP {resp.status_code} from student: {resp.text[:500]}")
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except (requests.ConnectionError, requests.Timeout) as e:
                wait = min(15 * (attempt + 1), 60)
                print(f"  [{attempt+1}/{max_retries}] Connection error, retry in {wait}s: {e}")
                time.sleep(wait)
        raise RuntimeError(f"Student unreachable after {max_retries} retries")

    def _groq_call(messages: list, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Call Groq API with exponential backoff on 429 rate limits."""
        for attempt in range(8):
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": groq_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=60,
            )
            if resp.status_code == 429:
                wait = min(2 ** attempt * 5, 120)
                print(f"  [Groq 429] Rate limited, waiting {wait}s (attempt {attempt+1}/8)")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        resp.raise_for_status()
        return ""

    def query_teacher(question: str) -> str:
        return _groq_call([
            {"role": "system", "content": "You are a helpful and concise assistant."},
            {"role": "user", "content": question},
        ], max_tokens=512, temperature=0.7)

    def teacher_grade(question: str, answer: str) -> dict:
        raw = _groq_call([
            {"role": "system", "content": GRADING_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nStudent Response: {answer}"},
        ], max_tokens=200, temperature=0.0)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
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

    # Teacher baseline — same questions, Teacher answers, then self-grade
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

    # Log to MLflow
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("Distillation-Eval-Hub")
        with mlflow.start_run(run_name=f"pipeline-eval-{model_version}"):
            mlflow.set_tag("model_version", model_version)
            mlflow.set_tag("eval_type", "pipeline_benchmark")
            mlflow.log_metric("student_avg_score", round(student_avg, 4))
            mlflow.log_metric("teacher_avg_score", round(teacher_avg, 4))
            mlflow.log_metric("score_gap", round(teacher_avg - student_avg, 4))
            for i, r in enumerate(results):
                mlflow.log_metric(f"q{i+1}_student_score", r["student_score"])
                mlflow.log_metric(f"q{i+1}_teacher_score", r["teacher_score"])
            mlflow.log_dict({"results": results}, "eval_results.json")
        print(f"MLflow run logged to {mlflow_tracking_uri}")
    else:
        print("mlflow_tracking_uri not set — skipping MLflow logging")

    return {
        "num_questions": len(results),
        "student_avg_score": round(student_avg, 2),
        "teacher_avg_score": round(teacher_avg, 2),
        "score_gap": round(teacher_avg - student_avg, 2),
        "results": results,
    }
