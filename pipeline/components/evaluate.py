"""
KFP Component 4 -- Evaluate

Sends test questions to the Student model one by one, has the 70B Teacher
grade each response, and prints clear per-question results.
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["requests"],
)
def evaluate(
    student_url: str,
    groq_api_key: str,
    groq_model: str,
    test_questions: list,
) -> dict:
    """Send test questions to Student, have Teacher grade responses."""
    import json
    import time
    import requests

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

    def teacher_grade(question: str, answer: str) -> dict:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": groq_model,
                "messages": [
                    {"role": "system", "content": GRADING_PROMPT},
                    {
                        "role": "user",
                        "content": f"Question: {question}\n\nStudent Response: {answer}",
                    },
                ],
                "max_tokens": 200,
                "temperature": 0.0,
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
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
            "score": grade["score"],
            "reason": grade.get("reason", ""),
        })

    scores = [r["score"] for r in results if isinstance(r.get("score"), (int, float))]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for i, r in enumerate(results):
        print(f"  Q{i+1}: {r['score']}/10 -- {r.get('reason', '')}")
    print(f"\n  Average Score: {avg_score:.2f}/10")
    print("=" * 60)

    return {
        "num_questions": len(results),
        "avg_score": round(avg_score, 2),
        "results": results,
    }
