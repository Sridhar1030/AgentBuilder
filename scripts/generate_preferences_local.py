#!/usr/bin/env python3
"""
Generate DPO preference pairs locally using Ollama as teacher & grader.

Usage:
    python scripts/generate_preferences_local.py

Requires:
    - Ollama running locally with llama3.1:8b-instruct-q4_K_M
    - Student model port-forwarded to localhost:8080
    - MinIO port-forwarded or route available

Output:
    Uploads preference JSONL to MinIO at s3://mlflow-artifacts/preferences/pref-local-<id>.jsonl
"""

import json
import os
import sys
import time
import uuid
import concurrent.futures

import boto3
import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
STUDENT_URL = os.environ.get("STUDENT_URL", "http://localhost:8080")

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://minio-api-sridharproject.apps.sridhartest-pool-7f6n4.aws.rh-ods.com")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "minioadmin123")
PREF_BUCKET = "mlflow-artifacts"

QUESTION_BANK = os.path.join(os.path.dirname(__file__), "..", "data", "kubeflow_questions.json")
MIN_SCORE_GAP = 2


def ollama_chat(messages: list, max_tokens: int = 512, temperature: float = 0.7) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def query_teacher(question: str) -> str:
    return ollama_chat(
        [
            {"role": "system", "content": "You are a helpful, accurate, and concise Kubeflow expert."},
            {"role": "user", "content": question},
        ],
        max_tokens=512,
        temperature=0.7,
    )


def query_student(question: str) -> str:
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{STUDENT_URL}/v1/chat/completions",
                json={
                    "model": "/mnt/models",
                    "messages": [{"role": "user", "content": question}],
                    "max_tokens": 256,
                    "temperature": 0.3,
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  Student query failed (attempt {attempt+1}): {e}")
            time.sleep(5)
    return ""


GRADING_PROMPT = (
    "You are an expert Kubeflow grader. Rate the following AI response to the given question "
    "on a scale of 1 to 10, where 1 is completely wrong/unhelpful and 10 is perfect.\n"
    'Respond with ONLY a JSON object: {"score": <number>, "reason": "<brief reason>"}'
)


def grade(question: str, answer: str) -> int:
    raw = ollama_chat(
        [
            {"role": "system", "content": GRADING_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nResponse: {answer}"},
        ],
        max_tokens=200,
        temperature=0.0,
    )
    try:
        score_str = raw.strip()
        if score_str.startswith("```"):
            score_str = score_str.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return int(json.loads(score_str)["score"])
    except (json.JSONDecodeError, KeyError, ValueError):
        for token in raw.split():
            try:
                val = int(token.strip('",.:'))
                if 1 <= val <= 10:
                    return val
            except ValueError:
                continue
        return 5


def load_questions() -> list[str]:
    with open(QUESTION_BANK) as f:
        data = json.load(f)
    questions = data.get("all_questions", [])
    if not questions and "topics" in data:
        for topic_qs in data["topics"].values():
            questions.extend(topic_qs)
    return questions


def process_question(i: int, total: int, question: str) -> dict | None:
    tag = f"[{i+1}/{total}]"
    print(f"{tag} {question[:80]}...")

    student_answer = query_student(question)
    if not student_answer:
        print(f"  {tag} Student returned empty, skipping")
        return None

    teacher_answer = query_teacher(question)
    if not teacher_answer:
        print(f"  {tag} Teacher returned empty, skipping")
        return None

    student_score = grade(question, student_answer)
    teacher_score = grade(question, teacher_answer)
    gap = teacher_score - student_score

    status = "PAIR" if gap >= MIN_SCORE_GAP else "skip"
    print(f"  {tag} teacher={teacher_score} student={student_score} gap={gap} -> {status}")

    if gap >= MIN_SCORE_GAP:
        return {
            "prompt": question,
            "chosen": teacher_answer,
            "rejected": student_answer,
            "source": "local_ollama",
            "teacher_score": teacher_score,
            "student_score": student_score,
            "score_gap": gap,
        }
    return None


def main():
    questions = load_questions()
    print(f"Loaded {len(questions)} questions from {QUESTION_BANK}")
    print(f"Ollama model: {OLLAMA_MODEL}")
    print(f"Student URL: {STUDENT_URL}")
    print(f"Min score gap: {MIN_SCORE_GAP}")
    print("=" * 70)

    preference_pairs = []
    start_time = time.time()

    for i, q in enumerate(questions):
        result = process_question(i, len(questions), q)
        if result:
            preference_pairs.append(result)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60
            print(f"\n--- Progress: {i+1}/{len(questions)}, {len(preference_pairs)} pairs, "
                  f"{rate:.1f} q/min, elapsed {elapsed/60:.1f}m ---\n")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Done! {len(preference_pairs)} preference pairs from {len(questions)} questions "
          f"({elapsed/60:.1f} min)")

    if not preference_pairs:
        print("No preference pairs generated. Exiting.")
        sys.exit(1)

    run_id = uuid.uuid4().hex[:12]
    pref_key = f"preferences/pref-local-{run_id}.jsonl"
    body = "\n".join(json.dumps(p) for p in preference_pairs)

    local_path = f"/tmp/pref-local-{run_id}.jsonl"
    with open(local_path, "w") as f:
        f.write(body)
    print(f"Saved locally: {local_path}")

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            verify=False,
        )
        s3.put_object(Bucket=PREF_BUCKET, Key=pref_key, Body=body.encode())
        pref_s3_path = f"s3://{PREF_BUCKET}/{pref_key}"
        print(f"Uploaded to MinIO: {pref_s3_path}")
    except Exception as e:
        print(f"MinIO upload failed: {e}")
        print(f"Use the local file at {local_path} to upload manually.")
        sys.exit(1)

    score_gaps = [p["score_gap"] for p in preference_pairs]
    avg_gap = sum(score_gaps) / len(score_gaps)
    print(f"\nStats:")
    print(f"  Total pairs: {len(preference_pairs)}")
    print(f"  Avg score gap: {avg_gap:.1f}")
    print(f"  Min gap: {min(score_gaps)}, Max gap: {max(score_gaps)}")
    print(f"  S3 path: s3://{PREF_BUCKET}/{pref_key}")


if __name__ == "__main__":
    main()
