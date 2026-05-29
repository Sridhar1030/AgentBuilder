"""
KFP Component -- Eval Dataset (agent-eval-harness /eval-dataset)

Generates and expands training data based on the analysis from eval-analyze.
Uses the teacher model to create new training examples focused on weak
categories identified by the analysis step.

From Antonin's doc:
  "eval-dataset: Generate and expand test cases using production traces
   and SDG Hub"

In practice this component:
  1. Reads the analysis dict (which categories are weak)
  2. Generates new code diffs + teacher reviews for those categories
  3. Uploads them to MinIO as supplemental JSONL training data
  4. Returns the S3 path of the new data
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["requests", "boto3"],
)
def eval_dataset(
    analysis: dict,
    teacher_api_url: str,
    teacher_model: str,
    teacher_api_key: str,
    teacher_system_prompt: str,
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    data_bucket: str,
    output_prefix: str,
    model_version: str = "unknown",
    max_new_examples: int = 10,
) -> str:
    """Generate supplemental training data for weak categories.

    Returns the S3 path of the generated JSONL file (empty string if
    no generation was needed).
    """
    import json
    import sys
    import time
    import requests
    import boto3

    sys.stdout.reconfigure(line_buffering=True)

    print("=" * 60)
    print("EVAL-DATASET (agent-eval-harness)")
    print("=" * 60)

    weaknesses = analysis.get("weaknesses", [])
    weak_categories = analysis.get("weak_categories", [])
    status = analysis.get("status", "no_data")

    if status == "no_data" or (not weaknesses and not weak_categories):
        print("  No weaknesses found -- skipping data generation")
        print("=" * 60)
        return ""

    print(f"  Weaknesses: {len(weaknesses)}")
    print(f"  Weak categories: {weak_categories}")

    api_url = teacher_api_url.rstrip("/")
    if not api_url.endswith("/v1/chat/completions"):
        api_url = api_url.rstrip("/") + "/v1/chat/completions"
    api_headers = {"Content-Type": "application/json"}
    if teacher_api_key:
        api_headers["Authorization"] = f"Bearer {teacher_api_key}"

    CATEGORY_TEMPLATES = {
        "bug": "Write a short Go/Python code diff (under 30 lines) that introduces a subtle bug: {detail}. The diff should use proper unified diff format with --- and +++ headers.",
        "security": "Write a short Go/Python code diff (under 30 lines) that introduces a security vulnerability: {detail}. Use unified diff format.",
        "performance": "Write a short code diff (under 30 lines) that introduces a performance issue: {detail}. Use unified diff format.",
        "kubernetes": "Write a short Kubernetes YAML or Go code diff (under 30 lines) that has a configuration issue: {detail}. Use unified diff format.",
        "reliability": "Write a short code diff (under 30 lines) that introduces a reliability issue like missing error handling or resource leaks: {detail}. Use unified diff format.",
        "style": "Write a short code diff (under 30 lines) with a style/readability issue: {detail}. Use unified diff format.",
        "clean": "Write a short, well-written code diff (under 30 lines) that is correct and clean with no issues. Use unified diff format.",
    }

    BUG_DETAILS = [
        "nil pointer dereference", "unclosed file handle",
        "integer overflow", "race condition",
        "missing error check", "off-by-one error",
        "SQL injection vulnerability", "hardcoded credentials",
        "missing TLS configuration", "unbounded loop",
    ]

    # Determine which categories need more data
    target_categories = []
    for w in weaknesses:
        judge = w.get("judge", "")
        if "correctness" in judge:
            target_categories.extend(["bug", "security", "reliability"])
        elif "quality" in judge:
            target_categories.extend(["bug", "performance"])
        elif "conciseness" in judge:
            target_categories.append("clean")

    if not target_categories:
        target_categories = ["bug", "security", "reliability"]

    # Deduplicate while preserving order
    seen = set()
    unique_categories = []
    for c in target_categories:
        if c not in seen:
            seen.add(c)
            unique_categories.append(c)
    target_categories = unique_categories

    print(f"  Target categories for new data: {target_categories}")

    def teacher_call(messages, max_tokens=512, temperature=0.7):
        for attempt in range(5):
            try:
                resp = requests.post(
                    api_url, headers=api_headers,
                    json={"model": teacher_model, "messages": messages,
                          "max_tokens": max_tokens, "temperature": temperature},
                    timeout=300)
                if resp.status_code >= 500 or resp.status_code == 429:
                    time.sleep(2 ** attempt * 5)
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(2 ** attempt * 5)
        return None

    generated = []
    examples_per_category = max(1, max_new_examples // len(target_categories))

    for cat in target_categories:
        template = CATEGORY_TEMPLATES.get(cat, CATEGORY_TEMPLATES["bug"])
        for i in range(examples_per_category):
            if len(generated) >= max_new_examples:
                break

            detail = BUG_DETAILS[i % len(BUG_DETAILS)]
            diff_prompt = template.format(detail=detail)

            print(f"  Generating {cat} example {i+1}/{examples_per_category}...")

            diff = teacher_call([
                {"role": "system", "content": "You are a code example generator. Output ONLY the diff, no explanation."},
                {"role": "user", "content": diff_prompt},
            ], max_tokens=400, temperature=0.8)

            if not diff:
                print(f"    Failed to generate diff, skipping")
                continue

            review = teacher_call([
                {"role": "system", "content": teacher_system_prompt},
                {"role": "user", "content": f"Review this code diff:\n\n```diff\n{diff}\n```"},
            ], max_tokens=200, temperature=0.3)

            if not review:
                print(f"    Failed to generate review, skipping")
                continue

            chatml = (
                f"<|im_start|>system\n{teacher_system_prompt}<|im_end|>\n"
                f"<|im_start|>user\nReview this code diff:\n\n```diff\n{diff}\n```<|im_end|>\n"
                f"<|im_start|>assistant\n{review}<|im_end|>"
            )

            generated.append({
                "text": chatml,
                "category": cat,
                "source": "eval-dataset-sdg",
                "model_version": model_version,
            })
            print(f"    Generated ({len(review.split())} words)")
            time.sleep(1)

    if not generated:
        print("  No examples generated")
        print("=" * 60)
        return ""

    output_key = f"{output_prefix}eval-dataset-{model_version}.jsonl"
    body = "\n".join(json.dumps(r) for r in generated)

    s3 = boto3.client(
        "s3", endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )
    s3.put_object(Bucket=data_bucket, Key=output_key, Body=body.encode())

    output_path = f"s3://{data_bucket}/{output_key}"
    print(f"\n  Generated {len(generated)} examples")
    print(f"  Categories: {dict((c, sum(1 for g in generated if g['category'] == c)) for c in target_categories)}")
    print(f"  Uploaded to: {output_path}")
    print("=" * 60)

    return output_path
