#!/usr/bin/env python3
"""
enrich_with_teacher.py
──────────────────────
Sends each raw code review example through the Ollama teacher model
(qwen2.5-coder:7b) to produce structured reviews.

Features:
  - Checkpoint/resume: saves progress incrementally, skips already-enriched rows
  - Configurable concurrency (default 2 for CPU-only Ollama)
  - Timeout + retry per request
  - Stats logging every 25 examples

Usage:
    python scripts/enrich_with_teacher.py \
        --input data/kubeflow_reviews.json \
        --output data/kubeflow_reviews_enriched.json \
        --ollama-url http://localhost:11434 \
        --workers 2

    python scripts/enrich_with_teacher.py \
        --input data/hf_supplement.json \
        --output data/hf_supplement_enriched.json \
        --ollama-url http://localhost:11434 \
        --workers 2
"""

import argparse
import json
import time
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

os.environ["PYTHONUNBUFFERED"] = "1"

import requests

ENRICHMENT_PROMPT = """You are a senior code reviewer specializing in Go, Python, and Kubernetes.

Given this code diff and the original reviewer's comment, create a structured review.

File: {file_path}
Language: {language}

Diff:
```
{diff}
```

Original reviewer comment:
{comment}

Respond with ONLY a structured review in this exact format (no extra text):

**Issue:** <clear one-sentence description of the problem>
**Why it matters:** <1-2 sentences on impact: reliability, security, performance, maintainability>
**Severity:** <critical|high|medium|low|nitpick>
**Category:** <bug|security|performance|reliability|style|refactor|kubernetes>
**Suggestion:** <concrete fix or improvement, include code if relevant>"""


def row_key(row: dict) -> str:
    """Stable unique key for a record to support checkpoint/resume."""
    parts = f"{row.get('repo_name','')}/{row.get('pr_number','')}/{row.get('file_path','')}/{row.get('comment_line','')}"
    return hashlib.md5(parts.encode()).hexdigest()


def build_diff_text(row: dict) -> str:
    diff = row.get("diff_context", "") or ""
    if not diff:
        before = row.get("before_code", "")
        after = row.get("after_code", "")
        if before or after:
            lines = []
            for line in before.split("\n"):
                lines.append(f"- {line}")
            for line in after.split("\n"):
                lines.append(f"+ {line}")
            diff = "\n".join(lines)
    return diff[:3000]


def call_ollama(prompt: str, model: str, url: str, timeout: int = 120) -> str | None:
    """Call Ollama generate API with retry."""
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except (requests.RequestException, KeyError) as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  [WARN] Ollama call failed after 3 attempts: {e}", file=sys.stderr)
                return None


def enrich_row(row: dict, model: str, url: str) -> dict:
    """Enrich a single row with teacher-generated structured review."""
    diff = build_diff_text(row)
    comment = row.get("reviewer_comment", "")

    if not diff or not comment:
        return row

    prompt = ENRICHMENT_PROMPT.format(
        file_path=row.get("file_path", "unknown"),
        language=row.get("language", "unknown"),
        diff=diff,
        comment=comment[:1000],
    )

    response = call_ollama(prompt, model, url)

    enriched = dict(row)
    if response and "**Issue:**" in response:
        enriched["reviewer_comment_raw"] = comment
        enriched["reviewer_comment"] = response
        enriched["enriched"] = True
    else:
        enriched["enriched"] = False

    return enriched


def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    print(f"Loading {input_path}...", flush=True)
    with open(input_path) as f:
        data = json.load(f)
    print(f"  {len(data)} records to enrich", flush=True)

    done_keys: set[str] = set()
    results: list[dict] = []
    if checkpoint_path.exists() and not args.restart:
        with open(checkpoint_path) as f:
            results = json.load(f)
        done_keys = {row_key(r) for r in results}
        print(f"  Resuming: {len(done_keys)} already enriched, {len(data) - len(done_keys)} remaining", flush=True)

    remaining = [r for r in data if row_key(r) not in done_keys]
    if not remaining:
        print("All records already enriched!", flush=True)
        save_final(results, output_path, checkpoint_path)
        return

    print(f"\nEnriching {len(remaining)} examples with {args.model} ({args.workers} workers)...", flush=True)
    print(f"  Ollama: {args.ollama_url}", flush=True)
    print(f"  Estimated time: ~{len(remaining) * 30 / args.workers / 60:.0f} minutes", flush=True)
    sys.stdout.flush()

    lock = Lock()
    enriched_count = 0
    failed_count = 0
    start_time = time.time()

    def process(row):
        return enrich_row(row, args.model, args.ollama_url)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, r): i for i, r in enumerate(remaining)}

        for future in as_completed(futures):
            result = future.result()
            with lock:
                results.append(result)
                if result.get("enriched"):
                    enriched_count += 1
                else:
                    failed_count += 1

                total_done = enriched_count + failed_count
                if total_done % 25 == 0 or total_done == len(remaining):
                    elapsed = time.time() - start_time
                    rate = total_done / elapsed if elapsed > 0 else 0
                    eta = (len(remaining) - total_done) / rate / 60 if rate > 0 else 0
                    print(
                        f"  [{total_done}/{len(remaining)}] "
                        f"enriched={enriched_count} failed={failed_count} "
                        f"rate={rate:.1f}/s ETA={eta:.0f}m",
                        flush=True,
                    )

                if total_done % 100 == 0:
                    with open(checkpoint_path, "w") as f:
                        json.dump(results, f)

    save_final(results, output_path, checkpoint_path)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed/60:.1f} minutes", flush=True)
    print(f"  Enriched: {enriched_count}/{len(remaining)}", flush=True)
    print(f"  Failed (kept raw): {failed_count}", flush=True)


def save_final(results: list[dict], output_path: Path, checkpoint_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} records to {output_path}", flush=True)

    if checkpoint_path.exists():
        with open(checkpoint_path, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich code review data with teacher LLM")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file")
    parser.add_argument("--output", "-o", required=True, help="Output enriched JSON file")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen2.5-coder:7b-instruct-q4_K_M")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--restart", action="store_true", help="Ignore checkpoint, start fresh")
    main(parser.parse_args())
