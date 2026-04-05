#!/usr/bin/env python3
"""
prepare_hf_supplement.py
────────────────────────
Downloads ronantakizawa/github-codereview from HuggingFace, filters for
Go/Python/YAML, applies quality thresholds, excludes repos already mined
by mine_github_reviews.py, and produces a stratified sample.

Output is in the same 20-column schema as the HF dataset (and as our
GitHub-mined data) so both sources can be concatenated directly.

Requirements:
    pip install datasets

Usage:
    python scripts/prepare_hf_supplement.py \
        --mined data/kubeflow_reviews.json \
        --output data/hf_supplement.json \
        --max-examples 8000
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_mined_repos(mined_path: str) -> set[str]:
    """Load repo names already mined so we can exclude them."""
    if not mined_path or not Path(mined_path).exists():
        return set()
    with open(mined_path) as f:
        data = json.load(f)
    repos = {r["repo_name"].lower() for r in data if r.get("repo_name")}
    print(f"Excluding {len(repos)} already-mined repos: {repos}")
    return repos


def main(args):
    from datasets import load_dataset

    print("Loading ronantakizawa/github-codereview from HuggingFace...")
    ds = load_dataset("ronantakizawa/github-codereview", split="train")
    print(f"Total rows in dataset: {len(ds)}")

    mined_repos = load_mined_repos(args.mined)

    ALLOWED_LANGUAGES = {"Go", "Python", "YAML"}

    # Priority categories get higher sampling weight
    PRIORITY_TYPES = {"security", "performance", "bug"}

    print("\nFiltering...")
    candidates = []
    stats = Counter()

    for row in ds:
        stats["total"] += 1

        lang = row.get("language", "")
        if lang not in ALLOWED_LANGUAGES:
            stats["filtered_language"] += 1
            continue

        if row.get("is_negative", False):
            stats["filtered_negative"] += 1
            continue

        qs = row.get("quality_score", 0) or 0
        if qs < 0.4:
            stats["filtered_quality"] += 1
            continue

        cl = row.get("comment_length", 0) or 0
        if cl < 50:
            stats["filtered_short"] += 1
            continue

        repo = (row.get("repo_name") or "").lower()
        if repo in mined_repos:
            stats["filtered_duplicate_repo"] += 1
            continue

        stats["passed"] += 1
        candidates.append(row)

    print(f"\nFilter stats:")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")

    print(f"\nCandidates after filtering: {len(candidates)}")

    # Stratified sampling by comment_type
    by_type = defaultdict(list)
    for row in candidates:
        ct = row.get("comment_type", "other") or "other"
        by_type[ct].append(row)

    print(f"\nCandidates by comment_type:")
    for ct, rows in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"  {ct}: {len(rows)}")

    max_n = args.max_examples
    sampled = []

    # Give priority types 2x their proportional share
    type_counts = {ct: len(rows) for ct, rows in by_type.items()}
    total_candidates = sum(type_counts.values())

    type_budgets = {}
    for ct, count in type_counts.items():
        proportion = count / total_candidates
        if ct in PRIORITY_TYPES:
            proportion *= 2.0
        type_budgets[ct] = proportion

    budget_sum = sum(type_budgets.values())
    for ct in type_budgets:
        type_budgets[ct] = int((type_budgets[ct] / budget_sum) * max_n)

    # Ensure we don't exceed available candidates per type
    for ct in type_budgets:
        type_budgets[ct] = min(type_budgets[ct], len(by_type[ct]))

    print(f"\nSampling budgets (target {max_n}):")
    for ct, budget in sorted(type_budgets.items(), key=lambda x: -x[1]):
        print(f"  {ct}: {budget} (available: {len(by_type[ct])})")

    random.seed(42)
    for ct, budget in type_budgets.items():
        rows = by_type[ct]
        random.shuffle(rows)
        sampled.extend(rows[:budget])

    random.shuffle(sampled)
    print(f"\nTotal sampled: {len(sampled)}")

    # Convert to list of dicts (HF dataset rows are already dict-like)
    output = []
    for row in sampled:
        output.append({
            "before_code": row.get("before_code", ""),
            "reviewer_comment": row.get("reviewer_comment", ""),
            "after_code": row.get("after_code", ""),
            "diff_context": row.get("diff_context", ""),
            "file_path": row.get("file_path", ""),
            "comment_line": row.get("comment_line", 0),
            "language": row.get("language", ""),
            "quality_score": row.get("quality_score", 0.0),
            "comment_type": row.get("comment_type", "other"),
            "comment_length": row.get("comment_length", 0),
            "before_lines": row.get("before_lines", 0),
            "after_lines": row.get("after_lines", 0),
            "is_negative": row.get("is_negative", False),
            "pr_title": row.get("pr_title", ""),
            "pr_number": row.get("pr_number", 0),
            "repo_name": row.get("repo_name", ""),
            "repo_stars": row.get("repo_stars", 0),
            "repo_language": row.get("repo_language", ""),
            "reviewer_username": row.get("reviewer_username", ""),
            "author_username": row.get("author_username", ""),
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(output)} examples to {args.output}")

    print(f"\nFinal distribution:")
    print(f"  By language: {dict(Counter(e['language'] for e in output))}")
    print(f"  By comment_type: {dict(Counter(e['comment_type'] for e in output))}")
    avg_qs = sum(e["quality_score"] for e in output) / len(output)
    print(f"  Quality score: min={min(e['quality_score'] for e in output):.2f} "
          f"avg={avg_qs:.2f} max={max(e['quality_score'] for e in output):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter & sample HF code review dataset as supplement"
    )
    parser.add_argument(
        "--mined", "-m",
        default="data/kubeflow_reviews.json",
        help="Path to GitHub-mined data (to exclude duplicate repos)",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/hf_supplement.json",
        help="Output JSON path (default: data/hf_supplement.json)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=8000,
        help="Max examples to sample (default: 8000)",
    )
    main(parser.parse_args())
