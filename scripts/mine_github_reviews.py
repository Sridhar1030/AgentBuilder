#!/usr/bin/env python3
"""
mine_github_reviews.py
─────────────────────
Mines PR review comments from target GitHub repos and outputs them
in the **exact same schema** as ronantakizawa/github-codereview so
the two sources can be concatenated into one training set.

Requirements:
    pip install requests
    export GITHUB_TOKEN=ghp_...   (or use `gh auth token`)

Usage:
    python scripts/mine_github_reviews.py \
        --output data/kubeflow_reviews.json \
        --max-prs-per-repo 150 \
        --max-examples 200
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import requests

# ── Target repositories ──────────────────────────────────────────────
REPOS = [
    "kubeflow/trainer",
    "kubeflow/pipelines",
    "ray-project/kuberay",
    "kserve/kserve",
    "tektoncd/pipeline",
]

VALID_EXTENSIONS = {
    ".go": "Go",
    ".py": "Python",
    ".yaml": "YAML",
    ".yml": "YAML",
}

NOISE_PATTERNS = re.compile(
    r"^(lgtm|looks good|\+1|nit|thanks|done|fixed\.?|sgtm|ack|ptal|wfm)$",
    re.IGNORECASE,
)

BOT_USERS = {
    "dependabot[bot]",
    "github-actions[bot]",
    "codecov[bot]",
    "stale[bot]",
    "mergify[bot]",
}

MIN_COMMENT_LENGTH = 30


# ── GitHub helpers ────────────────────────────────────────────────────
def get_token():
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    try:
        result = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        sys.exit("ERROR: Set GITHUB_TOKEN or install/auth gh CLI.")


def gh_get(session: requests.Session, url: str, params: dict | None = None):
    """GET with rate-limit back-off."""
    while True:
        resp = session.get(url, params=params)
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - int(time.time()), 5)
            print(f"  ⏳ Rate-limited — sleeping {wait}s")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()


def get_repo_meta(session, repo):
    data = gh_get(session, f"https://api.github.com/repos/{repo}")
    return {
        "stars": data.get("stargazers_count", 0),
        "language": data.get("language", ""),
    }


def get_merged_prs(session, repo, max_pages):
    prs = []
    for page in range(1, max_pages + 1):
        data = gh_get(
            session,
            f"https://api.github.com/repos/{repo}/pulls",
            params={
                "state": "closed",
                "sort": "updated",
                "direction": "desc",
                "per_page": 100,
                "page": page,
            },
        )
        if not data:
            break
        for pr in data:
            if pr.get("merged_at"):
                prs.append(pr)
    return prs


def get_review_comments(session, repo, pr_number):
    comments = []
    page = 1
    while True:
        data = gh_get(
            session,
            f"https://api.github.com/repos/{repo}/pulls/{pr_number}/comments",
            params={"per_page": 100, "page": page},
        )
        if not data:
            break
        comments.extend(data)
        if len(data) < 100:
            break
        page += 1
    return comments


# ── Parsing helpers ───────────────────────────────────────────────────
def detect_language(file_path: str) -> str | None:
    for ext, lang in VALID_EXTENSIONS.items():
        if file_path.endswith(ext):
            return lang
    return None


def is_useful_comment(body: str) -> bool:
    text = body.strip()
    if len(text) < MIN_COMMENT_LENGTH:
        return False
    if text.startswith("```suggestion"):
        return False
    first_line = text.split("\n")[0].strip().lower()
    if NOISE_PATTERNS.match(first_line):
        return False
    return True


def classify_comment(body: str) -> str:
    """Heuristic comment_type classifier matching HF dataset categories."""
    lower = body.lower()
    if "?" in body:
        return "question"
    if any(w in lower for w in ["suggest", "consider", "how about", "what about", "maybe", "could we"]):
        return "suggestion"
    if any(w in lower for w in ["bug", "error", "wrong", "incorrect", "broken", "fail", "crash", "nil pointer", "panic"]):
        return "bug"
    if any(w in lower for w in ["secur", "vuln", "inject", "auth", "permiss", "credential", "secret", "token leak"]):
        return "security"
    if any(w in lower for w in ["perf", "slow", "optim", "cache", "allocat", "o(n", "latency", "throughput"]):
        return "performance"
    if any(w in lower for w in ["race", "concurrent", "deadlock", "reliab", "retry", "timeout", "graceful"]):
        return "reliability"
    if any(w in lower for w in ["style", "naming", "readab", "convention", "godoc", "docstring", "comment", "typo"]):
        return "style"
    if any(w in lower for w in ["refactor", "simplif", "duplicat", "extract", "inline", "consolidat"]):
        return "refactoring"
    return "other"


def parse_diff_hunk(diff_hunk: str):
    """
    Split a unified diff hunk into before_code and after_code,
    mirroring what the HF dataset stores.
    """
    before_lines = []
    after_lines = []

    for line in diff_hunk.split("\n"):
        if line.startswith("@@"):
            continue
        if line.startswith("-"):
            before_lines.append(line[1:])
        elif line.startswith("+"):
            after_lines.append(line[1:])
        else:
            # context line (present in both)
            content = line[1:] if line.startswith(" ") else line
            before_lines.append(content)
            after_lines.append(content)

    return "\n".join(before_lines), "\n".join(after_lines)


def compute_quality_score(comment: str, diff_hunk: str) -> float:
    """Simple heuristic quality score [0..1] matching HF dataset range."""
    score = 0.3
    if len(comment) > 50:
        score += 0.1
    if len(comment) > 100:
        score += 0.1
    if len(comment) > 200:
        score += 0.1
    if "```" in comment:
        score += 0.05
    if any(w in comment.lower() for w in ["bug", "security", "race", "nil", "error", "crash"]):
        score += 0.1
    if "?" not in comment:
        score += 0.05
    if len(diff_hunk) > 200:
        score += 0.1
    return min(round(score, 4), 1.0)


# ── Main pipeline ─────────────────────────────────────────────────────
def mine(args):
    token = get_token()
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )

    max_pages = (args.max_prs_per_repo + 99) // 100
    dataset = []
    stats = Counter()

    for repo in REPOS:
        print(f"\n{'─'*60}")
        print(f"Mining {repo}...")
        meta = get_repo_meta(session, repo)
        print(f"  Stars: {meta['stars']}  Language: {meta['language']}")

        prs = get_merged_prs(session, repo, max_pages)
        print(f"  Merged PRs fetched: {len(prs)}")
        stats["prs_fetched"] += len(prs)

        for pr in prs:
            if len(dataset) >= args.max_examples:
                break

            pr_number = pr["number"]
            pr_title = pr.get("title", "")
            pr_author = pr.get("user", {}).get("login", "")

            comments = get_review_comments(session, repo, pr_number)
            stats["comments_raw"] += len(comments)

            for c in comments:
                if len(dataset) >= args.max_examples:
                    break

                user = c.get("user", {}).get("login", "")
                if user in BOT_USERS:
                    stats["filtered_bot"] += 1
                    continue

                file_path = c.get("path", "")
                language = detect_language(file_path)
                if language is None:
                    stats["filtered_language"] += 1
                    continue

                body = c.get("body", "")
                if not is_useful_comment(body):
                    stats["filtered_quality"] += 1
                    continue

                diff_hunk = c.get("diff_hunk", "")
                if not diff_hunk:
                    stats["filtered_no_diff"] += 1
                    continue

                before_code, after_code = parse_diff_hunk(diff_hunk)
                comment_line = c.get("line") or c.get("original_line") or 0
                quality_score = compute_quality_score(body, diff_hunk)
                comment_type = classify_comment(body)

                record = {
                    "before_code": before_code,
                    "reviewer_comment": body,
                    "after_code": after_code,
                    "diff_context": diff_hunk,
                    "file_path": file_path,
                    "comment_line": comment_line,
                    "language": language,
                    "quality_score": quality_score,
                    "comment_type": comment_type,
                    "comment_length": len(body),
                    "before_lines": before_code.count("\n") + 1,
                    "after_lines": after_code.count("\n") + 1,
                    "is_negative": False,
                    "pr_title": pr_title,
                    "pr_number": pr_number,
                    "repo_name": repo,
                    "repo_stars": meta["stars"],
                    "repo_language": meta["language"],
                    "reviewer_username": user,
                    "author_username": pr_author,
                }
                dataset.append(record)
                stats["accepted"] += 1

            if len(dataset) % 25 == 0 and len(dataset) > 0:
                print(f"  ... {len(dataset)} examples collected so far")

        if len(dataset) >= args.max_examples:
            print(f"\n  Reached target of {args.max_examples} examples — stopping.")
            break

    # ── Save ──────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(dataset, f, indent=2)

    # ── Report ────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"DONE — {len(dataset)} examples saved to {args.output}")
    print(f"{'═'*60}")
    print(f"\nPipeline stats:")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")

    print(f"\nBy repo:")
    for k, v in Counter(e["repo_name"] for e in dataset).most_common():
        print(f"  {k}: {v}")

    print(f"\nBy language:")
    for k, v in Counter(e["language"] for e in dataset).most_common():
        print(f"  {k}: {v}")

    print(f"\nBy comment_type:")
    for k, v in Counter(e["comment_type"] for e in dataset).most_common():
        print(f"  {k}: {v}")

    print(f"\nQuality score range: {min(e['quality_score'] for e in dataset):.2f} – {max(e['quality_score'] for e in dataset):.2f}")
    avg_qs = sum(e["quality_score"] for e in dataset) / len(dataset)
    print(f"Quality score mean:  {avg_qs:.2f}")

    # ── Verify schema matches HF ──────────────────────────────────────
    hf_cols = [
        "before_code", "reviewer_comment", "after_code", "diff_context",
        "file_path", "comment_line", "language", "quality_score",
        "comment_type", "comment_length", "before_lines", "after_lines",
        "is_negative", "pr_title", "pr_number", "repo_name", "repo_stars",
        "repo_language", "reviewer_username", "author_username",
    ]
    our_cols = list(dataset[0].keys())
    assert our_cols == hf_cols, f"Schema mismatch!\n  Ours: {our_cols}\n  HF:   {hf_cols}"
    print("\n✅ Schema matches ronantakizawa/github-codereview exactly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mine PR reviews from GitHub repos → HF-compatible JSON"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/kubeflow_reviews.json",
        help="Output JSON path (default: data/kubeflow_reviews.json)",
    )
    parser.add_argument(
        "--max-prs-per-repo",
        type=int,
        default=150,
        help="Max merged PRs to scan per repo (default: 150)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=200,
        help="Stop after collecting this many examples (default: 200)",
    )
    mine(parser.parse_args())
