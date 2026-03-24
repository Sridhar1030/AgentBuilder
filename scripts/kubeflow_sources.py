#!/usr/bin/env python3
"""
Step 1.1 — Kubeflow Source Collector

Crawls Kubeflow-ecosystem documentation from four source types and writes
every document as one JSON-Lines record to data/raw_sources.jsonl.

Sources:
  1. Kubeflow docs site  — parse sitemap at kubeflow.org, fetch & strip HTML
  2. KServe docs site    — parse sitemap at kserve.github.io
  3. GitHub repos        — READMEs, docs/ markdown, release notes
  4. YouTube transcripts — curated KubeCon / community talks

Each line: {"url", "project", "doc_type", "title", "content"}

Usage:
    python scripts/kubeflow_sources.py [--output data/raw_sources.jsonl]

Environment variables (optional):
    GITHUB_TOKEN  — raises GitHub API rate limit from 60 → 5000 req/hr
"""

import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from bs4 import BeautifulSoup

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    HAS_YT = True
except ImportError:
    HAS_YT = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KUBEFLOW_SITEMAP = "https://www.kubeflow.org/sitemap.xml"
KSERVE_SITEMAP = "https://kserve.github.io/website/sitemap.xml"

GITHUB_REPOS = [
    {
        "owner": "kubeflow", "repo": "pipelines",
        "project": "kubeflow-pipelines",
        "files": [
            "README.md", "CHANGELOG.md", "developer_guide.md",
            "sdk/python/README.md",
        ],
        "tree_filter": lambda p: (
            p.startswith("docs/") and p.endswith(".md")
        ),
    },
    {
        "owner": "kubeflow", "repo": "training-operator",
        "project": "training-operator",
        "files": ["README.md", "CHANGELOG.md"],
        "tree_filter": lambda p: (
            (p.startswith("docs/") or p.startswith("sdk/")) and p.endswith(".md")
        ),
    },
    {
        "owner": "kserve", "repo": "kserve",
        "project": "kserve",
        "files": ["README.md", "CHANGELOG.md"],
        "tree_filter": lambda p: p.startswith("docs/") and p.endswith(".md"),
    },
    {
        "owner": "kubeflow", "repo": "katib",
        "project": "katib",
        "files": ["README.md"],
        "tree_filter": lambda p: p.startswith("docs/") and p.endswith(".md"),
    },
    {
        "owner": "kubeflow", "repo": "model-registry",
        "project": "model-registry",
        "files": ["README.md"],
        "tree_filter": lambda p: p.startswith("docs/") and p.endswith(".md"),
    },
    {
        "owner": "kubeflow", "repo": "manifests",
        "project": "kubeflow-manifests",
        "files": ["README.md"],
        "tree_filter": None,
    },
    {
        "owner": "kubeflow", "repo": "notebooks",
        "project": "kubeflow-notebooks",
        "files": ["README.md"],
        "tree_filter": lambda p: p.startswith("docs/") and p.endswith(".md"),
    },
]

YOUTUBE_VIDEOS = [
    {"id": "LqhtmG8jRRc", "title": "End to End LLMOps with Kubeflow", "project": "kubeflow-general"},
    {"id": "bzu2Qqv4Ij0", "title": "Kubeflow 1.9 Release Overview", "project": "kubeflow-general"},
    {"id": "TQypOccQ3lc", "title": "KServe & Kubeflow Model Deployment", "project": "kserve"},
    {"id": "tRINql1jW-I", "title": "Building a ML Pipeline with Kubeflow", "project": "kubeflow-pipelines"},
    {"id": "6wWdNg0GMV4", "title": "Intro to Kubeflow", "project": "kubeflow-general"},
    {"id": "cTZArDgbIWw", "title": "Kubeflow Pipelines Tutorial", "project": "kubeflow-pipelines"},
    {"id": "dOkAFMRBMrg", "title": "Katib Hyperparameter Tuning on Kubernetes", "project": "katib"},
]

CURATED_KF_PAGES = [
    "/docs/started/introduction/",
    "/docs/started/installing-kubeflow/",
    "/docs/started/architecture/",
    "/docs/components/pipelines/",
    "/docs/components/pipelines/overview/",
    "/docs/components/pipelines/concepts/component/",
    "/docs/components/pipelines/concepts/pipeline/",
    "/docs/components/pipelines/concepts/run/",
    "/docs/components/pipelines/concepts/experiment/",
    "/docs/components/pipelines/user-guides/",
    "/docs/components/pipelines/user-guides/components/",
    "/docs/components/pipelines/user-guides/core-functions/compile-a-pipeline/",
    "/docs/components/pipelines/user-guides/core-functions/create-a-pipeline/",
    "/docs/components/pipelines/user-guides/core-functions/connect-api/",
    "/docs/components/pipelines/legacy-v1/",
    "/docs/components/trainer/",
    "/docs/components/trainer/overview/",
    "/docs/components/trainer/user-guides/",
    "/docs/components/trainer/user-guides/fine-tuning/",
    "/docs/components/trainer/operator-guides/",
    "/docs/components/trainer/operator-guides/installation/",
    "/docs/components/trainer/operator-guides/migration/",
    "/docs/components/trainer/operator-guides/job-template/",
    "/docs/components/katib/",
    "/docs/components/katib/overview/",
    "/docs/components/katib/user-guides/hp-tuning/configure-experiment/",
    "/docs/components/katib/user-guides/hp-tuning/using-early-stopping/",
    "/docs/components/katib/user-guides/nas/configure-experiment/",
    "/docs/external-add-ons/kserve/",
    "/docs/external-add-ons/kserve/kserve/",
    "/docs/components/notebooks/",
    "/docs/components/notebooks/overview/",
    "/docs/components/model-registry/",
    "/docs/components/model-registry/overview/",
    "/docs/components/multi-tenancy/",
    "/docs/components/multi-tenancy/overview/",
    "/docs/distributions/",
]

CURATED_KSERVE_PAGES = [
    "/website/docs/intro",
    "/website/docs/getting-started/quickstart-guide",
    "/website/docs/modelserving/v1beta1/serving-runtimes",
    "/website/docs/modelserving/v1beta1/transformer/torchserve",
    "/website/docs/modelserving/v1beta1/sklearn/v2",
    "/website/docs/modelserving/autoscaling/autoscaling",
    "/website/docs/modelserving/detect/alibi_detect",
    "/website/docs/modelserving/explainer/alibi",
    "/website/docs/modelserving/inference_graph",
    "/website/docs/modelserving/control_plane",
    "/website/docs/modelserving/data_plane/v2_protocol",
    "/website/docs/modelserving/storage/storagecontainers",
    "/website/docs/admin/serverless/serverless",
    "/website/docs/admin/modelmesh",
    "/website/docs/admin/kubernetes_deployment",
]

USER_AGENT = "KubeflowSourceCollector/1.0 (training-data-gen)"
REQUEST_DELAY = 0.5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session(github_token: str | None = None) -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    if github_token:
        s.headers["Authorization"] = f"token {github_token}"
    return s


def _fetch(session: requests.Session, url: str, timeout: int = 30) -> str | None:
    try:
        r = session.get(url, timeout=timeout, allow_redirects=True)
        if r.status_code == 200:
            return r.text
        print(f"  [WARN] HTTP {r.status_code} for {url}")
    except Exception as exc:
        print(f"  [WARN] Failed {url}: {exc}")
    return None


def _extract_html_content(html: str, url: str) -> tuple[str, str]:
    """Return (title, text_content) from an HTML page."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    if not title:
        t = soup.find("title")
        title = t.get_text(strip=True) if t else url.split("/")[-2] or url

    content_el = (
        soup.find("article")
        or soup.find("div", class_="td-content")
        or soup.find("div", class_="markdown")
        or soup.find("main")
        or soup.find("div", {"role": "main"})
    )
    if content_el is None:
        content_el = soup.body or soup

    text = content_el.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


def _parse_sitemap(xml_text: str) -> list[str]:
    """Extract <loc> URLs from an XML sitemap (handles namespaces)."""
    urls: list[str] = []
    try:
        root = ET.fromstring(xml_text)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        for loc in root.findall(".//sm:loc", ns):
            if loc.text:
                urls.append(loc.text.strip())
        if not urls:
            for loc in root.iter():
                if loc.tag.endswith("}loc") or loc.tag == "loc":
                    if loc.text:
                        urls.append(loc.text.strip())
    except ET.ParseError as exc:
        print(f"  [WARN] Sitemap XML parse error: {exc}")

    cleaned: list[str] = []
    for u in urls:
        # Kubeflow sitemap has a known bug: URLs look like
        # "https://www.kubeflow.orghttps://www.kubeflow.org/docs/..."
        idx = u.find("https://", 1)
        if idx > 0:
            u = u[idx:]
        cleaned.append(u)
    return cleaned


def _project_from_url(url: str) -> str:
    if "kserve" in url:
        return "kserve"
    for seg in ["pipelines", "trainer", "training-operator", "katib",
                "notebooks", "model-registry", "manifests", "multi-tenancy"]:
        if seg in url:
            return seg
    return "kubeflow-general"


# ---------------------------------------------------------------------------
# Source collectors
# ---------------------------------------------------------------------------

def collect_docs_site(
    session: requests.Session,
    sitemap_url: str,
    base_url: str,
    curated_paths: list[str],
    site_name: str,
    delay: float,
) -> list[dict]:
    """Fetch doc pages from a site via sitemap, with curated fallback."""
    records: list[dict] = []
    print(f"\n[{site_name}] Fetching sitemap: {sitemap_url}")
    xml = _fetch(session, sitemap_url)
    urls = _parse_sitemap(xml) if xml else []

    if urls:
        urls = [u for u in urls if "/docs/" in u]
        print(f"  Found {len(urls)} doc URLs in sitemap")
    else:
        print(f"  Sitemap unavailable — using {len(curated_paths)} curated URLs")
        urls = [base_url.rstrip("/") + p for p in curated_paths]

    seen = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        html = _fetch(session, url)
        if not html:
            continue
        title, content = _extract_html_content(html, url)
        if len(content) < 100:
            continue
        records.append({
            "url": url,
            "project": _project_from_url(url),
            "doc_type": "docs",
            "title": title,
            "content": content,
        })
        time.sleep(delay)

    print(f"  Collected {len(records)} pages from {site_name}")
    return records


def collect_github_repos(session: requests.Session, delay: float) -> list[dict]:
    """Fetch READMEs, docs/ markdown, and other key files from GitHub repos."""
    records: list[dict] = []
    print("\n[GitHub] Fetching repo documentation...")

    for repo_cfg in GITHUB_REPOS:
        owner = repo_cfg["owner"]
        repo = repo_cfg["repo"]
        project = repo_cfg["project"]
        print(f"  {owner}/{repo}...")

        for fpath in repo_cfg.get("files", []):
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/{fpath}"
            content = _fetch(session, raw_url)
            if not content:
                raw_url = raw_url.replace("/master/", "/main/")
                content = _fetch(session, raw_url)
            if content and len(content) > 50:
                records.append({
                    "url": raw_url,
                    "project": project,
                    "doc_type": "github-file",
                    "title": f"{owner}/{repo}/{fpath}",
                    "content": content,
                })
            time.sleep(delay)

        tree_filter = repo_cfg.get("tree_filter")
        if tree_filter:
            for branch in ("master", "main"):
                tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
                tree_raw = _fetch(session, tree_url)
                if tree_raw:
                    break
            if tree_raw:
                try:
                    tree = json.loads(tree_raw)
                    paths = [
                        n["path"]
                        for n in tree.get("tree", [])
                        if n["type"] == "blob" and tree_filter(n["path"])
                    ]
                    already = {f for f in repo_cfg.get("files", [])}
                    for fpath in paths:
                        if fpath in already:
                            continue
                        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{fpath}"
                        content = _fetch(session, raw_url)
                        if content and len(content) > 50:
                            records.append({
                                "url": raw_url,
                                "project": project,
                                "doc_type": "github-docs",
                                "title": f"{owner}/{repo}/{fpath}",
                                "content": content,
                            })
                        time.sleep(delay)
                except json.JSONDecodeError:
                    print(f"  [WARN] Could not parse tree for {owner}/{repo}")

    print(f"  Collected {len(records)} GitHub documents")
    return records


def collect_github_releases(session: requests.Session, delay: float) -> list[dict]:
    """Fetch the last 10 releases per major repo."""
    records: list[dict] = []
    repos = [
        ("kubeflow", "pipelines", "kubeflow-pipelines"),
        ("kubeflow", "training-operator", "training-operator"),
        ("kserve", "kserve", "kserve"),
        ("kubeflow", "katib", "katib"),
    ]
    print("\n[GitHub Releases] Fetching release notes...")

    for owner, repo, project in repos:
        url = f"https://api.github.com/repos/{owner}/{repo}/releases?per_page=10"
        raw = _fetch(session, url)
        if not raw:
            continue
        try:
            releases = json.loads(raw)
            for rel in releases:
                body = rel.get("body", "").strip()
                tag = rel.get("tag_name", "unknown")
                name = rel.get("name", tag)
                if len(body) < 50:
                    continue
                records.append({
                    "url": rel.get("html_url", url),
                    "project": project,
                    "doc_type": "release-notes",
                    "title": f"{owner}/{repo} {name}",
                    "content": body,
                })
        except json.JSONDecodeError:
            print(f"  [WARN] Bad JSON for releases of {owner}/{repo}")
        time.sleep(delay)

    print(f"  Collected {len(records)} release notes")
    return records


def collect_youtube(delay: float) -> list[dict]:
    """Fetch transcripts for curated Kubeflow talks."""
    records: list[dict] = []
    if not HAS_YT:
        print("\n[YouTube] Skipping — youtube-transcript-api not installed")
        return records

    print(f"\n[YouTube] Fetching {len(YOUTUBE_VIDEOS)} transcripts...")
    yt_api = YouTubeTranscriptApi()
    for vid in YOUTUBE_VIDEOS:
        try:
            fetched = yt_api.fetch(vid["id"])
            text = " ".join(s.text for s in fetched.snippets)
            if len(text) < 100:
                continue
            records.append({
                "url": f"https://www.youtube.com/watch?v={vid['id']}",
                "project": vid["project"],
                "doc_type": "youtube",
                "title": vid["title"],
                "content": text,
            })
            print(f"  ✓ {vid['title']}")
        except Exception as exc:
            print(f"  [WARN] Transcript unavailable for {vid['id']}: {exc}")
        time.sleep(delay)

    print(f"  Collected {len(records)} transcripts")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect Kubeflow documentation sources")
    parser.add_argument("--output", default="data/raw_sources.jsonl")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY)
    parser.add_argument("--github-token", default=os.getenv("GITHUB_TOKEN"))
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sess = _session(args.github_token)
    all_records: list[dict] = []

    all_records.extend(
        collect_docs_site(
            sess, KUBEFLOW_SITEMAP, "https://www.kubeflow.org",
            CURATED_KF_PAGES, "Kubeflow Docs", args.delay,
        )
    )
    all_records.extend(
        collect_docs_site(
            sess, KSERVE_SITEMAP, "https://kserve.github.io",
            CURATED_KSERVE_PAGES, "KServe Docs", args.delay,
        )
    )
    all_records.extend(collect_github_repos(sess, args.delay))
    all_records.extend(collect_github_releases(sess, args.delay))
    all_records.extend(collect_youtube(args.delay))

    with open(args.output, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n{'='*60}")
    print(f"Total documents collected: {len(all_records)}")
    by_type: dict[str, int] = {}
    by_project: dict[str, int] = {}
    for r in all_records:
        by_type[r["doc_type"]] = by_type.get(r["doc_type"], 0) + 1
        by_project[r["project"]] = by_project.get(r["project"], 0) + 1
    print("By type:")
    for k, v in sorted(by_type.items()):
        print(f"  {k:20s} {v}")
    print("By project:")
    for k, v in sorted(by_project.items()):
        print(f"  {k:25s} {v}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
