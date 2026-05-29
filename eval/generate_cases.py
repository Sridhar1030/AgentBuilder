"""Generate agent-eval-harness case directories from test_questions.json.

Creates the directory structure that the harness's score.py expects:

    eval/dataset/cases/
      case-001-slug/
        input.yaml      # diff, category, expected_behavior
        annotations.yaml # has_bug, expected_issues, category
        reference.md     # placeholder for teacher's gold-standard review

Run from project root:
    python3 eval/generate_cases.py
"""

import json
import re
from pathlib import Path

import yaml


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
    return slug[:max_len].rstrip('-')


def main():
    project_root = Path(__file__).resolve().parent.parent
    tq_path = project_root / "pipeline" / "domain" / "test_questions.json"
    cases_dir = project_root / "eval" / "dataset" / "cases"

    with open(tq_path) as f:
        questions = json.load(f)

    cases_dir.mkdir(parents=True, exist_ok=True)

    for i, entry in enumerate(questions, 1):
        case_id = entry.get("case_id", f"case-{i:03d}")
        case_dir = cases_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        # Extract diff from the question text
        question = entry["question"]
        diff_match = re.search(r'```diff\n(.*?)```', question, re.DOTALL)
        diff = diff_match.group(1).strip() if diff_match else question

        # input.yaml -- what the runner sends to the model
        input_data = {
            "diff": question,
            "category": entry.get("category", "unknown"),
            "expected_behavior": entry.get("expected_behavior", ""),
        }
        with open(case_dir / "input.yaml", "w") as f:
            yaml.dump(input_data, f, default_flow_style=False, width=120)

        # annotations.yaml -- metadata for judges
        annotations = {
            "has_bug": entry.get("has_bug", True),
            "expected_issues": entry.get("expected_issues", []),
            "category": entry.get("category", "unknown"),
        }
        with open(case_dir / "annotations.yaml", "w") as f:
            yaml.dump(annotations, f, default_flow_style=False)

        # reference.md -- placeholder for gold-standard review
        ref_path = case_dir / "reference.md"
        if not ref_path.exists():
            expected = entry.get("expected_behavior", "")
            ref_path.write_text(
                f"<!-- Gold-standard review (generate with teacher model) -->\n"
                f"<!-- Expected: {expected} -->\n"
            )

        print(f"  {case_id}: input.yaml, annotations.yaml, reference.md")

    print(f"\nGenerated {len(questions)} cases in {cases_dir}")


if __name__ == "__main__":
    main()
