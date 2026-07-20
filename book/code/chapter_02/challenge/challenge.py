"""Chapter 2 challenge exercise -- reference solution.

Challenge: extend build_training_examples.py to also handle the
completion report (FORGE-16A-78-32_Completion_003_2021-01-06.pdf),
which uses different field labels ("Present Ops:" / "Next 24 Hours:"
and a two-digit year) instead of the drilling reports'
"PRESENT OPERATIONS:" / "ACTIVITY PLANNED:". This is a real, genuine
data-format difference in this book's own archive, not a made-up
exercise.

Usage:
    python code/chapter_02/challenge/challenge.py
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_training_examples import SAMPLE_SET_DIR, extract_text  # noqa: E402

DRILLING_PATTERNS = {
    "well_name": r"WELL NAME:\s*(.+?)\s+JOB:",
    "rpt_date": r"RPT DATE:\s*([\d/]+)",
    "rpt_num": r"RPT NUM\.?:\s*(\d+)",
    "present_operations": r"PRESENT OPERATIONS:\s*(.+?)\nACTIVITY PLANNED",
    "activity_planned": r"ACTIVITY PLANNED:\s*(.+?)\n",
}

COMPLETION_PATTERNS = {
    "well_name": r"WELL NAME:\s*(.+?)\s+JOB:",
    "rpt_date": r"RPT DATE:\s*([\d/]+)",
    "rpt_num": r"RPT NUM\.?:\s*(\d+)",
    "present_operations": r"Present Ops:\s*(.+?)\nNext 24 Hours",
    "activity_planned": r"Next 24 Hours:\s*(.+?)\n",
}


def extract_fields(text: str, patterns: dict) -> dict:
    fields = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        fields[name] = match.group(1).strip() if match else None
    return fields


def normalize_date(raw_date: str) -> str:
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(raw_date, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {raw_date!r}")


def build_examples_for_report(pdf_path: Path) -> list[dict]:
    text = extract_text(pdf_path)

    # Try the drilling-report layout first, then the completion-report layout.
    fields = extract_fields(text, DRILLING_PATTERNS)
    if not fields["present_operations"]:
        fields = extract_fields(text, COMPLETION_PATTERNS)
    if not fields["present_operations"] or not fields["activity_planned"]:
        return []

    context = (
        f"Well: {fields['well_name']} | Report #{fields['rpt_num']} | "
        f"Date: {normalize_date(fields['rpt_date'])}"
    )
    return [
        {
            "instruction": "What are the present operations reported on this well?",
            "input": context,
            "output": fields["present_operations"],
        },
        {
            "instruction": "What activity is planned next?",
            "input": context,
            "output": fields["activity_planned"],
        },
    ]


def main() -> None:
    examples = []
    skipped = []
    for pdf_path in sorted(SAMPLE_SET_DIR.glob("*.pdf")):
        report_examples = build_examples_for_report(pdf_path)
        if report_examples:
            examples.extend(report_examples)
        else:
            skipped.append(pdf_path.name)

    print(f"Built {len(examples)} training examples ({len(skipped)} report(s) skipped: {skipped})\n")
    completion_examples = [e for e in examples if "2021-01-06" in e["input"]]
    for example in completion_examples:
        print(json.dumps(example, indent=2))


if __name__ == "__main__":
    main()
