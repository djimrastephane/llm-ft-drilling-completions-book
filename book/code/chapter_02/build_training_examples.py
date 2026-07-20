"""Chapter 2: Turning Drilling & Completions Reports into Training Examples.

Extracts each report's own self-reported PRESENT OPERATIONS and ACTIVITY
PLANNED fields with pdfplumber and turns them into instruction/response
training examples. Every `output` value is a report's own words,
verbatim -- nothing here is a fabricated label.

Only handles the "DAILY DRILLING REPORT" field layout; see this
chapter's Production Reality section and challenge exercise for the
completion report, which uses different field labels.

Usage:
    python code/chapter_02/build_training_examples.py
"""

import json
import re
from datetime import datetime
from pathlib import Path

import pdfplumber

BOOK_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_SET_DIR = BOOK_ROOT / "datasets" / "sample_training_set"
OUTPUT_PATH = BOOK_ROOT / "datasets" / "training_examples" / "sample_training_examples.jsonl"

FIELD_PATTERNS = {
    "well_name": r"WELL NAME:\s*(.+?)\s+JOB:",
    "rpt_date": r"RPT DATE:\s*([\d/]+)",
    "rpt_num": r"RPT NUM\.?:\s*(\d+)",
    "present_operations": r"PRESENT OPERATIONS:\s*(.+?)\nACTIVITY PLANNED",
    "activity_planned": r"ACTIVITY PLANNED:\s*(.+?)\n",
}


def extract_text(pdf_path: Path) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return pdf.pages[0].extract_text()


def extract_fields(text: str) -> dict:
    fields = {}
    for name, pattern in FIELD_PATTERNS.items():
        match = re.search(pattern, text)
        fields[name] = match.group(1).strip() if match else None
    return fields


def normalize_date(raw_date: str) -> str:
    return datetime.strptime(raw_date, "%m/%d/%Y").strftime("%Y-%m-%d")


def build_examples_for_report(pdf_path: Path) -> list[dict]:
    text = extract_text(pdf_path)
    fields = extract_fields(text)

    if not fields["present_operations"] or not fields["activity_planned"]:
        # Not every report in this archive uses this field layout -- see
        # Production Reality below and code/chapter_02/challenge/challenge.py.
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


def build_training_examples(sample_dir: Path = SAMPLE_SET_DIR) -> list[dict]:
    examples = []
    skipped = []
    for pdf_path in sorted(sample_dir.glob("*.pdf")):
        report_examples = build_examples_for_report(pdf_path)
        if report_examples:
            examples.extend(report_examples)
        else:
            skipped.append(pdf_path.name)
    if skipped:
        print(f"Skipped {len(skipped)} report(s) with an unrecognized field layout: {skipped}")
    return examples


def save_examples_jsonl(examples: list[dict], output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")


def main() -> None:
    examples = build_training_examples()
    save_examples_jsonl(examples)
    print(f"\nBuilt {len(examples)} training examples -> {OUTPUT_PATH}\n")
    for example in examples[:2]:
        print(json.dumps(example, indent=2))
        print()


if __name__ == "__main__":
    main()
