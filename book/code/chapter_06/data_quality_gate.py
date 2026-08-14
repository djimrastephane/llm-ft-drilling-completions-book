"""Chapter 6: A Data Quality Gate for Training Data.

Runs Chapter 2's own field extractor across the full 76-report Utah
FORGE archive (not just the 10-report sample) and checks two things a
training pipeline should never assume are true just because extraction
"succeeded": that every report actually produced usable fields, and
that no field value is silently duplicated across reports in a way
that deserves a second look before it gets trained on twice.

Usage:
    python code/chapter_06/data_quality_gate.py
"""

import sys
from collections import defaultdict
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_02"))

from build_training_examples import extract_fields, extract_text  # noqa: E402

FULL_SET_DIR = BOOK_ROOT / "datasets" / "full_training_set"


def build_archive_records(full_dir: Path = FULL_SET_DIR) -> list[dict]:
    """Extract fields from every report and tag each with a pass/fail status."""
    records = []
    for pdf_path in sorted(full_dir.glob("*.pdf")):
        text = extract_text(pdf_path)
        fields = extract_fields(text)
        status = "ok" if fields["present_operations"] and fields["activity_planned"] else "unrecognized_format"

        record = dict(fields)
        record["file"] = pdf_path.name
        record["status"] = status
        record["rpt_num"] = int(fields["rpt_num"]) if fields["rpt_num"] else None
        records.append(record)
    return records


def find_duplicate_values(records: list[dict], field_name: str) -> dict[str, list[int]]:
    """Field values that appear on more than one successfully-extracted report."""
    by_value = defaultdict(list)
    for record in records:
        if record["status"] == "ok" and record[field_name]:
            by_value[record[field_name]].append(record["rpt_num"])
    return {value: sorted(nums) for value, nums in by_value.items() if len(nums) > 1}


def classify_duplicate(rpt_nums: list[int]) -> str:
    """Consecutive report numbers usually mean one operation spanning days
    (benign); any gap means the same exact text recurred independently,
    which deserves a human look before training on it twice.
    """
    gaps = [b - a for a, b in zip(rpt_nums, rpt_nums[1:])]
    return "consecutive" if all(gap == 1 for gap in gaps) else "needs_review"


def run_quality_gate(full_dir: Path = FULL_SET_DIR) -> dict:
    records = build_archive_records(full_dir)
    ok_records = [r for r in records if r["status"] == "ok"]
    failed_records = [r for r in records if r["status"] != "ok"]

    duplicates = {}
    for field_name in ("present_operations", "activity_planned"):
        for value, nums in find_duplicate_values(records, field_name).items():
            duplicates[(field_name, value)] = classify_duplicate(nums), nums

    return {
        "total": len(records),
        "passed": len(ok_records),
        "failed": failed_records,
        "duplicates": duplicates,
    }


def main() -> None:
    report = run_quality_gate()
    print(f"Extraction: {report['passed']}/{report['total']} reports passed")
    if report["failed"]:
        print(f"  Failed (unrecognized format): {[r['file'] for r in report['failed']]}")

    needs_review = {k: v for k, v in report["duplicates"].items() if v[0] == "needs_review"}
    consecutive = {k: v for k, v in report["duplicates"].items() if v[0] == "consecutive"}

    print(f"\nDuplicate field values: {len(consecutive)} consecutive (likely a continued operation, no action needed)")
    print(f"Duplicate field values: {len(needs_review)} needs_review (same text, non-consecutive reports)")
    for (field_name, value), (_, nums) in needs_review.items():
        print(f"  {field_name}, reports {nums}: {value!r}")


if __name__ == "__main__":
    main()
