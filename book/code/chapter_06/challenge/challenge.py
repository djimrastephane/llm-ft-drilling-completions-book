"""Chapter 6 challenge exercise -- reference solution.

Challenge: add a third check to the quality gate -- report dates should
strictly increase alongside report numbers. A scrambled or misfiled
archive (two reports swapped, a report from the wrong well mixed in)
would show up here as an out-of-order or repeated date. Confirm this
book's own archive passes cleanly.

Usage:
    python code/chapter_06/challenge/challenge.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "chapter_02"))

from build_training_examples import normalize_date  # noqa: E402
from data_quality_gate import FULL_SET_DIR, build_archive_records  # noqa: E402


def check_chronological_order(records: list[dict]) -> list[tuple[int, int, str, str]]:
    """Return (prev_rpt_num, rpt_num, prev_date, date) for any pair that's out of order."""
    ok_records = sorted(
        (r for r in records if r["status"] == "ok" and r["rpt_date"]),
        key=lambda r: r["rpt_num"],
    )
    issues = []
    prev_num, prev_date = None, None
    for record in ok_records:
        date = normalize_date(record["rpt_date"])
        if prev_date is not None and date <= prev_date:
            issues.append((prev_num, record["rpt_num"], prev_date, date))
        prev_num, prev_date = record["rpt_num"], date
    return issues


def main() -> None:
    records = build_archive_records(FULL_SET_DIR)
    issues = check_chronological_order(records)

    if issues:
        print(f"{len(issues)} chronological issue(s) found:")
        for prev_num, num, prev_date, date in issues:
            print(f"  Report #{prev_num} ({prev_date}) -> Report #{num} ({date})")
    else:
        print(f"Chronological order: {sum(1 for r in records if r['status'] == 'ok')} reports checked, 0 issues -- dates strictly increase with report number")


if __name__ == "__main__":
    main()
