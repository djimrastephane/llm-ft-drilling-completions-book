"""Tests for Chapter 6: A Data Quality Gate for Training Data.

No model needed -- every test here runs against the full 76-report
archive with plain PDF extraction, so none of this file is marked slow.
"""

import pytest

from data_quality_gate import (
    build_archive_records,
    classify_duplicate,
    find_duplicate_values,
    run_quality_gate,
)


@pytest.fixture(scope="module")
def archive_records():
    return build_archive_records()


def test_build_archive_records_covers_the_full_archive(archive_records):
    assert len(archive_records) == 76


def test_extraction_pass_rate_matches_real_run(archive_records):
    passed = [r for r in archive_records if r["status"] == "ok"]
    failed = [r for r in archive_records if r["status"] != "ok"]

    assert len(passed) == 75
    assert [r["file"] for r in failed] == ["FORGE-16A-78-32_Completion_003_2021-01-06.pdf"]


def test_find_duplicate_values_detects_known_non_consecutive_duplicate(archive_records):
    duplicates = find_duplicate_values(archive_records, "present_operations")

    assert duplicates["CIRCULATE FOR TEMPERATURE"] == [49, 70]


def test_classify_duplicate_consecutive_reports_are_not_flagged_for_review():
    assert classify_duplicate([5, 6]) == "consecutive"
    assert classify_duplicate([32, 33]) == "consecutive"


def test_classify_duplicate_non_consecutive_reports_need_review():
    assert classify_duplicate([49, 70]) == "needs_review"
    assert classify_duplicate([47, 59]) == "needs_review"


def test_run_quality_gate_matches_real_run():
    report = run_quality_gate()

    assert report["total"] == 76
    assert report["passed"] == 75
    assert len(report["failed"]) == 1

    statuses = [status for status, _ in report["duplicates"].values()]
    assert statuses.count("consecutive") == 4
    assert statuses.count("needs_review") == 2
