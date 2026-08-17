"""Tests for Chapter 13: Continuous Fine-Tuning -- Keeping the Model Current.

Archive-splitting logic needs no model and isn't marked slow. The full
train-then-compare run is marked slow and needs Chapter 13's own
checkpoints to exist (checkpoints/ is gitignored -- CI and a fresh
clone won't have run this chapter's ~20-minute script), so it skips
itself rather than re-running a real training job inside a test.

    pytest -v -m "not slow"
"""

import pytest

from continuous_finetune import RUNS_DIR, build_examples, split_archive_by_cutoff


def test_split_archive_by_cutoff_excludes_the_held_out_report():
    current_reports, new_reports = split_archive_by_cutoff()

    assert all(r["rpt_num"] != 37 for r in current_reports)
    assert all(r["rpt_num"] != 37 for r in new_reports)


def test_split_archive_by_cutoff_splits_chronologically_by_report_number():
    current_reports, new_reports = split_archive_by_cutoff(cutoff=60)

    assert all(r["rpt_num"] < 60 for r in current_reports)
    assert all(r["rpt_num"] >= 60 for r in new_reports)
    assert len(current_reports) == 57
    assert len(new_reports) == 17


def test_split_archive_by_cutoff_covers_the_whole_non_held_out_archive():
    current_reports, new_reports = split_archive_by_cutoff()

    assert len(current_reports) + len(new_reports) == 74


def test_build_examples_matches_chapter_7s_totals_when_combined():
    current_reports, new_reports = split_archive_by_cutoff()

    current_examples = build_examples(current_reports)
    new_examples = build_examples(new_reports)

    assert len(current_examples) == 490
    assert len(new_examples) == 179
    assert len(current_examples) + len(new_examples) == 669  # Chapter 7/8's full "at scale" total


@pytest.mark.slow
def test_continuous_finetune_run_produces_a_comparable_updated_checkpoint():
    runs = sorted(RUNS_DIR.glob("run_*"))
    if not runs:
        pytest.skip("No Chapter 13 run found -- run code/chapter_13/continuous_finetune.py first")

    run_dir = runs[-1]
    assert (run_dir / "checkpoint_2").exists()  # the "current" model
    assert (run_dir / "checkpoint_4").exists()  # the "updated" model, after the new batch
