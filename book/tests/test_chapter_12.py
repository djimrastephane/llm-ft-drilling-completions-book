"""Tests for Chapter 12: Detecting Drift Across Model Versions.

Comparison logic needs no model and isn't marked slow. The full
version-comparison run against Chapter 8's real checkpoints is marked
slow, skipping itself if no checkpoint exists (checkpoints/ is
gitignored -- CI and a fresh clone won't have run Chapter 8's
~30-minute script).

    pytest -v -m "not slow"
"""

import pytest

from detect_model_drift import CHECKPOINTS_DIR, compare_versions, load_version, summarize_version
from eval_finetuned_model import build_held_out_eval_set


def test_compare_versions_flags_improved_metric():
    before = {"exact_match": 0, "avg_overlap": 0.35, "perplexity": 28.0}
    after = {"exact_match": 0, "avg_overlap": 0.35, "perplexity": 25.0}

    directions = compare_versions(before, after)

    assert directions["perplexity"] == "improved"
    assert directions["exact_match"] == "unchanged"
    assert directions["avg_overlap"] == "unchanged"


def test_compare_versions_flags_regressed_overlap_even_when_perplexity_improves():
    before = {"exact_match": 0, "avg_overlap": 0.35, "perplexity": 28.0}
    after = {"exact_match": 0, "avg_overlap": 0.17, "perplexity": 25.0}

    directions = compare_versions(before, after)

    assert directions["avg_overlap"] == "regressed"
    assert directions["perplexity"] == "improved"


def test_compare_versions_higher_exact_match_is_improved():
    before = {"exact_match": 2, "avg_overlap": 0.2, "perplexity": 30.0}
    after = {"exact_match": 5, "avg_overlap": 0.2, "perplexity": 30.0}

    assert compare_versions(before, after)["exact_match"] == "improved"


@pytest.fixture(scope="module")
def latest_run_checkpoints():
    runs = sorted(CHECKPOINTS_DIR.glob("run_*"))
    if not runs:
        pytest.skip("No Chapter 8 checkpoint found -- run code/chapter_08/finetune_at_scale.py first")
    return sorted(runs[-1].glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[1]))


@pytest.mark.slow
def test_summarize_version_returns_all_three_metrics(latest_run_checkpoints):
    model, tokenizer = load_version(latest_run_checkpoints[0])
    eval_set = build_held_out_eval_set()

    summary = summarize_version(model, tokenizer, eval_set)

    assert set(summary) == {"exact_match", "avg_overlap", "perplexity"}
    assert summary["perplexity"] > 0


@pytest.mark.slow
def test_checkpoint_1_to_2_shows_a_real_metric_disagreement(latest_run_checkpoints):
    eval_set = build_held_out_eval_set()

    model_1, tokenizer = load_version(latest_run_checkpoints[0])
    summary_1 = summarize_version(model_1, tokenizer, eval_set)

    model_2, _ = load_version(latest_run_checkpoints[1])
    summary_2 = summarize_version(model_2, tokenizer, eval_set)

    directions = compare_versions(summary_1, summary_2)

    assert directions["perplexity"] == "improved"
    assert directions["avg_overlap"] == "regressed"
