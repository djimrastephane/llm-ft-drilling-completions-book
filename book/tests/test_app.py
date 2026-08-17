"""Tests for the companion app's helpers.py.

app/helpers.py isn't a chapter script, so it has its own test file
rather than a chapter_NN-numbered one. Checkpoint-discovery and scoring
logic need no model and aren't marked slow. Generation/evaluation tests
are marked `slow` (need the real base model); anything needing a real
fine-tuned checkpoint skips cleanly if none exists yet (checkpoints/ is
gitignored -- CI and a fresh clone won't have run Chapter 5/8/13's
scripts).

    pytest -v -m "not slow"
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pytest

import helpers


def test_available_checkpoints_is_empty_when_nothing_exists_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "CHAPTER_05_ADAPTER", tmp_path / "chapter_05_lora")
    monkeypatch.setattr(helpers, "CHAPTER_08_RUNS", tmp_path / "chapter_08")
    monkeypatch.setattr(helpers, "CHAPTER_13_RUNS", tmp_path / "chapter_13")

    assert helpers.available_checkpoints() == {}


def test_available_checkpoints_lists_chapter_05_once_its_adapter_dir_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "CHAPTER_05_ADAPTER", tmp_path / "chapter_05_lora")
    monkeypatch.setattr(helpers, "CHAPTER_08_RUNS", tmp_path / "missing_08")
    monkeypatch.setattr(helpers, "CHAPTER_13_RUNS", tmp_path / "missing_13")
    (tmp_path / "chapter_05_lora").mkdir()

    checkpoints = helpers.available_checkpoints()

    assert list(checkpoints) == ["Chapter 5 -- first LoRA fine-tune (16 examples)"]
    assert checkpoints["Chapter 5 -- first LoRA fine-tune (16 examples)"] == tmp_path / "chapter_05_lora"


def test_available_checkpoints_finds_the_newest_run_and_highest_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "CHAPTER_05_ADAPTER", tmp_path / "missing_05")
    monkeypatch.setattr(helpers, "CHAPTER_08_RUNS", tmp_path / "chapter_08")
    monkeypatch.setattr(helpers, "CHAPTER_13_RUNS", tmp_path / "missing_13")
    ch8 = tmp_path / "chapter_08"
    (ch8 / "run_20260101_000000" / "checkpoint_1").mkdir(parents=True)
    (ch8 / "run_20260101_000000" / "checkpoint_2").mkdir(parents=True)
    (ch8 / "run_20260201_000000" / "checkpoint_1").mkdir(parents=True)

    checkpoints = helpers.available_checkpoints()

    label = "Chapter 8 -- fine-tuned at scale (669 examples)"
    assert checkpoints[label] == ch8 / "run_20260201_000000" / "checkpoint_1"


def test_score_against_reference_matches_chapter_11s_own_scoring():
    from eval_finetuned_model import exact_match
    from traceable_outputs import faithfulness_score

    generated = "Trip out of hole with BHA #18."
    expected = "Trip out of hole with BHA #18. Stop at 5,800' and circulate."

    score = helpers.score_against_reference(generated, expected)

    assert score["exact_match"] == exact_match(generated, expected)
    assert score["overlap_score"] == faithfulness_score(generated, expected)


@pytest.mark.slow
def test_generate_answer_returns_nonempty_text():
    model, tokenizer = helpers.load_base_model()

    answer = helpers.generate_answer(
        model,
        tokenizer,
        "What happened on this well during this time window?",
        "Well: FORGE 16A [78]-32 | Report #37 | Time: 20:30-21:30",
    )

    assert isinstance(answer, str) and answer.strip()


@pytest.mark.slow
def test_evaluate_checkpoint_returns_the_three_real_metrics_in_range():
    checkpoints = helpers.available_checkpoints()
    if not checkpoints:
        pytest.skip("No fine-tuned checkpoint found -- run Chapter 5, 8, or 13's script first")
    label, checkpoint_dir = next(iter(checkpoints.items()))
    eval_set = helpers.build_held_out_eval_set()

    summary = helpers.evaluate_checkpoint(label, checkpoint_dir, eval_set)

    assert set(summary) == {"exact_match", "avg_overlap", "perplexity"}
    assert 0 <= summary["exact_match"] <= len(eval_set)
    assert 0.0 <= summary["avg_overlap"] <= 1.0
    assert summary["perplexity"] > 0
