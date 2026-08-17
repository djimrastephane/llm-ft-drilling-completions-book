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


def test_parse_input_context_extracts_report_date_and_time_window():
    parsed = helpers.parse_input_context(
        "Well: FORGE 16A [78]-32 | Report #76 | Date: 2021-01-03 | Time: 16:00-21:30 (part 1 of 2)"
    )

    assert parsed == {
        "well_name": "FORGE 16A [78]-32",
        "report_num": 76,
        "date": "2021-01-03",
        "time_window": "16:00-21:30",
        "chunk_part": "1/2",
    }


def test_parse_input_context_handles_the_chapter_2_style_with_no_time_window():
    parsed = helpers.parse_input_context("Well: FORGE 16A [78]-32 | Report #3 | Date: 2020-10-22")

    assert parsed["report_num"] == 3
    assert parsed["time_window"] is None
    assert parsed["chunk_part"] is None


def test_parse_input_context_returns_all_none_for_unrecognized_text():
    parsed = helpers.parse_input_context("not a real input string")

    assert parsed == {"well_name": None, "report_num": None, "date": None, "time_window": None, "chunk_part": None}


def test_dataset_examples_sample_bucket_matches_chapter_2s_real_count():
    examples = helpers.dataset_examples("sample")

    assert len(examples) == 16
    assert all(e["report_num"] is not None for e in examples)
    assert not any(e["held_out"] for e in examples)  # Chapter 2 never includes report #37


def test_dataset_examples_held_out_bucket_is_all_report_37():
    examples = helpers.dataset_examples("held_out")

    assert len(examples) == 8
    assert all(e["report_num"] == 37 and e["held_out"] for e in examples)


def test_documented_findings_cover_every_real_retrieval_test_case():
    test_case_labels = {label for label, *_ in helpers.RETRIEVAL_TEST_CASES}

    assert set(helpers.DOCUMENTED_FINDINGS) == test_case_labels
    for kind, finding in helpers.DOCUMENTED_FINDINGS.values():
        assert kind in {"success", "failure", "blind spot"}
        assert finding.strip()


def test_pairwise_answer_similarity_ranks_near_identical_answers_highest():
    pairs = helpers.pairwise_answer_similarity(
        {
            "q1": "Trip out of hole with BHA #18.",
            "q2": "Trip out of hole with BHA #19.",
            "q3": "Rig up Schlumberger to run logs.",
        }
    )

    assert len(pairs) == 3
    top = pairs[0]
    assert {top["question_a"], top["question_b"]} == {"q1", "q2"}
    assert top["similarity"] > 0.5


def test_pairwise_answer_similarity_is_symmetric_regardless_of_input_order():
    forward = helpers.pairwise_answer_similarity({"a": "trip out of hole", "b": "trip out of hole again"})
    backward = helpers.pairwise_answer_similarity({"b": "trip out of hole again", "a": "trip out of hole"})

    assert forward[0]["similarity"] == backward[0]["similarity"]


def test_short_version_label_shortens_base_and_chapter_labels():
    assert helpers.short_version_label("Base model (no fine-tuning)") == "Base"
    assert helpers.short_version_label("Chapter 5 -- first LoRA fine-tune (16 examples)") == "Ch5"
    assert helpers.short_version_label("Chapter 13 -- continuous fine-tune (latest)") == "Ch13"


def test_relative_change_matches_hand_computed_percentages():
    assert helpers.relative_change(159.91, 112.28) == pytest.approx(-0.2979, abs=1e-3)
    assert helpers.relative_change(112.28, 25.03) == pytest.approx(-0.7771, abs=1e-3)
    assert helpers.relative_change(25.03, 25.73) == pytest.approx(0.0280, abs=1e-3)


def test_relative_change_is_none_when_starting_from_zero():
    assert helpers.relative_change(0.0, 0.562) is None


def test_evaluation_snapshot_finds_the_real_best_version_per_metric():
    summaries = {
        "Base model (no fine-tuning)": {"exact_match": 0, "avg_overlap": 0.0, "perplexity": 159.91},
        "Chapter 5 -- first LoRA fine-tune (16 examples)": {"exact_match": 0, "avg_overlap": 0.562, "perplexity": 112.28},
        "Chapter 8 -- fine-tuned at scale (669 examples)": {"exact_match": 0, "avg_overlap": 0.167, "perplexity": 25.03},
        "Chapter 13 -- continuous fine-tune (latest)": {"exact_match": 0, "avg_overlap": 0.125, "perplexity": 25.73},
    }

    snapshot = helpers.evaluation_snapshot(summaries)

    assert snapshot["best_perplexity"] == (["Chapter 8 -- fine-tuned at scale (669 examples)"], 25.03)
    assert snapshot["best_overlap"] == (["Chapter 5 -- first LoRA fine-tune (16 examples)"], 0.562)
    assert len(snapshot["best_exact_match"][0]) == 4  # a real tie -- every version scored 0
    assert snapshot["latest"] == "Chapter 13 -- continuous fine-tune (latest)"


def test_latest_regressed_on_both_is_true_for_the_books_own_ch8_to_ch13_transition():
    summaries = {
        "Chapter 8 -- fine-tuned at scale (669 examples)": {"exact_match": 0, "avg_overlap": 0.167, "perplexity": 25.03},
        "Chapter 13 -- continuous fine-tune (latest)": {"exact_match": 0, "avg_overlap": 0.125, "perplexity": 25.73},
    }

    assert helpers.latest_regressed_on_both(summaries) is True


def test_latest_regressed_on_both_is_false_when_only_one_version_exists():
    assert helpers.latest_regressed_on_both({"Base model (no fine-tuning)": {"exact_match": 0, "avg_overlap": 0.0, "perplexity": 100.0}}) is False


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
