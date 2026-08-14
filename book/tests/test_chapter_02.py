"""Tests for Chapter 2: Turning Drilling & Completions Reports into Training Examples."""

import json

from build_training_examples import (
    HELD_OUT_REPORT,
    SAMPLE_SET_DIR,
    build_examples_for_report,
    build_training_examples,
    extract_fields,
    extract_text,
    normalize_date,
    save_examples_jsonl,
)

STUCK_PIPE_REPORT = SAMPLE_SET_DIR / "FORGE-16A-78-32_Drilling_038_2020-11-26.pdf"
COMPLETION_REPORT = SAMPLE_SET_DIR / "FORGE-16A-78-32_Completion_003_2021-01-06.pdf"


def test_extract_fields_from_real_stuck_pipe_report():
    text = extract_text(STUCK_PIPE_REPORT)
    fields = extract_fields(text)

    assert fields["well_name"] == "FORGE 16A [78]-32"
    assert fields["rpt_num"] == "38"
    assert fields["present_operations"] == "PIPE FREE, TRIP OUT OF HOLE FOR BHA INSPECTION"
    assert fields["activity_planned"] == "INSPECT BHA, TRIP IN HOLE DRILL CURVE"


def test_normalize_date():
    assert normalize_date("11/26/2020") == "2020-11-26"


def test_build_examples_for_report_returns_two_examples():
    examples = build_examples_for_report(STUCK_PIPE_REPORT)

    assert len(examples) == 2
    for example in examples:
        assert set(example.keys()) == {"instruction", "input", "output"}
        assert "Report #38" in example["input"]
        assert "2020-11-26" in example["input"]


def test_build_examples_for_report_skips_unrecognized_layout():
    # The completion report uses different field labels ("Present Ops:",
    # "Next 24 Hours:") than the drilling-report patterns this chapter's
    # main script handles -- see the challenge exercise.
    assert build_examples_for_report(COMPLETION_REPORT) == []


def test_build_training_examples_over_sample_set():
    examples = build_training_examples()

    # 9 of the 10 sample reports use the drilling-report layout; the
    # completion report is skipped, and 1 drilling report (HELD_OUT_REPORT)
    # is deliberately reserved for Chapter 5's held-out generalization
    # check -- leaving 8 reports, 16 examples.
    assert len(examples) == 16
    outputs = [example["output"] for example in examples]
    assert "PIPE FREE, TRIP OUT OF HOLE FOR BHA INSPECTION" in outputs


def test_build_training_examples_never_includes_the_held_out_report():
    examples = build_training_examples()

    assert all("Report #37" not in example["input"] for example in examples)


def test_held_out_report_still_has_two_extractable_examples():
    # Reserved from training, but still a normal, recognized drilling
    # report -- Chapter 5 needs its examples for the generalization check.
    examples = build_examples_for_report(SAMPLE_SET_DIR / HELD_OUT_REPORT)

    assert len(examples) == 2
    for example in examples:
        assert "Report #37" in example["input"]


def test_save_examples_jsonl_round_trips(tmp_path):
    examples = build_training_examples()
    output_path = tmp_path / "examples.jsonl"

    save_examples_jsonl(examples, output_path)

    loaded = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert loaded == examples
