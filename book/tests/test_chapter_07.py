"""Tests for Chapter 7: Formatting and Chunking a Training Set at Scale.

No model needed -- everything here is plain PDF text extraction and
string processing, so none of this file is marked slow.
"""

import pytest

from build_training_examples import extract_text
from data_quality_gate import FULL_SET_DIR
from format_training_chunks import (
    build_timeline_examples_for_report,
    build_training_set_at_scale,
    chunk_text,
    contains_cid_artifact,
    parse_timeline_entries,
)

STUCK_PIPE_REPORT = FULL_SET_DIR / "FORGE-16A-78-32_Drilling_038_2020-11-26.pdf"


def test_chunk_text_returns_whole_text_when_under_the_limit():
    assert chunk_text("short text", max_chars=300) == ["short text"]


def test_chunk_text_splits_at_word_boundaries_not_mid_word():
    text = "one two three four five six seven eight nine ten"
    chunks = chunk_text(text, max_chars=15)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 15
        assert not chunk.startswith(" ") and not chunk.endswith(" ")
    assert " ".join(chunks) == text


def test_contains_cid_artifact_detects_undecoded_glyphs():
    assert contains_cid_artifact("Time(cid:0)WOB(K) Rotary") is True
    assert contains_cid_artifact("Normal operational text") is False


def test_parse_timeline_entries_finds_the_real_stuck_pipe_entry():
    text = extract_text(STUCK_PIPE_REPORT)
    entries = parse_timeline_entries(text)

    stuck_entry = next(e for e in entries if e["from_time"] == "23:30" and e["to_time"] == "04:00")
    assert "lost tool face and became" in stuck_entry["text"]
    assert "stuck" in stuck_entry["text"]


def test_build_timeline_examples_for_report_matches_real_run():
    examples, artifacts_filtered = build_timeline_examples_for_report(STUCK_PIPE_REPORT)

    assert len(examples) == 10  # 10 timeline entries in this report, none needed chunking
    assert artifacts_filtered == 0
    for example in examples:
        assert set(example.keys()) == {"instruction", "input", "output"}
        assert "Report #38" in example["input"]


def test_build_training_set_at_scale_never_includes_the_held_out_report():
    examples, _, _ = build_training_set_at_scale()

    assert all("Report #37 " not in example["input"] for example in examples)


def test_build_training_set_at_scale_matches_real_run():
    examples, skipped, artifacts_filtered = build_training_set_at_scale()

    assert skipped == ["FORGE-16A-78-32_Completion_003_2021-01-06.pdf"]
    assert artifacts_filtered == 3
    assert len(examples) == 669

    chunked = sum(1 for e in examples if "part" in e["input"])
    assert chunked == 125
