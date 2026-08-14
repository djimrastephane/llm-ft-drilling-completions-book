"""Tests for Chapter 3: Baseline Prompting -- What the Model Gets Wrong Before Fine-Tuning.

Marked `slow`: exercises the base model loaded in Chapter 1. Skip locally
if you're offline or on modest hardware:

    pytest -v -m "not slow"
"""

import json

import pytest

from baseline_prompting import (
    ask_baseline,
    build_prompt,
    run_baseline,
    save_results_jsonl,
)
from build_training_examples import build_training_examples
from load_local_model import MODEL_NAME, load_model_and_tokenizer

EXAMPLE = {
    "instruction": "What are the present operations reported on this well?",
    "input": "Well: FORGE 16A [78]-32 | Report #38 | Date: 2020-11-26",
    "output": "PIPE FREE, TRIP OUT OF HOLE FOR BHA INSPECTION",
}


@pytest.fixture(scope="module")
def loaded_model():
    return load_model_and_tokenizer(MODEL_NAME)


def test_build_prompt_combines_instruction_and_input():
    prompt = build_prompt(EXAMPLE)

    assert prompt == (
        "What are the present operations reported on this well?\n"
        "Well: FORGE 16A [78]-32 | Report #38 | Date: 2020-11-26"
    )


def test_build_prompt_never_leaks_the_expected_output():
    # The whole point of a baseline: the prompt must not contain the
    # answer, or the "test" would be meaningless.
    prompt = build_prompt(EXAMPLE)
    assert EXAMPLE["output"] not in prompt


def test_save_results_jsonl_round_trips(tmp_path):
    results = [
        {
            "instruction": EXAMPLE["instruction"],
            "input": EXAMPLE["input"],
            "expected": EXAMPLE["output"],
            "base_model_answer": "some generated text",
            "matched_expected": False,
        }
    ]
    output_path = tmp_path / "baseline_results.jsonl"

    save_results_jsonl(results, output_path)

    loaded = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert loaded == results


@pytest.mark.slow
def test_ask_baseline_returns_expected_shape(loaded_model):
    model, tokenizer = loaded_model
    result = ask_baseline(model, tokenizer, EXAMPLE, max_new_tokens=20)

    assert set(result.keys()) == {
        "instruction",
        "input",
        "expected",
        "base_model_answer",
        "matched_expected",
    }
    assert result["expected"] == EXAMPLE["output"]
    assert isinstance(result["base_model_answer"], str)
    assert len(result["base_model_answer"].strip()) > 0
    assert isinstance(result["matched_expected"], bool)


@pytest.mark.slow
def test_run_baseline_over_sample_set_never_matches_without_report_text(loaded_model):
    # The base model is only ever given each report's metadata here, never
    # its actual text -- so it has no way to produce the exact reported
    # value. A real run against this book's sample archive confirms 0/18.
    model, tokenizer = loaded_model
    examples = build_training_examples()

    results = run_baseline(model, tokenizer, examples)

    assert len(results) == len(examples) == 18
    assert sum(r["matched_expected"] for r in results) == 0
