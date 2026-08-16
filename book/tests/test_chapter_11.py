"""Tests for Chapter 11: Evaluating a Fine-Tuned Domain Model.

Held-out-set construction and scoring logic need no model and aren't
marked slow. The full evaluation run against Chapter 8's real
checkpoint is marked slow, skipping itself if no checkpoint exists
(checkpoints/ is gitignored -- CI and a fresh clone won't have run
Chapter 8's ~30-minute script).

    pytest -v -m "not slow"
"""

import math

import pytest

from eval_finetuned_model import build_held_out_eval_set, evaluate, exact_match, perplexity
from hybrid_rag_finetune import latest_checkpoint
from load_local_model import MODEL_NAME, load_model_and_tokenizer


def test_build_held_out_eval_set_returns_real_examples_from_report_37():
    eval_set = build_held_out_eval_set()

    assert len(eval_set) == 8
    assert all("Report #37" in e["input"] for e in eval_set)
    assert all(e["output"].strip() for e in eval_set)


def test_exact_match_is_case_insensitive_substring_check():
    assert exact_match("Trip out of hole with BHA #18.", "trip out of hole") is True
    assert exact_match("Circulate to cool the tools.", "trip out of hole") is False


@pytest.fixture(scope="module")
def loaded_checkpoint():
    try:
        checkpoint_dir = latest_checkpoint()
    except FileNotFoundError:
        pytest.skip("No Chapter 8 checkpoint found -- run code/chapter_08/finetune_at_scale.py first")

    from peft import PeftModel

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    return PeftModel.from_pretrained(model, checkpoint_dir), tokenizer


@pytest.mark.slow
def test_evaluate_scores_every_example_in_the_held_out_set(loaded_checkpoint):
    lora_model, tokenizer = loaded_checkpoint
    eval_set = build_held_out_eval_set()

    results = evaluate(lora_model, tokenizer, eval_set)

    assert len(results) == len(eval_set)
    assert all(0.0 <= r["overlap_score"] <= 1.0 for r in results)
    assert all(isinstance(r["exact_match"], bool) for r in results)


@pytest.mark.slow
def test_fine_tuned_perplexity_is_lower_than_base_model_on_held_out_text(loaded_checkpoint):
    lora_model, tokenizer = loaded_checkpoint
    eval_set = build_held_out_eval_set()
    texts = [e["output"] for e in eval_set]

    base_model, _ = load_model_and_tokenizer(MODEL_NAME)
    ppl_base = perplexity(base_model, tokenizer, texts)
    ppl_finetuned = perplexity(lora_model, tokenizer, texts)

    assert ppl_finetuned < ppl_base
    assert ppl_finetuned > 0 and not math.isnan(ppl_finetuned)
