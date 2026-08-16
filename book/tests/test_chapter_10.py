"""Tests for Chapter 10: Traceable Outputs and Hallucination Mitigation.

Faithfulness-scoring logic needs no model and isn't marked slow. The
full end-to-end check against Chapter 9's real test cases needs Chapter
8's fine-tuned checkpoint and is marked slow, skipping itself if no
checkpoint exists (checkpoints/ is gitignored -- CI and a fresh clone
won't have run Chapter 8's ~30-minute script).

    pytest -v -m "not slow"
"""

import pytest

from hybrid_rag_finetune import (
    INSTRUCTION,
    TEST_CASES,
    build_bm25_index,
    build_retrieval_corpus,
    latest_checkpoint,
)
from load_local_model import MODEL_NAME, load_model_and_tokenizer
from traceable_outputs import answer_with_traceable_sources, content_words, faithfulness_score


def test_content_words_drops_stopwords_and_instruction_boilerplate():
    words = content_words("What happened on this well during this time window?")

    assert words == set()


def test_content_words_keeps_real_content():
    words = content_words("Trip out of hole with BHA #18")

    assert "trip" in words
    assert "bha" in words
    assert "with" not in words  # stopword


def test_faithfulness_score_is_1_when_every_answer_word_appears_in_source():
    score = faithfulness_score("circulate to cool the tools", "circulate to cool the directional tools at depth")

    assert score == 1.0


def test_faithfulness_score_is_0_for_unrelated_text():
    score = faithfulness_score("fishing operations with BHA 33", "drilling ahead with new bit at surface")

    assert score == 0.0


def test_faithfulness_score_is_0_for_empty_answer():
    assert faithfulness_score("", "some source text") == 0.0


def test_faithfulness_score_is_partial_for_partly_supported_answer():
    score = faithfulness_score("trip out of hole with fishing bha", "pick up fishing bha and trip in hole")

    assert 0.0 < score < 1.0


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
def test_traceable_sources_flags_the_stuck_pipe_answer_as_ungrounded(loaded_checkpoint):
    lora_model, tokenizer = loaded_checkpoint
    corpus = build_retrieval_corpus()
    bm25 = build_bm25_index(corpus)

    _label, input_context, query, _target = TEST_CASES[1]  # #38, stuck pipe -- Chapter 9 found this one unfaithful
    result = answer_with_traceable_sources(lora_model, tokenizer, INSTRUCTION, input_context, query, corpus, bm25)

    assert result["grounded"] is False
    assert result["verified_sources"] == []


@pytest.mark.slow
def test_traceable_sources_grounds_the_held_out_report_correctly(loaded_checkpoint):
    lora_model, tokenizer = loaded_checkpoint
    corpus = build_retrieval_corpus()
    bm25 = build_bm25_index(corpus)

    _label, input_context, query, target_report = TEST_CASES[0]  # #37, held-out
    result = answer_with_traceable_sources(lora_model, tokenizer, INSTRUCTION, input_context, query, corpus, bm25)

    assert result["grounded"] is True
    assert any(c["report_num"] == target_report for c in result["verified_sources"])
