"""Tests for Chapter 1: Loading and Running Your First Local LLM.

Marked `slow`: the first run downloads Qwen/Qwen2.5-1.5B-Instruct
(a few GB) and every run loads it into memory. Skip locally if you're
offline or on modest hardware:

    pytest -v -m "not slow"
"""

import pytest

from load_local_model import MODEL_NAME, generate_reply, load_model_and_tokenizer


@pytest.fixture(scope="module")
def loaded_model():
    return load_model_and_tokenizer(MODEL_NAME)


@pytest.mark.slow
def test_load_model_and_tokenizer_returns_usable_pair(loaded_model):
    model, tokenizer = loaded_model
    assert model is not None
    assert tokenizer is not None
    # A chat template is required for generate_reply()'s prompt formatting.
    assert tokenizer.chat_template is not None


@pytest.mark.slow
def test_generate_reply_returns_nonempty_text(loaded_model):
    model, tokenizer = loaded_model
    reply = generate_reply(model, tokenizer, "Say hello in one short sentence.", max_new_tokens=30)
    assert isinstance(reply, str)
    assert len(reply.strip()) > 0


@pytest.mark.slow
def test_generate_reply_is_deterministic_with_greedy_decoding(loaded_model):
    model, tokenizer = loaded_model
    question = "What does MD stand for in a drilling report?"
    first = generate_reply(model, tokenizer, question, max_new_tokens=20)
    second = generate_reply(model, tokenizer, question, max_new_tokens=20)
    assert first == second
