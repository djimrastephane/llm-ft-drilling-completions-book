"""Tests for Chapter 4: Tokenization and Embeddings for Domain Fine-Tuning.

Marked `slow`: exercises the base model loaded in Chapter 1. Skip locally
if you're offline or on modest hardware:

    pytest -v -m "not slow"
"""

import pytest
import torch

from load_local_model import MODEL_NAME, load_model_and_tokenizer
from tokenize_and_embed import (
    cosine_similarity,
    raw_embedding_similarities,
    term_embedding,
    tokenize_with_pieces,
)


@pytest.fixture(scope="module")
def loaded_model():
    return load_model_and_tokenizer(MODEL_NAME)


def test_cosine_similarity_of_identical_vectors_is_one():
    vector = torch.tensor([1.0, 2.0, 3.0])
    assert cosine_similarity(vector, vector) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.slow
def test_tokenize_with_pieces_splits_oilfield_acronym_into_two_pieces(loaded_model):
    _, tokenizer = loaded_model
    assert tokenize_with_pieces(tokenizer, "BHA") == ["B", "HA"]


@pytest.mark.slow
def test_tokenize_with_pieces_keeps_common_word_whole(loaded_model):
    _, tokenizer = loaded_model
    assert tokenize_with_pieces(tokenizer, "pipe") == ["pipe"]


@pytest.mark.slow
def test_term_embedding_mean_pools_to_one_vector_per_term(loaded_model):
    model, tokenizer = loaded_model
    vector = term_embedding(model, tokenizer, "BHA")

    assert vector.shape == (model.get_input_embeddings().weight.shape[1],)


@pytest.mark.slow
def test_raw_embedding_similarity_matches_real_run(loaded_model):
    # Real run against the base model: true synonym phrasing ("trip out of
    # hole" vs "pull out of hole") clusters strongly in raw input
    # embeddings; an oilfield acronym and its own spelled-out meaning
    # ("BHA" vs "bottom hole assembly") does not, because they share no
    # tokens. Deterministic (no sampling) -- a fixed weight lookup, not a
    # generated answer -- so this is an exact-value regression check,
    # with a small tolerance for floating-point differences across
    # hardware/BLAS backends.
    model, tokenizer = loaded_model
    results = {
        (r["term_a"], r["term_b"]): r["cosine_similarity"]
        for r in raw_embedding_similarities(
            model, tokenizer, [("trip out of hole", "pull out of hole"), ("BHA", "bottom hole assembly")]
        )
    }

    assert results[("trip out of hole", "pull out of hole")] == pytest.approx(0.709, abs=0.01)
    assert results[("BHA", "bottom hole assembly")] == pytest.approx(-0.035, abs=0.01)
    assert (
        results[("trip out of hole", "pull out of hole")]
        > results[("BHA", "bottom hole assembly")]
    )
