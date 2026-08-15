"""Tests for Chapter 9: Hybrid System — Combining Fine-Tuning with Retrieval.

Corpus-building and retrieval-logic tests need no model and aren't
marked slow. Generation tests are marked `slow` (need the base model);
the one test that needs Chapter 8's actual fine-tuned checkpoint skips
itself if no checkpoint exists (checkpoints/ is gitignored -- CI and a
fresh clone won't have run Chapter 8's ~30-minute script).

    pytest -v -m "not slow"
"""

import pytest

from hybrid_rag_finetune import (
    TEST_CASES,
    build_bm25_index,
    build_grounded_prompt,
    build_retrieval_corpus,
    latest_checkpoint,
    retrieve,
)
from load_local_model import MODEL_NAME, load_model_and_tokenizer


@pytest.fixture(scope="module")
def loaded_model():
    return load_model_and_tokenizer(MODEL_NAME)


def test_build_retrieval_corpus_includes_the_held_out_report():
    corpus = build_retrieval_corpus()

    assert len(corpus) == 677
    assert any(c["report_num"] == 37 for c in corpus)  # held out from training, not from retrieval


def test_retrieve_returns_k_results_ranked_by_score():
    corpus = [
        {"report_num": 1, "from_time": "06:00", "to_time": "07:00", "text": "circulate for temperature"},
        {"report_num": 2, "from_time": "06:00", "to_time": "07:00", "text": "drilling ahead with BHA"},
        {"report_num": 3, "from_time": "06:00", "to_time": "07:00", "text": "rig up Schlumberger to run logs"},
    ]
    bm25 = build_bm25_index(corpus)

    results = retrieve("circulate temperature", corpus, bm25, k=2)

    assert len(results) == 2
    assert results[0]["report_num"] == 1
    assert results[0]["score"] >= results[1]["score"]


def test_build_grounded_prompt_includes_instruction_input_and_sources():
    retrieved = [{"report_num": 5, "from_time": "10:00", "to_time": "11:00", "text": "trip out of hole"}]

    prompt = build_grounded_prompt("What happened?", "Well: X | Report #5", retrieved)

    assert "What happened?" in prompt
    assert "Well: X | Report #5" in prompt
    assert "Report #5" in prompt
    assert "trip out of hole" in prompt


def test_bm25_beats_naive_metadata_query_on_the_held_out_report():
    # The specific failure mode this chapter's design fixed: using the
    # templated instruction/input itself (near-identical across hundreds
    # of chunks) as the retrieval query finds nothing useful; a real
    # information-need query does.
    corpus = build_retrieval_corpus()
    bm25 = build_bm25_index(corpus)

    metadata_query = "What happened on this well during this time window? Well: FORGE 16A [78]-32 | Report #37 | Time: 20:30-21:30"
    real_query = "trip out of hole stop at 5800 circulate cool hole and tools"

    metadata_hit = 37 in [r["report_num"] for r in retrieve(metadata_query, corpus, bm25, k=3)]
    real_hit = 37 in [r["report_num"] for r in retrieve(real_query, corpus, bm25, k=3)]

    assert metadata_hit is False
    assert real_hit is True


@pytest.mark.slow
def test_retrieval_finds_the_correct_report_for_all_test_cases():
    corpus = build_retrieval_corpus()
    bm25 = build_bm25_index(corpus)

    for _label, _input_context, query, target_report in TEST_CASES:
        top_reports = [r["report_num"] for r in retrieve(query, corpus, bm25, k=3)]
        assert target_report in top_reports


@pytest.mark.slow
def test_answer_with_retrieval_cites_its_sources(loaded_model):
    from hybrid_rag_finetune import answer_with_retrieval

    model, tokenizer = loaded_model
    corpus = build_retrieval_corpus()
    bm25 = build_bm25_index(corpus)

    result = answer_with_retrieval(
        model,
        tokenizer,
        "What happened on this well during this time window?",
        "Well: FORGE 16A [78]-32 | Report #37 | Time: 20:30-21:30",
        "trip out of hole stop at 5800 circulate cool hole and tools",
        corpus,
        bm25,
    )

    assert isinstance(result["answer"], str) and result["answer"].strip()
    assert result["sources"][0]["report_num"] == 37


@pytest.mark.slow
def test_chapter_08_checkpoint_reproduces_its_grounded_answer():
    try:
        checkpoint_dir = latest_checkpoint()
    except FileNotFoundError:
        pytest.skip("No Chapter 8 checkpoint found -- run code/chapter_08/finetune_at_scale.py first")

    from peft import PeftModel

    from hybrid_rag_finetune import answer_with_retrieval

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    lora_model = PeftModel.from_pretrained(model, checkpoint_dir)
    corpus = build_retrieval_corpus()
    bm25 = build_bm25_index(corpus)

    result = answer_with_retrieval(
        lora_model,
        tokenizer,
        "What happened on this well during this time window?",
        "Well: FORGE 16A [78]-32 | Report #37 | Time: 20:30-21:30",
        "trip out of hole stop at 5800 circulate cool hole and tools",
        corpus,
        bm25,
    )

    assert "5,800" in result["answer"] or "5800" in result["answer"]
