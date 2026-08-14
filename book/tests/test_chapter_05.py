"""Tests for Chapter 5: Your First LoRA Fine-Tune.

Marked `slow`: exercises the base model loaded in Chapter 1. These tests
check the fine-tuning *mechanism* on a tiny subset (a couple of examples,
a few epochs) -- they do not reproduce the full chapter's 16-example,
20-epoch run, which takes several minutes. That run's real numbers are
documented (and reproducible) in chapters/chapter_05.qmd itself.

Skip locally if you're offline or on modest hardware:

    pytest -v -m "not slow"

`first_lora_finetune` imports `peft` at module load time, which would
otherwise turn a missing `peft` into a hard collection error for this
whole file -- breaking `pytest -m "not slow"` even though none of that
file's `not slow` tests actually need it. `importorskip` turns a
missing `peft` into a clean skip instead.
"""

import os

# Must happen before peft (which imports transformers) -- see
# load_local_model.py's own note on the same guard.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import pytest

peft = pytest.importorskip("peft")

import torch  # noqa: E402

from build_training_examples import build_training_examples  # noqa: E402
from load_local_model import MODEL_NAME, load_model_and_tokenizer  # noqa: E402
from first_lora_finetune import build_lora_model, build_training_ids, fine_tune, score  # noqa: E402


@pytest.fixture(scope="module")
def loaded_model():
    return load_model_and_tokenizer(MODEL_NAME)


def test_score_counts_matches_out_of_total():
    results = [{"matched_expected": True}, {"matched_expected": False}, {"matched_expected": True}]
    assert score(results) == (2, 3)


@pytest.mark.slow
def test_build_training_ids_masks_only_the_prompt(loaded_model):
    _, tokenizer = loaded_model
    example = {
        "instruction": "What are the present operations reported on this well?",
        "input": "Well: FORGE 16A [78]-32 | Report #38 | Date: 2020-11-26",
        "output": "PIPE FREE, TRIP OUT OF HOLE FOR BHA INSPECTION",
    }

    input_ids, labels = build_training_ids(tokenizer, example)

    assert input_ids.shape == labels.shape
    masked_length = (labels[0] == -100).sum().item()
    assert 0 < masked_length < input_ids.shape[1]
    # Everything after the masked prefix should be trained on, i.e. match
    # the real input_ids (not masked out).
    assert torch.equal(labels[0, masked_length:], input_ids[0, masked_length:])


@pytest.mark.slow
def test_build_lora_model_only_trains_a_small_fraction_of_parameters(loaded_model):
    model, _ = loaded_model
    lora_model = build_lora_model(model)

    trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_model.parameters())

    assert 0 < trainable < total
    assert trainable / total < 0.01  # LoRA should touch well under 1% of parameters


@pytest.mark.slow
def test_fine_tune_runs_and_reduces_loss_on_a_tiny_subset(loaded_model):
    model, tokenizer = loaded_model
    examples = build_training_examples()[:2]
    lora_model = build_lora_model(model)

    def mean_loss() -> float:
        lora_model.eval()
        with torch.no_grad():
            losses = [lora_model(input_ids=ids, labels=lbl).loss.item() for ids, lbl in map(lambda e: build_training_ids(tokenizer, e), examples)]
        return sum(losses) / len(losses)

    loss_before = mean_loss()
    fine_tune(lora_model, tokenizer, examples, num_epochs=5)
    loss_after = mean_loss()

    assert loss_after < loss_before
