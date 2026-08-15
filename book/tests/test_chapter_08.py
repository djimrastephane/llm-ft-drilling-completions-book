"""Tests for Chapter 8: Fine-Tuning at Scale with Checkpointing and Experiment Tracking.

Marked `slow`: exercises the base model loaded in Chapter 1. These
tests check the checkpointing/resume/logging *mechanism* on a tiny
subset (a handful of examples, 2-3 epochs) -- they do not reproduce the
full chapter's 669-example run, which takes about 30 minutes. That
run's real numbers are documented (and reproducible) in
chapters/chapter_08.qmd itself.

Skip locally if you're offline or on modest hardware:

    pytest -v -m "not slow"
"""

import pytest
import torch

from build_training_examples import HELD_OUT_REPORT
from data_quality_gate import FULL_SET_DIR
from finetune_at_scale import MetricsLogger, load_checkpoint, save_checkpoint, train_with_checkpoints
from first_lora_finetune import build_lora_model
from format_training_chunks import build_timeline_examples_for_report, build_training_set_at_scale
from load_local_model import MODEL_NAME, load_model_and_tokenizer


@pytest.fixture(scope="module")
def loaded_model():
    return load_model_and_tokenizer(MODEL_NAME)


def test_metrics_logger_writes_matching_csv_and_jsonl(tmp_path):
    logger = MetricsLogger(tmp_path)
    logger.log(step=5, epoch=1, train_loss=1.2345, held_out_exact_match="1/8", checkpoint="checkpoint_1")

    csv_text = logger.csv_path.read_text()
    jsonl_text = logger.jsonl_path.read_text()

    assert "epoch,train_loss" in csv_text
    assert "1.2345" in csv_text
    assert '"train_loss": 1.2345' in jsonl_text
    assert '"held_out_exact_match": "1/8"' in jsonl_text


def test_metrics_logger_accumulates_multiple_rows(tmp_path):
    logger = MetricsLogger(tmp_path)
    logger.log(step=5, epoch=1, train_loss=2.0, held_out_exact_match="0/8", checkpoint="checkpoint_1")
    logger.log(step=10, epoch=2, train_loss=1.0, held_out_exact_match="1/8", checkpoint="checkpoint_2")

    rows = logger.csv_path.read_text().splitlines()
    assert len(rows) == 3  # header + 2 rows


@pytest.mark.slow
def test_train_with_checkpoints_saves_a_checkpoint_per_epoch(loaded_model, tmp_path):
    model, tokenizer = loaded_model
    examples, _, _ = build_training_set_at_scale()
    held_out_examples, _ = build_timeline_examples_for_report(FULL_SET_DIR / HELD_OUT_REPORT)

    lora_model = build_lora_model(model)
    logger = MetricsLogger(tmp_path)
    train_with_checkpoints(lora_model, tokenizer, examples[:3], held_out_examples[:2], tmp_path, logger, num_epochs=2)

    assert (tmp_path / "checkpoint_1").is_dir()
    assert (tmp_path / "checkpoint_2").is_dir()
    assert (tmp_path / "checkpoint_1" / "adapter_model.safetensors").exists()

    rows = logger.csv_path.read_text().splitlines()
    assert len(rows) == 3  # header + 2 epochs


@pytest.mark.slow
def test_resuming_from_a_checkpoint_continues_training_not_restarts(loaded_model, tmp_path):
    # A real regression check: loss after resuming and training one more
    # epoch should be lower than loss at the checkpoint it resumed from
    # -- proof the resumed run continues learning, not starting over.
    model, tokenizer = loaded_model
    examples, _, _ = build_training_set_at_scale()
    held_out_examples, _ = build_timeline_examples_for_report(FULL_SET_DIR / HELD_OUT_REPORT)
    small_examples = examples[:3]

    lora_model = build_lora_model(model)
    logger = MetricsLogger(tmp_path)
    train_with_checkpoints(lora_model, tokenizer, small_examples, held_out_examples[:2], tmp_path, logger, num_epochs=2)
    loss_at_checkpoint_2 = float(logger._rows[-1]["train_loss"])

    del lora_model
    resumed_model, _ = load_model_and_tokenizer(MODEL_NAME)
    resumed_lora_model = load_checkpoint(resumed_model, tmp_path / "checkpoint_2")
    train_with_checkpoints(
        resumed_lora_model, tokenizer, small_examples, held_out_examples[:2], tmp_path, logger, num_epochs=1, start_epoch=2
    )
    loss_after_resume = float(logger._rows[-1]["train_loss"])

    assert loss_after_resume < loss_at_checkpoint_2
    assert (tmp_path / "checkpoint_3").is_dir()


@pytest.mark.slow
def test_save_checkpoint_then_load_checkpoint_reproduces_the_same_model(loaded_model, tmp_path):
    model, tokenizer = loaded_model
    lora_model = build_lora_model(model)
    checkpoint_dir = save_checkpoint(lora_model, tmp_path, epoch=1)

    reloaded = load_checkpoint(model, checkpoint_dir)

    original_params = dict(lora_model.named_parameters())
    for name, param in reloaded.named_parameters():
        if "lora" in name.lower():
            assert torch.equal(param, original_params[name])
