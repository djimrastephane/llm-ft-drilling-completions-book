"""Chapter 13: Continuous Fine-Tuning -- Keeping the Model Current.

Chapter 12 built the tool to compare two model versions; it never
asked where the new version comes from. This chapter closes that loop:
it simulates new reports arriving after an archive is already in
production (a real chronological split of this book's own 74
non-held-out reports, report #60 onward held back as "new"), trains a
"current" model on the archive as it stood before that point, continues
training on just the new reports the same way Chapter 8 already resumes
from a saved checkpoint, and runs Chapter 12's own comparison to decide
whether the updated model is actually worth deploying.

Usage:
    python code/chapter_13/continuous_finetune.py
"""

import sys
import time
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_01"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_02"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_03"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_05"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_06"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_07"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_08"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_09"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_10"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_11"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_12"))

# load_local_model sets USE_TF=0/USE_FLAX=0 before its own transformers
# import -- must stay first, see Chapter 3/5/8/... for why.
from load_local_model import MODEL_NAME, load_model_and_tokenizer  # noqa: E402

import torch  # noqa: E402

from build_training_examples import HELD_OUT_REPORT  # noqa: E402
from data_quality_gate import FULL_SET_DIR, build_archive_records  # noqa: E402
from detect_model_drift import compare_versions, load_version, summarize_version  # noqa: E402
from eval_finetuned_model import build_held_out_eval_set  # noqa: E402
from finetune_at_scale import MetricsLogger, load_checkpoint, save_checkpoint, train_with_checkpoints  # noqa: E402
from first_lora_finetune import build_lora_model  # noqa: E402
from format_training_chunks import build_timeline_examples_for_report  # noqa: E402

RUNS_DIR = BOOK_ROOT / "checkpoints" / "chapter_13"
CUTOFF_REPORT_NUM = 60  # reports before this simulate "already in production"; at/after simulate "just arrived"


def new_run_dir() -> Path:
    run_dir = RUNS_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def split_archive_by_cutoff(cutoff: int = CUTOFF_REPORT_NUM) -> tuple[list[dict], list[dict]]:
    """Chronological split of this book's real, quality-gated archive.
    Held-out report #37 is excluded from both halves, same as
    everywhere else in this book -- retraining on new data doesn't
    change what "held out" means.
    """
    records = build_archive_records()
    ok = [r for r in records if r["status"] == "ok" and r["file"] != HELD_OUT_REPORT]
    current_reports = [r for r in ok if r["rpt_num"] < cutoff]
    new_reports = [r for r in ok if r["rpt_num"] >= cutoff]
    return current_reports, new_reports


def build_examples(reports: list[dict]) -> list[dict]:
    examples = []
    for record in reports:
        report_examples, _artifacts_filtered = build_timeline_examples_for_report(FULL_SET_DIR / record["file"])
        examples.extend(report_examples)
    return examples


def main() -> None:
    torch.manual_seed(0)

    current_reports, new_reports = split_archive_by_cutoff()
    current_examples = build_examples(current_reports)
    new_examples = build_examples(new_reports)
    print(f"'Currently in production': {len(current_reports)} reports, {len(current_examples)} training examples")
    print(f"New reports just arrived:  {len(new_reports)} reports, {len(new_examples)} training examples")

    held_out_eval_set = build_held_out_eval_set()
    run_dir = new_run_dir()
    logger = MetricsLogger(run_dir)
    print(f"Run directory: {run_dir}")

    print("\nTraining the current model on the archive as it stood before the new reports arrived...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    lora_model = build_lora_model(model)
    train_with_checkpoints(lora_model, tokenizer, current_examples, held_out_eval_set, run_dir, logger, num_epochs=2)
    current_checkpoint = run_dir / "checkpoint_2"

    print("\nNew reports have arrived. Continuing training on just the new batch...")
    del model, lora_model
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    lora_model = load_checkpoint(model, current_checkpoint)
    train_with_checkpoints(lora_model, tokenizer, new_examples, held_out_eval_set, run_dir, logger, num_epochs=2, start_epoch=2)
    updated_checkpoint = save_checkpoint(lora_model, run_dir, epoch=4)

    print("\nComparing the current model against the updated one...")
    # load_version (not load_checkpoint) here deliberately: load_checkpoint
    # passes is_trainable=True, which leaves the model in train mode (so
    # dropout stays active) -- fine for resuming training above, wrong for
    # evaluation, where it makes perplexity noisy run to run for no reason.
    lora_current, tokenizer = load_version(current_checkpoint)
    summary_current = summarize_version(lora_current, tokenizer, held_out_eval_set)

    lora_updated, _ = load_version(updated_checkpoint)
    summary_updated = summarize_version(lora_updated, tokenizer, held_out_eval_set)

    print(f"Current model: {summary_current}")
    print(f"Updated model: {summary_updated}")
    print(f"Direction:     {compare_versions(summary_current, summary_updated)}")


if __name__ == "__main__":
    main()
