"""Chapter 13 challenge exercise: does the updated model regress on a
held-out report from the new batch itself, not just report #37?

Reference solution. Holds report #65 (2020-12-23, from the "new"
batch) out of the continued-training step the same way report #37 is
held out everywhere else, then evaluates the updated model against it.

Usage:
    python code/chapter_13/challenge/challenge.py
"""

import sys
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parents[3]
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
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_13"))

# load_local_model sets USE_TF=0/USE_FLAX=0 before its own transformers
# import -- must stay first.
from load_local_model import MODEL_NAME, load_model_and_tokenizer  # noqa: E402

from data_quality_gate import FULL_SET_DIR  # noqa: E402
from detect_model_drift import summarize_version  # noqa: E402
from finetune_at_scale import MetricsLogger, load_checkpoint, save_checkpoint, train_with_checkpoints  # noqa: E402
from format_training_chunks import build_timeline_examples_for_report  # noqa: E402
from continuous_finetune import RUNS_DIR, build_examples, split_archive_by_cutoff  # noqa: E402

NEW_BATCH_HELD_OUT_REPORT = "FORGE-16A-78-32_Drilling_065_2020-12-23.pdf"


def main() -> None:
    current_reports, new_reports = split_archive_by_cutoff()
    new_reports_minus_held_out = [r for r in new_reports if r["file"] != NEW_BATCH_HELD_OUT_REPORT]
    new_examples = build_examples(new_reports_minus_held_out)
    new_batch_held_out_examples, _ = build_timeline_examples_for_report(FULL_SET_DIR / NEW_BATCH_HELD_OUT_REPORT)
    print(f"New batch, held-out report #65 excluded: {len(new_examples)} training examples")
    print(f"Report #65 held-out check: {len(new_batch_held_out_examples)} examples")

    runs = sorted(RUNS_DIR.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No runs in {RUNS_DIR} -- run code/chapter_13/continuous_finetune.py first")
    current_checkpoint = runs[-1] / "checkpoint_2"

    run_dir = runs[-1] / "challenge"
    run_dir.mkdir(exist_ok=True)
    logger = MetricsLogger(run_dir)

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    lora_model = load_checkpoint(model, current_checkpoint)
    train_with_checkpoints(lora_model, tokenizer, new_examples, new_batch_held_out_examples, run_dir, logger, num_epochs=2)
    updated_checkpoint = save_checkpoint(lora_model, run_dir, epoch=2)

    summary = summarize_version(lora_model, tokenizer, new_batch_held_out_examples)
    print(f"\nUpdated model, evaluated against report #65 (never in this training run): {summary}")


if __name__ == "__main__":
    main()
