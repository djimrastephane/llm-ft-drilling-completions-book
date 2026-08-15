"""Chapter 8: Fine-Tuning at Scale with Checkpointing and Experiment Tracking.

Fine-tunes LoRA on Chapter 7's full 669-example training set (instead
of Chapter 5's 16), saving a checkpoint every epoch and logging metrics
to plain CSV/JSONL files -- transparent, portable, and readable in any
spreadsheet, no separate tool to install. Simulates a real crash and
restart partway through training: the base model is fully reloaded and
training resumes purely from a saved checkpoint on disk, not from
anything still held in memory.

The held-out check reuses Chapter 2's single held-out report,
Drilling_037, split by report exactly as it has been since Chapter 2 --
never by individual training example, which would let two chunks of
the same report land on both sides of the split and quietly leak
answers across it.

Usage:
    python code/chapter_08/finetune_at_scale.py
"""

import csv
import json
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

# load_local_model sets USE_TF=0/USE_FLAX=0 before its own transformers
# import -- see Chapter 3/5 for why this has to happen first.
from load_local_model import MODEL_NAME, load_model_and_tokenizer  # noqa: E402

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402

from baseline_prompting import run_baseline  # noqa: E402
from build_training_examples import HELD_OUT_REPORT  # noqa: E402
from data_quality_gate import FULL_SET_DIR  # noqa: E402
from first_lora_finetune import build_lora_model, build_training_ids, score  # noqa: E402
from format_training_chunks import build_timeline_examples_for_report, build_training_set_at_scale  # noqa: E402

RUNS_DIR = BOOK_ROOT / "checkpoints" / "chapter_08"
LEARNING_RATE = 2e-4
TRAINING_SAMPLE_SIZE = 50  # exact-match over all 669 examples takes ~30 min; a fixed sample stays honest and fast


def new_run_dir() -> Path:
    run_dir = RUNS_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_checkpoint(lora_model, run_dir: Path, epoch: int) -> Path:
    checkpoint_dir = run_dir / f"checkpoint_{epoch}"
    lora_model.save_pretrained(checkpoint_dir)
    return checkpoint_dir


def load_checkpoint(model, checkpoint_dir: Path):
    """Reload a LoRA adapter from disk onto a freshly loaded base model."""
    return PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)


class MetricsLogger:
    """Plain CSV + JSONL metrics -- transparent, portable, no extra tool required.

    A production setup might send these same rows to TensorBoard,
    Weights & Biases, MLflow, or an internal dashboard instead; a CSV a
    reader can open directly in Excel is the right starting point here.
    """

    FIELDS = ["step", "epoch", "train_loss", "held_out_exact_match", "checkpoint"]

    def __init__(self, run_dir: Path):
        self.csv_path = run_dir / "metrics.csv"
        self.jsonl_path = run_dir / "metrics.jsonl"
        self._rows: list[dict] = []

    def log(self, **kwargs) -> None:
        self._rows.append({field: kwargs.get(field, "") for field in self.FIELDS})
        self._write()

    def _write(self) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self._rows)
        with self.jsonl_path.open("w", encoding="utf-8") as f:
            for row in self._rows:
                f.write(json.dumps(row) + "\n")


def fine_tune_epoch(lora_model, tokenizer, examples: list[dict], optimizer) -> float:
    lora_model.train()
    epoch_loss = 0.0
    for example in examples:
        input_ids, labels = build_training_ids(tokenizer, example)
        loss = lora_model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
    lora_model.eval()
    return epoch_loss / len(examples)


def train_with_checkpoints(
    lora_model, tokenizer, examples: list[dict], held_out_examples: list[dict], run_dir: Path, logger: MetricsLogger, num_epochs: int, start_epoch: int = 0
) -> None:
    # A fresh optimizer every call, deliberately: this simple version
    # doesn't persist Adam's momentum across a resume, only the model
    # weights -- see this chapter's Production Reality section.
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=LEARNING_RATE)
    for i in range(num_epochs):
        epoch = start_epoch + i + 1
        t0 = time.time()
        avg_loss = fine_tune_epoch(lora_model, tokenizer, examples, optimizer)
        checkpoint_dir = save_checkpoint(lora_model, run_dir, epoch)
        matched, total = score(run_baseline(lora_model, tokenizer, held_out_examples))
        logger.log(
            step=epoch * len(examples),
            epoch=epoch,
            train_loss=round(avg_loss, 4),
            held_out_exact_match=f"{matched}/{total}",
            checkpoint=checkpoint_dir.name,
        )
        print(f"  epoch {epoch}: avg loss {avg_loss:.4f}, held-out {matched}/{total} ({time.time() - t0:.0f}s) -> {checkpoint_dir.name}")


def main() -> None:
    torch.manual_seed(0)

    print("Loading Chapter 7's training set...")
    examples, skipped, artifacts_filtered = build_training_set_at_scale()
    held_out_examples, _ = build_timeline_examples_for_report(FULL_SET_DIR / HELD_OUT_REPORT)
    stride = max(1, len(examples) // TRAINING_SAMPLE_SIZE)
    training_sample = examples[::stride][:TRAINING_SAMPLE_SIZE]
    print(f"{len(examples)} training examples, {len(held_out_examples)} held-out examples (report #37, never trained on)")

    run_dir = new_run_dir()
    logger = MetricsLogger(run_dir)
    print(f"Run directory: {run_dir}")

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    print("\nBefore fine-tuning:")
    sample_before = score(run_baseline(model, tokenizer, training_sample))
    held_out_before = score(run_baseline(model, tokenizer, held_out_examples))
    print(f"  Training sample baseline ({len(training_sample)} examples): {sample_before[0]}/{sample_before[1]}")
    print(f"  Held-out baseline: {held_out_before[0]}/{held_out_before[1]}")

    lora_model = build_lora_model(model)
    lora_model.print_trainable_parameters()

    print("\nTraining, epochs 1-2...")
    train_with_checkpoints(lora_model, tokenizer, examples, held_out_examples, run_dir, logger, num_epochs=2)

    print("\nSimulating a crash and restart: reloading the base model from scratch,")
    print("then resuming purely from checkpoint_2 on disk...")
    del model, lora_model
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    lora_model = load_checkpoint(model, run_dir / "checkpoint_2")

    print("Resuming training, epoch 3...")
    train_with_checkpoints(lora_model, tokenizer, examples, held_out_examples, run_dir, logger, num_epochs=1, start_epoch=2)

    print("\nAfter fine-tuning:")
    sample_after = score(run_baseline(lora_model, tokenizer, training_sample))
    held_out_after = score(run_baseline(lora_model, tokenizer, held_out_examples))
    print(f"  Training sample recall ({len(training_sample)} examples): {sample_after[0]}/{sample_after[1]}")
    print(f"  Held-out generalization: {held_out_after[0]}/{held_out_after[1]}")
    print(f"\nMetrics -> {logger.csv_path}")


if __name__ == "__main__":
    main()
