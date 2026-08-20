"""Export real base-vs-fine-tuned answers for the Mathematica companion notebook.

Reuses Chapter 5's own real training and held-out example sets, and its
own already-trained checkpoint at checkpoints/chapter_05_lora, to
generate a fresh before/after comparison -- the same real questions and
the same real LoRA adapter behind this book's own headline
`13/16` (trained) / `0/2` (held-out) result, re-run today rather than
copied from the book text.

This script does not retrain anything: checkpoints/chapter_05_lora
already exists from running code/chapter_05/first_lora_finetune.py, so
this just loads it and asks it the same questions Chapter 5 asks, the
same way Chapter 5 itself does (its own run_baseline function, once
against the base model and once against the fine-tuned one).

Usage:
    python mathematica/export_before_after.py
"""

import json
import sys
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parents[1] / "book"
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_01"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_02"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_03"))

# load_local_model sets USE_TF=0/USE_FLAX=0 before its own transformers
# import -- must stay first, see Chapter 3/5's own scripts for why.
from load_local_model import MODEL_NAME, load_model_and_tokenizer  # noqa: E402

from peft import PeftModel  # noqa: E402

from baseline_prompting import run_baseline  # noqa: E402
from build_training_examples import (  # noqa: E402
    HELD_OUT_REPORT,
    SAMPLE_SET_DIR,
    build_examples_for_report,
    build_training_examples,
)

ADAPTER_DIR = BOOK_ROOT / "checkpoints" / "chapter_05_lora"
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "before_after_examples.json"


def to_notebook_rows(results: list[dict], group: str) -> list[dict]:
    return [
        {
            "group": group,
            "question": r["input"],
            "expected": r["expected"],
            "base_model_answer": r["base_model_answer"],
            "base_matched": r["matched_expected"],
        }
        for r in results
    ]


def merge_finetuned(rows: list[dict], finetuned_results: list[dict]) -> None:
    for row, r in zip(rows, finetuned_results):
        row["finetuned_model_answer"] = r["base_model_answer"]
        row["finetuned_matched"] = r["matched_expected"]


def main() -> None:
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"{ADAPTER_DIR} not found -- run code/chapter_05/first_lora_finetune.py first "
            "(see the book's Chapter 5 for what it does and how long it takes)."
        )

    print("Loading base model...")
    base_model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    training_examples = build_training_examples()
    held_out_examples = build_examples_for_report(SAMPLE_SET_DIR / HELD_OUT_REPORT)
    print(f"{len(training_examples)} training examples, {len(held_out_examples)} held-out examples")

    print("Running base model on both sets (before fine-tuning)...")
    training_before = run_baseline(base_model, tokenizer, training_examples)
    held_out_before = run_baseline(base_model, tokenizer, held_out_examples)

    rows = to_notebook_rows(training_before, "training") + to_notebook_rows(held_out_before, "held_out")

    print(f"Loading fine-tuned checkpoint from {ADAPTER_DIR}...")
    lora_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    lora_model.eval()

    print("Running fine-tuned model on both sets (after fine-tuning)...")
    training_after = run_baseline(lora_model, tokenizer, training_examples)
    held_out_after = run_baseline(lora_model, tokenizer, held_out_examples)

    merge_finetuned(rows[: len(training_examples)], training_after)
    merge_finetuned(rows[len(training_examples) :], held_out_after)

    summary = {
        "training_before": sum(r["base_matched"] for r in rows if r["group"] == "training"),
        "training_after": sum(r["finetuned_matched"] for r in rows if r["group"] == "training"),
        "training_total": len(training_examples),
        "held_out_before": sum(r["base_matched"] for r in rows if r["group"] == "held_out"),
        "held_out_after": sum(r["finetuned_matched"] for r in rows if r["group"] == "held_out"),
        "held_out_total": len(held_out_examples),
        "adapter": str(ADAPTER_DIR.relative_to(BOOK_ROOT.parent)),
        "model": MODEL_NAME,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({"summary": summary, "examples": rows}, indent=2))

    print(f"\nTraining:  {summary['training_before']}/{summary['training_total']} -> "
          f"{summary['training_after']}/{summary['training_total']}")
    print(f"Held-out:  {summary['held_out_before']}/{summary['held_out_total']} -> "
          f"{summary['held_out_after']}/{summary['held_out_total']}")
    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
