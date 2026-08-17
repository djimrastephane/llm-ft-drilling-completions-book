"""Chapter 12 challenge exercise: does the same metric disagreement
show up comparing two completely different training regimes, not just
adjacent epochs of the same run?

Reference solution. Compares Chapter 5's 16-example LoRA fine-tune
against Chapter 8's "at scale" checkpoint (669 examples, 3 epochs) as
two real model versions, using this chapter's own comparison harness.

Usage:
    python code/chapter_12/challenge/challenge.py
"""

import sys
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_01"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_02"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_06"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_07"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_09"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_10"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_11"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_12"))

# load_local_model sets USE_TF=0/USE_FLAX=0 before its own transformers
# import -- must stay first.
from load_local_model import MODEL_NAME, load_model_and_tokenizer  # noqa: E402, F401

from detect_model_drift import compare_versions, load_version, summarize_version  # noqa: E402
from eval_finetuned_model import build_held_out_eval_set  # noqa: E402


def main() -> None:
    eval_set = build_held_out_eval_set()

    model_ch5, tokenizer_ch5 = load_version(BOOK_ROOT / "checkpoints" / "chapter_05_lora")
    summary_ch5 = summarize_version(model_ch5, tokenizer_ch5, eval_set)
    print(f"Chapter 5 (16 examples): {summary_ch5}")

    runs = sorted((BOOK_ROOT / "checkpoints" / "chapter_08").glob("run_*"))
    latest_ch8_checkpoint = sorted(runs[-1].glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[1]))[-1]
    model_ch8, tokenizer_ch8 = load_version(latest_ch8_checkpoint)
    summary_ch8 = summarize_version(model_ch8, tokenizer_ch8, eval_set)
    print(f"Chapter 8 (at scale): {summary_ch8}")

    print(f"\nChapter 5 -> Chapter 8: {compare_versions(summary_ch5, summary_ch8)}")


if __name__ == "__main__":
    main()
