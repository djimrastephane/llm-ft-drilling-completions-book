"""Chapter 11 challenge exercise: does perplexity track training
progress even though held-out exact-match never moves?

Reference solution. Computes perplexity on the same 8-example held-out
evaluation set against each of Chapter 8's 3 real checkpoints (one per
training epoch).

Usage:
    python code/chapter_11/challenge/challenge.py
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

# load_local_model sets USE_TF=0/USE_FLAX=0 before its own transformers
# import -- must stay first.
from load_local_model import MODEL_NAME, load_model_and_tokenizer  # noqa: E402

from peft import PeftModel  # noqa: E402

from eval_finetuned_model import build_held_out_eval_set, perplexity  # noqa: E402

CHECKPOINTS_DIR = BOOK_ROOT / "checkpoints" / "chapter_08"


def main() -> None:
    eval_set = build_held_out_eval_set()
    texts = [e["output"] for e in eval_set]

    runs = sorted(CHECKPOINTS_DIR.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No runs in {CHECKPOINTS_DIR} -- run code/chapter_08/finetune_at_scale.py first")
    run_dir = runs[-1]
    checkpoints = sorted(run_dir.glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[1]))

    _, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    for checkpoint_dir in checkpoints:
        model, _ = load_model_and_tokenizer(MODEL_NAME)
        lora_model = PeftModel.from_pretrained(model, checkpoint_dir)
        ppl = perplexity(lora_model, tokenizer, texts)
        print(f"{checkpoint_dir.name}: perplexity {ppl:.2f}")


if __name__ == "__main__":
    main()
