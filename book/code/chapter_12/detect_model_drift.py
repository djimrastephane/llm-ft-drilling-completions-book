"""Chapter 12: Detecting Drift Across Model Versions.

Chapter 11 built an evaluation harness for one model version, one
snapshot in time. This chapter runs that exact harness across multiple
real versions -- Chapter 8's 3 epoch checkpoints -- and finds a real
disagreement between two of its own metrics: perplexity says
checkpoint_2 is an improvement over checkpoint_1; the overlap score
says it's a regression. Both are true, on the same two real
checkpoints, and neither number alone explains why.

Usage:
    python code/chapter_12/detect_model_drift.py
"""

import sys
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_01"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_02"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_06"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_07"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_09"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_10"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_11"))

# load_local_model sets USE_TF=0/USE_FLAX=0 before its own transformers
# import -- must stay first, see Chapter 3/5/8/9/10/11 for why.
from load_local_model import MODEL_NAME, load_model_and_tokenizer  # noqa: E402

from peft import PeftModel  # noqa: E402

from eval_finetuned_model import build_held_out_eval_set, evaluate, perplexity  # noqa: E402

CHECKPOINTS_DIR = BOOK_ROOT / "checkpoints" / "chapter_08"


def load_version(checkpoint_dir: Path):
    base_model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    return PeftModel.from_pretrained(base_model, checkpoint_dir), tokenizer


def summarize_version(model, tokenizer, eval_set: list[dict]) -> dict:
    """Chapter 11's three metrics, collapsed to one summary per model version."""
    results = evaluate(model, tokenizer, eval_set)
    texts = [e["output"] for e in eval_set]
    return {
        "exact_match": sum(r["exact_match"] for r in results),
        "avg_overlap": sum(r["overlap_score"] for r in results) / len(results),
        "perplexity": perplexity(model, tokenizer, texts),
    }


def compare_versions(before: dict, after: dict) -> dict:
    """Direction per metric between two version summaries -- lower
    perplexity is better, higher exact-match/overlap is better.
    Metrics disagreeing on direction is the signal worth investigating
    by hand; this function only reports direction, it doesn't judge.
    """

    def direction(before_value, after_value, lower_is_better=False):
        if after_value == before_value:
            return "unchanged"
        improved = after_value < before_value if lower_is_better else after_value > before_value
        return "improved" if improved else "regressed"

    return {
        "exact_match": direction(before["exact_match"], after["exact_match"]),
        "avg_overlap": direction(before["avg_overlap"], after["avg_overlap"]),
        "perplexity": direction(before["perplexity"], after["perplexity"], lower_is_better=True),
    }


def main() -> None:
    eval_set = build_held_out_eval_set()

    runs = sorted(CHECKPOINTS_DIR.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No runs in {CHECKPOINTS_DIR} -- run code/chapter_08/finetune_at_scale.py first")
    run_dir = runs[-1]
    checkpoints = sorted(run_dir.glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[1]))

    summaries = {}
    for checkpoint_dir in checkpoints:
        model, tokenizer = load_version(checkpoint_dir)
        summaries[checkpoint_dir.name] = summarize_version(model, tokenizer, eval_set)
        print(f"{checkpoint_dir.name}: {summaries[checkpoint_dir.name]}")

    print()
    names = list(summaries)
    for before_name, after_name in zip(names, names[1:]):
        directions = compare_versions(summaries[before_name], summaries[after_name])
        print(f"{before_name} -> {after_name}: {directions}")


if __name__ == "__main__":
    main()
