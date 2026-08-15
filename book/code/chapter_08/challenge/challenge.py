"""Chapter 8 challenge exercise -- reference solution.

Challenge: prove a saved checkpoint is actually self-sufficient --
reload the base model completely fresh, load a specific checkpoint
(checkpoint_1, from after epoch 1 of the most recent run) from disk
with no other state carried over, and confirm its held-out score
matches what metrics.csv logged for that exact epoch. A checkpoint that
can't reproduce its own recorded number isn't trustworthy.

Usage:
    python code/chapter_08/challenge/challenge.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "chapter_01"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "chapter_02"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "chapter_03"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "chapter_06"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "chapter_07"))

from load_local_model import MODEL_NAME, load_model_and_tokenizer  # noqa: E402

from baseline_prompting import run_baseline  # noqa: E402
from build_training_examples import HELD_OUT_REPORT  # noqa: E402
from data_quality_gate import FULL_SET_DIR  # noqa: E402
from finetune_at_scale import RUNS_DIR, load_checkpoint  # noqa: E402
from first_lora_finetune import score  # noqa: E402
from format_training_chunks import build_timeline_examples_for_report  # noqa: E402


def latest_run_dir() -> Path:
    runs = sorted(RUNS_DIR.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No runs found in {RUNS_DIR} -- run code/chapter_08/finetune_at_scale.py first")
    return runs[-1]


def logged_held_out_score(run_dir: Path, epoch: int) -> str:
    with (run_dir / "metrics.csv").open() as f:
        for row in csv.DictReader(f):
            if int(row["epoch"]) == epoch:
                return row["held_out_exact_match"]
    raise ValueError(f"No epoch {epoch} row in {run_dir / 'metrics.csv'}")


def main() -> None:
    run_dir = latest_run_dir()
    checkpoint_dir = run_dir / "checkpoint_1"
    logged_score = logged_held_out_score(run_dir, epoch=1)

    print(f"Reloading base model fresh, then {checkpoint_dir.name} from {run_dir.name}...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    lora_model = load_checkpoint(model, checkpoint_dir)

    held_out_examples, _ = build_timeline_examples_for_report(FULL_SET_DIR / HELD_OUT_REPORT)
    matched, total = score(run_baseline(lora_model, tokenizer, held_out_examples))

    print(f"Recomputed held-out score: {matched}/{total}")
    print(f"metrics.csv logged: {logged_score}")
    assert f"{matched}/{total}" == logged_score, "Reloaded checkpoint doesn't reproduce its own logged score"
    print("Match -- this checkpoint reproduces its own recorded result from disk alone.")


if __name__ == "__main__":
    main()
