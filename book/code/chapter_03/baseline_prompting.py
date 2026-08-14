"""Chapter 3: Baseline Prompting -- What the Model Gets Wrong Before Fine-Tuning.

Runs the unmodified base model (loaded exactly as in Chapter 1) against
the exact same instruction/input prompts Chapter 2 turned into training
examples, and records what it says -- without ever having seen this
well's reports. This is the fixed, saved baseline Chapter 5's
fine-tuned model gets compared against afterward.

Usage:
    python code/chapter_03/baseline_prompting.py
"""

import json
import sys
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_01"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_02"))

from build_training_examples import build_training_examples  # noqa: E402
from load_local_model import MODEL_NAME, generate_reply, load_model_and_tokenizer  # noqa: E402

OUTPUT_PATH = BOOK_ROOT / "datasets" / "training_examples" / "baseline_results.jsonl"


def build_prompt(example: dict) -> str:
    """Turn a Chapter 2 training example into the question the baseline model sees."""
    return f"{example['instruction']}\n{example['input']}"


def ask_baseline(model, tokenizer, example: dict, max_new_tokens: int = 60) -> dict:
    """Ask the base model one Chapter 2 example's question and record the result."""
    prompt = build_prompt(example)
    answer = generate_reply(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    expected = example["output"]
    return {
        "instruction": example["instruction"],
        "input": example["input"],
        "expected": expected,
        "base_model_answer": answer,
        "matched_expected": expected.strip().lower() in answer.strip().lower(),
    }


def run_baseline(model, tokenizer, examples: list[dict]) -> list[dict]:
    return [ask_baseline(model, tokenizer, example) for example in examples]


def save_results_jsonl(results: list[dict], output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def main() -> None:
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    examples = build_training_examples()

    results = run_baseline(model, tokenizer, examples)
    save_results_jsonl(results)

    matched = sum(r["matched_expected"] for r in results)
    print(f"Baseline: {matched}/{len(results)} answers contained the report's actual value verbatim")
    print(f"Saved -> {OUTPUT_PATH}\n")
    print(json.dumps(results[0], indent=2))


if __name__ == "__main__":
    main()
