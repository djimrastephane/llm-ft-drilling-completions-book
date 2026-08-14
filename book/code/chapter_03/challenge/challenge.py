"""Chapter 3 challenge exercise -- reference solution.

Challenge: baseline_prompting.py only gives the base model each report's
metadata (well, report number, date) -- never the report's actual text --
so a 0% match rate mostly proves the model has no way to know the
answer, not that fine-tuning is the only fix. This script re-runs the
same questions, but with each report's full extracted text included as
context, to see how much of that gap closes without any fine-tuning at
all.

Usage:
    python code/chapter_03/challenge/challenge.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "chapter_01"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "chapter_02"))

from baseline_prompting import build_prompt  # noqa: E402
from build_training_examples import build_training_examples, extract_text  # noqa: E402
from load_local_model import MODEL_NAME, generate_reply, load_model_and_tokenizer  # noqa: E402

SAMPLE_SET_DIR = Path(__file__).resolve().parents[3] / "datasets" / "sample_training_set"


def build_prompt_with_context(example: dict, report_text: str) -> str:
    """Same question as baseline_prompting.py, plus the report's own full text."""
    return f"{build_prompt(example)}\n\nReport text:\n{report_text}"


def run_with_context(model, tokenizer, examples: list[dict]) -> list[dict]:
    # One extraction per report, reused for both examples (present ops,
    # activity planned) that report produces.
    text_by_report = {}
    results = []
    for example in examples:
        report_num = example["input"].split("Report #")[1].split(" ")[0]
        if report_num not in text_by_report:
            pdf_path = next(SAMPLE_SET_DIR.glob(f"*_{report_num.zfill(3)}_*.pdf"))
            text_by_report[report_num] = extract_text(pdf_path)

        prompt = build_prompt_with_context(example, text_by_report[report_num])
        answer = generate_reply(model, tokenizer, prompt, max_new_tokens=60)
        expected = example["output"]
        results.append(
            {
                "input": example["input"],
                "instruction": example["instruction"],
                "expected": expected,
                "model_answer_with_context": answer,
                "matched_expected": expected.strip().lower() in answer.strip().lower(),
            }
        )
    return results


def main() -> None:
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    examples = build_training_examples()

    results = run_with_context(model, tokenizer, examples)
    matched = sum(r["matched_expected"] for r in results)

    print(f"With report text included: {matched}/{len(results)} answers contained the expected value verbatim")
    print("(baseline_prompting.py, with no report text, scored 0/18 -- see Chapter 3)")


if __name__ == "__main__":
    main()
