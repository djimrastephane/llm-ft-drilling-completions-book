"""Chapter 11: Evaluating a Fine-Tuned Domain Model.

Chapter 10 checked faithfulness on 4 hand-picked test cases -- useful,
but not a real evaluation: those 4 were chosen because Chapter 9
already knew what to look for in each. This chapter builds a real
evaluation harness: a held-out set of questions with known-correct
answers, built the same way Chapter 7 built its "at scale" training
set but run on the one report (Drilling_037) that has never appeared
in any training set in this book -- scored two ways (exact-match and
partial credit), plus perplexity, a third metric that doesn't need a
"correct answer" at all.

Usage:
    python code/chapter_11/eval_finetuned_model.py
"""

import math
import sys
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_01"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_02"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_06"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_07"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_09"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_10"))

# load_local_model sets USE_TF=0/USE_FLAX=0 before its own transformers
# import -- must stay first, see Chapter 3/5/8/9/10 for why.
from load_local_model import MODEL_NAME, generate_reply, load_model_and_tokenizer  # noqa: E402

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402

from build_training_examples import HELD_OUT_REPORT  # noqa: E402
from data_quality_gate import FULL_SET_DIR  # noqa: E402
from format_training_chunks import build_timeline_examples_for_report  # noqa: E402
from hybrid_rag_finetune import latest_checkpoint  # noqa: E402
from traceable_outputs import faithfulness_score  # noqa: E402


def build_held_out_eval_set() -> list[dict]:
    """The held-out report's own real timeline examples -- built with
    Chapter 7's exact chunker, on the one report never included in
    either Chapter 5's or Chapter 8's training set.
    """
    examples, _artifacts_filtered = build_timeline_examples_for_report(FULL_SET_DIR / HELD_OUT_REPORT)
    return examples


def exact_match(generated: str, expected: str) -> bool:
    return expected.strip().lower() in generated.strip().lower()


def evaluate(model, tokenizer, eval_set: list[dict]) -> list[dict]:
    results = []
    for example in eval_set:
        prompt = f"{example['instruction']}\n{example['input']}"
        generated = generate_reply(model, tokenizer, prompt, max_new_tokens=60)
        results.append(
            {
                "input": example["input"],
                "expected": example["output"],
                "generated": generated,
                "exact_match": exact_match(generated, example["output"]),
                "overlap_score": faithfulness_score(generated, example["output"]),
            }
        )
    return results


def perplexity(model, tokenizer, texts: list[str]) -> float:
    """Lower means the model found this text less surprising on
    average -- a generalization signal that doesn't need a "correct
    answer" at all, just real text the model never trained on. Averages
    loss (not perplexity itself) across examples before the one final
    exponentiation, which is the mathematically sound way to combine
    per-example log-losses into a single score.
    """
    total_loss, total_examples = 0.0, 0
    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            loss = model(input_ids=input_ids, labels=input_ids).loss
        total_loss += loss.item()
        total_examples += 1
    return math.exp(total_loss / total_examples)


def main() -> None:
    eval_set = build_held_out_eval_set()
    print(f"Held-out evaluation set: {len(eval_set)} real examples from a report never used in any training set")

    checkpoint_dir = latest_checkpoint()
    print(f"Loading fine-tuned model from {checkpoint_dir}...")
    base_model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    lora_model = PeftModel.from_pretrained(base_model, checkpoint_dir)

    results = evaluate(lora_model, tokenizer, eval_set)
    exact_matches = sum(r["exact_match"] for r in results)
    avg_overlap = sum(r["overlap_score"] for r in results) / len(results)
    print(f"\nExact-match: {exact_matches}/{len(results)}")
    print(f"Average overlap score: {avg_overlap:.2f}")
    for r in results:
        label = "match" if r["exact_match"] else f"{r['overlap_score']:.2f}"
        print(f"  [{label:>5}] {r['input']}")

    held_out_texts = [e["output"] for e in eval_set]
    fresh_base_model, _ = load_model_and_tokenizer(MODEL_NAME)
    ppl_base = perplexity(fresh_base_model, tokenizer, held_out_texts)
    ppl_finetuned = perplexity(lora_model, tokenizer, held_out_texts)
    print(f"\nPerplexity on held-out text -- base model: {ppl_base:.2f}, fine-tuned: {ppl_finetuned:.2f}")


if __name__ == "__main__":
    main()
