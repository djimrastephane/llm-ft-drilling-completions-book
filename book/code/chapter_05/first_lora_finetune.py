"""Chapter 5: Your First LoRA Fine-Tune.

Fine-tunes the base model (loaded exactly as in Chapter 1) on Chapter
2's 16 training examples with LoRA, then reruns Chapter 3's baseline
harness twice: once against those same 16 training examples (training
recall), and once against Chapter 2's held-out report, Drilling_037
(generalization) -- a report the fine-tuned model never trained on.

This training loop intentionally never moves the model or tensors to a
GPU or Apple Silicon (MPS) device -- everything runs on CPU, same as
Chapter 1's default. That's a deliberate choice, not an oversight: it
keeps this chapter's ~5-minute runtime and its exact printed numbers
reproducible for every reader on the same hardware baseline the rest of
the book assumes, rather than depending on whichever accelerator (or
none) a given machine happens to have.

Usage:
    python code/chapter_05/first_lora_finetune.py
"""

import sys
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_01"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_02"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_03"))

# load_local_model sets USE_TF=0/USE_FLAX=0 before its own transformers
# import -- importing it first protects every later import in this file
# (including peft, which also imports transformers) from the same
# TensorFlow-probing crash described in Chapter 3.
from load_local_model import MODEL_NAME, load_model_and_tokenizer  # noqa: E402

import torch  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402

from baseline_prompting import run_baseline  # noqa: E402
from build_training_examples import (  # noqa: E402
    HELD_OUT_REPORT,
    SAMPLE_SET_DIR,
    build_examples_for_report,
    build_training_examples,
)

ADAPTER_OUTPUT_DIR = BOOK_ROOT / "checkpoints" / "chapter_05_lora"

LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
)

NUM_EPOCHS = 20
LEARNING_RATE = 2e-4


def build_training_ids(tokenizer, example: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Chat-format one training example and mask the prompt out of the loss.

    Only the assistant's own answer should be trained on -- the model
    isn't supposed to learn to predict the question, just the response.
    """
    user_message = f"{example['instruction']}\n{example['input']}"
    prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": example["output"]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt_only, return_tensors="pt", add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    labels = full_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    return full_ids, labels


def build_lora_model(model):
    return get_peft_model(model, LORA_CONFIG)


def fine_tune(lora_model, tokenizer, examples: list[dict], num_epochs: int = NUM_EPOCHS) -> None:
    lora_model.train()
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=LEARNING_RATE)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for example in examples:
            input_ids, labels = build_training_ids(tokenizer, example)
            loss = lora_model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{num_epochs}: avg loss {epoch_loss / len(examples):.4f}")
    lora_model.eval()


def score(results: list[dict]) -> tuple[int, int]:
    return sum(r["matched_expected"] for r in results), len(results)


def main() -> None:
    torch.manual_seed(0)
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    examples = build_training_examples()
    held_out_examples = build_examples_for_report(SAMPLE_SET_DIR / HELD_OUT_REPORT)

    print("Before fine-tuning:")
    training_before = score(run_baseline(model, tokenizer, examples))
    held_out_before = score(run_baseline(model, tokenizer, held_out_examples))
    print(f"  Training baseline:  {training_before[0]}/{training_before[1]}")
    print(f"  Held-out baseline:  {held_out_before[0]}/{held_out_before[1]}")

    print("\nFine-tuning with LoRA...")
    lora_model = build_lora_model(model)
    lora_model.print_trainable_parameters()
    fine_tune(lora_model, tokenizer, examples)

    ADAPTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lora_model.save_pretrained(ADAPTER_OUTPUT_DIR)

    print("\nAfter fine-tuning:")
    training_after = score(run_baseline(lora_model, tokenizer, examples))
    held_out_after = score(run_baseline(lora_model, tokenizer, held_out_examples))
    print(f"  Training recall:         {training_after[0]}/{training_after[1]}")
    print(f"  Held-out generalization: {held_out_after[0]}/{held_out_after[1]}")
    print(f"\nAdapter saved -> {ADAPTER_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
