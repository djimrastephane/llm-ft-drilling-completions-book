"""Chapter 1: Loading and Running Your First Local LLM.

Loads a small, open-weight instruction-tuned model with `transformers`
and generates a reply entirely on your own machine -- no API key, no
network call once the model is downloaded and cached.

This is the exact base model the rest of this book fine-tunes, starting
in Chapter 5.

Usage:
    python code/chapter_01/load_local_model.py
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """Download (first run only, then cached) and load the base model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    return model, tokenizer


def generate_reply(model, tokenizer, user_message: str, max_new_tokens: int = 340) -> str:
    """Send one user message through the model's chat template and return its reply."""
    messages = [{"role": "user", "content": user_message}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy decoding -- deterministic, same output every run
    )
    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main() -> None:
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer()
    print(f"Loaded {MODEL_NAME} in {time.time() - t0:.1f}s")

    question = (
        "A driller's report says: 'During the slide lost tool face and "
        "became stuck.' What should the crew watch for on the next curve "
        "section?"
    )
    print(f"\nQuestion: {question}\n")

    t1 = time.time()
    answer = generate_reply(model, tokenizer, question)
    print(f"Model answer (generated in {time.time() - t1:.1f}s):\n{answer}")


if __name__ == "__main__":
    main()
