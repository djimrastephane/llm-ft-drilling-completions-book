"""Chapter 1 challenge exercise -- reference solution.

Challenge: run the same question through the base model twice -- once
with greedy decoding (do_sample=False, deterministic, what the chapter
used) and once with sampling (do_sample=True) -- and compare. This
previews a generation setting later chapters (5, 9, 11) come back to.

Usage:
    python code/chapter_01/challenge/challenge.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from load_local_model import MODEL_NAME, load_model_and_tokenizer  # noqa: E402


def generate_with_settings(model, tokenizer, user_message: str, *, do_sample: bool, max_new_tokens: int = 200) -> str:
    messages = [{"role": "user", "content": user_message}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")

    generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        generate_kwargs.update(temperature=0.8, top_p=0.9)

    torch.manual_seed(0)  # makes the sampled run reproducible run-to-run, for teaching purposes
    output_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main() -> None:
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    question = "In one sentence, what does 'trip out of hole' mean?"

    greedy = generate_with_settings(model, tokenizer, question, do_sample=False)
    sampled = generate_with_settings(model, tokenizer, question, do_sample=True)

    print("Question:", question)
    print("\n--- Greedy (do_sample=False) ---")
    print(greedy)
    print("\n--- Sampled (do_sample=True, temperature=0.8) ---")
    print(sampled)
    print(
        "\nGreedy decoding always picks the single most likely next word, so "
        "it gives the same answer every run -- useful for this book's "
        "reproducible examples. Sampling introduces controlled randomness, "
        "so re-running with a different seed can change the wording (though "
        "usually not the substance) of the answer."
    )


if __name__ == "__main__":
    main()
