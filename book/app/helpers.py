"""App-only glue logic for the base-vs-fine-tuned companion app.

STATUS: placeholder -- not implemented yet. Intended to hold model
loading and side-by-side comparison logic with no Streamlit dependency,
so it stays unit-testable, mirroring the previous book's app/helpers.py.
"""


def run_comparison(question: str) -> dict:
    """Return the base model's and fine-tuned model's answers to `question`.

    STATUS: not implemented. Once Chapters 1 and 5 exist, this should
    load both models (or reuse cached ones) and return something like:
        {"question": question, "base_answer": ..., "finetuned_answer": ...}
    """
    raise NotImplementedError("run_comparison() is not implemented yet.")
