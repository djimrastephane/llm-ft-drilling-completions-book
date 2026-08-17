"""App-only glue logic for the base-vs-fine-tuned companion app.

No Streamlit import in this file, on purpose -- every function here is
directly unit-testable without a running app, mirroring how every
chapter's own code stays testable. Model loading, generation, retrieval,
and scoring are never reimplemented here: they're imported straight from
the book's own chapter code, the same way `tests/conftest.py` makes them
importable for the test suite.

Checkpoint loading for evaluation/inference always goes through Chapter
12's `load_version()`, never Chapter 8's `load_checkpoint()`. Chapter 8's
loader sets `is_trainable=True` (needed there, since it resumes
training) which leaves dropout active and makes generation
non-deterministic -- a real bug Chapter 13's own CHANGELOG entry
documents catching and fixing. This app must not reintroduce it.
"""

import sys
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = BOOK_ROOT / "checkpoints"

for _n in range(1, 14):
    _chapter_dir = BOOK_ROOT / "code" / f"chapter_{_n:02d}"
    if _chapter_dir.is_dir() and str(_chapter_dir) not in sys.path:
        sys.path.insert(0, str(_chapter_dir))

# load_local_model sets USE_TF=0/USE_FLAX=0 before its own transformers
# import -- must stay first, same reason every chapter script that
# touches peft/sentence_transformers imports it before anything else.
from load_local_model import MODEL_NAME, generate_reply, load_model_and_tokenizer  # noqa: E402

from detect_model_drift import compare_versions, load_version, summarize_version  # noqa: E402
from eval_finetuned_model import build_held_out_eval_set, exact_match, perplexity  # noqa: E402
from hybrid_rag_finetune import (  # noqa: E402
    INSTRUCTION as RETRIEVAL_INSTRUCTION,
    TEST_CASES as RETRIEVAL_TEST_CASES,
    build_bm25_index,
    build_retrieval_corpus,
)
from traceable_outputs import answer_with_traceable_sources, faithfulness_score  # noqa: E402

CHAPTER_05_ADAPTER = CHECKPOINTS_DIR / "chapter_05_lora"
CHAPTER_08_RUNS = CHECKPOINTS_DIR / "chapter_08"
CHAPTER_13_RUNS = CHECKPOINTS_DIR / "chapter_13"


def _latest_run_latest_checkpoint(runs_dir: Path) -> Path | None:
    """Newest run directory's highest-numbered checkpoint, or None if
    `runs_dir` doesn't exist yet or has no runs -- mirrors the exact glob
    pattern Chapter 9's `latest_checkpoint()` and Chapter 12's `main()`
    already use for Chapter 8's own checkpoints, generalized to also
    work for Chapter 13's identically-shaped `checkpoints/chapter_13/run_*/`.
    """
    if not runs_dir.is_dir():
        return None
    runs = sorted(runs_dir.glob("run_*"))
    if not runs:
        return None
    checkpoints = sorted(runs[-1].glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[1]))
    return checkpoints[-1] if checkpoints else None


def available_checkpoints() -> dict[str, Path]:
    """Real checkpoints that actually exist on this machine right now,
    keyed by a human-readable label. `checkpoints/` is gitignored -- a
    fresh clone of this repo has none of these until the reader runs the
    matching chapter script themselves.
    """
    options: dict[str, Path] = {}
    if CHAPTER_05_ADAPTER.is_dir():
        options["Chapter 5 -- first LoRA fine-tune (16 examples)"] = CHAPTER_05_ADAPTER
    ch8 = _latest_run_latest_checkpoint(CHAPTER_08_RUNS)
    if ch8 is not None:
        options["Chapter 8 -- fine-tuned at scale (669 examples)"] = ch8
    ch13 = _latest_run_latest_checkpoint(CHAPTER_13_RUNS)
    if ch13 is not None:
        options["Chapter 13 -- continuous fine-tune (latest)"] = ch13
    return options


def load_base_model():
    return load_model_and_tokenizer(MODEL_NAME)


def load_finetuned_model(checkpoint_dir: Path):
    return load_version(checkpoint_dir)


def generate_answer(model, tokenizer, instruction: str, input_context: str = "") -> str:
    prompt = f"{instruction}\n{input_context}" if input_context else instruction
    return generate_reply(model, tokenizer, prompt, max_new_tokens=60)


def score_against_reference(generated: str, expected: str) -> dict:
    """The exact two metrics Chapter 11's `evaluate()` computes per
    example, decomposed here so the UI can generate once and score the
    same text it already showed the reader, instead of regenerating
    inside a batch call.
    """
    return {
        "exact_match": exact_match(generated, expected),
        "overlap_score": faithfulness_score(generated, expected),
    }


def evaluate_checkpoint(checkpoint_label: str, checkpoint_dir: Path | None, eval_set: list[dict]) -> dict:
    """Chapter 12's `summarize_version()`, run against either the base
    model (`checkpoint_dir=None`) or a real fine-tuned checkpoint.
    """
    if checkpoint_dir is None:
        model, tokenizer = load_base_model()
    else:
        model, tokenizer = load_finetuned_model(checkpoint_dir)
    return summarize_version(model, tokenizer, eval_set)


__all__ = [
    "MODEL_NAME",
    "RETRIEVAL_INSTRUCTION",
    "RETRIEVAL_TEST_CASES",
    "answer_with_traceable_sources",
    "available_checkpoints",
    "build_bm25_index",
    "build_held_out_eval_set",
    "build_retrieval_corpus",
    "compare_versions",
    "evaluate_checkpoint",
    "generate_answer",
    "load_base_model",
    "load_finetuned_model",
    "perplexity",
    "score_against_reference",
]
