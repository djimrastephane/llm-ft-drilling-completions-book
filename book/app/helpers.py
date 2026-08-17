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

import re
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

from build_training_examples import HELD_OUT_REPORT, build_training_examples  # noqa: E402
from data_quality_gate import run_quality_gate  # noqa: E402
from detect_model_drift import compare_versions, load_version, summarize_version  # noqa: E402
from eval_finetuned_model import build_held_out_eval_set, exact_match, perplexity  # noqa: E402
from format_training_chunks import build_training_set_at_scale  # noqa: E402
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


_SHORT_LABEL_PATTERN = re.compile(r"Chapter (\d+)")


def short_version_label(label: str) -> str:
    """Compact, chart-axis-safe version label ("Base", "Ch5", "Ch8",
    "Ch13") derived from the full descriptive label -- full descriptions
    stay in the results table, only the chart axes get the short form.
    """
    if label.startswith("Base model"):
        return "Base"
    match = _SHORT_LABEL_PATTERN.match(label)
    return f"Ch{match.group(1)}" if match else label


def relative_change(before: float, after: float) -> float | None:
    """Fractional change from `before` to `after` (0.298 means +29.8%),
    or None when `before` is 0 -- percent change is undefined starting
    from zero, e.g. the base model's `0.0` overlap score. Callers should
    show the raw before/after values instead in that case.
    """
    if before == 0:
        return None
    return (after - before) / before


def evaluation_snapshot(summaries: dict[str, dict]) -> dict:
    """Which version wins each metric (ties included, since Chapter 11's
    exact-match is strict enough that every version can genuinely tie at
    `0`), plus which version is chronologically latest -- so "latest
    checkpoint" and "best checkpoint" can be compared directly instead of
    left for the reader to work out from a table.
    """

    def winners(metric: str, minimize: bool) -> tuple[list[str], float]:
        values = {label: s[metric] for label, s in summaries.items()}
        target = min(values.values()) if minimize else max(values.values())
        return [label for label, v in values.items() if v == target], target

    best_perplexity_labels, best_perplexity_value = winners("perplexity", minimize=True)
    best_overlap_labels, best_overlap_value = winners("avg_overlap", minimize=False)
    best_exact_match_labels, best_exact_match_value = winners("exact_match", minimize=False)

    return {
        "best_perplexity": (best_perplexity_labels, best_perplexity_value),
        "best_overlap": (best_overlap_labels, best_overlap_value),
        "best_exact_match": (best_exact_match_labels, best_exact_match_value),
        "latest": list(summaries)[-1],
    }


def latest_regressed_on_both(summaries: dict[str, dict]) -> bool:
    """True only if the chronologically latest version is strictly worse
    than the immediately preceding one on BOTH avg_overlap and perplexity
    -- the real "continued fine-tuning didn't automatically improve the
    model" finding this page surfaces rather than hides. False whenever
    fewer than two versions exist, so nothing is claimed without evidence.
    """
    labels = list(summaries)
    if len(labels) < 2:
        return False
    directions = compare_versions(summaries[labels[-2]], summaries[labels[-1]])
    return directions["avg_overlap"] == "regressed" and directions["perplexity"] == "regressed"


_INPUT_PATTERN = re.compile(
    r"Well: (?P<well_name>.+?) \| Report #(?P<report_num>\d+) \| Date: (?P<date>[\d-]+)"
    r"(?: \| Time: (?P<time_from>[\d:]+)-(?P<time_to>[\d:]+)(?: \(part (?P<part_i>\d+) of (?P<part_n>\d+)\))?)?$"
)

DATASET_BUCKETS = {
    "Chapter 2 -- sample set summaries (16 examples)": "sample",
    "Chapter 7 -- full-archive timeline examples (669 examples, the real training set)": "full",
    "Report #37 -- held out, never in any training set (8 examples)": "held_out",
}


def parse_input_context(input_context: str) -> dict:
    """Pull the real, structured fields back out of an example's `input`
    string -- the exact "Well: X | Report #N | Date: D[ | Time: HH:MM-HH:MM[
    (part i of n)]]" template Chapter 2 and Chapter 7 both build it from.
    Nothing here is invented: every field returned either came straight out
    of the string or is None because that example's template doesn't carry it.
    """
    match = _INPUT_PATTERN.match(input_context)
    if not match:
        return {"well_name": None, "report_num": None, "date": None, "time_window": None, "chunk_part": None}
    g = match.groupdict()
    time_window = f"{g['time_from']}-{g['time_to']}" if g["time_from"] else None
    chunk_part = f"{g['part_i']}/{g['part_n']}" if g["part_i"] else None
    return {
        "well_name": g["well_name"],
        "report_num": int(g["report_num"]),
        "date": g["date"],
        "time_window": time_window,
        "chunk_part": chunk_part,
    }


DOCUMENTED_FINDINGS = {
    "Report #37 (held-out)": (
        "success",
        "Chapter 9/10's baseline case: retrieval correctly grounds the "
        "answer in report #37 itself, the one report never in any "
        "training set in this book.",
    ),
    "Report #38 (stuck pipe)": (
        "success",
        "Chapter 10's real finding: no retrieved chunk crosses the "
        "faithfulness threshold here -- correctly flagged ungrounded "
        "rather than silently accepted as a citation.",
    ),
    "Report #21 (step rate test)": (
        "failure",
        "Chapter 10's headline finding: a fluent, real-sounding answer "
        "can verify faithful against the WRONG retrieved chunk (report "
        "#27, a blowout-preventer test retrieved alongside the real "
        "target) instead of the actual target, report #21. A plain "
        "grounded: True/False flag would have silently counted this as "
        "a success -- only checking which specific source an answer "
        "verifies against catches it.",
    ),
    "Report #49 (fishing)": (
        "blind spot",
        "Chapter 10's Field Notes: the faithfulness check can verify "
        "against the right report's text while still missing a genuine "
        "'trip in' vs. 'trip out' direction inversion buried in "
        "otherwise-matching words -- a known blind spot of word-overlap "
        "scoring. This page can't catch that either; only reading the "
        "actual answer text can.",
    ),
}


def pairwise_answer_similarity(labeled_answers: dict[str, str]) -> list[dict]:
    """How much each pair of generated answers overlaps with each other --
    Chapter 10's own word-overlap check (`faithfulness_score`), applied to
    a new pairing (answer vs. answer, not answer vs. source), the same
    kind of reuse Chapter 11 already does applying it to answer vs.
    expected-output instead. A high score between two DIFFERENT
    questions' answers is evidence of the "shape, not judgment" pattern
    Chapters 3/8/9/11/12 document: the model reproducing a memorized
    template regardless of what was actually asked. Symmetrized (scored
    both directions and averaged) since `faithfulness_score` itself isn't.
    """
    labels = list(labeled_answers)
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            text_a, text_b = labeled_answers[a], labeled_answers[b]
            similarity = (faithfulness_score(text_a, text_b) + faithfulness_score(text_b, text_a)) / 2
            pairs.append({"question_a": a, "question_b": b, "similarity": round(similarity, 3)})
    return sorted(pairs, key=lambda p: p["similarity"], reverse=True)


def dataset_examples(bucket: str) -> list[dict]:
    """Real training examples for one of the book's three real example
    sources (`DATASET_BUCKETS` values), each annotated with fields parsed
    straight out of its own `input` string plus whether Chapter 6's real
    quality gate flagged that report's fields as a non-consecutive
    duplicate. Extraction runs live against the same PDFs Chapter 2/6/7/11
    use -- nothing here is pre-generated or invented.
    """
    if bucket == "sample":
        examples = build_training_examples()
    elif bucket == "full":
        examples, _skipped, _filtered = build_training_set_at_scale()
    elif bucket == "held_out":
        examples = build_held_out_eval_set()
    else:
        raise ValueError(f"Unknown dataset bucket: {bucket!r}")

    gate = run_quality_gate()
    needs_review_reports = {
        num
        for (_field, _value), (classification, nums) in gate["duplicates"].items()
        if classification == "needs_review"
        for num in nums
    }

    annotated = []
    for example in examples:
        parsed = parse_input_context(example["input"])
        annotated.append(
            {
                **example,
                **parsed,
                "held_out": parsed["report_num"] == 37,  # report #37 == HELD_OUT_REPORT
                "flagged_for_review": parsed["report_num"] in needs_review_reports,
            }
        )
    return annotated


__all__ = [
    "DATASET_BUCKETS",
    "DOCUMENTED_FINDINGS",
    "MODEL_NAME",
    "RETRIEVAL_INSTRUCTION",
    "RETRIEVAL_TEST_CASES",
    "answer_with_traceable_sources",
    "available_checkpoints",
    "build_bm25_index",
    "build_held_out_eval_set",
    "build_retrieval_corpus",
    "compare_versions",
    "dataset_examples",
    "evaluate_checkpoint",
    "evaluation_snapshot",
    "faithfulness_score",
    "generate_answer",
    "latest_regressed_on_both",
    "load_base_model",
    "load_finetuned_model",
    "pairwise_answer_similarity",
    "parse_input_context",
    "perplexity",
    "relative_change",
    "score_against_reference",
    "short_version_label",
]
