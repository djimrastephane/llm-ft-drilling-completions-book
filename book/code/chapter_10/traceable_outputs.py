"""Chapter 10: Traceable Outputs and Hallucination Mitigation.

Chapter 9's Field Notes found a real gap: retrieval found the right
report in all 4 of its test cases, but the fine-tuned model's answer was
only actually consistent with that report's text in 1 of them. A
citation Chapter 9 attaches means "this is what was retrieved," not
"this is what the answer is based on" -- those can silently diverge.

This chapter builds the check: a lexical faithfulness score between an
answer and each source chunk it was shown, so a citation only survives
if the answer is actually traceable to it. An answer with no surviving
citation is flagged as unsupported instead of quietly kept.

Usage:
    python code/chapter_10/traceable_outputs.py
"""

import re
import sys
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_01"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_02"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_06"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_07"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_09"))

# load_local_model sets USE_TF=0/USE_FLAX=0 before its own transformers
# import -- see Chapter 3/5/8/9 for why this has to happen first.
from load_local_model import MODEL_NAME, generate_reply, load_model_and_tokenizer  # noqa: E402

from peft import PeftModel  # noqa: E402

from hybrid_rag_finetune import (  # noqa: E402
    INSTRUCTION,
    TEST_CASES,
    build_bm25_index,
    build_grounded_prompt,
    build_retrieval_corpus,
    latest_checkpoint,
    retrieve,
)

# Words that appear in every prompt's own instruction/input template
# ("What happened on this well during this time window?"), not because
# the retrieved text was actually used. Counting them as "supported"
# would make every answer look more grounded than it is, template
# boilerplate aside.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "during",
    "for", "from", "happened", "in", "is", "it", "its", "of", "on", "or",
    "report", "the", "this", "time", "to", "was", "well", "were", "what",
    "which", "window", "with",
}


def content_words(text: str) -> set[str]:
    """Lowercase words with punctuation stripped, minus stopwords and
    anything shorter than 3 letters.
    """
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def faithfulness_score(answer: str, source_text: str) -> float:
    """Fraction of the answer's content words that also appear in the
    source text -- 1.0 means every content word in the answer shows up
    somewhere in that source, 0.0 means none do. A simple word-overlap
    check, not a semantic one: it will miss a paraphrase and can be
    fooled by a source that happens to share vocabulary without
    supporting the actual claim. See Production Reality below.
    """
    answer_words = content_words(answer)
    if not answer_words:
        return 0.0
    return len(answer_words & content_words(source_text)) / len(answer_words)


def answer_with_traceable_sources(
    model, tokenizer, instruction: str, input_context: str, query: str, corpus: list[dict], bm25, k: int = 3, threshold: float = 0.5
) -> dict:
    """Like Chapter 9's `answer_with_retrieval`, but scores the answer
    against each individually retrieved chunk after generation, and only
    keeps the ones the answer is actually traceable to as `verified_sources`.
    `sources` (Chapter 9's field) named everything retrieval handed the
    model; `verified_sources` here names only what the answer used.
    """
    retrieved = retrieve(query, corpus, bm25, k=k)
    prompt = build_grounded_prompt(instruction, input_context, retrieved)
    answer = generate_reply(model, tokenizer, prompt, max_new_tokens=60)

    scored = sorted(
        (
            {**chunk, "faithfulness": faithfulness_score(answer, chunk["text"])}
            for chunk in retrieved
        ),
        key=lambda c: c["faithfulness"],
        reverse=True,
    )
    verified_sources = [c for c in scored if c["faithfulness"] >= threshold]

    return {
        "answer": answer,
        "retrieved": scored,
        "verified_sources": verified_sources,
        "grounded": len(verified_sources) > 0,
    }


def main() -> None:
    print("Building retrieval corpus...")
    corpus = build_retrieval_corpus()
    bm25 = build_bm25_index(corpus)

    checkpoint_dir = latest_checkpoint()
    print(f"Loading fine-tuned model from {checkpoint_dir}...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    lora_model = PeftModel.from_pretrained(model, checkpoint_dir)

    for label, input_context, query, target_report in TEST_CASES:
        result = answer_with_traceable_sources(lora_model, tokenizer, INSTRUCTION, input_context, query, corpus, bm25)
        print(f"\n=== {label} ===")
        print(f"Answer: {result['answer']}")
        print(f"Grounded: {result['grounded']}")
        for chunk in result["retrieved"]:
            mark = "verified" if chunk["faithfulness"] >= 0.5 else "not used"
            print(f"  [{mark:>8}] Report #{chunk['report_num']} {chunk['from_time']}-{chunk['to_time']} (faithfulness {chunk['faithfulness']:.2f})")


if __name__ == "__main__":
    main()
