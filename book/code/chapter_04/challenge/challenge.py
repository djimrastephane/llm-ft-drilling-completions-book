"""Chapter 4 challenge exercise -- reference solution.

Challenge: tokenize_and_embed.py only tests a handful of hand-picked
acronyms. Extend it to real acronyms pulled from this book's own sample
archive -- BOP (blowout preventer) and TVD (true vertical depth), both
of which appear dozens of times across the 9 drilling reports -- and see
whether either embedding approach recognizes them any better than the
Field notes' BOP example did.

Usage:
    python code/chapter_04/challenge/challenge.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "chapter_01"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "chapter_02"))

from build_training_examples import SAMPLE_SET_DIR, extract_text  # noqa: E402
from load_local_model import MODEL_NAME, load_model_and_tokenizer  # noqa: E402
from tokenize_and_embed import (  # noqa: E402
    raw_embedding_similarities,
    sentence_embedding_similarities,
    tokenize_with_pieces,
)

ARCHIVE_PAIRS = [
    ("BOP", "blowout preventer"),
    ("BOP", "birthday party"),  # see this chapter's Field notes
    ("TVD", "true vertical depth"),
]


def count_term_occurrences(term: str, sample_dir: Path = SAMPLE_SET_DIR) -> int:
    return sum(len(re.findall(term, extract_text(pdf_path))) for pdf_path in sorted(sample_dir.glob("*Drilling_*.pdf")))


def main() -> None:
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    print("Tokenization:")
    seen_terms = dict.fromkeys(term for term, _ in ARCHIVE_PAIRS)
    for term in seen_terms:
        print(f"  {term!r:6} -> {tokenize_with_pieces(tokenizer, term)}")

    print("\nRaw input-embedding cosine similarity:")
    for result in raw_embedding_similarities(model, tokenizer, ARCHIVE_PAIRS):
        print(f"  {result['term_a']!r:8} vs {result['term_b']!r:22} -> {result['cosine_similarity']:.3f}")

    print("\nSentence-embedding cosine similarity:")
    for result in sentence_embedding_similarities(ARCHIVE_PAIRS):
        print(f"  {result['term_a']!r:8} vs {result['term_b']!r:22} -> {result['cosine_similarity']:.3f}")

    print("\nOccurrences across the 9 drilling reports:")
    for term in seen_terms:
        print(f"  {term!r:6} -> {count_term_occurrences(term)}")


if __name__ == "__main__":
    main()
