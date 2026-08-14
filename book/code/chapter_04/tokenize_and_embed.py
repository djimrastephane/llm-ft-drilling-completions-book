"""Chapter 4: Tokenization and Embeddings for Domain Fine-Tuning.

Looks at what the base model actually reads before it ever generates a
reply: tokens (fixed sub-word pieces from a rulebook the model never
adapts) and embeddings (the numeric coordinates those pieces get looked
up to). Shows, with a real run, that neither the base model's raw
per-token embeddings nor a general-purpose sentence-embedding model
reliably recognize this book's oilfield shorthand (POOH, BHA, ...) as
equivalent to its own spelled-out meaning.

Usage:
    python code/chapter_04/tokenize_and_embed.py
"""

import sys
from pathlib import Path

import torch

BOOK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_01"))

from load_local_model import MODEL_NAME, load_model_and_tokenizer  # noqa: E402

OILFIELD_TERMS = ["POOH", "TOOH", "BHA", "WOB", "SPP", "RPM"]
PLAIN_TERMS = ["pipe", "hello", "drilling", "stuck"]

TERM_PAIRS = [
    ("POOH", "trip out of hole"),
    ("TOOH", "trip out of hole"),
    ("BHA", "bottom hole assembly"),
    ("BHA", "pizza delivery"),
    ("trip out of hole", "pull out of hole"),
]


def tokenize_with_pieces(tokenizer, text: str) -> list[str]:
    """Return the exact sub-word pieces the tokenizer splits text into."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.convert_ids_to_tokens(ids)


def term_embedding(model, tokenizer, term: str) -> torch.Tensor:
    """Mean-pool the model's raw, per-token input embeddings for a term."""
    ids = tokenizer.encode(term, add_special_tokens=False)
    input_ids = torch.tensor([ids])
    vectors = model.get_input_embeddings()(input_ids)[0]
    return vectors.mean(dim=0)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def raw_embedding_similarities(model, tokenizer, pairs: list[tuple[str, str]]) -> list[dict]:
    results = []
    for term_a, term_b in pairs:
        vec_a = term_embedding(model, tokenizer, term_a)
        vec_b = term_embedding(model, tokenizer, term_b)
        results.append({"term_a": term_a, "term_b": term_b, "cosine_similarity": cosine_similarity(vec_a, vec_b)})
    return results


def sentence_embedding_similarities(pairs: list[tuple[str, str]]) -> list[dict]:
    """Same comparison, but with a purpose-built sentence-embedding model."""
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    results = []
    for term_a, term_b in pairs:
        vec_a, vec_b = st_model.encode(term_a), st_model.encode(term_b)
        sim = float(
            torch.nn.functional.cosine_similarity(
                torch.tensor(vec_a).unsqueeze(0), torch.tensor(vec_b).unsqueeze(0)
            )
        )
        results.append({"term_a": term_a, "term_b": term_b, "cosine_similarity": sim})
    return results


def main() -> None:
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    print("Tokenization: oilfield shorthand vs. plain English")
    for term in OILFIELD_TERMS + PLAIN_TERMS:
        pieces = tokenize_with_pieces(tokenizer, term)
        print(f"  {term!r:12} -> {len(pieces)} token(s): {pieces}")

    print("\nRaw input-embedding cosine similarity (base model, before fine-tuning):")
    for result in raw_embedding_similarities(model, tokenizer, TERM_PAIRS):
        print(f"  {result['term_a']!r:20} vs {result['term_b']!r:20} -> {result['cosine_similarity']:.3f}")

    print("\nSentence-embedding cosine similarity (general-purpose, contextual):")
    for result in sentence_embedding_similarities(TERM_PAIRS):
        print(f"  {result['term_a']!r:20} vs {result['term_b']!r:20} -> {result['cosine_similarity']:.3f}")


if __name__ == "__main__":
    main()
