"""Chapter 9 challenge exercise -- reference solution.

Challenge: rerun this chapter's 4 test queries using dense
sentence-transformer embeddings (Chapter 4's tool) instead of BM25, and
compare retrieval accuracy directly.

Usage:
    python code/chapter_09/challenge/challenge.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# hybrid_rag_finetune imports load_local_model, which sets USE_TF=0/
# USE_FLAX=0 before its own transformers import -- importing it first
# protects sentence_transformers' later transformers import too. See
# Chapter 3/5/8/9 for the same guard.
from hybrid_rag_finetune import TEST_CASES, build_retrieval_corpus  # noqa: E402

import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402


def retrieve_dense(query: str, corpus: list[dict], embeddings: np.ndarray, st_model: SentenceTransformer, k: int = 3) -> list[dict]:
    query_vector = st_model.encode([query])[0]
    query_vector = query_vector / np.linalg.norm(query_vector)
    similarities = embeddings @ query_vector
    top_indices = np.argsort(-similarities)[:k]
    return [{**corpus[i], "score": float(similarities[i])} for i in top_indices]


def main() -> None:
    corpus = build_retrieval_corpus()
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = st_model.encode([c["text"] for c in corpus], show_progress_bar=False)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    hits = 0
    for label, _input_context, query, target_report in TEST_CASES:
        retrieved = retrieve_dense(query, corpus, embeddings, st_model, k=3)
        top_reports = [r["report_num"] for r in retrieved]
        hit = target_report in top_reports
        hits += hit
        print(f"{label}: query={query!r} -> top-3 reports {top_reports} (hit: {hit})")

    print(f"\nDense embedding retrieval: {hits}/{len(TEST_CASES)} hit rate")
    print("(BM25, this chapter's default, scored 4/4 on these same queries -- see the main chapter)")


if __name__ == "__main__":
    main()
