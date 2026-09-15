"""Export optional semantic-embedding comparisons for the Mathematica notebook.

This advanced export reads mathematica/data/before_after_examples.json and
adds a second, optional JSON file with sentence-transformer cosine
similarities and a 2D projection. It does not rerun the base or fine-tuned
LLM; it only embeds the already-exported query, reference, base answer, and
fine-tuned answer texts.

Usage:
    python mathematica/export_semantic_embeddings.py

The first run may download `all-MiniLM-L6-v2` through sentence-transformers.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

BOOK_ROOT = Path(__file__).resolve().parents[1] / "book"
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_01"))

# Sets USE_TF=0/USE_FLAX=0 before sentence_transformers imports transformers.
from load_local_model import MODEL_NAME  # noqa: F401,E402

from sentence_transformers import SentenceTransformer  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
BEFORE_AFTER_PATH = DATA_DIR / "before_after_examples.json"
OUTPUT_PATH = DATA_DIR / "semantic_embedding_examples.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def main() -> None:
    data = json.loads(BEFORE_AFTER_PATH.read_text())
    examples = data["examples"]

    texts: list[str] = []
    text_refs: list[tuple[int, str]] = []
    for index, example in enumerate(examples, start=1):
        for role, key in [
            ("Query", "query_text"),
            ("Reference", "expected"),
            ("Base answer", "base_model_answer"),
            ("Fine-tuned answer", "finetuned_model_answer"),
        ]:
            text_refs.append((index, role))
            texts.append(example[key])

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    by_example: dict[int, dict[str, np.ndarray]] = {}
    for (index, role), vector in zip(text_refs, embeddings):
        by_example.setdefault(index, {})[role] = vector

    rows = []
    for index, example in enumerate(examples, start=1):
        vectors = by_example[index]
        rows.append(
            {
                "row": index,
                "group": example["group"],
                "report_number": example["report_number"],
                "question_type": example["question_type"],
                "label": f"Report #{example['report_number']} | {example['question_type']} | {example['group']}",
                "base_matched": example["base_matched"],
                "finetuned_matched": example["finetuned_matched"],
                "semantic_similarity": {
                    "query_to_reference": cosine(vectors["Query"], vectors["Reference"]),
                    "query_to_base_answer": cosine(vectors["Query"], vectors["Base answer"]),
                    "query_to_finetuned_answer": cosine(vectors["Query"], vectors["Fine-tuned answer"]),
                    "base_answer_to_reference": cosine(vectors["Base answer"], vectors["Reference"]),
                    "finetuned_answer_to_reference": cosine(vectors["Fine-tuned answer"], vectors["Reference"]),
                },
            }
        )

    coordinates = PCA(n_components=2, random_state=0).fit_transform(embeddings)
    points = []
    for (index, role), point in zip(text_refs, coordinates):
        example = examples[index - 1]
        points.append(
            {
                "row": index,
                "role": role,
                "group": example["group"],
                "report_number": example["report_number"],
                "question_type": example["question_type"],
                "x": float(point[0]),
                "y": float(point[1]),
                "label": f"#{index} {role}: Report #{example['report_number']} {example['question_type']}",
            }
        )

    payload = {
        "model": EMBEDDING_MODEL,
        "source": str(BEFORE_AFTER_PATH.relative_to(Path(__file__).resolve().parents[1])),
        "note": (
            "Semantic similarity is a contextual sentence-embedding signal, not a factual correctness score. "
            "Use answer-to-reference similarity together with exact-match, word overlap, and source text."
        ),
        "examples": rows,
        "points_2d": points,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved optional semantic embedding data -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
