"""Chapter 10 challenge exercise: does the faithfulness check also catch
hallucination with no retrieval context at all?

Reference solution. Reuses Chapter 9's TEST_CASES and Chapter 10's
faithfulness_score, generating each answer WITHOUT retrieval (Chapter
9's ungrounded baseline) and scoring it against the actual target
report's own retrieved chunk -- the strongest possible case for the
answer to be "faithful," if it happened to be right by coincidence.

Usage:
    python code/chapter_10/challenge/challenge.py
"""

import sys
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_01"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_02"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_06"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_07"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_09"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_10"))

# load_local_model sets USE_TF=0/USE_FLAX=0 before its own transformers
# import -- must stay first.
from load_local_model import MODEL_NAME, generate_reply, load_model_and_tokenizer  # noqa: E402

from peft import PeftModel  # noqa: E402

from hybrid_rag_finetune import (  # noqa: E402
    INSTRUCTION,
    TEST_CASES,
    build_bm25_index,
    build_retrieval_corpus,
    latest_checkpoint,
    retrieve,
)
from traceable_outputs import faithfulness_score  # noqa: E402


def main() -> None:
    corpus = build_retrieval_corpus()
    bm25 = build_bm25_index(corpus)
    checkpoint_dir = latest_checkpoint()
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    lora_model = PeftModel.from_pretrained(model, checkpoint_dir)

    for label, input_context, query, target_report in TEST_CASES:
        answer = generate_reply(lora_model, tokenizer, f"{INSTRUCTION}\n{input_context}", max_new_tokens=60)
        target_chunks = [c for c in retrieve(query, corpus, bm25, k=5) if c["report_num"] == target_report]
        best = max((faithfulness_score(answer, c["text"]) for c in target_chunks), default=0.0)
        print(f"{label}:")
        print(f"  Without-retrieval answer: {answer}")
        print(f"  Best faithfulness against its own target report's text: {best:.2f}")


if __name__ == "__main__":
    main()
