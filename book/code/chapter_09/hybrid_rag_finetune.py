"""Chapter 9: Hybrid System — Combining Fine-Tuning with Retrieval.

Chapter 3 already showed fine-tuning alone is a weak tool for citing
which report an answer came from, and Chapter 8 measured exactly how
weak: 0/50 exact-match on its own training sample. This chapter builds
a real retrieval layer over the full archive -- every report Chapter
6's gate passed, held-out report included, since retrieval's job is to
find facts regardless of what the fine-tuned model happened to train
on -- and combines it with Chapter 8's fine-tuned checkpoint so an
answer can be both fluent in the domain (fine-tuning) and traceable to
a specific report and time window (retrieval).

Retrieval here uses BM25 keyword search (`rank_bm25`), not dense
sentence embeddings -- Chapter 4 already showed general-purpose
embeddings struggle with this archive's specific vocabulary, and this
chapter confirms it again at the sentence level before choosing BM25
instead. See "Why not dense embeddings?" in the chapter text.

Usage:
    python code/chapter_09/hybrid_rag_finetune.py
"""

import sys
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_01"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_02"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_06"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_07"))

# load_local_model sets USE_TF=0/USE_FLAX=0 before its own transformers
# import -- see Chapter 3/5/8 for why this has to happen first.
from load_local_model import MODEL_NAME, generate_reply, load_model_and_tokenizer  # noqa: E402

import numpy as np  # noqa: E402
from peft import PeftModel  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

from build_training_examples import HELD_OUT_REPORT, extract_text  # noqa: E402
from data_quality_gate import FULL_SET_DIR, build_archive_records  # noqa: E402
from format_training_chunks import chunk_text, contains_cid_artifact, parse_timeline_entries  # noqa: E402

CHECKPOINTS_DIR = BOOK_ROOT / "checkpoints" / "chapter_08"


def build_retrieval_corpus(full_dir: Path = FULL_SET_DIR) -> list[dict]:
    """Every quality-gated report's timeline chunks -- including the held-out
    report. Retrieval isn't training: it should be able to find facts about
    any report, whether or not the fine-tuned model ever trained on it.
    """
    records = build_archive_records(full_dir)
    corpus = []
    for record in records:
        if record["status"] != "ok":
            continue
        text = extract_text(full_dir / record["file"])
        for entry in parse_timeline_entries(text):
            for chunk in chunk_text(entry["text"]):
                if contains_cid_artifact(chunk):
                    continue
                corpus.append(
                    {
                        "report_num": record["rpt_num"],
                        "from_time": entry["from_time"],
                        "to_time": entry["to_time"],
                        "text": chunk,
                    }
                )
    return corpus


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    return BM25Okapi([chunk["text"].lower().split() for chunk in corpus])


def retrieve(query: str, corpus: list[dict], bm25: BM25Okapi, k: int = 3) -> list[dict]:
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(-scores)[:k]
    return [{**corpus[i], "score": float(scores[i])} for i in top_indices]


def build_grounded_prompt(instruction: str, input_context: str, retrieved_chunks: list[dict]) -> str:
    sources = "\n".join(
        f"[Report #{c['report_num']}, {c['from_time']}-{c['to_time']}]: {c['text']}" for c in retrieved_chunks
    )
    return f"{instruction}\n{input_context}\n\nRelevant report excerpts:\n{sources}"


def answer_with_retrieval(model, tokenizer, instruction: str, input_context: str, query: str, corpus: list[dict], bm25: BM25Okapi, k: int = 3) -> dict:
    """`query` drives retrieval; `instruction`/`input_context` drive generation.

    These are deliberately separate. Chapter 7's templated input
    ("Well: ... | Report #NN | Time: HH:MM-HH:MM") is nearly identical
    across hundreds of chunks, so using it as the retrieval query finds
    almost nothing useful -- it carries metadata, not the actual
    information need. `query` is what a user would really type.
    """
    retrieved = retrieve(query, corpus, bm25, k=k)
    prompt = build_grounded_prompt(instruction, input_context, retrieved)
    answer = generate_reply(model, tokenizer, prompt, max_new_tokens=60)
    return {
        "instruction": instruction,
        "input": input_context,
        "query": query,
        "answer": answer,
        "sources": [{"report_num": c["report_num"], "from_time": c["from_time"], "to_time": c["to_time"]} for c in retrieved],
    }


def latest_checkpoint() -> Path:
    runs = sorted(CHECKPOINTS_DIR.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No runs in {CHECKPOINTS_DIR} -- run code/chapter_08/finetune_at_scale.py first")
    checkpoints = sorted(runs[-1].glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[1]))
    return checkpoints[-1]


# (label, input_context, retrieval query, target report number)
TEST_CASES = [
    ("Report #37 (held-out)", "Well: FORGE 16A [78]-32 | Report #37 | Time: 20:30-21:30", "trip out of hole stop at 5800 circulate cool hole and tools", 37),
    ("Report #38 (stuck pipe)", "Well: FORGE 16A [78]-32 | Report #38 | Time: 23:30-04:00", "lost tool face and became stuck during the slide", 38),
    ("Report #21 (step rate test)", "Well: FORGE 16A [78]-32 | Report #21 | Time: 15:30-16:00", "step rate test holding gpm constant", 21),
    ("Report #49 (fishing)", "Well: FORGE 16A [78]-32 | Report #49 | Time: 23:30-04:00", "fishing operations to recover lost equipment", 49),
]
INSTRUCTION = "What happened on this well during this time window?"


def main() -> None:
    print("Building retrieval corpus...")
    corpus = build_retrieval_corpus()
    bm25 = build_bm25_index(corpus)
    print(f"{len(corpus)} retrievable chunks across {len({c['report_num'] for c in corpus})} reports")

    checkpoint_dir = latest_checkpoint()
    print(f"Loading fine-tuned model from {checkpoint_dir}...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    lora_model = PeftModel.from_pretrained(model, checkpoint_dir)

    print("\nRetrieving with the templated input itself as the query (metadata only):")
    for r in retrieve(f"{INSTRUCTION} {TEST_CASES[0][1]}", corpus, bm25, k=3):
        print(f"  [{r['score']:.2f}] Report #{r['report_num']} {r['from_time']}-{r['to_time']}: {r['text'][:70]}")

    retrieval_hits = 0
    for label, input_context, query, target_report in TEST_CASES:
        print(f"\n=== {label} ===")
        retrieved = retrieve(query, corpus, bm25, k=3)
        top_reports = [r["report_num"] for r in retrieved]
        hit = target_report in top_reports
        retrieval_hits += hit
        print(f"Query: {query!r} -> top-3 reports {top_reports} (hit: {hit})")

        answer_alone = generate_reply(lora_model, tokenizer, f"{INSTRUCTION}\n{input_context}", max_new_tokens=60)
        print(f"Without retrieval: {answer_alone}")

        result = answer_with_retrieval(lora_model, tokenizer, INSTRUCTION, input_context, query, corpus, bm25)
        print(f"With retrieval:    {result['answer']}")

    print(f"\nRetrieval found the correct report in the top 3 for {retrieval_hits}/{len(TEST_CASES)} test queries")


if __name__ == "__main__":
    main()
