"""Failure Analysis.

Where fine-tuning fails, not just where it works. Two real, live checks
against your selected checkpoint:

1. The 4 real Chapter 9/10 retrieval test cases -- including report
   #21, the book's own headline case of a fluent, verified-faithful
   answer grounded in the WRONG report -- run live, paired with what the
   book itself found running the same cases.
2. A live "shape, not judgment" detector: do different questions from
   Chapter 11's held-out set get suspiciously similar answers?

Nothing here is a fixed screenshot. Your checkpoint may or may not
reproduce the book's exact documented finding -- that difference is
itself real information, not a bug in this page.
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import streamlit as st

import helpers

st.set_page_config(page_title="Failure Analysis", page_icon="🛢️")
st.title("Failure Analysis")
st.caption(
    "A polished chatbot hides its failures. This page doesn't: every "
    "case below is run live against your selected checkpoint, including "
    "the ones the book itself got wrong."
)


@st.cache_resource(show_spinner="Loading checkpoint...")
def _cached_model(checkpoint_dir_str: str):
    return helpers.load_finetuned_model(checkpoint_dir_str)


@st.cache_resource(show_spinner="Building the Chapter 9 retrieval index (first run only)...")
def _cached_retrieval_index():
    corpus = helpers.build_retrieval_corpus()
    bm25 = helpers.build_bm25_index(corpus)
    return corpus, bm25


@st.cache_data(show_spinner=False)
def _cached_held_out_eval_set():
    return helpers.build_held_out_eval_set()


checkpoints = helpers.available_checkpoints()
if not checkpoints:
    st.warning(
        "No fine-tuned checkpoint found yet. `checkpoints/` is gitignored, "
        "so a fresh clone starts empty -- run one of these first, then "
        "reload this page:\n\n"
        "- `python code/chapter_05/first_lora_finetune.py` (~5 min on CPU)\n"
        "- `python code/chapter_08/finetune_at_scale.py` (~30 min on CPU)\n"
        "- `python code/chapter_13/continuous_finetune.py`"
    )
    st.stop()

checkpoint_label = st.selectbox("Fine-tuned checkpoint", list(checkpoints))
checkpoint_dir = checkpoints[checkpoint_label]
model, tokenizer = _cached_model(str(checkpoint_dir))

st.header("1. Known failure cases (Chapter 9 & 10)")
st.caption(
    "The same 4 real, curated retrieval test cases from Chapter 9/10 -- "
    "run live here, not replayed from a transcript."
)

corpus, bm25 = _cached_retrieval_index()

for label, input_context, query, target_report in helpers.RETRIEVAL_TEST_CASES:
    kind, finding = helpers.DOCUMENTED_FINDINGS[label]
    icon = {"success": "✅", "failure": "⚠️", "blind spot": "🔍"}[kind]
    with st.expander(f"{icon} {label}", expanded=(kind == "failure")):
        with st.spinner("Retrieving and generating..."):
            result = helpers.answer_with_traceable_sources(
                model, tokenizer, helpers.RETRIEVAL_INSTRUCTION, input_context, query, corpus, bm25
            )
        st.write(result["answer"])
        st.metric("Grounded", "yes" if result["grounded"] else "no")
        for chunk in result["retrieved"]:
            mark = "✅ verified" if chunk["faithfulness"] >= 0.5 else "— not used"
            wrong_target = chunk["report_num"] != target_report and chunk["faithfulness"] >= 0.5
            line = (
                f"{mark} · Report #{chunk['report_num']} {chunk['from_time']}-{chunk['to_time']} "
                f"(faithfulness {chunk['faithfulness']:.2f})"
            )
            if wrong_target:
                line += f" -- **not the real target (report #{target_report})**"
            st.caption(line)
        st.info(f"**What the book found here:** {finding}")

st.header("2. Shape, not judgment: do different questions get the same answer?")
st.caption(
    "Chapters 3/8/9/11/12 all document the same pattern: a small "
    "fine-tune can learn this archive's vocabulary and phrasing readily, "
    "while still answering different questions with a near-identical "
    "template. This runs your selected checkpoint against Chapter 11's "
    "real 8-question held-out set and checks how similar its answers "
    "are to each other -- not to the correct answer."
)

if st.button("Run shape check", type="primary"):
    eval_set = _cached_held_out_eval_set()
    answers = {}
    progress = st.progress(0.0)
    for i, example in enumerate(eval_set):
        with st.spinner(f"Generating answer {i + 1}/{len(eval_set)}..."):
            answers[example["input"]] = helpers.generate_answer(model, tokenizer, example["instruction"], example["input"])
        progress.progress((i + 1) / len(eval_set))
    progress.empty()
    st.session_state["failure_analysis_answers"] = answers

answers = st.session_state.get("failure_analysis_answers")
if answers:
    pairs = helpers.pairwise_answer_similarity(answers)
    df = pd.DataFrame(
        [
            {
                "Question A": p["question_a"],
                "Question B": p["question_b"],
                "Answer similarity": p["similarity"],
            }
            for p in pairs
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    high_similarity = [p for p in pairs if p["similarity"] >= 0.5]
    if high_similarity:
        st.warning(
            f"{len(high_similarity)} of {len(pairs)} question pairs scored "
            "0.5 or higher -- these two different questions got "
            "suspiciously similar answers, consistent with a memorized "
            "template rather than a judgment about each question's own "
            "content."
        )
        worst = high_similarity[0]
        st.write(f"**Most similar pair** ({worst['similarity']}):")
        st.write(f"- *{worst['question_a']}* → {answers[worst['question_a']]}")
        st.write(f"- *{worst['question_b']}* → {answers[worst['question_b']]}")
    else:
        st.success(
            "No question pair scored 0.5 or higher on this checkpoint -- "
            "no strong shape-not-judgment signal detected in this run."
        )
