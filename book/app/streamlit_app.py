"""Model Playground -- the companion app's landing page.

Ask the same real prompt of the base model and a fine-tuned checkpoint,
side by side, and see the book's own real evaluation metrics on the
answers wherever a reference makes that possible. Nothing here
regenerates or reimplements a chapter's logic -- every model load,
generation, retrieval, and score comes straight from `app/helpers.py`,
which itself only imports the book's own chapter code.

Run:
    streamlit run book/app/streamlit_app.py
"""

import streamlit as st

import helpers

st.set_page_config(page_title="Fine-Tuned Drilling & Completions LLM", page_icon="🛢️")


@st.cache_resource(show_spinner="Loading base model (first run only)...")
def _cached_base_model():
    return helpers.load_base_model()


@st.cache_resource(show_spinner="Loading fine-tuned checkpoint...")
def _cached_finetuned_model(checkpoint_dir_str: str):
    return helpers.load_finetuned_model(checkpoint_dir_str)


@st.cache_resource(show_spinner="Building the Chapter 9 retrieval index (first run only)...")
def _cached_retrieval_index():
    corpus = helpers.build_retrieval_corpus()
    bm25 = helpers.build_bm25_index(corpus)
    return corpus, bm25


@st.cache_data(show_spinner=False)
def _cached_held_out_eval_set():
    return helpers.build_held_out_eval_set()


st.title("Model Playground")
st.caption(
    "Base model → fine-tuned model → the same prompt → side by side. "
    "Every answer below is generated live on your own machine; nothing "
    "here is pre-recorded."
)

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

prompt_source = st.radio(
    "Prompt source",
    [
        "Held-out evaluation question (Chapter 11)",
        "Retrieval demo case (Chapter 9 & 10)",
        "Your own question",
    ],
    help=(
        "Held-out questions and retrieval demo cases come with a real "
        "reference answer or a curated retrieval query, so real scores "
        "can be shown. A free-text question has neither, so it's shown "
        "without a fabricated score."
    ),
)

instruction = input_context = query = expected = None
show_retrieval = False

if prompt_source == "Held-out evaluation question (Chapter 11)":
    eval_set = _cached_held_out_eval_set()
    labels = [e["input"] for e in eval_set]
    choice = st.selectbox("Question (from report #37, never in any training set)", labels)
    example = eval_set[labels.index(choice)]
    instruction = "What happened on this well during this time window?"
    input_context = example["input"]
    expected = example["output"]

elif prompt_source == "Retrieval demo case (Chapter 9 & 10)":
    case_labels = [c[0] for c in helpers.RETRIEVAL_TEST_CASES]
    choice = st.selectbox("Test case", case_labels)
    _, input_context, query, target_report = helpers.RETRIEVAL_TEST_CASES[case_labels.index(choice)]
    instruction = helpers.RETRIEVAL_INSTRUCTION
    show_retrieval = True
    st.caption(f"Retrieval query: *\"{query}\"* -- real answer lives in report #{target_report}")

else:
    instruction = st.text_input("Ask a drilling or completions question", "")
    if not instruction:
        st.stop()

base_model, base_tokenizer = _cached_base_model()
finetuned_model, finetuned_tokenizer = _cached_finetuned_model(str(checkpoint_dir))

with st.spinner("Generating..."):
    base_answer = helpers.generate_answer(base_model, base_tokenizer, instruction, input_context or "")
    finetuned_answer = helpers.generate_answer(finetuned_model, finetuned_tokenizer, instruction, input_context or "")

columns = st.columns(3 if show_retrieval else 2)

with columns[0]:
    st.subheader("Base model")
    st.write(base_answer)
    if expected is not None:
        score = helpers.score_against_reference(base_answer, expected)
        st.metric("Exact-match", "yes" if score["exact_match"] else "no")
        st.metric("Overlap score", f"{score['overlap_score']:.2f}")

with columns[1]:
    st.subheader("Fine-tuned model")
    st.write(finetuned_answer)
    if expected is not None:
        score = helpers.score_against_reference(finetuned_answer, expected)
        st.metric("Exact-match", "yes" if score["exact_match"] else "no")
        st.metric("Overlap score", f"{score['overlap_score']:.2f}")

if show_retrieval:
    corpus, bm25 = _cached_retrieval_index()
    with st.spinner("Retrieving and grounding..."):
        result = helpers.answer_with_traceable_sources(
            finetuned_model, finetuned_tokenizer, instruction, input_context, query, corpus, bm25
        )
    with columns[2]:
        st.subheader("Fine-tuned + retrieval")
        st.write(result["answer"])
        st.metric("Grounded", "yes" if result["grounded"] else "no")
        for chunk in result["retrieved"]:
            mark = "✅ verified" if chunk["faithfulness"] >= 0.5 else "— not used"
            st.caption(
                f"{mark} · Report #{chunk['report_num']} {chunk['from_time']}-{chunk['to_time']} "
                f"(faithfulness {chunk['faithfulness']:.2f})"
            )

if expected is not None:
    st.divider()
    st.caption(f"Reference answer (report #37's own real text): {expected}")

st.divider()
st.caption(
    "A base and fine-tuned answer that look similarly fluent but score "
    "very differently on exact-match or overlap is the book's own "
    "\"shape, not judgment\" pattern (Chapters 3, 8, 9, 11, 12): "
    "fine-tuning teaches this archive's vocabulary and phrasing readily, "
    "but recalling one specific fact from one specific report is a "
    "separate, harder problem -- see the Before/After Evaluation page "
    "for that measured across the whole held-out set, not one prompt."
)
