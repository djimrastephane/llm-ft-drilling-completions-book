"""Dataset Explorer.

Browse the book's own real training examples -- not a mock-up. Every
row is generated live from the real Utah FORGE PDFs the same way
Chapter 2, Chapter 6, and Chapter 7 build them, annotated only with
fields that are actually derivable: the report number, date, and time
window parsed straight out of each example's own `input` string, plus
whether Chapter 6's real quality gate flagged that report as a
non-consecutive duplicate. There's no domain/topic/difficulty label
here -- this book's real training data doesn't have one, so this page
doesn't invent one.
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import streamlit as st

import helpers

st.set_page_config(page_title="Dataset Explorer", page_icon="🛢️")
st.title("Dataset Explorer")
st.caption(
    "The behavior of a fine-tuned model depends on exactly what it was "
    "shown. These are the real instruction/input/output examples this "
    "book's own code builds from the real archive -- not a summary of "
    "them."
)


@st.cache_data(show_spinner="Extracting real examples from the archive (first run only)...")
def _cached_examples(bucket: str) -> list[dict]:
    return helpers.dataset_examples(bucket)


bucket_label = st.selectbox("Example set", list(helpers.DATASET_BUCKETS))
bucket = helpers.DATASET_BUCKETS[bucket_label]
examples = _cached_examples(bucket)

report_nums = sorted({e["report_num"] for e in examples if e["report_num"] is not None})
col1, col2 = st.columns([2, 1])
with col1:
    selected_reports = st.multiselect("Filter by report number", report_nums)
with col2:
    review_only = st.checkbox("Flagged for review only", value=False)
search = st.text_input("Search instruction/input/output text", "")

filtered = examples
if selected_reports:
    filtered = [e for e in filtered if e["report_num"] in selected_reports]
if review_only:
    filtered = [e for e in filtered if e["flagged_for_review"]]
if search:
    needle = search.lower()
    filtered = [
        e
        for e in filtered
        if needle in e["instruction"].lower() or needle in e["input"].lower() or needle in e["output"].lower()
    ]

st.write(f"{len(filtered)} of {len(examples)} real examples match.")

if filtered:
    table_df = pd.DataFrame(
        [
            {
                "Report #": e["report_num"],
                "Date": e["date"],
                "Time window": e["time_window"] or "—",
                "Chunk": e["chunk_part"] or "—",
                "Flagged for review": "yes" if e["flagged_for_review"] else "",
                "Output (preview)": (e["output"][:80] + "…") if len(e["output"]) > 80 else e["output"],
            }
            for e in filtered
        ]
    )
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.subheader("Inspect one example")
    row_labels = [f"#{i} -- Report {e['report_num']}, {e['date']}" for i, e in enumerate(filtered)]
    chosen = st.selectbox("Pick a row from the table above", row_labels)
    example = filtered[row_labels.index(chosen)]

    st.markdown(f"**Instruction:** {example['instruction']}")
    st.markdown(f"**Input:** `{example['input']}`")
    st.markdown(f"**Output (the real, expected answer):** {example['output']}")
    if example["held_out"]:
        st.info(
            "This example is from report #37 -- deliberately held out of "
            "every training run in this book, so Chapter 5+ can measure "
            "generalization, not just training recall."
        )
    if example["flagged_for_review"]:
        st.warning(
            f"Report #{example['report_num']}'s present-operations or "
            "activity-planned text is a near-duplicate of another, "
            "non-consecutive report's -- Chapter 6's quality gate flags "
            "this as a decision still owed to a human, not one it made "
            "automatically."
        )
else:
    st.info("No examples match these filters.")
