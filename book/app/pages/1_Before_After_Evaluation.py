"""Before vs. After Evaluation.

Instead of one cherry-picked prompt, this page runs Chapter 11's real
8-example held-out evaluation set (report #37, never in any training
set) against the base model and every fine-tuned checkpoint that
actually exists on this machine, and shows Chapter 12's own real
metric-disagreement finding rather than collapsing everything into one
score.
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import streamlit as st

import helpers

st.set_page_config(page_title="Before vs. After Evaluation", page_icon="🛢️")
st.title("Before vs. After Evaluation")
st.caption(
    "Chapter 12's summarize_version()/compare_versions(), run live against "
    "Chapter 11's real 8-question held-out set -- the same functions the "
    "book's own code uses, not a separate scoring path."
)


@st.cache_data(show_spinner=False)
def _cached_eval_set():
    return helpers.build_held_out_eval_set()


@st.cache_data(show_spinner=False)
def _cached_summary(version_label: str, checkpoint_dir_str: str | None) -> dict:
    # Loads a fresh base model per call, same as Chapter 12's own main()
    # loop does for each checkpoint it compares -- not an inefficiency
    # introduced here, the book's own script works the same way.
    checkpoint_dir = Path(checkpoint_dir_str) if checkpoint_dir_str else None
    return helpers.evaluate_checkpoint(version_label, checkpoint_dir, _cached_eval_set())


checkpoints = helpers.available_checkpoints()

versions: list[tuple[str, str | None]] = [("Base model (no fine-tuning)", None)]
versions += [(label, str(path)) for label, path in checkpoints.items()]

if not checkpoints:
    st.warning(
        "No fine-tuned checkpoint found yet -- only the base model can be "
        "shown. Run one of Chapter 5, 8, or 13's scripts first for a real "
        "before/after comparison (see the Model Playground page for exact "
        "commands)."
    )

st.write(f"Evaluating {len(versions)} version(s) against the real 8-question held-out set.")
run = st.button("Run evaluation", type="primary")

if run:
    summaries: dict[str, dict] = {}
    progress = st.progress(0.0)
    for i, (label, checkpoint_dir_str) in enumerate(versions):
        with st.spinner(f"Evaluating: {label}..."):
            summaries[label] = _cached_summary(label, checkpoint_dir_str)
        progress.progress((i + 1) / len(versions))
    progress.empty()

    st.session_state["before_after_summaries"] = summaries

summaries = st.session_state.get("before_after_summaries")
if summaries:
    rows = [
        {
            "Version": label,
            "Exact-match": f"{s['exact_match']}/{len(_cached_eval_set())}",
            "Avg. overlap score": round(s["avg_overlap"], 3),
            "Perplexity": round(s["perplexity"], 2),
        }
        for label, s in summaries.items()
    ]
    df = pd.DataFrame(rows).set_index("Version")
    st.dataframe(df, use_container_width=True)

    # st.bar_chart's Vega-Lite backend sorts the x-axis alphabetically by
    # label text, not by row order -- "Chapter 13" would otherwise sort
    # before "Chapter 5" as a string. A numeric prefix keeps the chart in
    # the same chronological order as the table above.
    ordered_labels = {label: f"{i}. {label}" for i, label in enumerate(summaries)}
    chart_df = pd.DataFrame(
        {
            "Avg. overlap score": {ordered_labels[label]: s["avg_overlap"] for label, s in summaries.items()},
            "Perplexity": {ordered_labels[label]: s["perplexity"] for label, s in summaries.items()},
        }
    ).reindex(list(ordered_labels.values()))
    st.bar_chart(chart_df)
    st.caption("Lower perplexity is better; higher overlap score is better.")

    labels = list(summaries)
    if len(labels) > 1:
        st.subheader("What changed between versions")
        st.caption(
            "Chapter 12's own finding, preserved here rather than collapsed "
            "into one number: these three metrics do not always agree on "
            "direction between two real versions."
        )
        for before_label, after_label in zip(labels, labels[1:]):
            directions = helpers.compare_versions(summaries[before_label], summaries[after_label])
            st.write(f"**{before_label} → {after_label}**")
            cols = st.columns(3)
            for col, (metric, direction) in zip(cols, directions.items()):
                col.metric(metric.replace("_", " "), direction)
