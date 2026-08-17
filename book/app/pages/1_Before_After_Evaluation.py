"""Before vs. After Evaluation.

Instead of one cherry-picked prompt, this page runs Chapter 11's real
8-example held-out evaluation set (report #37, never in any training
set) against the base model and every fine-tuned checkpoint that
actually exists on this machine, and shows Chapter 12's own real
metric-disagreement finding rather than collapsing everything into one
score -- including when the latest checkpoint isn't the best one.
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
    n_questions = len(_cached_eval_set())
    labels = list(summaries)
    short = {label: helpers.short_version_label(label) for label in labels}

    # --- Evaluation snapshot -------------------------------------------------
    st.subheader("Evaluation snapshot")
    snapshot = helpers.evaluation_snapshot(summaries)
    cols = st.columns(4)
    ppx_labels, ppx_value = snapshot["best_perplexity"]
    ov_labels, ov_value = snapshot["best_overlap"]
    em_labels, em_value = snapshot["best_exact_match"]
    cols[0].metric("Best perplexity", f"{ppx_value:.2f}", ", ".join(short[l] for l in ppx_labels))
    cols[1].metric("Best overlap", f"{ov_value:.3f}", ", ".join(short[l] for l in ov_labels))
    em_winner_text = "Tie" if len(em_labels) > 1 else short[em_labels[0]]
    cols[2].metric("Best exact-match", f"{em_value}/{n_questions}", em_winner_text)
    cols[3].metric("Latest checkpoint", short[snapshot["latest"]])

    # --- Results table ---------------------------------------------------
    rows = [
        {
            "Version": label,
            "Exact-match": f"{s['exact_match']}/{n_questions}",
            "Avg. overlap score": round(s["avg_overlap"], 3),
            "Perplexity": round(s["perplexity"], 2),
        }
        for label, s in summaries.items()
    ]
    df = pd.DataFrame(rows).set_index("Version")
    st.dataframe(df, use_container_width=True)
    st.caption(
        "Exact-match is intentionally strict (Chapter 3's rule): the "
        "generated answer has to contain the reference answer's exact "
        "wording, word for word, to count -- getting the substance right "
        "in different words still scores 0. That's why every version can "
        "show `0` here despite real, measurable differences in overlap "
        "and perplexity below."
    )

    # --- What changed between versions (moved up: this is the interpretation) --
    if len(labels) > 1:
        st.subheader("What changed between versions")
        st.caption(
            "Chapter 12's own finding, preserved here rather than collapsed "
            "into one number: these metrics do not always agree on "
            "direction between two real versions."
        )

        if helpers.latest_regressed_on_both(summaries):
            prev_label, latest_label = labels[-2], labels[-1]
            prev, latest = summaries[prev_label], summaries[latest_label]
            st.warning(
                f"**Latest does not mean best.** {short[latest_label]} regressed "
                f"on both average overlap ({prev['avg_overlap']:.3f} → "
                f"{latest['avg_overlap']:.3f}) and perplexity "
                f"({prev['perplexity']:.2f} → {latest['perplexity']:.2f}) "
                f"compared to {short[prev_label]} -- continued fine-tuning "
                "did not automatically improve this checkpoint on this "
                "held-out set."
            )

        for before_label, after_label in zip(labels, labels[1:]):
            before, after = summaries[before_label], summaries[after_label]
            directions = helpers.compare_versions(before, after)
            st.write(f"**{short[before_label]} → {short[after_label]}**")

            ppx_change = helpers.relative_change(before["perplexity"], after["perplexity"])
            ov_change = helpers.relative_change(before["avg_overlap"], after["avg_overlap"])

            cols = st.columns(3)
            ppx_arrow = "↓" if directions["perplexity"] == "improved" else "↑" if directions["perplexity"] == "regressed" else "="
            cols[0].metric(
                "Perplexity",
                f"{ppx_arrow} {abs(ppx_change) * 100:.1f}%" if ppx_change is not None else f"{after['perplexity']:.2f}",
                directions["perplexity"],
            )
            if ov_change is None:
                cols[1].metric("Avg. overlap", f"{before['avg_overlap']:.3f} → {after['avg_overlap']:.3f}", directions["avg_overlap"])
            else:
                ov_arrow = "↑" if directions["avg_overlap"] == "improved" else "↓" if directions["avg_overlap"] == "regressed" else "="
                cols[1].metric("Avg. overlap", f"{ov_arrow} {abs(ov_change) * 100:.1f}%", directions["avg_overlap"])
            cols[2].metric("Exact-match", f"{after['exact_match']}/{n_questions}", directions["exact_match"])

    # --- Charts: split, not overlaid (perplexity and overlap have very ---
    # --- different scales -- on one shared axis overlap is invisible) ----
    # st.bar_chart's Vega-Lite backend sorts the x-axis alphabetically by
    # label text, not by row order -- "Ch13" would otherwise sort before
    # "Ch5"/"Ch8" as a string. A short numeric prefix keeps the chart in
    # the same chronological order as the table, while staying far
    # shorter than the old full descriptive labels.
    chart_order = {label: f"{i}. {short[label]}" for i, label in enumerate(labels)}

    st.subheader("Perplexity by version")
    st.caption("Lower is better.")
    ppx_df = pd.DataFrame(
        {"Perplexity": {chart_order[label]: s["perplexity"] for label, s in summaries.items()}}
    ).reindex(list(chart_order.values()))
    st.bar_chart(ppx_df)

    st.subheader("Average overlap by version")
    st.caption("Higher is better.")
    ov_df = pd.DataFrame(
        {"Avg. overlap score": {chart_order[label]: s["avg_overlap"] for label, s in summaries.items()}}
    ).reindex(list(chart_order.values()))
    st.bar_chart(ov_df)
