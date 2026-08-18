"""Before vs. After Evaluation.

Instead of one cherry-picked prompt, this page runs the real 8-example
held-out evaluation set (report #37, never in any training set) against
the base model and every fine-tuned checkpoint that actually exists on
this machine. Written for a reader with no AI/programming background:
every metric is defined in plain language before it's used, one real
example answer is shown side by side so the numbers have something
concrete to anchor to, and a `0/8` score is explained rather than left
to look like a failure.
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import altair as alt
import pandas as pd
import streamlit as st

import helpers

st.set_page_config(page_title="Before vs. After Evaluation", page_icon="🛢️")
st.title("Before vs. After Evaluation")
st.write(
    "This page runs a real evaluation on whatever fine-tuned checkpoint "
    "you've actually trained -- not one hand-picked question, but a full "
    "8-question test set, scored three different ways so no single "
    "number can hide the real picture."
)
st.caption(
    "For the curious: every score below is computed live by this book's "
    "own Chapter 11/12 evaluation code -- not a separate scoring path "
    "invented for this page."
)

st.markdown(
    """
| Term | What it actually means |
|---|---|
| **Held-out set** | 8 real questions built from one report the model was deliberately never shown during training -- the fairest test of whether it learned something transferable, not just memorized answers it already had. |
| **Perplexity** | How "surprised" the model is by real report text it's never seen. Lower is better -- a low number means the model finds this operation's report language unsurprising, the way someone who's read years of tour sheets from the same rig can predict how the next line will read. |
| **Overlap score** | How much of the model's own wording actually shows up in the real, correct answer, from `0` (nothing matches) to `1` (every word matches). This book treats `0.5` and above as a reasonably faithful match to the real report text. |
| **Exact-match** | The strictest possible test: the model's answer has to contain the correct answer's exact wording, word for word, to count. A low score here does **not** mean the model failed -- it means the wording didn't happen to reuse the reference text verbatim; overlap score above is the more realistic signal. |
"""
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


@st.cache_data(show_spinner="Generating one real example with both models...")
def _cached_example(checkpoint_dir_str: str | None) -> dict:
    eval_set = _cached_eval_set()
    # The same real report #37, 20:30-21:30 example already quoted in
    # this repo's own root README -- reused here so the same concrete
    # answer anchors the numbers on this page too.
    example = next((e for e in eval_set if "20:30-21:30" in e["input"]), eval_set[0])

    base_model, base_tokenizer = helpers.load_base_model()
    base_answer = helpers.generate_answer(base_model, base_tokenizer, example["instruction"], example["input"])

    finetuned_answer = None
    if checkpoint_dir_str:
        ft_model, ft_tokenizer = helpers.load_finetuned_model(Path(checkpoint_dir_str))
        finetuned_answer = helpers.generate_answer(ft_model, ft_tokenizer, example["instruction"], example["input"])

    return {
        "input": example["input"],
        "expected": example["output"],
        "base_answer": base_answer,
        "finetuned_answer": finetuned_answer,
    }


checkpoints = helpers.available_checkpoints()

versions: list[tuple[str, str | None]] = [("Base model (no fine-tuning)", None)]
versions += [(label, str(path)) for label, path in checkpoints.items()]
path_by_label = dict(versions)

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
    cols[1].metric("Best overlap", f"{ov_value:.3f} (≈{ov_value * 100:.0f}%)", ", ".join(short[l] for l in ov_labels))
    em_winner_text = "Tie" if len(em_labels) > 1 else short[em_labels[0]]
    cols[2].metric("Best exact-match", f"{em_value}/{n_questions}", em_winner_text)
    cols[3].metric("Latest checkpoint", short[snapshot["latest"]])

    # --- A real example, side by side (concrete before the abstract) ----
    st.subheader("A real example, side by side")
    st.caption(
        "Numbers are easier to trust once you've seen what they're "
        "actually scoring. This is a real question from the held-out "
        "set, generated live just now by the base model and by your "
        "latest fine-tuned checkpoint."
    )
    latest_label = snapshot["latest"]
    example = _cached_example(path_by_label.get(latest_label))
    st.markdown(f"**Question asked:** `{example['input']}`")
    ex_cols = st.columns(2)
    with ex_cols[0]:
        st.markdown("**Base model answered:**")
        st.write(example["base_answer"])
    with ex_cols[1]:
        st.markdown(f"**{short[latest_label]} (latest fine-tuned) answered:**")
        if example["finetuned_answer"] is not None:
            st.write(example["finetuned_answer"])
            ex_score = helpers.score_against_reference(example["finetuned_answer"], example["expected"])
            st.caption(f"Overlap with the real answer below: {ex_score['overlap_score']:.2f} (≈{ex_score['overlap_score'] * 100:.0f}%)")
        else:
            st.write("*(no fine-tuned checkpoint selected)*")
    st.markdown(f"**What the report actually says:** {example['expected']}")

    # --- Results table ---------------------------------------------------
    st.subheader("Results table")
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
        "A `0/8` here is not a failure verdict. Exact-match is the "
        "strictest possible test: the model's answer has to contain the "
        "reference answer's exact wording, word for word, to count -- "
        "getting the substance right in different words still scores 0. "
        "Overlap score is the more realistic measure of how close the "
        "wording actually was."
    )

    # --- What changed between versions (moved up: this is the interpretation) --
    if len(labels) > 1:
        st.subheader("What changed between versions")
        st.caption(
            "These signals don't always move the same direction between "
            "two versions -- perplexity can improve while overlap gets "
            "worse, or the other way around. This page shows both "
            "instead of collapsing them into a single verdict."
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
    # Built with explicit Altair specs (not st.bar_chart) for two reasons:
    # a forced domainMin=0 on the y-axis guarantees bars are never drawn
    # from a non-zero baseline (a non-zero baseline can make an
    # improvement look like a missing value), and an explicit `sort=`
    # order fixes the x-axis without needing st.bar_chart's numeric-prefix
    # workaround, so the axis can show clean "Base"/"Ch5"/"Ch8" labels.
    order = [short[label] for label in labels]

    st.subheader("Perplexity by version")
    st.caption("Lower is better.")
    ppx_df = pd.DataFrame({"Version": order, "Perplexity": [summaries[label]["perplexity"] for label in labels]})
    ppx_chart = (
        alt.Chart(ppx_df)
        .mark_bar()
        .encode(
            # labelAngle=0 (horizontal, not Vega-Lite's default rotated
            # labels) -- rotated short labels were getting truncated to a
            # single trailing character ("Ch13" showing as just "3") in
            # this narrow a chart; horizontal labels also read more
            # naturally for a reader unfamiliar with rotated chart axes.
            x=alt.X("Version", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Perplexity", scale=alt.Scale(domainMin=0)),
        )
    )
    st.altair_chart(ppx_chart, use_container_width=True)

    st.subheader("Average overlap by version")
    st.caption("Higher is better.")
    # Column name has no "." -- Altair/Vega-Lite's shorthand field syntax
    # treats a period as nested-property notation (e.g. "a.b" means field
    # b inside a), so "Avg. overlap score" silently resolved to nothing
    # and rendered an empty chart. Caught while verifying this in the
    # browser: the perplexity chart above rendered fine (no "." in its
    # column name) while this one was blank.
    ov_df = pd.DataFrame({"Version": order, "Avg overlap score": [summaries[label]["avg_overlap"] for label in labels]})
    ov_chart = (
        alt.Chart(ov_df)
        .mark_bar()
        .encode(
            x=alt.X("Version", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Avg overlap score", scale=alt.Scale(domainMin=0)),
        )
    )
    st.altair_chart(ov_chart, use_container_width=True)
