# Companion App: Model Comparison Laboratory

**Status: V1, V2, and V3 frozen** -- three real, working pages complete
one coherent evidence chain: training data -> model behavior ->
failures. None of V1/V2/V3 is revisited for new *pages*; the freeze is
about scope (no Experiment Explorer, no generic fine-tuning platform),
not about polish -- V1's Before vs. After Evaluation was reworked after
shipping, per real usability feedback, into the definitive
model-evolution page described below.

The app now answers three clear questions, an unusually coherent
structure for a book companion:

- **Dataset Explorer (V2) — "What did we train on?"** Browse the real
  training examples Chapter 2/7 actually build, with real Chapter 6
  quality-gate flags and provenance, not invented metadata.
- **Before vs. After Evaluation (V1) — "What changed as we fine-tuned?"**
  Polished into the definitive model-evolution page rather than
  building a separate Experiment Explorer, and written for a reader
  with no AI/programming background: a plain-English glossary up top
  defines every metric before it's used (Perplexity, Overlap score,
  Exact-match, Held-out set), a real held-out example is generated live
  and shown side by side (base vs. latest fine-tuned answer, with the
  real correct answer alongside) so the numbers have something concrete
  to anchor to, and a `0/8` exact-match score is explicitly explained as
  not a failure verdict. An evaluation snapshot up top shows best
  perplexity/overlap/exact-match and the latest checkpoint side by side
  so disagreement between them is impossible to miss. Perplexity and
  average overlap are charted separately with a forced zero-baseline
  y-axis (their scales are roughly 25-160 vs. 0-0.6 -- one shared axis,
  or a non-zero baseline, both make overlap effectively invisible or
  make an improvement look like missing data), with relative change
  shown between consecutive versions, and an explicit callout whenever
  the latest checkpoint regresses on both continuous metrics compared
  to the one before it -- latest doesn't automatically mean best, and
  this page says so rather than hiding it.
- **Failure Analysis (V3) — "Where and how does it fail?"** The same 4
  real Chapter 9/10 retrieval test cases run live -- including report
  #21, the book's headline case of a fluent, verified-faithful answer
  grounded in the wrong report -- each paired with what the book itself
  found running that case, plus a live "shape, not judgment" detector
  across Chapter 11's held-out set.

An Experiment Explorer was considered for V4 and deliberately dropped:
Before vs. After Evaluation already has the necessary evidence once
polished, and a separate read-only checkpoint-config page would have
added a fourth "clear question" without adding a new one. A
"Fine-Tuning or RAG?" page remains planned, further out.

This progression -- book chapters -> real training code -> locally
generated checkpoints -> companion app -> live comparison -> reproduced
evaluation results -- makes the app part of the book's reproducibility
story, not a separate demo built beside it. Every page reads real,
locally generated artifacts (or generates straight from the real
archive) and never a pre-recorded number. Verified live: with Chapter
8's checkpoint, V3's report #21 case reproduces the book's exact
documented result -- "Test choke manifold at 5000psi," grounded: yes,
verified against report #27 at faithfulness 0.75, not the real target
-- and the shape detector's top-ranked pair reproduces the exact
"Production Casing Run Csg & Cement Rig up casing" answer already
quoted in this repo's own root README.

This is an educational companion, not a production system, mirroring
the author's previous book's companion app.

## Run it

```bash
pip install -r book/requirements.txt      # includes Streamlit
cd book
streamlit run app/streamlit_app.py
```

You'll get the most out of this with at least one real fine-tuned
checkpoint on disk -- `checkpoints/` is gitignored, so a fresh clone
starts empty. **You don't have to train anything before launching,
though**: every page that needs a checkpoint checks what actually
exists first, and if nothing does, shows setup instructions and stops
cleanly instead of crashing. Dataset Explorer needs no checkpoint at
all -- it reads straight from the real archive. To get a real
checkpoint, run any of:

```bash
python code/chapter_05/first_lora_finetune.py     # ~5 min on CPU, 16 examples
python code/chapter_08/finetune_at_scale.py       # ~30 min on CPU, 669 examples
python code/chapter_13/continuous_finetune.py     # continuous fine-tune
```

then reload the app -- no restart needed. Every checkpoint-dependent
page detects which checkpoints actually exist and only offers those; no
page assumes a specific one is present.

## What it reuses from the book

Nothing here reimplements the pipeline. Every model load, generation,
retrieval, and score is imported directly from the book's own chapter
code:

| Step | Comes from |
|---|---|
| Base model loading, generation | `code/chapter_01/load_local_model.py` |
| Sample-set training examples | `code/chapter_02/build_training_examples.py` |
| Quality-gate status, duplicate flags | `code/chapter_06/data_quality_gate.py` |
| Full-archive timeline training examples | `code/chapter_07/format_training_chunks.py` |
| Retrieval corpus, BM25 index | `code/chapter_09/hybrid_rag_finetune.py` |
| Grounded, traceable answers | `code/chapter_10/traceable_outputs.py` |
| Held-out eval set, exact-match | `code/chapter_11/eval_finetuned_model.py` |
| Checkpoint loading (eval-safe), before/after comparison | `code/chapter_12/detect_model_drift.py` |

Checkpoint loading always goes through Chapter 12's `load_version()`,
never Chapter 8's `load_checkpoint()` -- the latter sets
`is_trainable=True`, leaving dropout active and making generation
non-deterministic. That's a real bug Chapter 13's own CHANGELOG entry
documents catching and fixing; this app doesn't reintroduce it.

`app/helpers.py` holds all of this glue logic with **no Streamlit
import**, so it stays directly unit-testable (`tests/test_app.py`).
`app/streamlit_app.py` and `app/pages/*.py` are only the screens around
it, plus `st.cache_resource`/`st.cache_data` caching so the ~1.5B
parameter model isn't reloaded on every interaction.

## Metrics shown

Only metrics that already exist as real, reusable functions in this
book: **exact-match**, **overlap/faithfulness score**, **perplexity**,
and a **grounded/ungrounded** flag when retrieval is used. Nothing here
computes "instruction following," "terminology adherence," "expected
concepts covered," or a confidence score -- those aren't computed
anywhere in the book, and this app follows the same non-negotiable rule
the chapters do: never invent a number.

## Files

| File | Purpose |
|---|---|
| `streamlit_app.py` | Model Playground (landing page): checkpoint picker, prompt source, base vs. fine-tuned vs. optional retrieval |
| `pages/1_Before_After_Evaluation.py` | The definitive model-evolution page: live evaluation snapshot, split perplexity/overlap charts, relative change between versions, and an explicit "latest doesn't mean best" callout when it applies |
| `pages/2_Dataset_Explorer.py` | Browse the real training examples Chapter 2/7 build, filterable by report number, date, time window, and Chapter 6's real quality-gate flags |
| `pages/3_Failure_Analysis.py` | Live rerun of Chapter 9/10's 4 real retrieval test cases against your checkpoint, paired with the book's documented finding for each, plus a live shape-not-judgment detector |
| `helpers.py` | All model-loading, generation, retrieval, dataset, and scoring glue (no Streamlit; unit-testable) |

## Planned, not yet built

- **Fine-Tuning or RAG?** -- a decision-guide page cross-referencing
  Chapter 3's "when fine-tuning is, and is not, the right tool" section
  with the author's companion RAG book. An Experiment Explorer was
  considered and deliberately dropped -- see "Status" above.
