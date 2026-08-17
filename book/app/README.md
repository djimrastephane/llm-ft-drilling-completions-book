# Companion App: Model Comparison Laboratory

**Status: V1 and V2 frozen, V3 planned** -- three of six planned pages
are real and working. Neither V1 nor V2 is revisited for new features;
new pages ship as their own labeled version, the same one-piece-at-a-time
cadence the book's own chapters use.

- **V1 — "What changed after fine-tuning?"** -- **Model Playground**
  (base vs. fine-tuned vs. optional retrieval, on the same real prompt)
  and **Before vs. After Evaluation** (the book's own metrics run live
  across the full held-out set, not one cherry-picked example).
- **V2 — "What data produced that change?"** -- **Dataset Explorer**
  (browse the real training examples Chapter 2/7 actually build, with
  real Chapter 6 quality-gate flags and provenance, not invented
  metadata).
- **V3 (planned) — "Where does it fail?"** -- **Failure Analysis**:
  curated real wrong-answer cases already documented in the book's own
  Field Notes (e.g. Chapter 10's report #21 wrong-source case), plus live
  comparison for new prompts. Chosen deliberately over adding more
  general UI features, so the app completes one coherent evidence chain
  -- training data -> model behavior -> failures -- rather than growing
  into a generic fine-tuning platform.
- **Planned, further out** -- Experiment Explorer and a "Fine-Tuning or
  RAG?" page -- same honesty convention the book itself uses for chapter
  status, applied to this app.

This progression -- book chapters -> real training code -> locally
generated checkpoints -> companion app -> live comparison -> reproduced
evaluation results -- makes the app part of the book's reproducibility
story, not a separate demo built beside it. Every page reads real,
locally generated artifacts (or generates straight from the real
archive) and never a pre-recorded number.

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
though**: both pages check what actually exists first, and if nothing
does, they show setup instructions and stop cleanly instead of
crashing. To get a real checkpoint, run any of:

```bash
python code/chapter_05/first_lora_finetune.py     # ~5 min on CPU, 16 examples
python code/chapter_08/finetune_at_scale.py       # ~30 min on CPU, 669 examples
python code/chapter_13/continuous_finetune.py     # continuous fine-tune
```

then reload the app -- no restart needed. Both pages detect which
checkpoints actually exist and only offer those; no page assumes a
specific one is present.

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
| `pages/1_Before_After_Evaluation.py` | Live before/after evaluation across the real held-out set, for every checkpoint present |
| `pages/2_Dataset_Explorer.py` | Browse the real training examples Chapter 2/7 build, filterable by report number, date, time window, and Chapter 6's real quality-gate flags |
| `helpers.py` | All model-loading, generation, retrieval, dataset, and scoring glue (no Streamlit; unit-testable) |

## Planned, not yet built

- **V3 -- Failure Analysis** -- curated real wrong-answer cases already
  documented in the book's own Field Notes (e.g. Chapter 10's report
  #21 wrong-source case), plus live comparison for new prompts. See
  "Status" above for why this is next.
- **Experiment Explorer** -- read-only inspection of real checkpoint
  configs and logged metrics (no in-app training).
- **Fine-Tuning or RAG?** -- a decision-guide page cross-referencing
  Chapter 3's "when fine-tuning is, and is not, the right tool" section
  with the author's companion RAG book.
