# Changelog

All notable changes to *Fine-Tuning Local LLM for Drilling & Completions*
are documented in this file. See [RELEASE.md](RELEASE.md) for per-release
highlights in narrative form.

## [Unreleased]

### Added

- Initial repository scaffold: Quarto book project, chapter/code/test
  structure, CI workflows, licensing, and placeholder content for all
  14 planned chapters (Part 0 + Part I + Part II).
- Training dataset: the full 76-report Utah FORGE archive and the same
  10-report curated subset used in the author's previous book
  (`ddr-rag-book`), plus a script to reproduce the sample tier.
- **Part 0: Preparing Your Local LLM Workshop** (`chapters/chapter_00.qmd`)
  fully drafted — Python install, cloning the repository, virtual
  environment setup, installing requirements, a hardware (GPU/CPU)
  check, choosing an editor, running `setup_check.py`, and
  troubleshooting. A callout right after "Why this chapter exists" makes
  the already-know-Python shortcut path (skip 0.2/0.6/0.7, still do
  0.3/0.4/0.5/0.8) visually distinct instead of buried in prose.
- **Chapter 1: Loading and Running Your First Local LLM**
  (`chapters/chapter_01.qmd`) fully drafted, with working, tested code
  (`code/chapter_01/load_local_model.py`,
  `code/chapter_01/challenge/challenge.py`,
  `tests/test_chapter_01.py`). Establishes Qwen2.5-1.5B-Instruct as the
  base model this book fine-tunes starting in Chapter 5. Every quoted
  report excerpt and model output in this chapter is from a real,
  verified run against the committed sample dataset -- nothing invented.
- **Chapter 2: Turning Drilling & Completions Reports into Training
  Examples** (`chapters/chapter_02.qmd`) fully drafted, with working,
  tested code (`code/chapter_02/build_training_examples.py`,
  `code/chapter_02/challenge/challenge.py`, `tests/test_chapter_02.py`).
  Extracts each report's own self-reported PRESENT OPERATIONS / ACTIVITY
  PLANNED fields with `pdfplumber` and turns them into 16 real
  instruction/response training examples from 8 of the 10 sample
  reports (the completion report uses a different field layout, handled
  in the challenge exercise). One further drilling report,
  `Drilling_037`, is deliberately reserved as held-out data
  (`HELD_OUT_REPORT`) -- never included in training by any script in
  this book -- so Chapter 5 can measure generalization, not just
  training-set recall. Every field value and example count in the
  chapter is from a real, verified run.
- **Chapter 3: Baseline Prompting: What the Model Gets Wrong Before
  Fine-Tuning** (`chapters/chapter_03.qmd`) fully drafted, with working,
  tested code (`code/chapter_03/baseline_prompting.py`,
  `code/chapter_03/challenge/challenge.py`, `tests/test_chapter_03.py`).
  Runs the unmodified base model against Chapter 2's 16 training
  examples with each `output` withheld, and saves the results to
  `datasets/training_examples/training_baseline_results.jsonl` -- the
  fixed training baseline Chapter 5's fine-tuned model gets compared
  against. On a real run against the sample archive, the base model
  scores `0/16` on exact-match (it correctly declines to guess rather
  than hallucinating a wrong answer); the challenge exercise shows that
  score rising to `6/16` once each report's full text is included as
  context. The chapter is explicit that this is a *training* baseline,
  not evidence of generalization -- Chapter 5 has to report a second,
  separate score against the held-out report to make that claim
  honestly. Also fixes a real bug surfaced while drafting this chapter:
  `code/chapter_01/load_local_model.py` now disables `transformers`'
  TensorFlow/Flax backend probing (`USE_TF=0`, `USE_FLAX=0`), since an
  unrelated, broken TensorFlow install on a shared/Anaconda-style
  environment was segfaulting model loading for every chapter that
  imports it.
- Implemented `figures/diagrams/generate_diagrams.py` (previously an
  unimplemented stub): renders each chapter's opening pipeline diagram
  via TikZ -> pdflatex -> PDF, converted to light/dark themed SVGs for
  HTML with `pdftocairo`. Generated `pipeline_ch01`–`pipeline_ch13`
  (`.pdf`, `_light.svg`, `_dark.svg`), which was also required for the
  Quarto PDF book output to compile at all (missing diagram files
  previously failed the LaTeX build).
- **Chapter 4: Tokenization and Embeddings for Domain Fine-Tuning**
  (`chapters/chapter_04.qmd`) fully drafted, with working, tested code
  (`code/chapter_04/tokenize_and_embed.py`,
  `code/chapter_04/challenge/challenge.py`, `tests/test_chapter_04.py`).
  Shows, from real runs, exactly how the base model's tokenizer splits
  this book's oilfield shorthand (`POOH`, `BHA`, ... each split into two
  arbitrary sub-word pieces), then measures whether either the base
  model's raw input embeddings or a general-purpose sentence-embedding
  model (`sentence-transformers`) already recognize that shorthand as
  equivalent to its spelled-out meaning. Neither does reliably: `BHA` vs.
  `"bottom hole assembly"` scores `-0.035` raw / `0.115`
  sentence-embedding, both statistically indistinguishable from an
  unrelated phrase. Field notes: the safety-critical term `BOP`
  (blowout preventer) scores *lower* against its own meaning (`0.059`
  sentence-embedding) than against `"birthday party"` (`0.320`) --
  concrete evidence that off-the-shelf embeddings can't be assumed to
  already understand domain-specific, safety-relevant vocabulary.
- **Chapter 5: Your First LoRA Fine-Tune** (`chapters/chapter_05.qmd`)
  fully drafted, with working, tested code
  (`code/chapter_05/first_lora_finetune.py`,
  `code/chapter_05/challenge/challenge.py`, `tests/test_chapter_05.py`).
  Fine-tunes a LoRA adapter (`peft`, plain PyTorch training loop, no
  framework abstraction) on Chapter 2's 16 training examples, then
  reruns Chapter 3's exact baseline harness against both the training
  set and Chapter 2's held-out report for the first time. Real run:
  training recall `0/16` -> `13/16` after 20 epochs (2,179,072 /
  0.141% of parameters trainable); held-out generalization stays at
  `0/2` -- unchanged. Every one of the 5 wrong answers (3 training
  misses, both held-out misses) is traced to a real, verbatim answer
  borrowed from a *different* training report, never a novel guess --
  concrete, diagnostic evidence of memorization without generalization
  from a training set this small. Challenge exercise: a lighter
  `q_proj`/`v_proj`-only LoRA config trains half as many parameters
  (`0.0705%`) and scores `10/16`.
- **Chapter 6: A Data Quality Gate for Training Data**
  (`chapters/chapter_06.qmd`) fully drafted, with working, tested code
  (`code/chapter_06/data_quality_gate.py`,
  `code/chapter_06/challenge/challenge.py`, `tests/test_chapter_06.py`).
  Runs Chapter 2's field extractor across the full 76-report archive
  (not just the 10-report sample) for the first time: `75/76` reports
  pass extraction, and a cross-report duplicate-value check finds 6
  duplicate groups, 4 consecutive (a continued operation across days,
  no action needed) and 2 non-consecutive (flagged for human review,
  not auto-excluded) -- including report `#49`'s `present_operations`
  matching report `#70`'s, 21 days apart. Challenge exercise adds a
  chronological-order check, which the real archive passes cleanly (0
  issues across 75 reports).
- **Chapter 7: Formatting and Chunking a Training Set at Scale**
  (`chapters/chapter_07.qmd`) fully drafted, with working, tested code
  (`code/chapter_07/format_training_chunks.py`,
  `code/chapter_07/challenge/challenge.py`, `tests/test_chapter_07.py`).
  Goes beyond Chapter 2's two summary fields into each report's
  `TIME BREAKDOWN` operational log -- one training example per logged
  time window, split ("chunked") at word boundaries when an entry runs
  too long. Real run across the 74 reports Chapter 6's gate and Chapter
  2's held-out reservation leave available: `669` training examples
  (over 40x Chapter 2's 16, from the same archive), `125` requiring
  chunking, and `3` chunks filtered out for containing `(cid:N)`
  undecoded-glyph artifacts -- a real PDF-extraction defect found inside
  the single longest entry in the archive (report `#21`, a step-rate-test
  sub-table `pdfplumber` couldn't decode). Challenge exercise reruns
  with a smaller chunk size (150 vs. 300 chars): `855` examples, `405`
  chunked.
- **Chapter 8: Fine-Tuning at Scale with Checkpointing and Experiment
  Tracking** (`chapters/chapter_08.qmd`) fully drafted, with working,
  tested code (`code/chapter_08/finetune_at_scale.py`,
  `code/chapter_08/challenge/challenge.py`, `tests/test_chapter_08.py`).
  Fine-tunes on Chapter 7's full 669-example training set (not Chapter
  5's 16-example proof of concept), checkpointing every epoch and
  logging plain CSV/JSONL metrics -- no TensorBoard/W&B/MLflow
  dependency. Simulates a real crash by fully reloading the base model
  and resuming purely from a saved checkpoint on disk. Real run (3
  epochs, ~32 minutes on CPU): training loss falls `2.755` ->
  `2.164` -> `1.821`, but exact-match recall on a fixed 50-example
  training sample stays `0/50` throughout -- a genuinely different,
  and equally honest, result from Chapter 5's `0/16` -> `13/16`, and
  informative precisely because loss and exact-match don't move
  together at this scale. Held-out generalization (Chapter 2's
  reserved report) stays `0/8`. Field notes: at this stage the model
  answers different, unrelated questions with nearly identical
  domain-flavored phrases ("Production Casing... Rig up casing/cement")
  -- evidence it has learned the archive's vocabulary and phrasing
  before it has learned which facts belong to which report. Challenge
  exercise: a checkpoint reloaded on a fully fresh process reproduces
  its own logged held-out score exactly.
- **Chapter 9: Hybrid System: Combining Fine-Tuning with Retrieval**
  (`chapters/chapter_09.qmd`) fully drafted, with working, tested code
  (`code/chapter_09/hybrid_rag_finetune.py`,
  `code/chapter_09/challenge/challenge.py`, `tests/test_chapter_09.py`).
  Builds a BM25 keyword-retrieval index over all 75 quality-gated
  reports -- held-out report included, since retrieval should find
  facts regardless of what the fine-tuned model trained on -- and
  combines it with Chapter 8's fine-tuned checkpoint. Real run:
  BM25 finds the correct report in the top 3 for `4/4` real test
  queries; dense sentence embeddings only `3/4` (missing exactly the
  query that ranked report `#38`'s real answer `340th` out of 677 with
  embeddings vs. `1st` with BM25 -- the same domain-vocabulary gap
  Chapter 4 found, now confirmed at the sentence level). Caught and
  fixed a real design bug while verifying: using the fine-tuned
  model's templated input as the retrieval query finds nothing useful,
  since that string is nearly identical across hundreds of chunks --
  retrieval needs a real information-need query, kept separate from
  the model's generation prompt. Field notes: across the 4 test cases,
  retrieval found the correct source every time, but the fine-tuned
  model only faithfully used that source some of the time -- finding
  the right document and using it faithfully are measurably different
  problems, which Chapter 10 picks up directly. Chapters 10–13 remain
  placeholders.
- **A real figure for Chapter 5's "Engineering Translation: LoRA
  adapter" callout** (`figures/diagrams/lora_lowrank_ch05.pdf` /
  `_light.svg` / `_dark.svg`, embedded in `chapters/chapter_05.qmd`
  right after the callout). Reviewed as a plain-HTML sketch first
  (light/dark tokens and a two-option comparison against the book's own
  diagram palette) before being built for real. Two rows: Table A x
  Table B = a full-size correction (each table drawn 2 cells thick so
  it doesn't read as rank 1), then frozen base weights + that
  correction = what the model reads at inference -- so the diagram
  shows the correction being added on top, not swapping the original
  weights out. `figures/diagrams/generate_diagrams.py` gains
  `render_lora_diagram()` and a dedicated `LORA_TIKZ_TEMPLATE`, since
  the existing `render_diagram()` only draws straight vertical box
  chains and this is a two-dimensional grid layout; extends
  `LIGHT_PALETTE`/`DARK_PALETTE` with `soft_fill`/`strong_border` for
  the correction and "what's actually used" cell styles, ignored by the
  existing chain renderer.
- Revised the "Engineering Translation: LoRA adapter" callout's wording
  a second time, per review: "two shorter lists" (rank-1 only, and
  never actually shown being added to the base weights) became "two
  smaller tables" -- accurate at Chapter 5's real rank 8, not just
  rank 1 -- with an explicit new sentence that LoRA's correction is
  added on top of the frozen weights, not swapped in for them. The new
  figure above makes both points visually as well as in prose.
- Third review pass on the same figure and callout: the diagram's
  "computed, never stored" label overstated a real implementation
  detail (some PEFT workflows do merge/materialize the combined
  weights, e.g. for export) -- changed to "produced from A x B", and
  the matching prose changed from "without ever storing that full-size
  version directly" to "without needing to store that full-size
  version separately". "What the model reads at inference" became
  "Effective weights used at inference" (the model uses combined
  weights, it doesn't "read" a table). The Step 2 label became "Frozen
  base weights + LoRA correction = adapted model behavior" to state
  the addition as its own equation. Figure caption (HTML and PDF)
  rewritten to: "LoRA trains two small adapter tables. Their product
  creates a full-size correction, which is added to the frozen base
  weights. The original model is not rewritten."
- **A second figure for Chapter 5, Step 3: a grouped before/after bar
  chart of the chapter's headline result**
  (`figures/diagrams/recall_gap_ch05.pdf` / `_light.svg` / `_dark.svg`,
  embedded right after "What just happened?", before the "Be precise
  about what this run actually shows" paragraph). Reviewed first as an
  HTML sketch alongside a loss-curve alternative; picked as higher
  priority because it carries the chapter's actual thesis (training
  recall moves, held-out generalization doesn't) in one glance, not
  just "loss fell." Plots the chapter's own real counts --
  training-set recall `0/16 -> 13/16`, held-out generalization
  `0/2 -> 0/2`, unchanged -- as two bar pairs on a shared 0-100% axis,
  with the real fraction labeled on every bar so the underlying counts
  are never hidden behind a percentage. `generate_diagrams.py` gains
  `render_recall_gap_diagram()` and `RECALL_GAP_TIKZ_TEMPLATE`, reusing
  the LoRA figure's existing `trained`/`frozenfill` color roles (solid
  = measured result, outline = untouched baseline) rather than
  inventing a third color language for this chapter.
- The loss-curve alternative (Chapter 5's `epoch and loss` callout,
  plotting the 5 real logged loss values against epoch) was reviewed
  and deliberately not built yet -- ranked lower because Chapter 5
  already prints those 5 values directly and a falling number is
  easier to accept on its own than the recall/generalization split
  above. Left as a documented, ready-to-build option if wanted later.
- **Chapter 10: Traceable Outputs and Hallucination Mitigation**
  (`chapters/chapter_10.qmd`) fully drafted, with working, tested code
  (`code/chapter_10/traceable_outputs.py`,
  `code/chapter_10/challenge/challenge.py`, `tests/test_chapter_10.py`,
  `notebooks/chapter_10_explore.ipynb`). Builds a plain word-overlap
  faithfulness check -- `content_words()` strips stopwords and the
  instruction template's own boilerplate, `faithfulness_score()` scores
  an answer against a single source chunk -- and applies it to Chapter
  9's real hybrid system, keeping only the citations an answer is
  actually traceable to (`verified_sources`) instead of everything
  retrieval handed the model. Real run against Chapter 9's exact 4 test
  cases: report `#37` correctly verified faithful (`1.00`) against its
  own report; report `#38` correctly flagged `grounded: False`, no
  chunk crosses threshold, matching Chapter 9's own human judgment;
  report `#49` verified against its real target but the check can't
  catch a genuine "trip in" vs. "trip out" direction inversion buried
  in otherwise-matching words (Field Notes). Report `#21` is the
  headline finding: the grounded answer *"Test choke manifold at
  5000psi"* is real, fluent, and verified faithful -- against report
  `#27` (`0.75`), a blowout-preventer test also retrieved alongside the
  actual target, report `#21` (`0.25`). A plain `grounded: True`/`False`
  flag would have silently counted this as a success; only checking
  which specific source an answer verifies against catches it. Caught
  and fixed a transcription error while verifying: an early draft
  claimed the `#27` comparison scored `1.0`; the real run returns
  `0.75`, because the answer's own `"5000psi"` (no space) doesn't token-match
  the source's `"5000 psi"` (two words) -- corrected in the chapter
  text, and used as an additional real illustration of the check's
  lexical limits. Challenge exercise: Chapter 9's un-grounded baseline
  answers (no retrieval at all) score `0.17`, `0.17`, `0.00`, `0.17`
  against their own target reports -- all well under the `0.5`
  threshold, confirming the check generalizes as a real hallucination
  detector, not something narrowly tuned to Chapter 9's specific setup.
  Glossary gains "Grounded / grounding" and "Faithfulness" entries.
  Both READMEs, the chapter table, and test counts (52 -> 60 across
  Chapters 1-10) updated to match.
- **Chapter 11: Evaluating a Fine-Tuned Domain Model**
  (`chapters/chapter_11.qmd`) fully drafted, with working, tested code
  (`code/chapter_11/eval_finetuned_model.py`,
  `code/chapter_11/challenge/challenge.py`, `tests/test_chapter_11.py`,
  `notebooks/chapter_11_explore.ipynb`). Builds a real, 8-example
  held-out evaluation set by calling Chapter 7's own
  `build_timeline_examples_for_report` directly on the held-out report
  -- the exact same 8 examples Chapter 8's training script already
  built internally for its own `held_out_exact_match` logging, never
  looked at past that one column until now. Scores it three ways:
  exact-match (reused from Chapter 3), a partial-credit overlap score
  (Chapter 10's `faithfulness_score`, reused unchanged but applied to a
  different comparison -- answer vs. known-correct output, not answer
  vs. retrieved source), and perplexity (new). Real run: exact-match
  `0/8`, average overlap `0.17` -- every generated answer collapses to
  one of two near-identical generic phrases regardless of the actual
  question, the same "shape, not judgment" pattern Chapter 3/9 already
  found. Headline finding: perplexity on the identical held-out text
  falls from `159.91` (base model) to `25.03` (fine-tuned) -- a real,
  substantial change the exact-match number completely hides. Field
  Notes reads Chapter 8's own `metrics.csv` all the way across:
  `held_out_exact_match` stayed `0/8` at every one of its 3 logged
  epochs, while perplexity on the same text fell at every epoch
  (`27.99 -> 25.40 -> 25.03`), tracking the falling training loss
  almost exactly. Challenge exercise reproduces that exact table by
  running perplexity against all 3 of Chapter 8's real checkpoints.
  Glossary gains an "Evaluation set" entry. Both READMEs, the chapter
  table, and test counts (60 -> 64 across Chapters 1-11) updated to
  match.
- **Chapter 12: Detecting Drift Across Model Versions**
  (`chapters/chapter_12.qmd`) fully drafted, with working, tested code
  (`code/chapter_12/detect_model_drift.py`,
  `code/chapter_12/challenge/challenge.py`, `tests/test_chapter_12.py`,
  `notebooks/chapter_12_explore.ipynb`). `summarize_version()` collapses
  Chapter 11's three metrics into one comparable summary per model
  version; `compare_versions()` reports a direction
  (improved/regressed/unchanged) per metric between two versions,
  deliberately without collapsing disagreement into a single verdict.
  Real run across Chapter 8's 3 epoch checkpoints found a genuine
  disagreement: `checkpoint_1 -> checkpoint_2` shows `avg_overlap`
  regressing (`0.354 -> 0.167`) while `perplexity` improves (`27.99 ->
  25.40`), on the same 8 held-out questions. Field Notes investigates
  by reading the actual generated text: both checkpoints still produce
  one memorized-sounding template regardless of the question (the same
  "shape, not judgment" pattern from Chapters 3/9/11) -- the template
  itself just changed between epochs, and the new one happens to share
  less vocabulary with the real answers, which is what the overlap
  score is actually measuring. Challenge exercise confirms the same
  disagreement pattern holds comparing two entirely different training
  regimes (Chapter 5's 16-example fine-tune vs. Chapter 8's "at scale"
  checkpoint: `avg_overlap` `0.5625 -> 0.167`, `perplexity` `112.28 ->
  25.03`), not just adjacent epochs of one run. Glossary's "Drift"
  entry extended to define "regression" alongside it. Both READMEs, the
  chapter table, and test counts (64 -> 69 across Chapters 1-12)
  updated to match.
- **Chapter 13: Continuous Fine-Tuning -- Keeping the Model Current**
  (`chapters/chapter_13.qmd`) fully drafted, with working, tested code
  (`code/chapter_13/continuous_finetune.py`,
  `code/chapter_13/challenge/challenge.py`, `tests/test_chapter_13.py`,
  `notebooks/chapter_13_explore.ipynb`) -- **the book's final chapter;
  Part 0 and all 13 chapters are now written, tested, and passing CI.**
  Simulates new reports arriving with a real chronological split of
  this book's own 74 non-held-out reports (`57` "currently in
  production" before report `#60`, `17` "just arrived" at/after it,
  `490` and `179` real training examples respectively -- `669` total,
  matching Chapter 7/8's own count exactly). Trains a "current" model
  on the first half, continues training it on the new batch using
  Chapter 8's exact checkpoint-and-resume mechanism unchanged, then
  runs Chapter 12's comparison to decide whether the result is worth
  deploying. Real run: training completed cleanly, loss falling at
  every epoch (`3.03 -> 2.37 -> 1.87 -> 1.51`), and the result still
  regressed on `avg_overlap` (`0.464 -> 0.125`) while perplexity
  improved only marginally (`25.92 -> 25.73`, under `1%`) -- the same
  metric-disagreement pattern Chapter 12 found, now confirmed a third
  time on a genuinely new, real scenario. Field Notes traces the
  regression to a real cause: `20` of the new batch's `179` examples
  mention casing/cement, concentrated at the archive's end as the well
  nears total depth -- a real, sensible operational-phase shift, not
  an unexplainable artifact. Challenge exercise builds a second
  held-out check from the new batch's own report `#65` (`20` examples)
  and finds its perplexity (`16.56`) notably lower than either model
  scored against report `#37` (`~25.8`), showing operational-phase
  match matters more than training recency alone.
  **Caught and fixed a real bug while verifying:** the comparison
  step's first draft reused Chapter 8's `load_checkpoint` (which sets
  `is_trainable=True`) to load checkpoints for pure evaluation, leaving
  the model in train mode with dropout active -- `model.training`
  printed `True` where it should have printed `False`, and perplexity
  varied slightly (`~25.79-25.86`) from run to run as a result. Fixed
  by using Chapter 12's `load_version` (no `is_trainable`) for
  evaluation-only loads instead, confirmed fully deterministic across
  repeated runs afterward; Chapters 9-12's own code never had this bug
  (none of them pass `is_trainable=True`). Both READMEs, `RELEASE.md`,
  the chapter table, and test counts (69 -> 74 across all 13 chapters)
  updated to reflect the book's completion. Glossary gains a
  "Continuous fine-tuning" entry.
- **Companion app V1** (`book/app/`): the previous three-file placeholder
  ("not implemented yet") is now a real, working Streamlit app with two
  pages -- a **Model Playground** (landing page) and a **Before vs.
  After Evaluation** page. Nothing here reimplements the book's
  pipeline: every model load, generation, retrieval, and score is
  imported directly from the chapter code that already exists (Chapters
  1, 9, 10, 11, 12), the same way `tests/conftest.py` makes it
  importable for the test suite. Checkpoint loading for
  inference/evaluation always goes through Chapter 12's `load_version()`,
  never Chapter 8's `load_checkpoint()` (which sets `is_trainable=True`
  and leaves dropout active) -- the exact bug Chapter 13's own entry
  above documents catching and fixing; the app doesn't reintroduce it.
  Model Playground offers three real prompt sources -- Chapter 11's
  8-question held-out set (with real exact-match/overlap scoring),
  Chapter 9/10's 4 curated retrieval test cases (with a real
  `grounded` flag and per-source faithfulness scores), and a free-text
  question (explicitly labeled "no ground truth available," not scored)
  -- and a checkpoint selector that only lists what actually exists on
  this machine (`checkpoints/` is gitignored). Verified live in the
  browser against this machine's real checkpoints: selecting the
  retrieval demo case for report #37 reproduced the exact transcript
  already published in both READMEs ("Trip out of hole with BHA #18.
  Stop at 5,800' and circulate to cool hole and tools.", `grounded:
  True`); the Before/After Evaluation page's live-computed perplexity
  for Chapter 8's checkpoint came back `159.91` (base) `-> 25.03`
  (fine-tuned) -- the same numbers already published from a direct run
  of `eval_finetuned_model.py` -- and surfaced the same
  `avg_overlap`-regressed/`perplexity`-improved disagreement between
  Chapter 5 and Chapter 8 that Chapter 12's own chapter text describes.
  Caught and fixed one real bug while verifying: `st.bar_chart`'s
  Vega-Lite backend sorts its x-axis alphabetically by label text, not
  by row order, so "Chapter 13" was rendering before "Chapter 5" and
  "Chapter 8" as a string; fixed with an explicit numeric sort prefix on
  the chart labels only, table and comparison text unaffected. New
  `tests/test_app.py` (fast checkpoint-discovery/scoring tests, plus
  `slow`-marked generation/evaluation tests that skip cleanly if no
  checkpoint exists), and a new `pandas` dependency for the evaluation
  table/chart. Dataset Explorer, Experiment Explorer, Failure Analysis,
  and a "Fine-Tuning or RAG?" page remain planned, not built -- see
  `book/app/README.md`.
- **Companion app V2: Dataset Explorer** (`book/app/pages/2_Dataset_Explorer.py`).
  V1 is now frozen -- no further features land on it; new app work ships
  as V2 and beyond, the same one-piece-at-a-time cadence the book's own
  chapters use. Browses three real example sets, generated live from the
  real archive rather than read from a pre-generated file (none of
  `datasets/training_examples/*.jsonl` are committed -- they're
  gitignored, generated artifacts): Chapter 2's 16-example sample-set
  summaries, Chapter 7's 669-example full-archive timeline set (the real
  training set), and report #37's own 8 held-out examples. A new
  `helpers.parse_input_context()` regex-parses each example's own
  `input` string back into its real, already-present fields (report
  number, date, time window, chunk part) -- no domain/topic/difficulty
  label is added, because this book's real training data
  (`{instruction, input, output}` only) has no such field. Cross-
  references Chapter 6's real `run_quality_gate()` duplicate-detection
  to flag which examples belong to a report whose fields non-
  consecutively duplicate another report's, surfacing the same "decision
  still owed to a human" framing Chapter 6 itself uses. Verified live:
  filtering the full 669-example set to "flagged for review only" lands
  on exactly `38` examples across `4` unique report numbers (`47, 49,
  59, 70`) -- consistent with Chapter 6's own published `2` non-
  consecutive duplicate groups (2 groups × 2 reports each). New tests in
  `tests/test_app.py` for `parse_input_context()` and `dataset_examples()`
  (fast -- PDF extraction only, no model, matching the existing
  chapter-6/7 test convention of not marking extraction-only tests
  `slow`). Both READMEs updated to describe the V1/V2 split.
- **Companion app V3: Failure Analysis** (`book/app/pages/3_Failure_Analysis.py`).
  V2 is now frozen alongside V1. This completes the app's evidence
  chain -- training data (V2) -> model behavior (V1) -> failures (V3) --
  chosen deliberately over adding more general UI features. Two live
  sections, both run against whichever real checkpoint the reader
  selects, never a fixed screenshot: (1) the same 4 real Chapter 9/10
  retrieval test cases already wired into the V1 Playground, each paired
  with a new `helpers.DOCUMENTED_FINDINGS` entry quoting what the book
  itself found running that exact case; (2) a new
  `helpers.pairwise_answer_similarity()` -- Chapter 10's own
  `faithfulness_score()` reused on a new pairing (generated answer vs.
  generated answer, not answer vs. source, the same kind of reuse
  Chapter 11 already does applying it to answer vs. expected-output) --
  run across Chapter 11's real 8-question held-out set to detect the
  "shape, not judgment" pattern Chapters 3/8/9/11/12 all document.
  Verified live in the browser against Chapter 8's real checkpoint: the
  report #21 case reproduced the book's exact documented result --
  "Test choke manifold at 5000psi," `grounded: True`, verified against
  report #27 at faithfulness `0.75`, not the real target -- and the
  shape detector's top-ranked pair reproduced the exact "Production
  Casing Run Csg & Cement Rig up casing" answer already quoted as the
  headline example in this repo's own root README, this time surfaced
  as evidence of memorization rather than as a win. New tests for
  `DOCUMENTED_FINDINGS` (covers every real test case) and
  `pairwise_answer_similarity()` (ranks near-identical answers highest,
  symmetric regardless of input order). Both READMEs updated to
  describe the finished V1/V2/V3 progression.
- **Book cover** (`figures/cover.jpg`, `_quarto.yml`'s `cover-image`).
  Generated via a new `render_cover()` in
  `figures/diagrams/generate_diagrams.py`, reusing the exact box/arrow
  TikZ style every chapter's own pipeline diagram already uses (scaled
  up) instead of introducing a new illustration style: title and
  subtitle taken verbatim from `_quarto.yml`, and a 3-stage vertical
  chain -- "Raw Field Reports" -> "Local LLM + LoRA Fine-Tuning" ->
  "Traceable, Deployed Answers" -- built from vocabulary already used by
  this book's own per-chapter diagrams (Ch1/2's "Raw Reports", Ch5's
  "LoRA Fine-Tune", Ch10's "Traceable Answer"), not invented copy.
  `standalone` auto-sizes to content like every other diagram here,
  landing at a natural 1250x1446px portrait ratio. Verified in a real
  `quarto render`: the image shows correctly on the HTML welcome page
  (`class="quarto-cover-image"`). Discovered along the way that Quarto's
  `cover-image` field only reaches HTML/EPUB output for book projects --
  it has no effect on the PDF, whose title page is plain text
  (title/subtitle/author) regardless of the setting. Added a PDF-only
  cover page via `_quarto.yml`'s `format.pdf.include-in-header`, hooked
  through `\AtBeginDocument` rather than `include-before-body` --
  pandoc's own `\maketitle` fires at `\begin{document}`, before any
  `include-before-body` content, so a naive `include-before-body`
  attempt put the cover *after* the text title page instead of before
  it (verified via a temporary `keep-tex: true` render and direct
  `xelatex` inspection); `\AtBeginDocument` fires at the same point
  `\maketitle` does and wins the race. Verified in the rendered PDF:
  page 1 is the cover, page 2 a blank verso (the twoside `scrbook`
  class's standard recto/verso front-matter convention, not a bug), page
  3 the existing text title page -- both formats now share the same
  cover, `pytest -m "not slow"` still passes (67 tests), and the full
  book still renders in one `quarto render` pass (both formats, no
  wiped output).
- **Companion app screenshots** (`figures/app_screenshot_playground.jpg`,
  `figures/app_screenshot_evaluation.jpg`), captured live from the
  actual running app against real checkpoints, not staged or invented --
  the same "never fabricate a result" rule the chapters follow. Model
  Playground's screenshot shows the real report #37 held-out question
  answered by both the base model and Chapter 5's checkpoint, with real
  exact-match/overlap scores; embedded in `index.qmd`'s "What this
  becomes" section, right where the app is first introduced. Before vs.
  After Evaluation's screenshot captures the exact "latest does not mean
  best" regression callout and per-transition percentage changes across
  Base/Ch5/Ch8/Ch13 -- the same real regression Chapter 13's own
  version-comparison already found in that chapter's own code output;
  embedded in Chapter 13's "See it side by side" section right after the
  paragraph that references it. Caught a real LaTeX bug while verifying
  the PDF render: an inline code span (`` `compare_versions()` ``) in
  the Chapter 13 figure's caption crashed `xelatex` with "Undefined
  control sequence `\SQSPL@scan`" -- `_quarto.yml`'s `seqsplit`-based
  `\texttt` redefinition (added for wrapping long inline code in body
  text) isn't `\protect`-safe inside a LaTeX caption's moving argument;
  reworded the caption to avoid inline code formatting rather than
  touching the global macro. Verified both figures render as proper
  numbered Quarto figures in HTML (`Figure 1`, `Figure 13.1`) and in the
  PDF at the correct pages (18, 171), `pytest -m "not slow"` still
  passes (67 tests), and the full book still renders in one
  `quarto render` pass.

### Changed

- **Chapter 3** gains a new "When fine-tuning is, and is not, the right
  tool" section, after "Why not just eyeball a few answers?". Scopes
  what fine-tuning is actually good for (domain vocabulary, writing
  style, measurable before/after, staying local) against what it's
  weaker at (recalling one specific fact from one specific report,
  citing sources, replacing retrieval, proving generalization from a
  small dataset) -- so the chapter doesn't imply fine-tuning alone
  solves factual lookup, and points ahead to Chapter 9's combination of
  fine-tuning with retrieval. Written after Chapter 5 shipped, so it
  can point at that chapter's real held-out result as concrete
  evidence rather than a promise.
- Both `README.md` (root) and `book/README.md` restructured to follow
  the same section design as the author's previous book,
  [`ddr-rag-book`](https://github.com/djimrastephane/ddr-rag-book):
  added a "Why Not Just Prompt a Cloud AI Assistant?" section and a
  "What You're Building" transcript box (the real Chapter 9 with/without
  retrieval answer on held-out report `#37`, plus Chapter 5's `0/16 →
  13/16` training-recall vs. `0/2` held-out result), a Project Map
  table, a Reader Contract, a full per-chapter "How Long Does Each
  Chapter Take" table (reusing each chapter's own in-book "Estimated
  time" status-strip value), a First Success Checkpoint, Common Reader
  Journeys, What Makes This Book Different, a dedicated Companion
  Pipeline section, and a Bonus Material section -- none of it invented,
  every quoted number and transcript pulled from this book's own
  already-shipped chapter content.
- Root README's wording audited against the same zero-programming-
  experience bar CLAUDE.md holds chapter prose to, since it's a reader's
  first contact with the book, before Part 0 has taught them anything to
  lean on. Glossed every previously-bare acronym/term on first use
  (`LoRA`, weights, hallucination, retrieval, grounded, exact-match,
  held-out/generalization, CI, VRAM, perplexity, BM25, dense sentence
  embeddings, `avg_overlap`), reusing the exact plain-language phrasing
  already established in `appendix_b_glossary.qmd` and each chapter's
  own Engineering Translation callouts rather than inventing new
  wording. Caught and fixed a real accuracy issue along the way: the
  README claimed the book fine-tunes with "LoRA/QLoRA," but no chapter
  script actually loads a quantized model -- `bitsandbytes`/QLoRA is
  only an optional, unused install path (Chapter 0 says as much) --
  corrected to "LoRA." Also reworded flatter, spec-sheet-style sections
  ("Who This Book Is For," "What Makes This Book Different") to match
  the book's own direct, evidence-first voice instead of reading like
  generic project-README copy -- "Who This Book Is For" now ties into
  the book's recurring four-engineer cast (Oumy, Mike, Sarah, Sean)
  instead of a generic job-title list. Brought `book/README.md`'s
  duplicate "chapter map" evidence paragraph in sync with the same
  glosses. Added a "README maintenance" section to `CLAUDE.md`
  codifying this accessibility/accuracy/consistency check for future
  README edits, and fixed `CLAUDE.md`'s own stale "Chapters 11-13 are
  placeholder" status line in the process (all 13 chapters are actually
  drafted, tested, and passing CI per `git log`).
- Moved the CI/publish badges in the root README from ~420 lines down
  (under "Table of Contents," past every section a first-time visitor
  actually reads) to directly under the title, the standard GitHub spot
  -- and fixed the one cross-reference that pointed at their old
  location. `book/README.md` had no badges at all; added the same four
  (Linux/Windows/macOS tests + publish), in the same order, for
  consistency between the two.
- **Connected the companion app to the book's own reader-facing text**,
  not just repo-level READMEs. `index.qmd`'s "What this becomes" section
  described the app as "planned" and pointed to Appendix A "once it
  exists" -- both stale (the app is real, and Appendix A never actually
  had app setup content); rewrote to describe what V1/V2/V3 actually do
  and point to `app/README.md` instead. Chapter 1's Production Reality
  aside grouped "the companion app in `app/`" in with systems that
  handle multi-request concurrency and hot-swapping checkpoints without
  a restart -- inaccurate now that the app is real and known not to do
  either (it's explicitly a single-reader local tool); reworded so the
  app is named as an example of what *isn't* solved, not what is. Added
  a new "See it side by side" section to Chapter 13 (the last chapter,
  positioned before the existing "Suggested next step" so that closing
  message stays last) pointing to the app now that a reader has a real
  checkpoint from every chapter to load into it. Verified both edits
  with a real `quarto render` of the affected pages (not just the
  `.qmd` source) before committing.
- **Reworked Before vs. After Evaluation (V1) into the definitive
  model-evolution page**, per real usability feedback after checking it
  in the browser. The single combined bar chart made overlap
  effectively invisible -- perplexity ranges roughly `25` to `160`,
  average overlap `0` to `0.562`, one shared axis flattens the smaller
  series to nothing -- split into two separate charts, each with its
  own scale. Added `helpers.relative_change()` (percent change between
  consecutive versions, `None` when the baseline is `0` rather than
  dividing by zero) so each version transition reads as "perplexity
  improved 77.7%, overlap regressed 70.3%" instead of just a direction
  word. Added `helpers.evaluation_snapshot()` (best perplexity/overlap/
  exact-match, ties included, plus which version is chronologically
  latest) as a summary block at the top. Added
  `helpers.latest_regressed_on_both()` and a matching UI callout: on
  this machine's real checkpoints, Chapter 13 regresses on both average
  overlap (`0.167` -> `0.125`) and perplexity (`25.03` -> `25.73`)
  compared to Chapter 8 -- continued fine-tuning did not automatically
  improve the model, and the page now says so explicitly ("latest does
  not mean best") instead of leaving the reader to notice it in a
  table. Moved "What changed between versions" to immediately after the
  results table, since it's the interpretation that matters, not an
  afterthought below two charts. Added an explanatory caption for why
  every version can show `0/8` exact-match despite real, measurable
  differences elsewhere -- Chapter 3's exact-match rule requires the
  reference answer's exact wording as a substring, so being right in
  different words still scores 0. Added `helpers.short_version_label()`
  ("Base"/"Ch5"/"Ch8"/"Ch13") for chart axes -- the old full descriptive
  labels were truncating -- full descriptions stay in the results
  table. Re-caught the same `st.bar_chart` alphabetical-x-axis bug
  fixed once already for this page (see above): short labels alone
  sort "Ch13" before "Ch5"/"Ch8" as a string, so the same numeric-prefix
  fix had to be reapplied on top of the short labels. An Experiment
  Explorer was considered for a V4 and deliberately dropped -- this
  page already has the necessary evidence once polished; both READMEs
  updated to describe the finished three-page evidence chain (Dataset
  Explorer / Before vs. After Evaluation / Failure Analysis) instead.
  New tests for all four new pure functions.
- **Made Before vs. After Evaluation understandable without an AI/
  programming background**, per real feedback from checking the page as
  a non-technical oil & gas reader. Added a plain-English glossary at
  the top defining Held-out set, Perplexity, Overlap score, and
  Exact-match before any of them are used, in operational-analogy terms
  (e.g. perplexity described the way a driller who's read years of tour
  sheets from the same rig can predict how the next line will read, not
  in NLP terms). Added a live "real example, side by side" section --
  the same report #37, 20:30-21:30 question already quoted in this
  repo's own root README, generated fresh by the base model and the
  latest fine-tuned checkpoint, with the real correct answer alongside
  -- so a reader can judge the qualitative difference immediately
  instead of only looking at abstract scores. Reworded the exact-match
  caption to lead with "a `0/8` here is not a failure verdict" instead
  of explaining the strict rule first. De-emphasized code/chapter
  references (function names, "Chapter 12's own finding") from the
  primary captions into smaller secondary notes, leading with plain
  business language instead. Caught and fixed two real chart bugs while
  verifying this in the browser: (1) the average-overlap chart rendered
  completely empty because its column name, `"Avg. overlap score"`,
  contained a period -- Altair/Vega-Lite's shorthand field syntax reads
  a `.` as nested-property notation, so the field silently resolved to
  nothing; renamed the column to drop the period. (2) Chart x-axis
  labels were truncating to a single trailing character (`"Ch13"`
  rendering as just `"3"`) under Vega-Lite's default rotated-label
  layout; fixed by forcing horizontal labels (`labelAngle=0`), which
  also reads more naturally for a reader unfamiliar with rotated chart
  axes. New dependency: `altair` (already an implicit Streamlit
  dependency, now imported and pinned directly since the charts are
  built with explicit Altair specs instead of `st.bar_chart`, needed for
  the guaranteed-zero-baseline y-axis and explicit label-order control).

### Fixed

- Review pass on Chapter 5's recall-gap bar chart: the y-axis said
  "share answered correctly," broader than what the chapter actually
  measures (the strict `matched_expected` exact-match rule from
  Chapter 3) -- changed to "exact-match score" (and the `fig-alt` text
  to match). The two near-zero bars (`0/16`, `0/2`, `0/2`) were drawn
  as barely-visible slivers that risked reading as small nonzero
  values; replaced with a distinct flat marker (a bold rounded tick at
  the baseline) so a zero result reads unambiguously as zero, not as
  noise. Removed the inline-code styling around `0/16`/`13/16`/`0/2` in
  the figure caption -- code styling reads as a value to type, not a
  measured result; left it in place in the "Be precise about what this
  run actually shows" paragraph below, where it's consistent with the
  rest of the chapter's prose.
- **Chapter 5's two new figures (the LoRA diagram and the recall-gap
  bar chart) were numbered in the PDF but not in HTML.** The PDF's
  native single-image figure syntax numbered them correctly ("Figure
  5.1", "Figure 5.2"), but the HTML embed's light/dark image pair broke
  Quarto's caption/crossref detection -- `fig-cap` silently became an
  inert `data-fig-cap` attribute, so the HTML side fell back to a
  hand-written, unnumbered "*Figure: ...*" line instead. Fixed by
  moving both images inside a shared `#fig-...` div (one per
  blank-line-separated paragraph, no `#fig-` id of their own), which
  Quarto merges into a single real numbered figure instead of treating
  them as separate `(a)`/`(b)` subfigures -- HTML now shows the same
  "Figure 5.1"/"Figure 5.2" as the PDF, and both formats now share the
  same stable `#fig-lora-lowrank` / `#fig-recall-gap` ids. Checked
  every other chapter for the same manual-caption pattern; these were
  the only two.
- **"LoRA" was used from Chapter 0 onward (including the book's own
  table of contents) but the acronym itself was never expanded where a
  reader would actually encounter it.** Chapter 5's "Engineering
  Translation: LoRA adapter" callout explained the mechanism (a small,
  frozen-base "correction sheet") but not what the letters stood for or
  why it's called "low-rank" -- only Appendix B's glossary spelled out
  "Low-Rank Adaptation," and a first-time reader has no reason to check
  it before or during Chapter 5. Added "LoRA stands for Low-Rank
  Adaptation" to that callout, plus a plain-language explanation of what
  "low-rank" means (the correction sheet is rebuilt from two much
  shorter lists of numbers multiplied together -- like a large table
  rebuilt from a row list and a column list instead of every cell typed
  out by hand -- with the **rank**, `r=8` in Step 2, being how long
  those lists are), matching the book's existing physical-analogy style
  instead of introducing matrix-multiplication terminology unexplained.
- **Chapter numbering was off by one in rendered HTML/PDF output.**
  `chapters/chapter_00.qmd` (Part 0) sat inside Quarto's normal numbered
  chapter sequence, so it silently became "Chapter 1" and pushed every
  real chapter's rendered number one ahead of its own status-strip label
  (conceptual Chapter 6 rendered as "6" in its status strip but "7" in
  the book's actual heading/TOC). Marked Part 0's heading `.unnumbered`;
  verified in both HTML and PDF that Chapter 1 now renders as "1" and
  Chapter 6 as "6".
- `tests/test_chapter_05.py` imports `first_lora_finetune`, which
  imports `peft` at module load time -- on a machine without `peft`
  installed, that turned into a hard collection error for the whole
  file (and, per pytest's collection behavior, could abort the entire
  `pytest -m "not slow"` run, not just skip this one file). Now uses
  `pytest.importorskip("peft")`, ordered after the same `USE_TF=0`
  guard `load_local_model.py` sets, so a missing `peft` skips cleanly
  instead of segfaulting or erroring out collection.
- Chapter 5's training loop never moved the model or tensors to a GPU
  or MPS device, silently running CPU-only regardless of what hardware
  was available -- correct behavior, but undocumented. Now stated
  explicitly, in both the code's module docstring and the chapter text,
  as a deliberate reproducibility choice, not an oversight.
- Chapter 4's Field notes overstated what its embedding-similarity
  numbers prove -- claiming a specific, unverified cause (character-level
  surface similarity) for why `"birthday party"` scored higher than
  `"blowout preventer"`. Narrowed the claim to what the numbers actually
  show: the embeddings don't encode the domain equivalence strongly
  enough for this task, without asserting a specific mechanism this
  chapter never actually inspected.
- Chapter 6's closing sections ("WHAT YOU BUILT", "Suggested next step")
  described the gate as having "decided which reports are trustworthy"
  and Chapter 7 building on "the 75 reports this gate actually passed"
  -- blurring the fact that 2 of those 75 still carry an unresolved
  `needs_review` flag. Reworded both to state plainly that those 2
  flags are a decision still owed to a human, not one the gate already
  made.
- Chapter 0's optional Ollama install (0.9) told readers to pull
  `qwen2.5:7b-instruct` -- a different, much larger model than the
  `Qwen2.5-1.5B-Instruct` this book actually loads and fine-tunes from
  Chapter 1 onward. Changed to `qwen2.5:1.5b-instruct` so an optional
  side-by-side comparison via Ollama uses the same model size, not a
  bigger one that would quietly look better for the wrong reason.
- Chapter 5 defined what an epoch *is* ("one full pass through the
  training data") but never explained why more than one pass is
  needed -- a natural question for a zero-ML-background reader, and
  the whole reason `NUM_EPOCHS=20` is set the way it is. Extended the
  "epoch and loss" callout to explain that each pass only nudges the
  adapter's weights a small amount, so the loss falling from `4.7173`
  to `0.1201` over 20 epochs is that accumulation made visible. Chapter
  8 uses "epoch" as its central organizing concept (checkpoint every
  epoch, log every epoch) without ever restating or cross-referencing
  Chapter 5's definition; its "checkpoint" callout now does both.
- "Why fine-tune a pretrained model instead of training one from
  scratch on our own reports" was only ever answered once, in the
  Preface, in a single dense sentence aimed at a reader who already
  knows what "parameter-efficient" and "foundation model" mean --
  never in the plain-language, problem-first style the actual chapters
  use, and never at the point (Chapter 1, right where
  Qwen2.5-1.5B-Instruct is introduced) where a first-time reader would
  naturally ask it. Added a new "Engineering Translation: pretraining"
  callout and a "Why not train a model from scratch on our own
  reports?" section there, before "Why not just use a cloud AI
  assistant?" -- distinct from Chapter 5's separate "Why LoRA instead
  of updating the whole model?" (that's full vs. parameter-efficient
  fine-tuning of an already-pretrained model; this is pretrained vs.
  from-scratch in the first place).
- That new "pretraining" callout conflated pretraining with instruction
  tuning, crediting pretraining alone with teaching the model to
  "follow instructions" -- technically two separate stages (the second
  is the reason for the `-Instruct` in the model's name). Split them
  out explicitly. Separately, "how this operation talks and what its
  reports actually say" overclaimed fine-tuning's reliability at exact
  factual recall, contradicting Chapter 3's "when fine-tuning is, and
  is not, the right tool" and Chapter 5's own measured result (`0/50`
  exact-match at scale). Narrowed to what fine-tuning actually teaches
  reliably -- vocabulary, sequencing, report shape -- with a forward
  pointer to where the factual-recall distinction is actually covered.
- Both `README.md` (root) and `book/README.md` still described Chapters
  1–9 in future/conditional tense ("not yet written", "will be
  tested", "Once chapter code and fixtures exist"), even though Part 0
  and Chapters 1–9 have been fully drafted, tested, and green on CI
  since Chapter 9 shipped. Rewrote both to present-tense, verified
  language: a per-chapter status table with ✅/— columns, the real
  `52`-test count across Chapters 1–9, and real numbers pulled from
  the chapters themselves (Chapter 6's `75/76` gate pass rate and `6`
  duplicate field-value groups, Chapter 8's `2.755 → 2.164 → 1.821`
  training-loss curve, Chapter 9's `4/4` vs `3/4` BM25-vs-dense
  retrieval result) in place of the old placeholder chapter map.
- The GitHub issue tracker showed 82 open issues even though Part 0 and
  every chapter, the diagrams pipeline, all appendices, and the
  companion app (V1-V3) are drafted, verified, and passing CI. The
  tracker was a granular project-planning breakdown created in one
  batch on 2026-07-20 (per-chapter tracking issues plus five sub-tasks
  each -- Write narrative / Verify code / Create figures / Review /
  Final proof -- for Chapters 3-13, plus infrastructure issues for
  diagrams, appendices, and the companion app); only the earliest four
  issues (Part 0, Chapter 1, Chapter 2, the training dataset) were ever
  closed as the corresponding work shipped, and closing lapsed after
  that even though the work kept shipping. Closed the 80 issues that
  map to now-completed work, each with a comment pointing to the
  specific commit that completed it (e.g. Chapter draft commits like
  `e1af03c`, the diagrams-pipeline commit `4e3b17f`, the companion app's
  `745cb75`/`4a5d9bc`/`1a4535e`/`36e3f6a`). Verified against the real
  repository state before closing anything -- e.g. confirmed
  `figures/cover.jpg` does not exist and no companion-app screenshots
  are referenced from any chapter -- so 2 issues (`#1` book cover,
  `#80` companion app screenshots) were left open as genuinely
  unfinished work, not closed along with the rest.
