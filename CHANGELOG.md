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

### Fixed

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
