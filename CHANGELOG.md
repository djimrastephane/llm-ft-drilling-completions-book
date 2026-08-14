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
  chunked. Chapters 8–13 remain placeholders.

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

### Fixed

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
