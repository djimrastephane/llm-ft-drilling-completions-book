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
  troubleshooting.
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
  PLANNED fields with `pdfplumber` and turns them into 18 real
  instruction/response training examples from 9 of the 10 sample
  reports (the completion report uses a different field layout, handled
  in the challenge exercise). Every field value and example count in
  the chapter is from a real, verified run.
- **Chapter 3: Baseline Prompting: What the Model Gets Wrong Before
  Fine-Tuning** (`chapters/chapter_03.qmd`) fully drafted, with working,
  tested code (`code/chapter_03/baseline_prompting.py`,
  `code/chapter_03/challenge/challenge.py`, `tests/test_chapter_03.py`).
  Runs the unmodified base model against Chapter 2's 18 training
  examples with each `output` withheld, and saves the results to
  `datasets/training_examples/baseline_results.jsonl` -- the fixed
  baseline Chapter 5's fine-tuned model gets compared against. On a
  real run against the sample archive, the base model scores `0/18`
  on exact-match (it correctly declines to guess rather than
  hallucinating a wrong answer); the challenge exercise shows that
  score rising to `6/18` once each report's full text is included as
  context. Also fixes a real bug surfaced while drafting this chapter:
  `code/chapter_01/load_local_model.py` now disables `transformers`'
  TensorFlow/Flax backend probing (`USE_TF=0`, `USE_FLAX=0`), since an
  unrelated, broken TensorFlow install on a shared/Anaconda-style
  environment was segfaulting model loading for every chapter that
  imports it. Chapters 4–13 remain placeholders.
