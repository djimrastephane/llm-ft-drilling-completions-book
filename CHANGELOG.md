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
  Chapters 2–13 remain placeholders.
