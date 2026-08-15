# Fine-Tuning Local LLM for Drilling & Completions

[![Code tests Linux](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-linux.yml/badge.svg)](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-linux.yml)
[![Code tests macOS](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-macos.yml/badge.svg)](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-macos.yml)
[![Code tests Windows](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-windows.yml/badge.svg)](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-windows.yml)
[![Publish book to GitHub Pages](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/publish.yml/badge.svg)](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/publish.yml)

> **Status: in progress.** Part 0 and Chapters 1–9 are written, tested,
> and passing CI on Linux/macOS/Windows; Chapters 10–13 are still
> placeholders. Every quoted number in Chapters 1–9 comes from a real,
> reproducible run of this repository's own code — see
> [RELEASE.md](RELEASE.md) / [CHANGELOG.md](CHANGELOG.md) for the full
> history of what's landed.
>
> The "Publish book to GitHub Pages" workflow is manual-only
> (`workflow_dispatch`) until the remaining chapters are drafted — its
> badge reflects the last manual run, not every push to `main`.

This repository will contain the chapters, code, and training data for
**Fine-Tuning Local LLM for Drilling & Completions** — a hands-on,
build-as-you-go book that teaches drilling and completions engineers how
to fine-tune and run a private, local large language model on their own
operational data, assuming zero prior programming or machine-learning
experience.

It follows the same build-as-you-go structure as the author's previous
book, [*Building Industrial RAG Systems from Daily Drilling
Reports*](https://github.com/djimrastephane/ddr-rag-book) (`ddr-rag-book`):
a Quarto book, one chapter at a time, each shipping runnable code, tests,
and exercises — but this time the destination is a small, fine-tuned
language model that runs entirely on your own machine, not a retrieval
pipeline over a document archive.

Part II's "industrializing" chapters (6–13: data quality gates, evidence
traceability, evaluation, drift detection, continuous retraining) are
informed by the author's private companion project
[`industrial-ddr-finetuning`](https://github.com/djimrastephane/industrial-ddr-finetuning),
which already runs a schema-v2 extraction pipeline (per-field status,
verbatim evidence spans, automated validation, and a review workflow)
over this same Utah FORGE archive. That repo is private (its `raw_data/`
mirrors the same public PDFs used here); this book's Part II chapters
teach the underlying techniques from scratch rather than depending on it.

### What You're Building

Starting from a general-purpose local model that answers drilling and
completions questions generically (or gets the shorthand and the
operational context wrong), you fine-tune it — chapter by chapter — into a
model that has actually learned your domain's language, reports, and
judgment calls. No cloud training service, no proprietary API, no data
leaving your machine: every chapter runs against models and hardware you
control.

- Link to the [official source code repository](https://github.com/djimrastephane/llm-ft-drilling-completions-book)
- License: code is [MIT](LICENSE); the book's text is [CC BY 4.0](LICENSE-CONTENT.md)

To get a copy of this repository, click the [Download ZIP](https://github.com/djimrastephane/llm-ft-drilling-completions-book/archive/refs/heads/main.zip) button, or run the following in a terminal:

```bash
git clone https://github.com/djimrastephane/llm-ft-drilling-completions-book.git
```

Never used a terminal or Git before? That's exactly what **Start Here**
and **Part 0** below are for — nothing past this point assumes you
already know how.

---

# Start Here

This README has one job: get you to successfully complete Part 0 and
into Chapter 1. If this is your first Python project, do these steps in
order:

1. Read [Part 0: Preparing Your Local LLM Workshop](book/chapters/chapter_00.qmd) — installs Python, clones this repository, and gets your environment ready. No prior experience assumed.
2. Run `setup_check.py` — one command that confirms everything is working.
3. Continue to [Chapter 1: Loading and Running Your First Local LLM](book/chapters/chapter_01.qmd) — load a real open-weight model and generate your first answer, entirely on your own machine.
4. Keep going through Chapter 9 (see the [Table of Contents](#table-of-contents) below) — each chapter builds directly on the last one's saved output.

| Step | Typical time |
|---|---|
| Part 0 | ~30–45 minutes |
| Chapters 1–9, typing every example | several hours across multiple sessions — Chapters 5 and 8 each include a real fine-tuning run (~5 min and ~30 min on CPU) |

You don't need to understand everything before you start — you need to
run the first command. Everything else follows from there.

---

# Your Learning Journey

```
Base Local LLM
   ↓
Domain Training Data
   ↓
Baseline Prompting (what it gets wrong)
   ↓
Tokenization & Embeddings
   ↓
First LoRA Fine-Tune
   ↓
Data Quality & Scale
   ↓
Hybrid Fine-Tuning + Retrieval
   ↓
Traceable, Evaluated, Continuously Updated Model
```

## What You Will Build and Learn

Chapters 1–9 (below) already do this, for real, against this book's own
Utah FORGE archive — Chapters 10–13 are still planned. By the end of the
finished book you should be able to:

- Run a general-purpose local LLM and evaluate its out-of-the-box
  answers to drilling and completions questions ✅
- Turn raw drilling and completions reports into a training dataset ✅
- Run a first parameter-efficient (LoRA) fine-tune of a local model ✅
- Apply data quality gates to training data before fine-tuning ✅
- Fine-tune at scale with checkpointing and experiment tracking ✅
- Combine a fine-tuned model with retrieval (hybrid RAG + fine-tuning) ✅
- Evaluate a fine-tuned model's quality and detect hallucinations
  (planned — Chapters 10–11)
- Detect drift across model versions and keep the model current as new
  reports arrive (planned — Chapters 12–13)

## Who This Book Is For

This book is written for:

- Drilling Engineers
- Completion Engineers
- Intervention Engineers
- Production Engineers
- Digital Oilfield Professionals
- Energy Data Scientists

If you've ever wanted a model that actually understands your rig-floor
shorthand and your operator's reporting style — instead of a generic
chatbot — this book is for you.

## Who This Book Is Not For

This book is probably not for you if:

- you want a theoretical machine learning textbook
- you want a mathematical treatment of transformers
- you want to pretrain a foundation model from scratch
- you already fine-tune production-scale LLMs professionally

None of that is a criticism — it just means your time is better spent
elsewhere.

## Expected Background

**Helpful:**

- operational experience
- experience reading drilling and completions reports
- comfort in Excel
- curiosity

**Not required:**

- Python
- AI or machine learning
- software engineering
- Git
- Linux

## Minimum Computer Requirements

**Minimum:**

- 16 GB RAM
- 20 GB free disk space
- a modern CPU

**Recommended:**

- A GPU with at least 8 GB VRAM (NVIDIA/CUDA or Apple Silicon via MPS)
  makes fine-tuning chapters faster. Every chapter through Chapter 9 has
  actually been run and verified CPU-only on ordinary laptop hardware —
  Chapter 5's fine-tune takes about 5 minutes, Chapter 8's "at scale"
  run about 30–35 minutes.

**No cloud account required. No paid API required.** Everything in this
book is designed to run locally, using small open-weight models and
parameter-efficient fine-tuning (LoRA/QLoRA) so a single consumer GPU — or
patience, on CPU — is enough.

## Choose Your Workshop

| Environment | Recommended for |
|---|---|
| Jupyter Notebook | Learning and experimentation |
| VS Code | General coding |
| PyCharm Community | Larger projects |
| Positron | Data science workflows |
| Terminal only | Minimal setup |

[Part 0](book/chapters/chapter_00.qmd) covers general setup, with a
short dedicated walkthrough for each option in Appendices A1–A5,
mirroring the previous book.

---

# Table of Contents

Part 0 and Chapters 1–9 are written, tested, and passing CI; Chapters
10–13 are still placeholders. For the full repository layout (folder
tree, part/chapter file map) see [`book/README.md`](book/README.md).

[![Publish book to GitHub Pages](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/publish.yml/badge.svg)](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/publish.yml)
[![Code tests Linux](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-linux.yml/badge.svg)](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-linux.yml)
[![Code tests Windows](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-windows.yml/badge.svg)](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-windows.yml)
[![Code tests macOS](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-macos.yml/badge.svg)](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-macos.yml)

| Status | Chapter | Artifact |
|---|---|---|
| ✅ | [Part 0: Preparing Your Local LLM Workshop](book/chapters/chapter_00.qmd) | environment + hardware check (`setup_check.py`) |
| ✅ | [Ch 1: Loading and Running Your First Local LLM](book/chapters/chapter_01.qmd) | `code/chapter_01/load_local_model.py` |
| ✅ | [Ch 2: Turning Drilling & Completions Reports into Training Examples](book/chapters/chapter_02.qmd) | `code/chapter_02/build_training_examples.py` |
| ✅ | [Ch 3: Baseline Prompting — What the Model Gets Wrong Before Fine-Tuning](book/chapters/chapter_03.qmd) | `code/chapter_03/baseline_prompting.py` |
| ✅ | [Ch 4: Tokenization and Embeddings for Domain Fine-Tuning](book/chapters/chapter_04.qmd) | `code/chapter_04/tokenize_and_embed.py` |
| ✅ | [Ch 5: Your First LoRA Fine-Tune](book/chapters/chapter_05.qmd) | `code/chapter_05/first_lora_finetune.py` |
| ✅ | [Ch 6: A Data Quality Gate for Training Data](book/chapters/chapter_06.qmd) | `code/chapter_06/data_quality_gate.py` |
| ✅ | [Ch 7: Formatting and Chunking a Training Set at Scale](book/chapters/chapter_07.qmd) | `code/chapter_07/format_training_chunks.py` |
| ✅ | [Ch 8: Fine-Tuning at Scale — Checkpoints and Experiment Tracking](book/chapters/chapter_08.qmd) | `code/chapter_08/finetune_at_scale.py` |
| ✅ | [Ch 9: Hybrid System — Combining Fine-Tuning with Retrieval](book/chapters/chapter_09.qmd) | `code/chapter_09/hybrid_rag_finetune.py` |
| — | Ch 10: Traceable Outputs and Hallucination Mitigation | `code/chapter_10/traceable_outputs.py` |
| — | Ch 11: Evaluating a Fine-Tuned Domain Model | `code/chapter_11/eval_finetuned_model.py` |
| — | Ch 12: Detecting Drift Across Model Versions | `code/chapter_12/detect_model_drift.py` |
| — | Ch 13: Continuous Fine-Tuning — Keeping the Model Current | `code/chapter_13/continuous_finetune.py` |
| ✅ | Appendix A: Environment Setup | — |
| ✅ | Appendices A1–A5: Jupyter / VS Code / PyCharm / Positron / Terminal-only | — |
| ✅ | [Appendix B: Drilling, Completions & Fine-Tuning Glossary](book/appendix/appendix_b_glossary.qmd) | — |

Every ✅ chapter ships with working, tested code and a companion
notebook — see [Automated Tests](#automated-tests) below. Every quoted
number in a ✅ chapter comes from an actual run of that chapter's own
code against this repository's real Utah FORGE archive, not an
estimate.

## Companion App (planned)

An optional Streamlit companion app is planned under
[`book/app/`](book/app), reusing the book's own chapter code (base model
loading, fine-tuning, and evaluation), to show a question answered
side-by-side by the base model and the fine-tuned model. See
[`book/app/README.md`](book/app/README.md) for current status.

## Exercises

Every drafted chapter includes a **Practical exercise** and a
**Challenge exercise**, with reference solutions alongside each
chapter's code under `book/code/chapter_NN/challenge/`.

## Automated Tests

Every drafted chapter's real code is tested in
[`book/tests/`](book/tests) — 52 tests across Chapters 1–9 as of this
writing, run on Linux, Windows, and macOS on every push that touches
`book/**` (badges at the top of this README).

```bash
cd book
pip install -r requirements.txt
pytest -v
```

Chapters 5, 8, and 9 include tests marked `slow` (they load and
generate from the real base model, and Chapter 8's fine-tuning tests
take a few minutes) or that need a checkpoint from a previous chapter's
script to already exist on disk (skipped automatically if it doesn't —
see each test file's own docstring). Skip the slow ones locally with:

```bash
pytest -v -m "not slow"
```

## Questions, Feedback, and Contributing to This Repository

Questions, corrections, and feedback are all welcome via [GitHub
Issues](https://github.com/djimrastephane/llm-ft-drilling-completions-book/issues).
Progress is tracked on the repository's [GitHub Project
board](https://github.com/djimrastephane?tab=projects).

## A Note on AI-Assisted Development

The ideas, engineering examples, and technical validation are the
author's. AI tools including Claude were used to accelerate scaffolding,
coding, editing, and documentation tasks.

## Citation

Once published, this book can be cited as:

Chicago-style citation:

> Djimra Stephane Soulanoudjingar. *Fine-Tuning Local LLM for Drilling &
> Completions*. 2026. https://github.com/djimrastephane/llm-ft-drilling-completions-book.

BibTeX entry:

```bibtex
@book{llm-ft-drilling-completions-book,
  author  = {Djimra Stephane Soulanoudjingar},
  title   = {Fine-Tuning Local LLM for Drilling \& Completions},
  year    = {2026},
  github  = {https://github.com/djimrastephane/llm-ft-drilling-completions-book}
}
```
