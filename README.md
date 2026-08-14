# Fine-Tuning Local LLM for Drilling & Completions

> **Status: early draft.** Part 0 is written; Chapters 1–13 are still
> placeholders. This README describes the planned book so the structure
> below makes sense as it fills in. See [RELEASE.md](RELEASE.md) /
> [CHANGELOG.md](CHANGELOG.md) for what actually exists at any point in
> time.

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

This README has one job: get you to successfully complete Part 0. If
this is your first Python project, do these steps in order:

1. Read [Part 0: Preparing Your Local LLM Workshop](book/chapters/chapter_00.qmd) — installs Python, clones this repository, and gets your environment ready. No prior experience assumed.
2. Run `setup_check.py` — one command that confirms everything is working.
3. Continue to Chapter 1 (not yet written) once it lands.

| Step | Typical time |
|---|---|
| Part 0 | ~30–45 minutes |

You don't need to understand everything before you start — you need to
run the first command. Everything else follows from there.

---

# Your Learning Journey (planned)

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

## What You Will Build and Learn (planned)

By the end of the book you should be able to:

- ✓ Run a general-purpose local LLM and evaluate its out-of-the-box
  answers to drilling and completions questions
- ✓ Turn raw drilling and completions reports into a training dataset
- ✓ Run a first parameter-efficient (LoRA) fine-tune of a local model
- ✓ Apply data quality gates to training data before fine-tuning
- ✓ Fine-tune at scale with checkpointing and experiment tracking
- ✓ Combine a fine-tuned model with retrieval (hybrid RAG + fine-tuning)
- ✓ Evaluate a fine-tuned model's quality and detect hallucinations
- ✓ Detect drift across model versions and keep the model current as new
  reports arrive

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

## Minimum Computer Requirements (planned)

**Minimum:**

- 16 GB RAM
- 20 GB free disk space
- a modern CPU

**Recommended:**

- A GPU with at least 8 GB VRAM (NVIDIA/CUDA or Apple Silicon via MPS)
  makes fine-tuning chapters dramatically faster. CPU-only readers can
  still follow along with a very small base model, at the cost of slower
  training runs.

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

# Table of Contents (planned)

Part 0 is written; the chapter map below for Chapters 1–13 is still
planned, not yet drafted. For the full repository layout (folder tree,
part/chapter file map) see [`book/README.md`](book/README.md).

[![Publish book to GitHub Pages](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/publish.yml/badge.svg)](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/publish.yml)
[![Code tests Linux](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-linux.yml/badge.svg)](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-linux.yml)
[![Code tests Windows](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-windows.yml/badge.svg)](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-windows.yml)
[![Code tests macOS](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-macos.yml/badge.svg)](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-macos.yml)

| Chapter | Planned artifact |
|---|---|
| Part 0: Preparing Your Local LLM Workshop | environment + hardware check (`setup_check.py`) |
| Ch 1: Loading and Running Your First Local LLM | `code/chapter_01/load_local_model.py` |
| Ch 2: Turning Drilling & Completions Reports into Training Examples | `code/chapter_02/build_training_examples.py` |
| Ch 3: Baseline Prompting — What the Model Gets Wrong Before Fine-Tuning | `code/chapter_03/baseline_prompting.py` |
| Ch 4: Tokenization and Embeddings for Domain Fine-Tuning | `code/chapter_04/tokenize_and_embed.py` |
| Ch 5: Your First LoRA Fine-Tune | `code/chapter_05/first_lora_finetune.py` |
| Ch 6: Data Quality Gates for Training Data | `code/chapter_06/data_quality_gate.py` |
| Ch 7: Formatting and Chunking a Training Set at Scale | `code/chapter_07/format_training_chunks.py` |
| Ch 8: Fine-Tuning at Scale — Checkpoints and Experiment Tracking | `code/chapter_08/finetune_at_scale.py` |
| Ch 9: Hybrid System — Combining Fine-Tuning with Retrieval | `code/chapter_09/hybrid_rag_finetune.py` |
| Ch 10: Traceable Outputs and Hallucination Mitigation | `code/chapter_10/traceable_outputs.py` |
| Ch 11: Evaluating a Fine-Tuned Domain Model | `code/chapter_11/eval_finetuned_model.py` |
| Ch 12: Detecting Drift Across Model Versions | `code/chapter_12/detect_model_drift.py` |
| Ch 13: Continuous Fine-Tuning — Keeping the Model Current | `code/chapter_13/continuous_finetune.py` |
| Appendix A: Environment Setup | — |
| Appendices A1–A5: Jupyter / VS Code / PyCharm / Positron / Terminal-only | — |
| Appendix B: Drilling, Completions & Fine-Tuning Glossary | `book/appendix/appendix_b_glossary.qmd` |

## Companion App (planned)

An optional Streamlit companion app is planned under
[`book/app/`](book/app), reusing the book's own chapter code (base model
loading, fine-tuning, and evaluation), to show a question answered
side-by-side by the base model and the fine-tuned model. See
[`book/app/README.md`](book/app/README.md) for current status.

## Exercises

Every chapter will include a **Practical exercise** and a **Challenge
exercise**, with reference solutions alongside each chapter's code under
`book/code/chapter_NN/challenge/` — same convention as the previous book.

## Automated Tests

Every chapter's code will be tested in [`book/tests/`](book/tests) on
Linux, Windows, and macOS. Once dependencies and code exist:

```bash
cd book
pip install -r requirements.txt
pytest -v
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
