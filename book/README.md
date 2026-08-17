# Fine-Tuning Local LLM for Drilling & Completions

[![Code tests Linux](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-linux.yml/badge.svg)](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-linux.yml)
[![Code tests Windows](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-windows.yml/badge.svg)](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-windows.yml)
[![Code tests macOS](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-macos.yml/badge.svg)](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-macos.yml)
[![Publish book to GitHub Pages](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/publish.yml/badge.svg)](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/publish.yml)

Companion source for the book of the same name — a hands-on,
build-as-you-go guide that teaches drilling and completions engineers how
to fine-tune and run a private, local large language model on their own
operational data, assuming no prior programming experience.

Readers start with a general-purpose local model and a folder of drilling
and completions reports, and finish with a fine-tuned model that
understands their domain — one working chapter at a time, using real,
publicly available Daily Drilling Reports from Utah FORGE (a DOE-funded
geothermal research well) throughout.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

quarto render                       # builds the book to _book/
```

The sample training archive is already committed under `datasets/` — no
generation or download step needed.

New to Python or unsure where to start? **Part 0 — Preparing Your Local
LLM Workshop** (`chapters/chapter_00.qmd`) walks through the setup above
one command at a time, with plain-English explanations and no assumed
prior experience. It works with any editor — Jupyter Notebook, VS Code,
PyCharm Community, Positron, or a terminal alone — with a short dedicated
guide for each in Appendices A1–A5. [Appendix
A](appendix/appendix_a_environment_setup.qmd) covers what's left: the
dataset, rendering the book, and hardware notes.

## Project structure

```
book/
├── _quarto.yml              Quarto book configuration
├── index.qmd                Welcome / front page
├── preface.qmd
├── acknowledgements.qmd
├── references.bib           Bibliography
├── custom.scss               HTML theme overrides
│
├── chapters/                 One .qmd per chapter (see below)
├── templates/
│   └── chapter_template.qmd  Copy this to draft a new chapter
│
├── code/                     One subfolder per chapter, e.g. code/chapter_01/
│   ├── setup_check.py        Part 0's environment + hardware check
│   └── chapter_NN/
│       ├── *.py              Runnable scripts referenced in the chapter
│       └── challenge/         Reference solutions to challenge exercises
│
├── tests/                     pytest suite exercising every chapter's real code
│   ├── conftest.py            Shared fixtures (dataset paths, sys.path setup)
│   └── test_chapter_NN.py     One file per chapter with testable code
├── pytest.ini                 pytest configuration (testpaths, markers)
│
├── datasets/
│   ├── sample_training_set/   10 real, curated Utah FORGE reports (Part I)
│   └── full_training_set/     Full 76-report Utah FORGE archive (Part II)
│
├── notebooks/                 Interactive Jupyter/Quarto companion notebooks
├── figures/                   Book figures and diagrams
├── app/                       Streamlit companion app -- model comparison lab (see app/README.md)
│   ├── streamlit_app.py       Model Playground (landing page)
│   ├── pages/                 Additional pages, e.g. Before/After Evaluation
│   └── helpers.py             App glue logic, no Streamlit import -- unit-testable
└── appendix/
    ├── appendix_a_environment_setup.qmd    Dataset, rendering, hardware notes
    ├── appendix_a1_jupyter.qmd             Jupyter Notebook guide
    ├── appendix_a2_vscode.qmd              VS Code guide
    ├── appendix_a3_pycharm.qmd             PyCharm Community guide
    ├── appendix_a4_positron.qmd            Positron guide
    ├── appendix_a5_terminal.qmd            Terminal-only guide
    └── appendix_b_glossary.qmd             Drilling, completions & fine-tuning terms
```

## Chapter map

**Part 0 — Preparing Your Local LLM Workshop** (`chapters/chapter_00.qmd`,
✅ written): environment and hardware setup for readers with no prior
programming experience, IDE-agnostic — Jupyter Notebook, VS Code, PyCharm
Community, Positron, or terminal only, each with a short dedicated guide
in Appendices A1–A5.

**Part I — Foundations** (10 real, curated Utah FORGE reports — a
genuine stuck-pipe event and more — fully reproducible offline; ✅ all 5
chapters written and tested):

| Ch. | Artifact you build |
|---|---|
| 1 | Local model loading and inference script |
| 2 | Drilling & completions training-example builder |
| 3 | Baseline prompting harness (what the base model gets wrong) |
| 4 | Tokenization and embedding walkthrough |
| 5 | First LoRA fine-tune |

**Part II — Industrialising the System** (✅ all 8 chapters written and
tested; Chapters 6 and 9 in
particular are informed by the author's private companion project
**`industrial-ddr-finetuning`**, built specifically against this book's
public archive — real per-report extraction, a real 75/76 data-quality
gate result, and real, verified checkpointed fine-tuning and retrieval
numbers):

| Ch. | Status | Artifact you build |
|---|---|---|
| 6 | ✅ | Data quality gate for training data |
| 7 | ✅ | Training-set formatting and chunking at scale |
| 8 | ✅ | Fine-tuning at scale with checkpointing and experiment tracking |
| 9 | ✅ | Hybrid system combining fine-tuning with retrieval |
| 10 | ✅ | Traceable, hallucination-mitigated outputs |
| 11 | ✅ | Fine-tuned model evaluation harness |
| 12 | ✅ | Model drift detector across versions |
| 13 | ✅ | Continuous fine-tuning / retraining pipeline |

Each written Part II chapter includes a simplified, standalone
implementation in `code/chapter_NN/` (no external service required to
follow along) plus pointers to the exact files behind any private
companion-project number it cites. Where a chapter states a specific
number or result, it was independently verified against this book's own
code and real training data before being written down — this book does
not present fabricated or borrowed results as if they came from a real
run. A few examples (see the root [README.md](../README.md)'s "Table of
Contents" section for the same list with each metric explained in plain
language):

- Chapter 6's data quality gate found `75/76` reports pass extraction
  with `6` duplicate field-value groups.
- Chapter 8's "at scale" fine-tune measured training loss falling
  `2.755 → 2.164 → 1.821` across 3 real epochs.
- Chapter 9 measured BM25 keyword retrieval (ranking reports by shared
  words with the question) finding the correct source report `4/4`
  times on real test queries, against `3/4` for dense sentence
  embeddings (matching by meaning instead of exact wording).
- Chapter 10's faithfulness check caught a real answer that cited the
  correct report while actually being grounded in a different one.
- Chapter 11 measured perplexity (how "surprised" the model is by real
  text it never trained on — lower is better) on real held-out text
  falling from `159.91` (base model) to `25.03` (fine-tuned), even
  though exact-match on the same 8 questions stayed `0/8` at every
  training epoch.
- Chapter 12 found `avg_overlap` (how much of the model's wording
  overlaps with the known-correct answer) and `perplexity` disagreeing
  on direction between two real checkpoints of the same training run.
- Chapter 13 simulated new reports arriving and retrained on them, and
  the same disagreement showed up a third time on a real
  continuous-fine-tuning run.

## Relationship to the companion pipeline

[**`industrial-ddr-finetuning`**](https://github.com/djimrastephane/industrial-ddr-finetuning)
is a separate, private repository from this book. This book's own code
never depends on it — every chapter's `code/chapter_NN/` scripts run
standalone against the committed `datasets/` archive. Its
schema-v2 extraction pipeline (field-level status, verbatim evidence
spans, automated validation, review workflow) runs over this same public
FORGE archive; where Part II cites a number from it, the chapter text
says so explicitly. See `datasets/README.md` for how the two projects
relate.

## Recommended workflow in Positron (for authors drafting new chapters)

This section describes the author's own workflow for *writing* new
chapters — it is not a requirement for readers. Readers following the
book should start with Part 0 and Appendices A1–A5, which cover five
different editors on equal footing.

1. Open `book/` as the Positron workspace root.
2. Create and select the `.venv` interpreter (see Quickstart above).
3. Draft a chapter in `chapters/chapter_NN.qmd`, starting from
   `templates/chapter_template.qmd`.
4. Develop and test the chapter's code as a plain script in
   `code/chapter_NN/` first — run it from the Positron console — then
   move working code blocks into the `.qmd` once verified.
5. Use `quarto preview chapters/chapter_NN.qmd` for live-reloading a
   single chapter while writing.
6. Run `quarto render` before committing, to catch any chapter that
   fails to execute end to end.

## Reproducing the whole book

```bash
pip install -r requirements.txt
quarto render
```

Output is written to `_book/`. No API keys or paid services are required
— every chapter runs against small, open-weight local models
(`Qwen2.5-1.5B-Instruct`) and parameter-efficient (LoRA) fine-tuning.

## Running tests

`tests/` exercises the real functions in every chapter's
`code/chapter_NN/` script — 74 tests across all 13 chapters. CI runs the
full suite on Linux, Windows, and macOS on every push and pull request
that touches `book/**` (see `.github/workflows/tests-linux.yml`,
`tests-windows.yml`, `tests-macos.yml`), and all three are green.

```bash
pip install -r requirements.txt
pytest -v
```

Tests marked `slow` or `gpu` are expected to download a model or require
real fine-tuning hardware; skip them locally with
`pytest -v -m "not slow and not gpu"` if you're offline or on modest
hardware.

## License

Code (`code/`, `notebooks/`) is licensed under the [MIT License](../LICENSE).
The book's text (chapters, preface, appendices) is licensed under
[CC BY 4.0](../LICENSE-CONTENT.md). The Utah FORGE report data in
`datasets/` is public DOE-funded research data, not covered by either
license — see `datasets/README.md`.
