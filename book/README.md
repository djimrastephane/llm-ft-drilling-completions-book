# Fine-Tuning Local LLM for Drilling & Completions

Companion source for the book of the same name — a hands-on,
build-as-you-go guide that teaches drilling and completions engineers how
to fine-tune and run a private, local large language model on their own
operational data, assuming no prior programming experience.

Readers start with a general-purpose local model and a folder of drilling
and completions reports, and finish with a fine-tuned model that
understands their domain — one working chapter at a time.

**Status:** Part 0 and Chapters 1–9 are written, with working, tested
code and real, verified results; Chapters 10–13 are still placeholders.
See the [root README](../README.md#table-of-contents) for the
per-chapter status table, or [`CHANGELOG.md`](../CHANGELOG.md) for the
full history of what landed and why.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

quarto render                       # builds the book to _book/
```

New to Python or unsure where to start? **Part 0 — Preparing Your Local
LLM Workshop** (`chapters/chapter_00.qmd`) will walk through setup one
command at a time, with plain-English explanations and no assumed prior
experience, mirroring the previous book's Part 0.

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
programming experience, IDE-agnostic, each with a short dedicated guide
in Appendices A1–A5.

**Part I — Foundations** (✅ all 5 chapters written and tested):

| Ch. | Artifact you build |
|---|---|
| 1 | Local model loading and inference script |
| 2 | Drilling & completions training-example builder |
| 3 | Baseline prompting harness (what the base model gets wrong) |
| 4 | Tokenization and embedding walkthrough |
| 5 | First LoRA fine-tune |

**Part II — Industrialising the System** (✅ Chapters 6–9 written and
tested; Chapters 10–13 still placeholders):

| Ch. | Status | Artifact you build |
|---|---|---|
| 6 | ✅ | Data quality gate for training data |
| 7 | ✅ | Training-set formatting and chunking at scale |
| 8 | ✅ | Fine-tuning at scale with checkpointing and experiment tracking |
| 9 | ✅ | Hybrid system combining fine-tuning with retrieval |
| 10 | — | Traceable, hallucination-mitigated outputs |
| 11 | — | Fine-tuned model evaluation harness |
| 12 | — | Model drift detector across versions |
| 13 | — | Continuous fine-tuning / retraining pipeline |

Each written Part II chapter includes a simplified, standalone
implementation in `code/chapter_NN/` (no external service required to
follow along). Every specific number or result a written chapter states
comes from an actual, independently-verified run of this book's own
code against real training data — this book does not present
fabricated or borrowed results as if they came from a real run. A few
examples: Chapter 6's data quality gate found `75/76` reports pass
extraction with `6` duplicate field-value groups; Chapter 8's "at
scale" fine-tune measured training loss falling `2.755 → 2.164 →
1.821` across 3 real epochs; Chapter 9 measured BM25 keyword retrieval
finding the correct source report `4/4` times on real test queries,
against `3/4` for dense sentence embeddings.

Chapters 6, 9, and 10 in particular are informed by the author's private
companion project [`industrial-ddr-finetuning`](https://github.com/djimrastephane/industrial-ddr-finetuning),
which runs a schema-v2 extraction pipeline (field-level status, verbatim
evidence spans, automated validation, review workflow) over this same
FORGE archive — see `book/datasets/README.md`. The book's own
implementations are written from scratch for teaching purposes, not
copied from it.

## Recommended workflow in Positron (for authors drafting new chapters)

This section describes the author's own workflow for *writing* new
chapters — it is not a requirement for readers.

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

## Running tests

`tests/` exercises the real functions in every chapter's
`code/chapter_NN/` script — 52 tests across Chapters 1–9. CI runs the
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
