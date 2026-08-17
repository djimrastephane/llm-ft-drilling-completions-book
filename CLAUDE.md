# CLAUDE.md

Guidance for Claude Code (or any other agent) working in this repository.

## What this repo is

*Fine-Tuning Local LLM for Drilling & Completions* — a hands-on Quarto
book teaching drilling/completions engineers with zero programming
experience to fine-tune and run a local LLM on their own operational
data. Companion source lives under `book/`; see `book/README.md` for
the full folder layout and `README.md` (root) for the reader-facing
front door and chapter status table — keep both in sync when a
chapter's status changes.

**Status**: Part 0 and all 13 chapters are written, tested, and passing
CI. Always check `CHANGELOG.md`'s `## [Unreleased]` section and
`git log` for the current real state — don't trust a stale memory of
"which chapters are done" (this file's own status line has gone stale
before; re-verify against `git log` rather than assuming this note is
current).

This project's sibling/predecessor is
[`ddr-rag-book`](https://github.com/djimrastephane/ddr-rag-book) (same
author, same structure, RAG instead of fine-tuning). When in doubt
about a convention, that repo is precedent, not this file.

## The one non-negotiable rule

**Every number, quote, or result stated in a chapter must come from an
actual run of this book's own code against real data — never
estimated, assumed, or invented.** This has caught real bugs before
(Chapter 7's example indexing, Chapter 9's retrieval-query design) and
real transcription errors (Chapter 10's `1.0` that was actually
`0.75`). When drafting or editing chapter prose:
- Run the exact code block shown in the `.qmd`, in sequence, and copy
  its real output — don't hand-write a plausible-looking transcript.
- If a claim can't currently be verified against real code, don't make
  it, or mark it as a documented future limitation instead.
- A surprising or "wrong" result is usually better chapter content
  than a clean one — this book's voice explicitly keeps honest
  failures and limitations rather than editing them out (see Chapter
  5's held-out generalization staying at `0/2`, Chapter 9's Field
  Notes on faithfulness, Chapter 10's `#21` wrong-source case).

## The other non-negotiable rule: who this is actually for

The reader is a **drilling, completions, or well-intervention engineer
with no-to-low programming experience** — not a software engineer, not
an ML practitioner, not someone who already reads code comfortably.
That's not a disclaimer, it's a constraint on every sentence written in
this repo: chapter prose, callouts, diagram labels and captions,
READMEs, error messages a reader might see, all of it.

What that means concretely:
- Explain a new concept with a physical/operational analogy the reader
  already has (a correction sheet, a pre-shift equipment check, a
  spreadsheet pivot table) **before** using its technical name — never
  the other way around.
- No unexplained jargon, no bare math/notation, no "matrix
  multiplication" or "outer product" dropped in without translation —
  even when it would be the fastest way to explain something to a
  technical reader. If a sentence needs a software/ML background to
  parse, rewrite it, don't caveat it.
- When technical precision and reader accessibility genuinely pull in
  different directions, accessibility wins for the *explanation*;
  precision gets preserved as a footnote-style caveat in Production
  Reality, not by making the main explanation harder to read.
- This has been gotten wrong and fixed before: Chapter 5's LoRA
  "low-rank" explanation originally read "two much narrower matrices
  multiplied together" — accurate, but exactly the kind of phrase this
  reader would bounce off. It was rewritten to "two much shorter lists
  of numbers... the way a large table can sometimes be rebuilt from a
  row list and a column list," matching the rest of the book's
  physical-analogy voice. Check new or edited prose against that bar,
  not against what would satisfy a software engineer reviewing it.
- "Comfort in Excel" is explicitly part of the reader's expected
  background (see the root README's "Expected Background" section) —
  spreadsheet/table analogies land; software-engineering ones don't.
- **Code blocks inside a chapter get comments only where the *why* is
  genuinely non-obvious** (e.g. `traceable_outputs.py`'s `STOPWORDS`
  comment explaining why instruction-template boilerplate is excluded)
  — not a line-by-line narration of what each line does. That's
  deliberate, not an oversight: the surrounding "What problem are we
  solving? / Inputs / Expected Output / What just happened?" prose
  carries the beginner-level explanation instead, right before and
  after the code block. A reader who can't yet parse Python leans on
  that prose scaffold, not on inline comments — so don't try to make a
  code block self-explanatory by piling comments into it; make sure
  the prose around it actually does that job, and keep the code itself
  clean.

## README maintenance

`README.md` (root) and `book/README.md` are not exempt from "the other
non-negotiable rule" above — the root README in particular is the
reader's first contact with the book, before Part 0 has taught them
anything, so it can't lean on a concept a later chapter defines. Whenever
either README is edited, re-read the full diff for:

- **Jargon defined before use, every time, including acronyms.** LoRA,
  QLoRA, RAG, BM25, perplexity, weights, checkpoint, hallucination,
  grounded/grounding, faithfulness, exact-match, held-out,
  generalization, open-weight, instruction-tuned, API, CI, VRAM — none
  of these get to appear bare on first use. Reuse the exact
  plain-language phrasing already established in
  `appendix/appendix_b_glossary.qmd` and each chapter's own Engineering
  Translation callouts (e.g. Chapter 9's BM25 callout, Chapter 11's
  perplexity callout) rather than inventing new wording — that's also
  what keeps the README and the chapters saying the same thing about the
  same concept.
- **Technique claims match what the code actually does, not what's
  aspirationally installed.** E.g. `requirements.txt` installs
  `bitsandbytes` for optional QLoRA support, but no chapter script
  actually loads a quantized model — a README line claiming
  "LoRA/QLoRA" is inaccurate; say "LoRA" (caught and fixed 2026-08-17).
- **Name/terminology consistency** between the two READMEs, the
  glossary, and the chapters themselves — same spelling of "Utah FORGE,"
  the same model name (`Qwen2.5-1.5B-Instruct`), the same companion
  project name (`industrial-ddr-finetuning`), the same chapter titles as
  the actual `.qmd` headings.
- **Natural flow, read start to finish as prose**, not just checked
  clause by clause — a parenthetical gloss on every third word reads
  worse than the jargon it's fixing. Prefer rephrasing a sentence to
  avoid the term entirely (e.g. "no paid, metered connection to someone
  else's hosted model" instead of "no paid API") over piling on
  parentheticals, and only add an explicit `**term** (meaning)` gloss
  where the term itself needs to stay visible (it's reused later, or
  it's the name of something the reader will see elsewhere, like a
  chapter title or a printed metric name).

## Chapter-writing conventions

Every chapter starts from `book/templates/chapter_template.qmd`, whose
comments are the authoritative style guide (audience, required
sections, persona rules). Highlights:

- **Audience**: zero Python/software-engineering experience, strong
  operational + Excel experience, curious about AI. Problem first,
  technology second — never introduce a concept before the reader has
  felt the failure that makes it necessary.
- **Structure per chapter**: pipeline diagram → chapter-status strip →
  Learning objectives → Operational Problem (one persona asking a real
  question) → Example → Theory (Engineering Translation callouts) →
  Implementation (Step N: What problem / Inputs / Expected Output /
  code / What just happened / Production Reality) → Practical exercise
  → Field notes → Challenge exercise → Key takeaways → Repository
  files → CHECKPOINT + WHAT YOU BUILT → What can you do now → Suggested
  next step (bolded "Coming up in Chapter N+1:").
- **Personas** (`index.qmd`'s roster, rotate — don't always reuse one):
  Oumy (drilling), Mike (completions), Sarah (intervention), Sean
  (production). One persona per chapter's Operational Problem, chosen
  for narrative fit, never appearing in Theory/Production
  Reality/Field notes (those stay unattributed and independently
  verified).
- **Engineering Translation callouts**: one per new concept, physical/
  operational analogy (a correction sheet, a pre-shift equipment
  check), no unexplained jargon and no bare math notation — see
  Chapter 5's LoRA callout for the bar this is held to (reviewed and
  revised twice to keep it accessible).
- **Difficulty/time**: 🟢 Beginner / 🟠 Intermediate / 🔴 Advanced badge
  and an honest estimated-time range in the chapter-status strip; keep
  the root README's "How Long Does Each Chapter Take" table in sync.

## Technical gotchas

- **Import order**: `load_local_model.py` sets `USE_TF=0`/`USE_FLAX=0`
  before importing `transformers`, to avoid a segfault probing a
  broken TensorFlow install on shared/Anaconda-style environments. Any
  script or test that also imports `peft`/`sentence_transformers` must
  import `load_local_model` **first**. This has been the cause of real
  crashes multiple times when a new chapter's import order got this
  wrong.
- **Local dev environment**: the system `python3` on this machine has
  no ML deps installed. Use `/opt/anaconda3/bin/python3` (has
  `torch`/`peft`/`transformers`/etc.) for running chapter scripts,
  pytest, and notebook execution locally. CI installs `requirements.txt`
  into a fresh Python 3.11 venv, so this split is a local-machine
  quirk, not a CI concern.
- **Checkpoints are gitignored**: `checkpoints/chapter_05_lora/` and
  `checkpoints/chapter_08/run_*/` must exist locally to run
  checkpoint-dependent chapters (9, 10, ...). Chapter 8's training run
  takes ~30 min on CPU. Tests that need a checkpoint use
  `pytest.mark.slow` and skip cleanly (`pytest.skip(...)`) if none
  exists — never make a test hard-fail on a missing checkpoint.
- **Rendering the whole book**: use plain `quarto render` (no `--to`
  flag). `quarto render --to html` and `--to pdf` run as separate
  passes and each one wipes the other format's output file from
  `_book/`, even though cross-format links/downloads still reference
  it. This was a real, previously-diagnosed "the PDF download is
  broken" bug — it wasn't a real bug, just the wrong render command.
- **Figure captions in HTML with light/dark image pairs**: putting
  `fig-cap` directly on an `<img>` inside a `.pipeline-diagram` div
  does *not* produce a numbered Quarto figure — it silently becomes an
  inert `data-fig-cap` attribute. To get a real, numbered "Figure N.M"
  caption in both HTML and PDF: wrap both images in a shared
  `#fig-your-label` div, one image per blank-line-separated paragraph,
  with **no** `#fig-` id on the individual images (that triggers
  unwanted `(a)`/`(b)` subfigure treatment instead). See Chapter 5's
  `lora_lowrank_ch05`/`recall_gap_ch05` figures for the working
  pattern.

## Diagrams

`book/figures/diagrams/generate_diagrams.py` compiles TikZ → PDF (via
`pdflatex`) → light/dark SVG (via `pdftocairo`). Two renderer shapes
exist:
- `render_diagram()` / `DIAGRAMS` / `THEORY_DIAGRAMS`: simple vertical
  box chains (2–4 boxes), used for every chapter's opening pipeline
  diagram. Can't branch.
- Dedicated per-figure functions (`render_lora_diagram()`,
  `render_recall_gap_diagram()`) for anything else — a 2D grid, a bar
  chart, a number line. Each gets its own TikZ template string; there
  is no generic "arbitrary diagram" DSL, by design (avoid the
  abstraction until a second real use case actually needs it).
- `LIGHT_PALETTE`/`DARK_PALETTE` mirror `custom-light.scss`/
  `custom-dark.scss`'s actual colors — pull new colors (e.g. a warning
  accent) from the compiled Bootstrap CSS in `_book/site_libs/`, don't
  invent new ones.
- After adding a diagram, render it standalone first
  (`pdftoppm -png -r 200 name.pdf preview` + `Read` the PNG) to check
  spacing/overlap before wiring it into a chapter — TikZ text needs
  generous clearance (0.4–0.5 units) around box/grid edges or labels
  collide.
- New diagrams are prototyped as an HTML/SVG sketch (published as an
  Artifact) and reviewed *before* being built in TikZ — see Chapter
  5's and Chapter 6's diagram threads for the pattern. Chapter 6's
  number-line sketch was reworked into a simpler table + gap-bar
  design after review; that revision is parked, not yet built into the
  book.

## Testing

- `book/tests/conftest.py` auto-adds every `code/chapter_NN/` to
  `sys.path`, so tests import chapter modules directly
  (`from traceable_outputs import ...`), unlike the standalone scripts
  themselves which need manual `sys.path.insert` (they're meant to be
  run as `python code/chapter_NN/script.py`, not as a package).
  Reserve `pytest.mark.slow` for anything that loads/generates from the
  real model; `pytest.mark.gpu` for anything needing CUDA/MPS.
- Run fast tests: `pytest -v -m "not slow"`. Run everything (needs
  checkpoints + a real model download):
  `/opt/anaconda3/bin/python3 -m pytest tests/ -v`.
- One test file per chapter, matching the chapter's own real functions
  — never a simplified stand-in.

## Git / commit conventions

- Real conversational pattern in this repo: draft → verify by actually
  running the code → write CHANGELOG entry → commit → **ask before
  pushing** → push only once told to → check CI status
  (`gh run list --branch main --limit 3`) and report pass/fail, don't
  assume.
- `CHANGELOG.md`: entries append to the **end** of `## [Unreleased]`'s
  `### Added` section in chronological order (new chapters/features),
  with `### Changed`/`### Fixed` below for revisions to already-shipped
  content. Each entry is detailed — real numbers, what was verified,
  what was caught and fixed — not a one-line summary. Read a few
  existing entries before writing a new one to match depth and tone.
- Commit messages end with `Co-Authored-By: Claude Sonnet 5
  <noreply@anthropic.com>`. Never force-push; never amend a pushed
  commit — create a new commit instead.
- `checkpoints/`, `_book/`, `__pycache__/`, `.pytest_cache/` are
  gitignored — don't add them, and revert any non-deterministic PDF
  diff noise from re-running `generate_diagrams.py` on files you
  didn't mean to touch (`pdflatex` embeds a fresh timestamp/ID every
  run even when content is identical).

## Companion projects (context, not code dependencies)

- [`industrial-ddr-finetuning`](https://github.com/djimrastephane/industrial-ddr-finetuning) —
  private, real production-scale pipeline over this same Utah FORGE
  archive. Chapters 6 and 9 reference it for verified real-world
  numbers/technique. This book's own code never imports from or
  depends on it — every chapter script runs standalone against
  `book/datasets/`.
- `book/app/` — an optional Streamlit companion app, **not yet
  implemented** (`book/app/README.md` says so explicitly). Don't
  assume it exists or works.
