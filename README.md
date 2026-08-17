# Fine-Tuning Local LLM for Drilling & Completions

**First time here? Jump straight to [Start Here](#start-here).**

This repository contains the chapters, code, and training data for
**Fine-Tuning Local LLM for Drilling & Completions** — a hands-on,
build-as-you-go book that teaches drilling and completions engineers how
to fine-tune and run a private, local large language model on their own
operational data, assuming zero prior programming or machine-learning
experience.

In *Fine-Tuning Local LLM for Drilling & Completions*, you build a
working, fine-tuned local model from scratch, one chapter at a time:
loading a general-purpose **open-weight** model (one you download and
keep on your own machine, not just borrowed access to someone else's),
turning real reports into training data, measuring exactly what the base
model gets wrong, running your first **LoRA** fine-tune (a lightweight
way to teach it your operation's own wording without retraining the
whole model from scratch), then hardening that prototype at scale —
saving your progress as you go so a long training run doesn't have to
start over from zero, catching bad training data before it ever reaches
fine-tuning, and pairing it with a real search step so the final answer
comes back backed by real evidence, and traceable to the actual report
it came from — no cloud training service, no black-box trainer, no data
leaving your machine.

Every example uses real, publicly available Daily Drilling Reports from
**Utah FORGE** — a Department of Energy-funded geothermal research well —
not synthetic stand-ins. A real stuck-pipe event, a real report the
fine-tuned model never trained on, and a real test of whether it learned
anything transferable, rather than just memorizing what it was shown, all
appear exactly as they came out of this book's own code.

### Why Not Just Prompt a Cloud AI Assistant?

A fair question before installing anything: ChatGPT, Claude, and similar
hosted assistants already run large models that have been specifically
trained to hold a fluent conversation and follow instructions, not just
predict the next word. Why bother loading, and then fine-tuning, your
own?

Because a hosted assistant never actually learns your operation. Ask
`Qwen2.5-1.5B-Instruct` — the small, general-purpose model this book
starts from — about a real report from its own archive, and it answers
fluently and confidently about nothing in particular. A real report from
this book's sample archive says: *"During the slide lost tool face and
became stuck."* Asked what the crew should watch for on the next curve
section, the untouched base model replies:

```
Based on the information provided in the driller's report that "during
the slide lost tool face and became stuck," it suggests that there was
an issue with the drill bit or other tools getting caught during the
descent of the drill string...

1. Tool Catching: The drill bit or any attached tools may have become
   entangled due to debris, mud cake buildup, or other obstructions.
2. Drill String Configuration: There might be a misalignment or
   improper configuration of the drill string...
```

This fluent, confident, wrong answer is what the AI field calls a
**hallucination** — a model output that sounds right but isn't actually
supported by real evidence. Chapter 10 covers detecting and mitigating
it directly. Here, the underlying problem is simpler: "Trip out of hole"
— basic, everyday oilfield terminology — is treated like an English
idiom the model has to guess at, because its general training data
apparently didn't include enough real oilfield operational text to have
learned it. A hosted, pay-per-use connection to someone else's model
can't fix that either: you're given access to run the model, not the
**weights** themselves — the actual, millions-of-numbers-deep "knowledge"
stored inside it — so there's nothing to fine-tune (Chapter 1 covers this
in more depth). There's also a data question a hosted assistant forces on
you that a local model doesn't — every prompt, and every report excerpt
you paste into it, leaves your machine, and real drilling and completions
reports are usually confidential.

This book fixes both problems: it fine-tunes a small local model
directly on your own reports, chapter by chapter (Chapter 5 onward). And
because fine-tuning alone is a weak tool for citing an exact source,
Chapter 8 measured exactly how weak: `0/50` on a strict **exact-match**
test, where the model's answer had to match the report's exact wording,
word for word, to count. Chapter 9 pairs it with a real search step over
the report archive (**retrieval**, like a well-indexed filing cabinet)
so the final answer comes back **grounded** (actually built from a real,
cited report, not just a plausible guess) and traceable to the report it
came from.

### What You're Building

The box below is a **real, verified transcript from Chapter 9's own
code** — not an illustration, and not something invented for this
README. Report `#37` is the one report this book's fine-tuned model
never saw during training (Chapter 2 held it out on purpose). Asked what
happened on that report's `20:30–21:30` window:

```
Without retrieval (fine-tuned model alone):
  Production Casing Run Csg & Cement Rig up casing

With retrieval (Chapter 9's hybrid system):
  Trip out of hole with BHA #18. Stop at 5,800' and circulate to cool
  hole and tools.
  Source: Report #37, 20:30-21:30
```

The first answer is fluent, plausible, and wrong — a pattern borrowed
from a different report entirely. The second is close to word-for-word
what report `#37` actually says, found by a real keyword-retrieval index
built over the archive and handed to the fine-tuned model as grounding
text before it answered. Nothing about report `#37` was in the training
data; only the retrieval step was.

That's the destination — and, just as importantly, this book doesn't
stop at the win. Chapter 5 shows the same honest picture when the fix
*isn't* in place yet: on the 16 examples it actually trained on — an
open-book test, where it already saw the answer key — fine-tuning alone
raised the model's score from `0/16` to `13/16`. But on report `#37`,
which it never saw during training, that same fine-tuned model still
scored `0/2` — the real test of whether it learned anything
transferable, not just memorized what it was shown. Real, working code,
and the real limitations that motivate the next chapter — not a happy
path with the failures edited out.

- Link to the [official source code repository](https://github.com/djimrastephane/llm-ft-drilling-completions-book)
- License: code is [MIT](LICENSE); the book's text is [CC BY 4.0](LICENSE-CONTENT.md)
- Progress: Part 0 and all 13 chapters are written, tested, and passing
  CI (short for continuous integration — the project's automated tests,
  which re-run on every change to catch a broken example before you ever
  see it). See [CHANGELOG.md](CHANGELOG.md) for the full history of
  what's landed and why, and [RELEASE.md](RELEASE.md) for per-release
  highlights — there's no tagged release yet.
- The book isn't published to GitHub Pages yet — the "Publish book to
  GitHub Pages" workflow (badge below) is still manual-only
  (`workflow_dispatch`), even though every chapter is now drafted.
  Read chapters directly as `.qmd` source under
  [`book/chapters/`](book/chapters), or render the whole book locally
  with `quarto render` (see Quickstart in
  [`book/README.md`](book/README.md)).

To get a copy of this repository, click the [Download
ZIP](https://github.com/djimrastephane/llm-ft-drilling-completions-book/archive/refs/heads/main.zip)
button, or run the following in a terminal:

```bash
git clone https://github.com/djimrastephane/llm-ft-drilling-completions-book.git
```

Never used a terminal or Git before? That's exactly what **Start Here**
and **Part 0** below are for — nothing past this point assumes you
already know how.

### Project Map

A few things share the name "FORGE" or connect to a companion project in
this book — worth telling apart before you start:

| Piece | What it is | Where it's used |
|---|---|---|
| Ten-report sample archive | `book/datasets/sample_training_set/`, committed in this repo | The main path through Chapters 1–5 |
| 76-report full Utah FORGE archive | `book/datasets/full_training_set/`, committed in this repo | Scale exercises from Chapter 6 onward |
| Companion app | `book/app/`, in this repo | Planned base-vs-fine-tuned UI — not implemented yet, see [`book/app/README.md`](book/app/README.md) |
| Companion project (`industrial-ddr-finetuning`) | A separate, private repository | Referenced for real numbers/technique in Part II; not required to follow the book |
| Your own report archive | Not included — it's yours | The adaptation path after you finish the book |

Everything in the first two rows runs from this one repository. The
fourth is optional and lives elsewhere — see **Companion Pipeline**
below.

---

# Start Here

This README has one job: get you to successfully run Chapter 1.
Everything else in this file is reference material for later.

If this is your first Python project, do these steps in order:

1. Read [Part 0: Preparing Your Local LLM Workshop](book/chapters/chapter_00.qmd) — installs Python, clones this repository, and gets your environment ready. No prior experience assumed.
2. Run `setup_check.py` — one command that confirms everything is working before you touch a real model.
3. Work through [Chapter 1: Loading and Running Your First Local LLM](book/chapters/chapter_01.qmd) — load a real open-weight model on your own machine, and watch it get real oilfield shorthand wrong (see "Why Not Just Prompt a Cloud AI Assistant?" above).
4. Continue sequentially through [Chapter 13: Continuous Fine-Tuning](book/chapters/chapter_13.qmd), the book's last chapter — each chapter builds on the last one's saved output: a training example, a baseline result, a checkpoint, or a retrieval index.

| Step | Typical time |
|---|---|
| Part 0 | 30–45 min |
| Chapter 1 | 20–30 min |
| Chapter 2 | 30–40 min |
| Chapter 3 | 20–30 min |

See [How Long Does Each Chapter Take](#how-long-does-each-chapter-take)
below for the full breakdown through Chapter 9. You don't need to
understand everything before you start — you need to run the first
command. Everything else follows from there.

---

# Your Learning Journey

Each arrow below is one or more chapters of real, working code — not a
diagram of what's theoretically possible.

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

By the last arrow, you're not reading about fine-tuning — you built a
working system yourself, and you understand every piece of it.

## What You Will Build and Learn

By the end of the book you will have built eleven real, working
artifacts — not eleven topics you read about:

- ✓ **Local model loading and inference script** — run a general-purpose local LLM and evaluate its out-of-the-box answers to drilling and completions questions
- ✓ **Domain training-example builder** — turn raw drilling and completions reports into a training dataset
- ✓ **Baseline prompting harness** — measure exactly, and reproducibly, what the base model gets wrong before you touch it
- ✓ **First LoRA fine-tune** — a real, working parameter-efficient fine-tune, with an honest held-out generalization result
- ✓ **Data quality gate** — catch bad training data before it ever reaches fine-tuning
- ✓ **Checkpointed fine-tune at scale** — real experiment tracking and resumable checkpoints, no black-box trainer
- ✓ **Hybrid fine-tuning + retrieval system** — grounded, citable answers, demonstrated above
- ✓ **Faithfulness checker** — catches a real, fluent answer that cited the right report but was actually grounded in the wrong one
- ✓ **Evaluation harness** — a real 8-example held-out set scored three ways, showing that a single metric can make real training progress look like nothing happened
- ✓ **Model-version drift detector** — caught two of the book's own metrics disagreeing about whether a newer checkpoint is actually an improvement
- ✓ **Continuous fine-tuning loop** — simulated new reports arriving, retrained on them, and let the drift detector catch a real regression before it would have shipped

## Who This Book Is For

This book is written for drilling, completions, intervention, and
production engineers — and for the digital oilfield professionals and
energy data scientists who support them. You'll recognize the fit
quickly: every chapter opens with one of four recurring engineers (Oumy,
Mike, Sarah, Sean — you'll meet all four) asking the kind of question
you'd actually ask before trusting a model with your own reports, then
answers it with real code and a real result, not a hypothetical.

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

## Reader Contract

- **Part I** (Chapters 1–5) builds a working, fine-tuned prototype
  against the ten-report sample archive, including an honest measurement
  of where a small fine-tune does and doesn't generalize.
- **Part II** (Chapters 6–13) hardens that prototype against industrial
  failure modes: data quality gating, formatting and chunking at scale,
  checkpointed fine-tuning with experiment tracking, hybrid retrieval,
  traceability, evaluation, drift detection, and continuous retraining.
- Some of Part II's chapters are informed by the author's separate,
  private companion project `industrial-ddr-finetuning`, not by this
  repository's own code — each chapter says so where it applies.
- This book teaches an inspectable, from-scratch architecture and
  working components — not a full enterprise deployment with
  authentication, monitoring, permissions, governance, or support
  operations.

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

## How Long Does Each Chapter Take

| Chapter | Typical time |
|---|---|
| Part 0 | 30–45 min |
| Chapter 1 | 20–30 min |
| Chapter 2 | 30–40 min |
| Chapter 3 | 20–30 min |
| Chapter 4 | 30–40 min |
| Chapter 5 | 45–60 min |
| Chapter 6 | 30–40 min |
| Chapter 7 | 30–40 min |
| Chapter 8 | 45–60 min |
| Chapter 9 | 45–60 min |
| Chapter 10 | 30–40 min |
| Chapter 11 | 30–40 min |
| Chapter 12 | 30–40 min |
| Chapter 13 | 45–60 min |

There's no need to do this in one sitting — most readers spread it
across several days, a chapter or two at a time.

## Minimum Computer Requirements

**Minimum:**

- 16 GB RAM
- 20 GB free disk space
- a modern CPU

**Recommended:**

- A graphics card (GPU) with at least 8 GB of its own memory (VRAM) —
  either NVIDIA or an Apple Silicon Mac (M1/M2/M3/M4) — makes fine-tuning
  chapters faster, though none of it is required. Every chapter through
  Chapter 9 has actually been run and verified CPU-only (no graphics card
  at all, just the computer's regular processor) on ordinary laptop
  hardware — Chapter 5's fine-tune takes about 5 minutes that way,
  Chapter 8's "at scale" run about 30–35 minutes.

**No cloud account required. No paid, metered connection to someone
else's hosted model required.** Everything in this book is designed to
run locally, using small open-weight models and the same lightweight,
parameter-efficient fine-tuning (LoRA) introduced above, so a single
consumer GPU — or patience, on CPU — is enough.

## Choose Your Workshop

| Environment | Recommended for |
|---|---|
| [Jupyter Notebook](book/appendix/appendix_a1_jupyter.qmd) | Learning and experimentation |
| [VS Code](book/appendix/appendix_a2_vscode.qmd) | General coding |
| [PyCharm Community](book/appendix/appendix_a3_pycharm.qmd) | Larger projects |
| [Positron](book/appendix/appendix_a4_positron.qmd) | Data science workflows |
| [Terminal only](book/appendix/appendix_a5_terminal.qmd) | Minimal setup |

**There is no wrong choice. All examples run identically in every one of
these.** [Part 0](book/chapters/chapter_00.qmd) covers general setup;
each link above has a short, dedicated walkthrough for that specific
tool.

## First Success Checkpoint

You are ready for Chapter 1 when:

- ✅ Python runs
- ✅ `setup_check.py` runs successfully
- ✅ the sample training set exists in `book/datasets/sample_training_set/`
- ✅ your virtual environment is active

All four checks are covered in Part 0 — if any of them aren't true yet,
that's exactly what it's for.

## Common Reader Journeys

**"I am a drilling or completions engineer with no coding experience."**
→ Start with Part 0 and proceed sequentially, one chapter at a time.

**"I already know Python."**
→ Skip Part 0 and start directly at Chapter 1.

**"I already fine-tune LLMs professionally."**
→ Skim Part I for context — it's short, and Chapter 5's held-out result
is worth knowing before Part II — then start at Chapter 6, where data
quality gating, checkpointing at scale, and hybrid retrieval are
covered.

## What Makes This Book Different

- **Real reports, not synthetic stand-ins.** Every example traces back to
  an actual Utah FORGE Daily Drilling Report — public, DOE-funded data,
  so there's nothing confidential and nothing invented to work through.
- **Zero programming or machine-learning background assumed.** The first
  script you ever run is also the first Python you'll ever touch.
- **A training loop you can read line by line**, not a black-box
  trainer — every step is right there in front of you, not hidden inside
  someone else's library call.
- **Honest negative results, left in.** Chapter 8's `0/50` exact-match
  and Chapter 5's `0/2` held-out score are both still here, not edited
  out to make the book look cleaner than the real run was.
- **Industrial constraints taught for real** — data quality gating,
  checkpointing at scale, hybrid retrieval, traceability — not a toy
  happy path that stops working the moment you point it at your own
  archive.

---

# Table of Contents

Part 0 and all 13 chapters are written, tested, and passing CI. This
repository's `.qmd` chapter files are Quarto Markdown — GitHub's file
viewer shows them as plain unformatted source, since there's no
published GitHub Pages site yet (see "Progress" above). For the full
repository layout (folder tree, part/chapter file map) see
[`book/README.md`](book/README.md).

[![Publish book to GitHub Pages](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/publish.yml/badge.svg)](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/publish.yml)
[![Code tests Linux](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-linux.yml/badge.svg)](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-linux.yml)
[![Code tests Windows](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-windows.yml/badge.svg)](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-windows.yml)
[![Code tests macOS](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-macos.yml/badge.svg)](https://github.com/djimrastephane/llm-ft-drilling-completions-book/actions/workflows/tests-macos.yml)

The "Publish book to GitHub Pages" workflow is still manual-only
(`workflow_dispatch`) — its badge reflects the last manual run, not
every push to `main`. Every chapter is now drafted, so nothing is
blocking a real publish beyond deciding to trigger one.

| Status | Chapter | Main Code (Quick Access) | All Code + Supplementary |
|---|---|---|---|
| ✅ | [Part 0: Preparing Your Local LLM Workshop](book/chapters/chapter_00.qmd) | [setup_check.py](book/code/setup_check.py) | — |
| ✅ | [Ch 1: Loading and Running Your First Local LLM](book/chapters/chapter_01.qmd) | - [load_local_model.py](book/code/chapter_01/load_local_model.py)<br/>- [chapter_01_explore.ipynb](book/notebooks/chapter_01_explore.ipynb) | [./book/code/chapter_01](book/code/chapter_01) |
| ✅ | [Ch 2: Turning Drilling & Completions Reports into Training Examples](book/chapters/chapter_02.qmd) | - [build_training_examples.py](book/code/chapter_02/build_training_examples.py)<br/>- [chapter_02_explore.ipynb](book/notebooks/chapter_02_explore.ipynb) | [./book/code/chapter_02](book/code/chapter_02) |
| ✅ | [Ch 3: Baseline Prompting — What the Model Gets Wrong Before Fine-Tuning](book/chapters/chapter_03.qmd) | - [baseline_prompting.py](book/code/chapter_03/baseline_prompting.py)<br/>- [chapter_03_explore.ipynb](book/notebooks/chapter_03_explore.ipynb) | [./book/code/chapter_03](book/code/chapter_03) |
| ✅ | [Ch 4: Tokenization and Embeddings for Domain Fine-Tuning](book/chapters/chapter_04.qmd) | - [tokenize_and_embed.py](book/code/chapter_04/tokenize_and_embed.py)<br/>- [chapter_04_explore.ipynb](book/notebooks/chapter_04_explore.ipynb) | [./book/code/chapter_04](book/code/chapter_04) |
| ✅ | [Ch 5: Your First LoRA Fine-Tune](book/chapters/chapter_05.qmd) | - [first_lora_finetune.py](book/code/chapter_05/first_lora_finetune.py)<br/>- [chapter_05_explore.ipynb](book/notebooks/chapter_05_explore.ipynb) | [./book/code/chapter_05](book/code/chapter_05) |
| ✅ | [Ch 6: A Data Quality Gate for Training Data](book/chapters/chapter_06.qmd) | - [data_quality_gate.py](book/code/chapter_06/data_quality_gate.py)<br/>- [chapter_06_explore.ipynb](book/notebooks/chapter_06_explore.ipynb) | [./book/code/chapter_06](book/code/chapter_06) |
| ✅ | [Ch 7: Formatting and Chunking a Training Set at Scale](book/chapters/chapter_07.qmd) | - [format_training_chunks.py](book/code/chapter_07/format_training_chunks.py)<br/>- [chapter_07_explore.ipynb](book/notebooks/chapter_07_explore.ipynb) | [./book/code/chapter_07](book/code/chapter_07) |
| ✅ | [Ch 8: Fine-Tuning at Scale — Checkpoints and Experiment Tracking](book/chapters/chapter_08.qmd) | - [finetune_at_scale.py](book/code/chapter_08/finetune_at_scale.py)<br/>- [chapter_08_explore.ipynb](book/notebooks/chapter_08_explore.ipynb) | [./book/code/chapter_08](book/code/chapter_08) |
| ✅ | [Ch 9: Hybrid System — Combining Fine-Tuning with Retrieval](book/chapters/chapter_09.qmd) | - [hybrid_rag_finetune.py](book/code/chapter_09/hybrid_rag_finetune.py)<br/>- [chapter_09_explore.ipynb](book/notebooks/chapter_09_explore.ipynb) | [./book/code/chapter_09](book/code/chapter_09) |
| ✅ | [Ch 10: Traceable Outputs and Hallucination Mitigation](book/chapters/chapter_10.qmd) | - [traceable_outputs.py](book/code/chapter_10/traceable_outputs.py)<br/>- [chapter_10_explore.ipynb](book/notebooks/chapter_10_explore.ipynb) | [./book/code/chapter_10](book/code/chapter_10) |
| ✅ | [Ch 11: Evaluating a Fine-Tuned Domain Model](book/chapters/chapter_11.qmd) | - [eval_finetuned_model.py](book/code/chapter_11/eval_finetuned_model.py)<br/>- [chapter_11_explore.ipynb](book/notebooks/chapter_11_explore.ipynb) | [./book/code/chapter_11](book/code/chapter_11) |
| ✅ | [Ch 12: Detecting Drift Across Model Versions](book/chapters/chapter_12.qmd) | - [detect_model_drift.py](book/code/chapter_12/detect_model_drift.py)<br/>- [chapter_12_explore.ipynb](book/notebooks/chapter_12_explore.ipynb) | [./book/code/chapter_12](book/code/chapter_12) |
| ✅ | [Ch 13: Continuous Fine-Tuning — Keeping the Model Current](book/chapters/chapter_13.qmd) | - [continuous_finetune.py](book/code/chapter_13/continuous_finetune.py)<br/>- [chapter_13_explore.ipynb](book/notebooks/chapter_13_explore.ipynb) | [./book/code/chapter_13](book/code/chapter_13) |
| ✅ | Appendix A: Environment Setup | — | [./book/appendix](book/appendix) |
| ✅ | Appendices A1–A5: Jupyter / VS Code / PyCharm / Positron / Terminal-only | — | [./book/appendix](book/appendix) |
| ✅ | [Appendix B: Drilling, Completions & Fine-Tuning Glossary](book/appendix/appendix_b_glossary.qmd) | — | [appendix_b_glossary.qmd](book/appendix/appendix_b_glossary.qmd) |

Every ✅ chapter ships with working, tested code and a companion
notebook — see [Automated Tests](#automated-tests) below. Every quoted
number in a ✅ chapter comes from an actual run of that chapter's own
code against this repository's real Utah FORGE archive, not an
estimate. A few examples:

- Chapter 6's data quality gate found `75/76` reports pass extraction,
  with `6` duplicate field-value groups.
- Chapter 8's "at scale" fine-tune measured **training loss** (a single
  number scoring how wrong the model's predictions were, which should
  generally fall as training continues, the way a golf score should
  generally fall as you improve) falling `2.755 → 2.164 → 1.821` across
  3 real epochs.
- Chapter 9 measured two different ways of searching the report archive
  for the right source: **BM25 keyword retrieval** (ranking reports by
  how many of the question's own words they contain, like a smarter
  Ctrl+F) found the correct source report `4/4` times on real test
  queries, against `3/4` for **dense sentence embeddings** (matching by
  meaning instead of exact wording).
- Chapter 10's faithfulness check caught a real answer that cited the
  correct report alongside the wrong one, and was actually grounded in
  the wrong one.
- Chapter 11 measured **perplexity** (a score for how "surprised" the
  model is by real text it never trained on — lower means it found the
  wording more expected) on real held-out text falling from `159.91`
  (base model) to `25.03` (fine-tuned) — even though exact-match on the
  same 8 questions stayed `0/8` at every epoch.
- Chapter 12 found two of those same metrics disagreeing between two
  real checkpoints — `avg_overlap` (how much of the model's wording
  overlaps with the known-correct answer) got worse while `perplexity`
  improved between epoch 1 and epoch 2 of the same training run.
- Chapter 13 simulated new reports arriving, retrained on them, and
  found the same disagreement a third time on a real
  continuous-fine-tuning run — a real update that trained cleanly and
  still regressed on `avg_overlap`.

## Companion Pipeline

[`industrial-ddr-finetuning`](https://github.com/djimrastephane/industrial-ddr-finetuning)
is a separate, private repository: a real, working pipeline that pulls
structured field values out of each report (with a record of exactly
which sentence each value came from, an automated check on the result,
and a human review step for anything uncertain), built against this
book's same public Utah FORGE archive. Chapters 6 and 9 in particular
reference it for verified real-world numbers and technique. This book's
own code never depends on it — every chapter's script in
`book/code/chapter_NN/` runs standalone against the committed sample or
full archive under `book/datasets/`. The book's own implementations are
written from scratch for teaching purposes, not copied from it.

### What this becomes

The companion pipeline applies the same fine-tuning, data-quality, and
retrieval logic the book builds to the operator's full report archive at
production scale — this book's chapters teach the underlying techniques
standalone, at a scope one reader can run and inspect on their own
machine.

## Companion App (planned)

An optional small web app (built with a Python tool called Streamlit) is
planned under [`book/app/`](book/app), reusing the book's own chapter
code (base model loading, fine-tuning, and evaluation), to show a
question answered side-by-side by the base model and the fine-tuned
model. Not implemented yet — see
[`book/app/README.md`](book/app/README.md) for current status and the
exact files it plans to reuse.

## Exercises

Every drafted chapter includes a **Practical exercise** and a
**Challenge exercise**, with reference solutions alongside each
chapter's code under `book/code/chapter_NN/challenge/`.

## Automated Tests

Every chapter's real code is tested in [`book/tests/`](book/tests) —
74 tests across all 13 chapters, run on Linux, Windows, and macOS on
every push that touches `book/**` (badges above). Run them yourself
from the `book/` directory:

```bash
pip install -r requirements.txt
pytest -v
```

Chapters 5, 8, 9, 10, 11, 12, and 13 include tests marked `slow` (they
load and generate from the real base model, and Chapter 8's and 13's
fine-tuning tests take several minutes) or that need a checkpoint from
a previous chapter's script to already exist on disk (skipped
automatically if it doesn't — see each test file's own docstring).
Skip the slow ones locally with:

```bash
pytest -v -m "not slow"
```

## Bonus Material

Every written chapter ends with a **Repository files** table listing the
exact files that back everything the chapter claims. A few worth knowing
about on their own:

- [`book/code/chapter_01/build_sample_training_set.py`](book/code/chapter_01/build_sample_training_set.py)
  reproduces this book's curated 10-report sample archive from the full
  76-report Utah FORGE archive.
- Each chapter's **Field notes** callout is a real, independently
  verified result checked against the actual archive before being
  written down — not an illustrative estimate.
- [`book/appendix/appendix_b_glossary.qmd`](book/appendix/appendix_b_glossary.qmd)
  covers drilling, completions, and fine-tuning terminology used
  throughout the book.

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
