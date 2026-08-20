# Mathematica Companion: Fine-Tuning Before/After Explorer

An optional companion notebook, separate from `book/app/` (the
Streamlit companion app) — for readers who'd rather explore this
book's fine-tuning result interactively in Wolfram Language than in a
browser. It shows the exact same real evidence Chapter 5 and the root
README quote: the base model against Chapter 5's LoRA fine-tuned
checkpoint, on the real questions from the real 16-report training set
and the 2 held-out questions from a report the model never trained on.

**This is real, not simulated.** Nothing in the notebook is invented
or hand-picked to look good. `export_before_after.py` reuses this
book's own functions — `code/chapter_01/load_local_model.py`,
`code/chapter_02/build_training_examples.py`,
`code/chapter_03/baseline_prompting.py` — and the real trained adapter
at `book/checkpoints/chapter_05_lora`, and asks it the same questions
Chapter 5's own script asks. Re-running the export script regenerates
the data fresh from a real local run, not a cached or fixed example
set.

## What's in this folder

| File | What it is |
|---|---|
| `export_before_after.py` | Python script that generates the real before/after data (needs `book/.venv` and a real Chapter 5 checkpoint) |
| `data/before_after_examples.json` | The exported data — 16 real training examples + 2 real held-out examples, each with the question, the report's real reference answer, the actual report excerpt that answer comes from (so you can visually confirm it's really there), the base model's real answer, and the fine-tuned model's real answer |
| `FineTuning_Before_After_Explorer.nb` | The notebook itself |

## Prerequisites

You need a Wolfram Language notebook front end to open and interact
with `.nb` files:

- **Wolfram Mathematica** (paid), or
- **[Wolfram Engine](https://www.wolfram.com/engine/)** (free for
  developers) paired with a notebook interface — either the Wolfram
  Notebook front end, or a Jupyter installation with the
  [Wolfram Language kernel for Jupyter](https://github.com/WolframResearch/WolframLanguageForJupyter)
  installed.

Unlike the rest of this book, this one piece isn't free-and-open by
default — Mathematica itself is commercial software. The free Wolfram
Engine path above exists specifically so this notebook doesn't require
a paid license to try.

## Running it

1. Open `FineTuning_Before_After_Explorer.nb` in your Wolfram notebook
   front end.
2. Evaluate the notebook top to bottom (`Shift+Enter` on each input
   cell, or **Evaluate Notebook** from the Evaluation menu).
3. Use the dropdown in Section 3 to step through all 18 real questions
   — 16 the model trained on, 2 it never saw — and compare the base
   model's answer, the fine-tuned model's answer, and the report's
   actual reference text side by side.

The notebook loads `data/before_after_examples.json` using
`NotebookDirectory[]`, so it works regardless of where you've cloned
this repository — you don't need to edit any paths.

## Regenerating the data

The committed `data/before_after_examples.json` was generated from a
real run against this repository's own Chapter 5 checkpoint. To
regenerate it yourself (for example after retraining Chapter 5, or
adapting this to your own reports):

```bash
cd book
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python code/chapter_05/first_lora_finetune.py   # only if checkpoints/chapter_05_lora doesn't exist yet, ~5 min on CPU
cd ..
python mathematica/export_before_after.py
```

## Scope

This notebook covers Chapter 5's fine-tuning effect specifically — the
book's first, simplest before/after comparison. It doesn't cover
retrieval (Chapter 9), traceability (Chapter 10), the full evaluation
harness (Chapter 11), or drift detection (Chapter 12); those stay in
the book and the Streamlit companion app.

## A note on verification

This notebook's Wolfram Language code was written from language
knowledge rather than executed against a live Wolfram kernel — no
kernel was available in the environment it was built in. The Python
export script *was* actually run, and `data/before_after_examples.json`
reflects a real local run against this repository's real checkpoint
(reproducing the book's own `13/16` trained / `0/2` held-out result
exactly). The notebook itself has since been opened and evaluated
successfully in a real Wolfram front end by the author. If you still
hit an issue opening or evaluating it, please open one in the
companion repository.
