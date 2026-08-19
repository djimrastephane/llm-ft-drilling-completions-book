# Troubleshooting Guide

This page collects the errors readers are most likely to hit while
working through the book, in one place. Two chapters have their own
troubleshooting sections already — [Part 0, Section
0.11](book/chapters/chapter_00.qmd) for first-time setup, and [Appendix
A, Section 5](book/appendix/appendix_a_environment_setup.qmd) for
rendering and fine-tuning packages — but both are `.qmd` files, which
GitHub's file viewer shows as plain unformatted source rather than a
readable page. This file repeats the same tables in plain Markdown so
they're readable directly on GitHub, plus a few issues that don't
belong to either chapter.

All commands below assume you're inside the `book/` folder with your
virtual environment active (`(.venv)` showing in your prompt) — see
[Part 0](book/chapters/chapter_00.qmd) if you haven't gotten there yet.

&nbsp;
## Setup (Python, cloning, virtual environment)

| Symptom | Likely cause |
|---|---|
| `command not found: python3` (Windows) | Python wasn't added to `PATH` during install — re-run the installer and check "Add python.exe to PATH". |
| `python3 --version` shows a version below 3.11 | You have an older Python 3 install alongside a newer one; try `python3.11 --version` explicitly, or reinstall from [python.org](https://www.python.org/downloads/). |
| Prompt never shows `(.venv)` after activating | Confirm you ran the activate command from *inside* `book/`, and that `.venv/` actually exists there. |
| `pip install -r requirements.txt` fails partway through | Re-run the same command — a partial download is the most common cause, and simply retrying often finishes the job. If it fails on `torch` specifically, try `pip install torch` alone first, then re-run the full install. |
| `setup_check.py` reports `[FAIL] <package> not importable` | `(.venv)` isn't active, or the requirements install didn't finish successfully — re-check the venv-activate and pip-install steps. |
| `ModuleNotFoundError: No module named 'torch.backends'` when checking hardware | Your `torch` install didn't complete — re-run `pip install -r requirements.txt` and check for `ERROR` lines. |

&nbsp;
## Rendering the book and running fine-tuning packages

| Symptom | Likely cause |
|---|---|
| `quarto render` fails on a code cell | Activate `.venv` before rendering — Quarto uses whichever Python is on `PATH`. Confirm with `which python` and `python -c "import transformers"`. |
| `ModuleNotFoundError: peft` or `trl` (Chapter 5 onward) | Re-run `pip install -r requirements.txt`; these are heavier dependencies that pull in `torch` and sometimes need a platform-specific `torch` install first. |
| `bitsandbytes` fails to import | It's Linux + NVIDIA GPU only (see `requirements.txt`). No chapter in this book actually loads a quantized model, so on macOS or CPU-only Linux this is expected — every chapter fine-tunes with full-precision LoRA instead. |
| Training is extremely slow | Confirm whether a GPU is actually being used (`torch.cuda.is_available()` or `torch.backends.mps.is_available()`); on CPU-only hardware this is expected, not a bug — Chapter 5's fine-tune runs in about 5 minutes on CPU, Chapter 8's "at scale" run in about 30–35 minutes. |
| The PDF download disappears after rendering, or cross-format links point at a missing file | Render with plain `quarto render` (no `--to` flag). `quarto render --to html` and `quarto render --to pdf` run as separate passes, and each one deletes the *other* format's output from `_book/` even though links between formats still point at it. This isn't a bug in the book — it's what those two flags actually do. |

&nbsp;
## Downloading the base model (Chapter 1 onward)

Chapter 1's `load_local_model.py` downloads `Qwen/Qwen2.5-1.5B-Instruct`
from Hugging Face the first time you run it — a few gigabytes — then
caches it locally, so every run after that is offline. If the first
download doesn't go smoothly:

| Symptom | Likely cause |
|---|---|
| Download hangs, times out, or fails partway | Usually a network interruption — re-run the same script; `transformers` resumes from what's already cached rather than starting over. |
| Download blocked entirely, or an SSL/certificate error | A corporate firewall or proxy blocking `huggingface.co`. Try from a different network, or set the standard `HTTPS_PROXY`/`HF_HUB_ENABLE_HF_TRANSFER` environment variables your network requires. |
| `OSError: No space left on device` | The model plus its cache needs a few gigabytes free, on top of this book's other requirements (see the root README's "Minimum Computer Requirements"). Free up disk space and re-run. |
| Everything after the first run should be instant and offline | If a later chapter still seems to be downloading, confirm you're using the same user account and machine as the first run — the cache is per-machine, typically under your home folder's `.cache/huggingface/` (or wherever `HF_HOME` points, if you've set it). |

&nbsp;
## Checkpoints and chapter order

`checkpoints/` is `.gitignore`d — a fresh clone starts with none, on
purpose, so nothing large or reader-specific ends up committed. Chapters
9 through 13 read a checkpoint that an earlier chapter's script
produced (Chapter 5's `first_lora_finetune.py`, Chapter 8's
`finetune_at_scale.py`, or Chapter 13's own
`continuous_finetune.py`). If you jump straight to a later chapter's
script without running the earlier one first, you'll get a clear error
naming the exact script to run — for example:

```
FileNotFoundError: No runs in checkpoints/chapter_08 -- run code/chapter_08/finetune_at_scale.py first
```

That's the intended behavior, not a bug — run the named script, then
re-run what you were doing. Working through the chapters in order (as
Start Here in the root README recommends) avoids this entirely, since
each chapter's checkpoint is already in place by the time the next one
needs it.

&nbsp;
## Running the test suite

```bash
pytest -v -m "not slow and not gpu"
```

skips anything that downloads or fine-tunes a real model, or needs a
CUDA/MPS device — the fast path for confirming your setup works.
Running the full suite (`pytest -v`, no `-m` filter) needs a real
checkpoint on disk for some tests; those skip cleanly with a message
rather than failing if the checkpoint isn't there yet.

If a test process crashes outright (not a normal pytest failure —
the process dies with no traceback) while both `faiss` and
`sentence-transformers`/`torch` are being imported together, this is a
known crash mode on macOS from a combination of these two libraries'
own threading behavior, seen in the author's previous book under the
same combination. It hasn't shown up in this repository's own CI runs,
but if you hit it locally, try setting `OMP_NUM_THREADS=1` before
running pytest:

```bash
OMP_NUM_THREADS=1 pytest -v
```

&nbsp;
## Silent crashes in a conda/Anaconda environment

If Python exits immediately with no error message at all — not even a
traceback — the moment you `import transformers` or `import peft` in
your *own* script or notebook cell (as opposed to running one of this
book's own scripts directly), the most common cause on a shared or
Anaconda-style environment is `transformers` probing for a TensorFlow
or Flax install and crashing on a broken one, even though this book
never uses either. This book's own scripts already guard against it —
`load_local_model.py` sets two environment variables before importing
`transformers`:

```python
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
```

If you're writing your own experimental cells that import
`transformers`, `peft`, or `sentence_transformers` directly, import
`load_local_model` first (or set those two environment variables
yourself) so the same guard applies.

&nbsp;
## Companion app (Streamlit)

```bash
pip install -r book/requirements.txt
cd book
streamlit run app/streamlit_app.py
```

The app is designed to run with zero checkpoints on disk — every
checkpoint-dependent page checks what actually exists first and shows
setup instructions instead of crashing if nothing does. If a page looks
empty or says no checkpoint is available, that's the app working
correctly, not an error; train one with any of the commands the page
itself suggests (or see `book/app/README.md`), then reload — no restart
needed.

&nbsp;
## Still stuck

Open a [GitHub
Issue](https://github.com/djimrastephane/llm-ft-drilling-completions-book/issues)
with what you ran, the full error text, and your OS/Python version —
see the root [README](README.md#questions-feedback-and-contributing-to-this-repository)
for the same link.
