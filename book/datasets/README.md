# Datasets

This folder is where the book's training data will live, mirroring the
two-tier layout of the author's previous book (`sample_ddrs/` /
`forge_archive/`):

- `sample_training_set/` — a small, curated set of instruction/response
  training examples for Part I (Chapters 1–5).
- `full_training_set/` — a larger training set for Part II's
  "at scale" chapters (6–13).

**Status: empty.** No data has been sourced or committed yet.

## Policy for what gets committed here

Daily drilling and completions reports are typically confidential.
Unlike the author's previous book, which could rely on a fully public
Department of Energy-funded archive (Utah FORGE), this book's dataset
has not been finalized. Before anything is committed to this folder in a
public repository, it must be either:

1. **Genuinely public data** the author has the right to redistribute
   (e.g. another public research well archive, or a synthetically
   generated training set that doesn't encode any real operator's
   confidential reports), or
2. **Fully synthetic** examples written or generated specifically for
   this book, containing no real, identifiable operational data.

Real, confidential reports from any operator must never be committed
here. If you (the author, or a reader following along with your own
data) need to experiment with confidential reports, keep them outside
this repository entirely and add their path to `.gitignore` locally —
see `appendix/appendix_a_environment_setup.qmd`, Section 4.

## Once data lands

Each chapter's **Repository files** table will list the exact files it
depends on here. If either tier needs to be rebuilt or regenerated from
source, the script that does it will live at
`code/chapter_01/build_sample_training_set.py` and
`code/chapter_08/build_full_training_set.py`, mirroring the previous
book's `build_sample_archive.py` / `build_full_archive.py` convention.
