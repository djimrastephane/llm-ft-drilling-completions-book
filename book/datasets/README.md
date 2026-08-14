# Datasets

This book reuses the same source archive as the author's previous book,
[`ddr-rag-book`](https://github.com/djimrastephane/ddr-rag-book): real,
publicly available Daily Drilling Reports and completion reports from
**Utah FORGE** (well FORGE 16A(78)-32), a Department of Energy-funded
enhanced geothermal system research well. No anonymisation is applied or
needed -- this is public data.

- `full_training_set/` — the full 76-report Utah FORGE archive. Source
  for Part II's "at scale" chapters (6–13).
- `sample_training_set/` — the same 10-report curated subset used in
  `ddr-rag-book`'s Part I (including the real stuck-pipe report,
  `Drilling_038`, and the packers-fail-to-fishing sequence,
  `Drilling_049`/`050`), for Chapters 1–5. Reproducible from
  `full_training_set/` via `code/chapter_01/build_sample_training_set.py`.

Part II's data-quality and traceability chapters (6, 9, 10) are informed
by [`industrial-ddr-finetuning`](https://github.com/djimrastephane/industrial-ddr-finetuning),
a private companion project that has already run a schema-v2 extraction
pipeline (field-level status, verbatim evidence spans, validation, review
workflow) over this same 76-report archive. This book teaches that
pipeline's techniques from scratch in Part II rather than reusing its code.

Both tiers are raw report PDFs, not yet training examples --
turning them into instruction/response training pairs is Chapter 2's
job. Derived artifacts (extracted text, formatted training files, model
checkpoints) are gitignored; see `book/.gitignore`.

## Policy for what gets committed here

This dataset is public DOE-funded research data, safe to commit, share,
and publish, exactly as in the previous book. That is *not* true the
moment this book's code is pointed at anyone's own organisation's
confidential reports instead:

- Never commit real, confidential reports, extracted text, training
  examples derived from them, or model checkpoints trained on them to a
  public repository.
- If you're experimenting on a shared machine, confirm your
  organisation's data classification policy covers running third-party
  Python packages (model downloads, tokenizers) against confidential
  well data.
- Add your own archive's path to `.gitignore` before dropping any
  confidential reports into a local `datasets/` folder for testing.
- Before fine-tuning on your own organisation's data, confirm you have
  the right to use it for this purpose -- that determination is outside
  the scope of this book.

## License note

The Utah FORGE report PDFs under `full_training_set/` and
`sample_training_set/` are public DOE-funded research data and are not
covered by either of this repository's licenses (MIT for code, CC BY 4.0
for book text) -- consult the original Utah FORGE source for its terms.
