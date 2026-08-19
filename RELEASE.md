# Release notes

Per-release highlights for *Fine-Tuning Local LLM for Drilling &
Completions*. See [CHANGELOG.md](CHANGELOG.md) for the full history.

## Unreleased

Nothing pending yet.

## [1.0.0] - 2026-08-19

Part 0 and all 13 chapters are written, tested, and passing CI on
Linux, macOS, and Windows -- the book's full content arc, start to
finish, is complete and published live to GitHub Pages.

### Highlights

Readers start with a general-purpose local model that gets real
oilfield shorthand wrong (Chapter 1), and finish with a fine-tuned
model wired into a continuous retraining loop that catches its own
regressions (Chapter 13) -- every number along the way independently
verified against this book's own code and the real, public Utah FORGE
archive, never invented.

- **Part 0 & Part I (Chapters 1-5)** build the first working
  prototype: load a local model, turn real reports into training
  data, measure what the base model gets wrong, and run a first LoRA
  fine-tune -- with an honest result (`0/16 -> 13/16` training recall,
  held-out generalization staying at `0/2`) rather than a happy path.
- **Part II (Chapters 6-13)** hardens that prototype: a data quality
  gate (`75/76` reports pass, `6` duplicate groups caught), formatting
  and chunking at scale, checkpointed fine-tuning with real experiment
  tracking, a hybrid fine-tuning + retrieval system (`4/4` BM25
  retrieval accuracy), a faithfulness checker that catches a real
  answer grounded in the wrong report, a proper evaluation harness
  (perplexity falling `159.91 -> 25.03` while exact-match stays
  `0/8`), a drift detector that catches two of its own metrics
  disagreeing between model versions, and a continuous fine-tuning
  loop that simulates new reports arriving and catches a real
  regression before it would have shipped.
- Dual licensing: MIT for code, CC BY 4.0 for book text.
- CI workflows for Linux/macOS/Windows test runs and a manually
  triggered GitHub Pages publish (`.github/workflows/`), live at
  <https://djimrastephane.github.io/llm-ft-drilling-completions-book/>.

See [CHANGELOG.md](CHANGELOG.md) for the detailed, chapter-by-chapter
history, including every bug caught and fixed along the way.
