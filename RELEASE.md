# Release notes

Per-release highlights for *Fine-Tuning Local LLM for Drilling &
Completions*. See [CHANGELOG.md](CHANGELOG.md) for the full history.

## Unreleased

Nothing pending yet.

## [1.1.0] - 2026-09-22

A hardening and trust release: no new chapters, but every existing
claim in the book got checked again by an independent pass, and the
gaps that check found got closed or documented honestly rather than
left implied.

### Highlights

- **An independent technical review of the whole repository**,
  published as `TECHNICAL_REVIEW.md` -- ML/fine-tuning correctness,
  LoRA/PEFT implementation, retrieval, evaluation methodology,
  drilling/completions engineering accuracy, data-leakage and source
  traceability, reproducibility, and editorial quality. It re-ran
  Chapter 5's full LoRA fine-tune end-to-end and reproduced its
  numbers digit-for-digit, independently hand-derived Chapter 5's
  trainable-parameter count and matched it exactly, independently
  reproduced Chapter 9's BM25-vs-dense-embedding comparison, and
  confirmed by direct grep that the held-out report is genuinely never
  leaked into training data. It raised 18 findings (1 CRITICAL, 2
  HIGH, 7 MEDIUM, 8 LOW) -- every one fixed, including a fabricated
  quote in Chapter 4 attributed to a report that never contained it,
  now replaced with a genuine, verified excerpt -- and the review's
  final assessment moved from "READY AFTER TECHNICAL CORRECTIONS" to
  **READY**.
- **A separate engineering QA/QC audit** independently verified every
  equation in the book's code (perplexity, cosine similarity, BM25
  retrieval, the faithfulness score, the drift comparator) and every
  drilling-data number quoted in the prose against its source PDF,
  closing its one real test gap and its two open `CANNOT VERIFY`
  items with real, run numbers instead of leaving them reasoned but
  unconfirmed. Published as a closed, public reference issue.
- **New `Appendix C: Known Limitations`**, written after review from
  an oil-and-gas professional and verified point-by-point against the
  book's own code: the field-extraction regexes are hardcoded to Utah
  FORGE's report layout, the faithfulness score is a plain word-overlap
  check that can score `0.80` on an answer with the opposite meaning of
  its source, training runs here are CPU/demo-scale only, checkpointing
  restores adapter weights but not optimizer state, and the base
  model's tokenizer still splits oilfield shorthand like `BHA` into
  unrelated word pieces. Six gaps in total, each already true of the
  book's code -- newly written down in one place instead of left
  implied.
- **A whole-book roadmap diagram** and a root-level `TROUBLESHOOTING.md`,
  both linked from the published book itself, not just the GitHub
  README.
- **A new optional Mathematica companion notebook**
  (`mathematica/FineTuning_Before_After_Explorer.nb`), reusing the
  book's own real functions and the real Chapter 5 adapter to let a
  reader step through all 18 real before/after questions interactively,
  with token-overlap and cosine-similarity metrics, an "answer
  microscope," and a failure-patterns view -- author-expanded across
  several rounds and confirmed opening and evaluating correctly in a
  real Wolfram front end.
- Two long-completed GitHub issues, left open after the work they
  tracked had already shipped, closed to match reality.
- Full fast test suite: 74 passing (was 67 at v1.0.0), plus 27 slower/
  model-dependent tests skipped by default -- 101 total (was 94).

See [CHANGELOG.md](CHANGELOG.md) for the detailed, entry-by-entry
history behind each of these.

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
