# Diagrams

Every chapter's opening pipeline diagram and any in-theory-section flow
diagrams are generated from `generate_diagrams.py` (TikZ → PDF →
SVG/PDF), the same pattern as the author's previous book. See
`templates/chapter_template.qmd` for exactly how a chapter references
these files (`pipeline_chNN_light.svg`, `pipeline_chNN_dark.svg`,
`pipeline_chNN.pdf`).

**Status:** `generate_diagrams.py` here is a skeleton only — it defines
the expected `DIAGRAMS` / `THEORY_DIAGRAMS` structure but does not yet
render anything. Port the previous book's rendering logic (or rewrite
it) before drafting chapters that need diagrams.
