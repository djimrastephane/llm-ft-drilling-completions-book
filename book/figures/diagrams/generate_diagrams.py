"""Generate this book's per-chapter pipeline and theory diagrams.

STATUS: skeleton only -- not implemented. The previous book
(ddr-rag-book) compiles each entry below via TikZ -> pdflatex -> PDF,
then converts to light/dark SVG variants for HTML and keeps the PDF for
LaTeX output. Port or rewrite that rendering pipeline before drafting
any chapter that references `pipeline_chNN_*` files.

Usage (once implemented):
    python figures/diagrams/generate_diagrams.py
"""

# Chapter-opening pipeline diagrams: chapter number -> ordered pipeline
# stage labels (2-4 short strings), rendered as a vertical box chain.
DIAGRAMS: dict[int, list[str]] = {
    1: ["Base Model", "Loaded Locally"],
    2: ["Raw Reports", "Training Examples"],
    3: ["Prompt", "Baseline Answer"],
    4: ["Text", "Tokens / Embeddings"],
    5: ["Base Model", "LoRA Fine-Tune"],
    6: ["Raw Examples", "Quality-Gated Set"],
    7: ["Training Set", "Formatted Chunks"],
    8: ["Training Run", "Checkpoints"],
    9: ["Fine-Tuned Model", "+ Retrieval"],
    10: ["Model Output", "Traceable Answer"],
    11: ["Fine-Tuned Model", "Evaluation Report"],
    12: ["Model v(N)", "Drift Report"],
    13: ["New Reports", "Updated Model"],
}

# Theory-section diagrams: chapter number -> list of (name, steps) for
# any in-chapter algorithm walkthroughs, beyond the opening pipeline
# diagram above. Populate as chapters are drafted.
THEORY_DIAGRAMS: dict[int, list[tuple[str, list[str]]]] = {}


def main() -> None:
    raise NotImplementedError(
        "Diagram rendering pipeline not implemented yet -- see this "
        "file's module docstring."
    )


if __name__ == "__main__":
    main()
