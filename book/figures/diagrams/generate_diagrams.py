"""Generate this book's per-chapter pipeline and theory diagrams.

Each DIAGRAMS entry is rendered as a small vertical box chain (2-4 boxes,
one arrow between each) via TikZ, compiled to PDF with pdflatex, and
converted to light/dark themed SVGs with pdftocairo for HTML. The PDF
book output embeds the light-themed PDF directly (see each chapter's
`.content-visible when-format="pdf"` block).

Usage:
    python figures/diagrams/generate_diagrams.py
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent

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
# diagram above. Populate as chapters are drafted. The renderer here only
# draws a straight vertical chain -- a diagram that branches (a yes/no
# decision) needs that support added first.
THEORY_DIAGRAMS: dict[int, list[tuple[str, list[str]]]] = {}

# Colors mirror custom-light.scss / custom-dark.scss so each diagram
# matches its page's theme instead of looking pasted in.
LIGHT_PALETTE = {"box_fill": "FFFFFF", "box_border": "0B5394", "text": "222222", "arrow": "0B5394"}
DARK_PALETTE = {"box_fill": "2B2E33", "box_border": "58A6FF", "text": "E8E8E8", "arrow": "58A6FF"}

TIKZ_TEMPLATE = r"""
\documentclass[tikz,border=4pt]{standalone}
\usepackage{xcolor}
\usepackage{sansmathfonts}
\renewcommand{\familydefault}{\sfdefault}
\definecolor{boxfill}{HTML}{%(box_fill)s}
\definecolor{boxborder}{HTML}{%(box_border)s}
\definecolor{textcolor}{HTML}{%(text)s}
\definecolor{arrowcolor}{HTML}{%(arrow)s}
\usetikzlibrary{arrows.meta, positioning}
\begin{document}
\begin{tikzpicture}[
    box/.style={
      rectangle, rounded corners=2pt, draw=boxborder, fill=boxfill,
      line width=0.9pt, minimum width=3.4cm, minimum height=1.05cm,
      text=textcolor, font=\small, align=center, inner sep=6pt
    },
    arr/.style={-{Stealth[length=2.2mm]}, arrowcolor, line width=0.9pt}
  ]
%(nodes)s
%(arrows)s
\end{tikzpicture}
\end{document}
"""


def _build_tex(labels: list[str], palette: dict[str, str]) -> str:
    nodes, arrows = [], []
    prev = None
    for i, label in enumerate(labels):
        name = f"n{i}"
        if prev is None:
            nodes.append(f"\\node[box] ({name}) {{{label}}};")
        else:
            nodes.append(f"\\node[box, below=0.7cm of {prev}] ({name}) {{{label}}};")
            arrows.append(f"\\draw[arr] ({prev}) -- ({name});")
        prev = name
    return TIKZ_TEMPLATE % {**palette, "nodes": "\n".join(nodes), "arrows": "\n".join(arrows)}


def _compile_tex_to_pdf(tex_source: str, workdir: Path, jobname: str) -> Path:
    (workdir / f"{jobname}.tex").write_text(tex_source)
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{jobname}.tex"],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    pdf_path = workdir / f"{jobname}.pdf"
    if not pdf_path.exists():
        raise RuntimeError(f"pdflatex failed for {jobname}:\n{result.stdout[-3000:]}")
    return pdf_path


def _pdf_to_svg(pdf_path: Path, svg_path: Path) -> None:
    subprocess.run(["pdftocairo", "-svg", str(pdf_path), str(svg_path)], check=True)


def render_diagram(name: str, labels: list[str]) -> None:
    """Render one diagram to <name>.pdf, <name>_light.svg, <name>_dark.svg in OUTPUT_DIR."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        light_pdf = _compile_tex_to_pdf(_build_tex(labels, LIGHT_PALETTE), workdir, f"{name}_light")
        dark_pdf = _compile_tex_to_pdf(_build_tex(labels, DARK_PALETTE), workdir, f"{name}_dark")

        shutil.copy(light_pdf, OUTPUT_DIR / f"{name}.pdf")
        _pdf_to_svg(light_pdf, OUTPUT_DIR / f"{name}_light.svg")
        _pdf_to_svg(dark_pdf, OUTPUT_DIR / f"{name}_dark.svg")


def main() -> None:
    for chapter_num, labels in DIAGRAMS.items():
        name = f"pipeline_ch{chapter_num:02d}"
        render_diagram(name, labels)
        print(f"Rendered {name} (.pdf, _light.svg, _dark.svg)")

    for chapter_num, diagrams in THEORY_DIAGRAMS.items():
        for diagram_name, labels in diagrams:
            render_diagram(diagram_name, labels)
            print(f"Rendered {diagram_name} (.pdf, _light.svg, _dark.svg)")


if __name__ == "__main__":
    main()
