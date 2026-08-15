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
# decision) needs that support added first. Diagrams that aren't a simple
# chain (e.g. the LoRA grid diagram below) get their own dedicated builder
# instead of being forced into this shape.
THEORY_DIAGRAMS: dict[int, list[tuple[str, list[str]]]] = {}

# Colors mirror custom-light.scss / custom-dark.scss so each diagram
# matches its page's theme instead of looking pasted in. soft_fill and
# strong_border extend the chain-diagram palette for the LoRA grid
# diagram below (a lighter tint for "computed, not stored" cells, and a
# stronger accent for the "what the model actually reads" cell) -- the
# chain renderer's TIKZ_TEMPLATE only references the four original keys,
# so it ignores the extra ones.
LIGHT_PALETTE = {
    "box_fill": "FFFFFF", "box_border": "0B5394", "text": "222222", "arrow": "0B5394",
    "soft_fill": "E6EEF7", "strong_border": "073763",
}
DARK_PALETTE = {
    "box_fill": "2B2E33", "box_border": "58A6FF", "text": "E8E8E8", "arrow": "58A6FF",
    "soft_fill": "2A3542", "strong_border": "8FC3FF",
}

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


# Chapter 5's "Engineering Translation: LoRA adapter" figure: a two-row
# grid diagram, not a chain. Row 1 shows Table A x Table B = a full-size
# correction (each table drawn 2 cells thick so it doesn't read as
# rank-1). Row 2 shows that correction added on top of the frozen base
# weights, not swapped in. Coordinates are hand-placed in cm; see the
# companion sketch this was reviewed from for the pixel-space version.
LORA_TIKZ_TEMPLATE = r"""
\documentclass[tikz,border=4pt]{standalone}
\usepackage{xcolor}
\usepackage{sansmathfonts}
\renewcommand{\familydefault}{\sfdefault}
\definecolor{trained}{HTML}{%(box_border)s}
\definecolor{corrfill}{HTML}{%(soft_fill)s}
\definecolor{frozenfill}{HTML}{%(box_fill)s}
\definecolor{strongline}{HTML}{%(strong_border)s}
\definecolor{textcolor}{HTML}{%(text)s}
\usetikzlibrary{calc}
\begin{document}
\begin{tikzpicture}[
    trainedcell/.style={fill=trained, draw=trained, line width=0.6pt},
    frozencell/.style={fill=frozenfill, draw=trained, line width=1pt},
    corrcell/.style={fill=corrfill, draw=trained, dashed, line width=0.6pt},
    combinedcell/.style={fill=corrfill, draw=strongline, dashed, line width=1pt},
    every node/.style={text=textcolor}
  ]

\foreach \i in {0,...,4} { \foreach \j in {0,...,1} {
  \draw[trainedcell] ($(0,0) + (\j*0.48,-\i*0.48)$) rectangle ++(0.42,0.42); } }
\foreach \i in {0,...,1} { \foreach \j in {0,...,4} {
  \draw[trainedcell] ($(1.60,-0.72) + (\j*0.48,-\i*0.48)$) rectangle ++(0.42,0.42); } }
\foreach \i in {0,...,4} { \foreach \j in {0,...,4} {
  \draw[corrcell] ($(4.64,0) + (\j*0.48,-\i*0.48)$) rectangle ++(0.42,0.42); } }

\node[font=\bfseries\small] at (3.49,0.90) {Step 1 -- the correction is built from two small tables};
\node[font=\Large] at (1.25,-1.17) {$\times$};
\node[font=\Large] at (4.29,-1.17) {$=$};
\node[font=\scriptsize] at (0.45,-2.70) {Table A};
\node[font=\scriptsize] at (0.45,-2.95) {trained, small};
\node[font=\scriptsize] at (2.77,-2.70) {Table B};
\node[font=\scriptsize] at (2.77,-2.95) {trained, small};
\node[font=\scriptsize] at (5.81,-2.70) {Full-size correction};
\node[font=\scriptsize] at (5.81,-2.95) {computed, never stored};

\draw[textcolor, dashed, line width=0.5pt] (-0.3,-3.30) -- (8.72,-3.30);
\node[font=\bfseries\small] at (4.21,-4.00) {Step 2 -- added on top, not swapped in};

\foreach \i in {0,...,4} { \foreach \j in {0,...,4} {
  \draw[frozencell] ($(0,-4.60) + (\j*0.48,-\i*0.48)$) rectangle ++(0.42,0.42); } }
\foreach \i in {0,...,4} { \foreach \j in {0,...,4} {
  \draw[corrcell] ($(3.04,-4.60) + (\j*0.48,-\i*0.48)$) rectangle ++(0.42,0.42); } }
\foreach \i in {0,...,4} { \foreach \j in {0,...,4} {
  \draw[combinedcell] ($(6.08,-4.60) + (\j*0.48,-\i*0.48)$) rectangle ++(0.42,0.42); } }

\node[font=\Large] at (2.69,-5.77) {$+$};
\node[font=\Large] at (5.73,-5.77) {$=$};
\node[font=\scriptsize] at (1.17,-7.30) {Base weights};
\node[font=\scriptsize] at (1.17,-7.55) {frozen, unchanged};
\node[font=\scriptsize] at (4.21,-7.30) {LoRA correction};
\node[font=\scriptsize] at (4.21,-7.55) {from Step 1};
\node[font=\scriptsize] at (7.25,-7.30) {What the model reads};
\node[font=\scriptsize] at (7.25,-7.55) {at inference};

\end{tikzpicture}
\end{document}
"""


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


def render_lora_diagram(name: str = "lora_lowrank_ch05") -> None:
    """Render Chapter 5's two-row LoRA grid figure to <name>.pdf, _light.svg, _dark.svg."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        light_pdf = _compile_tex_to_pdf(LORA_TIKZ_TEMPLATE % LIGHT_PALETTE, workdir, f"{name}_light")
        dark_pdf = _compile_tex_to_pdf(LORA_TIKZ_TEMPLATE % DARK_PALETTE, workdir, f"{name}_dark")

        shutil.copy(light_pdf, OUTPUT_DIR / f"{name}.pdf")
        _pdf_to_svg(light_pdf, OUTPUT_DIR / f"{name}_light.svg")
        _pdf_to_svg(dark_pdf, OUTPUT_DIR / f"{name}_dark.svg")


def main() -> None:
    for chapter_num, labels in DIAGRAMS.items():
        name = f"pipeline_ch{chapter_num:02d}"
        render_diagram(name, labels)
        print(f"Rendered {name} (.pdf, _light.svg, _dark.svg)")

    render_lora_diagram()
    print("Rendered lora_lowrank_ch05 (.pdf, _light.svg, _dark.svg)")

    for chapter_num, diagrams in THEORY_DIAGRAMS.items():
        for diagram_name, labels in diagrams:
            render_diagram(diagram_name, labels)
            print(f"Rendered {diagram_name} (.pdf, _light.svg, _dark.svg)")


if __name__ == "__main__":
    main()
