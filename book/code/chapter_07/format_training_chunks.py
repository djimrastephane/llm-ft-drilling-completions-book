"""Chapter 7: Formatting and Chunking a Training Set at Scale.

Chapter 2's two summary fields (PRESENT OPERATIONS, ACTIVITY PLANNED)
only capture a report's headline state -- every report also has a
`TIME BREAKDOWN` section: a real, multi-entry operational log, one row
per time window, with its own free-text narrative. This module turns
those rows into many more, richer training examples per report, and
splits ("chunks") any entry too long for one training example into
several, so nothing gets silently truncated.

Only processes reports that passed Chapter 6's quality gate, and never
Chapter 2's held-out report (Drilling_037) -- the "at scale" training
set this produces must respect the same reservation Chapter 5 already
fine-tuned against.

Usage:
    python code/chapter_07/format_training_chunks.py
"""

import json
import re
import sys
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_02"))
sys.path.insert(0, str(BOOK_ROOT / "code" / "chapter_06"))

from build_training_examples import HELD_OUT_REPORT, extract_fields, extract_text, normalize_date  # noqa: E402
from data_quality_gate import FULL_SET_DIR, build_archive_records  # noqa: E402

OUTPUT_PATH = BOOK_ROOT / "datasets" / "training_examples" / "full_training_examples.jsonl"

TIMELINE_SECTION_PATTERN = re.compile(
    r"TIME BREAKDOWN\n(?:FROM TO HRS PHASE CODE NPT OPERATIONS\n)?(.*?)\nTOTAL HRS", re.DOTALL
)
TIMELINE_ENTRY_PATTERN = re.compile(r"^(\d{2}:\d{2})\s+(\d{2}:\d{2})\s+([\d.]+)\s+(.*)", re.MULTILINE)

MAX_CHUNK_CHARS = 300


def parse_timeline_entries(text: str) -> list[dict]:
    """Split a report's TIME BREAKDOWN section into (from, to, text) entries."""
    section = TIMELINE_SECTION_PATTERN.search(text)
    if not section:
        return []
    body = section.group(1)
    matches = list(TIMELINE_ENTRY_PATTERN.finditer(body))

    entries = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        raw_text = f"{match.group(4)} {body[start:end].strip()}".strip()
        entries.append(
            {
                "from_time": match.group(1),
                "to_time": match.group(2),
                "text": re.sub(r"\s+", " ", raw_text),
            }
        )
    return entries


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into pieces no longer than max_chars, only at word boundaries."""
    words = text.split()
    chunks, current, current_len = [], [], 0
    for word in words:
        added_len = len(word) + (1 if current else 0)
        if current and current_len + added_len > max_chars:
            chunks.append(" ".join(current))
            current, current_len = [], 0
            added_len = len(word)
        current.append(word)
        current_len += added_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def contains_cid_artifact(text: str) -> bool:
    """(cid:N) markers are pdfplumber's stand-in for a glyph it couldn't decode --
    usually from an embedded sub-table -- never real report content.
    """
    return "(cid:" in text


def build_timeline_examples_for_report(pdf_path: Path) -> tuple[list[dict], int]:
    """Turn one report's timeline entries into instruction/response training examples.

    Returns (examples, artifacts_filtered) -- chunks containing an
    undecoded-glyph artifact are dropped rather than trained on, and
    counted so that filtering stays visible instead of silent.
    """
    text = extract_text(pdf_path)
    fields = extract_fields(text)
    if not fields["well_name"] or not fields["rpt_num"] or not fields["rpt_date"]:
        return [], 0

    context = f"Well: {fields['well_name']} | Report #{fields['rpt_num']} | Date: {normalize_date(fields['rpt_date'])}"

    examples, artifacts_filtered = [], 0
    for entry in parse_timeline_entries(text):
        chunks = chunk_text(entry["text"])
        for i, chunk in enumerate(chunks):
            if contains_cid_artifact(chunk):
                artifacts_filtered += 1
                continue
            time_label = f"Time: {entry['from_time']}-{entry['to_time']}"
            if len(chunks) > 1:
                time_label += f" (part {i + 1} of {len(chunks)})"
            examples.append(
                {
                    "instruction": "What happened on this well during this time window?",
                    "input": f"{context} | {time_label}",
                    "output": chunk,
                }
            )
    return examples, artifacts_filtered


def build_training_set_at_scale(full_dir: Path = FULL_SET_DIR) -> tuple[list[dict], list[str], int]:
    """Build timeline training examples for every quality-gated, non-held-out report."""
    records = build_archive_records(full_dir)
    examples, skipped, artifacts_filtered = [], [], 0
    for record in records:
        if record["file"] == HELD_OUT_REPORT:
            continue
        if record["status"] != "ok":
            skipped.append(record["file"])
            continue
        report_examples, report_artifacts = build_timeline_examples_for_report(full_dir / record["file"])
        examples.extend(report_examples)
        artifacts_filtered += report_artifacts
    return examples, skipped, artifacts_filtered


def save_examples_jsonl(examples: list[dict], output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")


def main() -> None:
    records = build_archive_records()
    processed_count = len(records) - sum(1 for r in records if r["status"] != "ok") - 1  # -1 for held-out

    examples, skipped, artifacts_filtered = build_training_set_at_scale()
    save_examples_jsonl(examples)

    chunked = sum(1 for e in examples if "part" in e["input"])
    print(f"Built {len(examples)} training examples from {processed_count} reports (excluding {len(skipped)} unrecognized, 1 held out)")
    print(f"{chunked} examples are chunks of a longer timeline entry (max {MAX_CHUNK_CHARS} chars each)")
    print(f"{artifacts_filtered} chunk(s) filtered out for containing undecoded-glyph artifacts")
    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
