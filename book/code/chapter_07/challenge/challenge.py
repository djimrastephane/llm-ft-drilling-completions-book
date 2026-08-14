"""Chapter 7 challenge exercise -- reference solution.

Challenge: format_training_chunks.py uses MAX_CHUNK_CHARS=300. Rerun the
same pipeline with a smaller chunk size (150 chars) and see how many
more, shorter examples that produces.

Usage:
    python code/chapter_07/challenge/challenge.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "chapter_02"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "chapter_06"))

from build_training_examples import HELD_OUT_REPORT, extract_fields, extract_text, normalize_date  # noqa: E402
from data_quality_gate import FULL_SET_DIR, build_archive_records  # noqa: E402
from format_training_chunks import chunk_text, contains_cid_artifact, parse_timeline_entries  # noqa: E402

SMALL_MAX_CHUNK_CHARS = 150


def build_examples_with_chunk_size(pdf_path: Path, max_chars: int) -> list[dict]:
    text = extract_text(pdf_path)
    fields = extract_fields(text)
    if not fields["well_name"] or not fields["rpt_num"] or not fields["rpt_date"]:
        return []

    context = f"Well: {fields['well_name']} | Report #{fields['rpt_num']} | Date: {normalize_date(fields['rpt_date'])}"

    examples = []
    for entry in parse_timeline_entries(text):
        chunks = chunk_text(entry["text"], max_chars=max_chars)
        for i, chunk in enumerate(chunks):
            if contains_cid_artifact(chunk):
                continue
            time_label = f"Time: {entry['from_time']}-{entry['to_time']}"
            if len(chunks) > 1:
                time_label += f" (part {i + 1} of {len(chunks)})"
            examples.append({"instruction": "What happened on this well during this time window?", "input": f"{context} | {time_label}", "output": chunk})
    return examples


def main() -> None:
    records = build_archive_records()
    examples = []
    for record in records:
        if record["file"] == HELD_OUT_REPORT or record["status"] != "ok":
            continue
        examples.extend(build_examples_with_chunk_size(FULL_SET_DIR / record["file"], SMALL_MAX_CHUNK_CHARS))

    chunked = sum(1 for e in examples if "part" in e["input"])
    print(f"With MAX_CHUNK_CHARS={SMALL_MAX_CHUNK_CHARS}: {len(examples)} examples, {chunked} are chunks")
    print("(format_training_chunks.py, with MAX_CHUNK_CHARS=300, produces 669 examples, 125 chunks -- see Chapter 7)")


if __name__ == "__main__":
    main()
