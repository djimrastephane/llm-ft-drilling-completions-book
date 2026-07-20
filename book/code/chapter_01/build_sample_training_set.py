"""Reproduce this book's curated 10-report sample training set.

`datasets/sample_training_set/` is a fixed, curated subset of
`datasets/full_training_set/` (the full 76-report Utah FORGE archive,
well FORGE 16A(78)-32) -- the same curated subset used by the author's
previous book, `ddr-rag-book`, including its real stuck-pipe
(Drilling_038) and packers-fail-to-fishing (Drilling_049/050) sequence.
Both tiers are already committed; this script exists so the sample set
can be regenerated if the full archive is ever extended or replaced.

Usage:
    python code/chapter_01/build_sample_training_set.py
"""

import shutil
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parents[2]
FULL_SET_DIR = BOOK_ROOT / "datasets" / "full_training_set"
SAMPLE_SET_DIR = BOOK_ROOT / "datasets" / "sample_training_set"

CURATED_FILENAMES = [
    "FORGE-16A-78-32_Completion_003_2021-01-06.pdf",
    "FORGE-16A-78-32_Drilling_003_2020-10-22.pdf",
    "FORGE-16A-78-32_Drilling_019_2020-11-07.pdf",
    "FORGE-16A-78-32_Drilling_036_2020-11-24.pdf",
    "FORGE-16A-78-32_Drilling_037_2020-11-25.pdf",
    "FORGE-16A-78-32_Drilling_038_2020-11-26.pdf",
    "FORGE-16A-78-32_Drilling_039_2020-11-27.pdf",
    "FORGE-16A-78-32_Drilling_048_2020-12-06.pdf",
    "FORGE-16A-78-32_Drilling_049_2020-12-07.pdf",
    "FORGE-16A-78-32_Drilling_050_2020-12-08.pdf",
]


def main() -> None:
    SAMPLE_SET_DIR.mkdir(parents=True, exist_ok=True)
    missing = [name for name in CURATED_FILENAMES if not (FULL_SET_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing from {FULL_SET_DIR}: {missing}. "
            "Populate datasets/full_training_set/ first."
        )

    for name in CURATED_FILENAMES:
        shutil.copyfile(FULL_SET_DIR / name, SAMPLE_SET_DIR / name)

    print(f"Copied {len(CURATED_FILENAMES)} reports into {SAMPLE_SET_DIR}")


if __name__ == "__main__":
    main()
