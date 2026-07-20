"""Shared fixtures for the book's code tests.

Every chapter's script in code/chapter_NN/ is written as a standalone
file, not a package -- readers run them directly with `python
code/chapter_NN/script.py`. To test the real functions inside them
without duplicating any code, each chapter's folder is added to
sys.path here, so tests can just `import load_local_model`, `import
build_training_examples`, and so on, exactly as if that file were the
only thing on the path -- which mirrors how a reader actually runs them.

STATUS: fixtures below are placeholders. Chapter 9 (hybrid fine-tuning +
retrieval) is the one most likely to import both faiss and
sentence-transformers/torch in the same process; if that combination
crashes on macOS the way it did in the author's previous book, set
OMP_NUM_THREADS=1 here before those imports happen, the same way this
file's ddr-rag-book counterpart does.
"""

import sys
from pathlib import Path

import pytest

BOOK_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = BOOK_ROOT / "datasets"
SAMPLE_TRAINING_SET_DIR = DATASETS_DIR / "sample_training_set"
FULL_TRAINING_SET_DIR = DATASETS_DIR / "full_training_set"

for _n in range(1, 14):
    _chapter_dir = BOOK_ROOT / "code" / f"chapter_{_n:02d}"
    if _chapter_dir.is_dir() and str(_chapter_dir) not in sys.path:
        sys.path.insert(0, str(_chapter_dir))


@pytest.fixture(scope="session")
def sample_training_set_dir() -> Path:
    return SAMPLE_TRAINING_SET_DIR


@pytest.fixture(scope="session")
def full_training_set_dir() -> Path:
    return FULL_TRAINING_SET_DIR
