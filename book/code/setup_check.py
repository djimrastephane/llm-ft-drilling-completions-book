"""Part 0: confirm the local LLM workshop is ready before Chapter 1.

Run this file after finishing Part 0 (virtual environment created and
active, packages installed) to confirm everything is in place, including
a plain-language read on what hardware is available for fine-tuning.

Usage:
    python code/setup_check.py

STATUS: minimal placeholder -- expand as Part 0 is written (e.g. a
Python-version check, a clearer table of what CPU-only readers should
expect chapter by chapter).
"""

import sys


def check_python_version() -> bool:
    ok = sys.version_info >= (3, 11)
    label = "OK" if ok else "FAIL (need Python 3.11+)"
    print(f"[{label}] Python {sys.version.split()[0]}")
    return ok


def check_core_packages() -> bool:
    packages = ["torch", "transformers", "peft", "datasets", "accelerate"]
    all_ok = True
    for name in packages:
        try:
            __import__(name)
            print(f"[OK] {name} importable")
        except ImportError:
            print(f"[FAIL] {name} not importable -- run: pip install -r requirements.txt")
            all_ok = False
    return all_ok


def check_hardware() -> None:
    try:
        import torch
    except ImportError:
        print("[SKIP] hardware check needs torch -- install requirements first")
        return

    if torch.cuda.is_available():
        print(f"[INFO] CUDA GPU available: {torch.cuda.get_device_name(0)}")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("[INFO] Apple Silicon MPS device available")
    else:
        print(
            "[INFO] No GPU detected -- CPU-only. Fine-tuning chapters will "
            "still work with a small model, just more slowly."
        )


def main() -> None:
    print("Fine-Tuning Local LLM for Drilling & Completions -- setup check\n")
    py_ok = check_python_version()
    pkgs_ok = check_core_packages()
    check_hardware()
    print()
    if py_ok and pkgs_ok:
        print("Workshop is ready.")
    else:
        print("Workshop is not ready yet -- see [FAIL] lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
