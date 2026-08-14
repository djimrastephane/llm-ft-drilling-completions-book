"""Chapter 5 challenge exercise -- reference solution.

Challenge: first_lora_finetune.py targets all four attention projections
(q_proj, k_proj, v_proj, o_proj). A more common minimal LoRA setup only
targets q_proj and v_proj. Rerun fine-tuning with that lighter config --
fewer trainable parameters -- and see whether training recall changes.

Usage:
    python code/chapter_05/challenge/challenge.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "chapter_01"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "chapter_02"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "chapter_03"))

from load_local_model import MODEL_NAME, load_model_and_tokenizer  # noqa: E402

import torch  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402

from baseline_prompting import run_baseline  # noqa: E402
from build_training_examples import build_training_examples  # noqa: E402
from first_lora_finetune import fine_tune, score  # noqa: E402

LIGHT_LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)


def main() -> None:
    torch.manual_seed(0)
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    examples = build_training_examples()

    lora_model = get_peft_model(model, LIGHT_LORA_CONFIG)
    lora_model.print_trainable_parameters()

    fine_tune(lora_model, tokenizer, examples)

    training_after = score(run_baseline(lora_model, tokenizer, examples))
    print(f"\nTraining recall (q_proj/v_proj only): {training_after[0]}/{training_after[1]}")
    print("(first_lora_finetune.py, targeting all four attention projections, scored 13/16 -- see Chapter 5)")


if __name__ == "__main__":
    main()
