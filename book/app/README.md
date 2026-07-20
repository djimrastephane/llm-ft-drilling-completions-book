# Base vs. Fine-Tuned Companion App

**Status: not implemented yet.** This is a placeholder for a small
Streamlit app, planned to show the book's payoff end to end: ask one
drilling or completions question and see the base model's answer next
to the fine-tuned model's answer, side by side.

This is planned as an educational companion, not a production system,
mirroring the author's previous book's companion app.

## Run it (once implemented)

```bash
pip install -r book/requirements.txt      # includes Streamlit
streamlit run book/app/streamlit_app.py
```

## What it will reuse from the book

Nothing here should reimplement the pipeline. The app is planned to
import the book's own code:

| Step | Comes from |
|---|---|
| Base model loading | `code/chapter_01/load_local_model.py` |
| Fine-tuned model loading | `code/chapter_05/first_lora_finetune.py` (or a later checkpoint) |
| Evaluation / scoring | `code/chapter_11/eval_finetuned_model.py` |

`app/helpers.py` will hold app-only glue logic (model loading, side-by-side
formatting). `app/streamlit_app.py` is only the screen around it.

## Files

| File | Purpose |
|---|---|
| `streamlit_app.py` | The UI: question box, base vs. fine-tuned answers side by side |
| `helpers.py` | Model-loading and comparison glue (no Streamlit; unit-testable) |
