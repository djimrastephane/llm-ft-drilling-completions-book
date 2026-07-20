"""Base vs. Fine-Tuned companion app.

STATUS: placeholder -- not implemented yet. See app/README.md for the
planned design (question in, base model's and fine-tuned model's
answers shown side by side).

Run once implemented:
    streamlit run book/app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="Fine-Tuned Drilling & Completions LLM", page_icon="🛢️")

st.title("Base vs. Fine-Tuned Drilling & Completions LLM")
st.info(
    "This companion app is not implemented yet. It's planned to load "
    "the base model from Chapter 1 and the fine-tuned model from "
    "Chapter 5+, then show their answers to the same question side by "
    "side. See book/app/README.md for details."
)

question = st.text_input("Ask a drilling or completions question", "")
if question:
    st.warning("Model comparison is not wired up yet -- see app/helpers.py.")
