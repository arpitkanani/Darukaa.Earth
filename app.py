"""Streamlit entrypoint for the Darukaa.Earth biodiversity assistant."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit as st
from pypdf.errors import PdfReadError

import ingest
import memory
from rag_engine import resources, stream_answer

st.set_page_config(page_title="Darukaa.Earth", page_icon="🌱")
st.title("Darukaa.Earth Biodiversity Intelligence")
st.caption("Evidence-backed land and ecosystem recommendations from the supplied knowledge base.")
memory.initialize()

with st.sidebar:
    st.header("Add knowledge")
    st.caption("Upload a PDF (maximum 20 MB) to add it to the current knowledge base.")
    uploaded_pdf = st.file_uploader(
        "Drag and drop a PDF here",
        type=["pdf"],
        accept_multiple_files=False,
        help="Only PDF files up to 20 MB are accepted.",
    )
    if uploaded_pdf is not None:
        if uploaded_pdf.size > ingest.MAX_UPLOAD_BYTES:
            st.error("This file is larger than the 20 MB limit.")
        elif st.button("Ingest PDF", type="primary", use_container_width=True):
            safe_name = Path(uploaded_pdf.name).name
            digest = hashlib.sha256(uploaded_pdf.getvalue()).hexdigest()[:12]
            destination = ingest.UPLOAD_DIR / f"{digest}_{safe_name}"
            ingest.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            try:
                destination.write_bytes(uploaded_pdf.getvalue())
                chunk_count = ingest.ingest_pdf(destination, source=safe_name)
                resources.cache_clear()
            except (OSError, ValueError, PdfReadError) as exc:
                cleanup_error = None
                if destination.exists():
                    try:
                        destination.unlink()
                    except OSError as unlink_error:
                        cleanup_error = unlink_error
                message = f"Could not ingest this PDF: {exc}"
                if cleanup_error:
                    message += f" The temporary upload could not be removed: {cleanup_error}"
                st.error(message)
            else:
                st.success(f"Added {safe_name} ({chunk_count} chunks).")

for item in memory.messages():
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

with st.expander("Structured input (optional)"):
    st.caption("Paste a JSON object with a required question and optional land context.")
    structured_text = st.text_area(
        "Land context and question (JSON)",
        height=220,
        placeholder='{"question": "How can I improve soil health?", "soil_organic_carbon": 0.3, "rainfall": "low", "crop": "wheat", "region": "semi-arid"}',
    )
    submit_json = st.button("Ask JSON question", type="primary")

question = st.chat_input("Ask about your land or ecosystem...")
structured = None
if structured_text.strip():
    try:
        parsed = json.loads(structured_text)
        if not isinstance(parsed, dict):
            raise ValueError("The JSON root must be an object.")
        json_question = parsed.pop("question", None)
        if json_question is not None and not isinstance(json_question, str):
            raise ValueError('The "question" value must be a string.')
        if submit_json:
            if not json_question or not json_question.strip():
                raise ValueError('Add a non-empty "question" field to submit JSON.')
            question = json_question.strip()
        structured = parsed
    except (json.JSONDecodeError, ValueError) as exc:
        if submit_json or question:
            st.error(f"Invalid structured JSON: {exc}")
            st.stop()

if question:
    memory.add("user", question)
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            stream, sources = stream_answer(question, structured)
            answer = st.write_stream(stream)
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            if sources:
                st.caption("Sources: " + ", ".join(sources))
            memory.add("assistant", answer)
