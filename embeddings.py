"""Gemini-hosted embeddings used for indexing and retrieval."""

from __future__ import annotations

import os
from collections.abc import Sequence

from google import genai
from google.genai import errors as genai_errors

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
BATCH_SIZE = 100


def _client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY to create Gemini embeddings."
        )
    return genai.Client(api_key=api_key)


def embed_texts(texts: Sequence[str], *, query: bool = False) -> list[list[float]]:
    """Embed text in batches using Gemini's free-tier embedding model."""
    if not texts:
        return []

    task_type = "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT"
    model_name = os.getenv("GEMINI_EMBEDDING_MODEL", EMBEDDING_MODEL)
    result: list[list[float]] = []
    try:
        client = _client()
        for start in range(0, len(texts), BATCH_SIZE):
            response = client.models.embed_content(
                model=model_name,
                contents=list(texts[start : start + BATCH_SIZE]),
                config={
                    "task_type": task_type,
                    "output_dimensionality": EMBEDDING_DIMENSIONS,
                },
            )
            result.extend(
                [list(embedding.values) for embedding in response.embeddings]
            )
    except genai_errors.APIError as error:
        raise RuntimeError(
            f"Gemini embedding failed for model '{model_name}'. "
            "Check your Gemini API key, model name, and free-tier quota."
        ) from error

    if len(result) != len(texts):
        raise RuntimeError(
            f"Gemini returned {len(result)} embeddings for {len(texts)} texts."
        )
    return result
