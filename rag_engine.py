"""Retrieval, grounded multi-metric reasoning, and streaming provider fallback."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

# sentence-transformers only needs the PyTorch path here. Disabling optional
# TensorFlow imports avoids unrelated local TensorFlow/protobuf conflicts.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import chromadb
from dotenv import load_dotenv
from groq import Groq
from google import genai
from google.genai import errors as genai_errors
from sentence_transformers import SentenceTransformer

import memory

ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
COLLECTION = "biodiversity_knowledge"
DB_DIR = os.path.join(ROOT, "chroma_db")
MODEL_NAME = "BAAI/bge-small-en-v1.5"

SYSTEM = """You are an evidence-grounded environmental scientist.
Use ONLY the retrieved source passages for factual claims and recommendations.
If the passages do not support a claim, say that evidence is unavailable.
Every substantive response must connect at least three metrics among soil health,
biodiversity, water/rainfall, land use, and human impact. Never give vague advice.
For each recommendation use exactly these labels:
What to do:
Why it works:
Which metric(s) it improves:
Source reference:
Time horizon: short / medium / long term
Cite only source names present in the retrieved passages; never invent page numbers,
sections, measurements, or citations. If the user lacks enough land, soil, climate,
or land-use context, ask one specific clarifying question instead of guessing."""


@lru_cache(maxsize=1)
def resources() -> tuple[Any, Any]:
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION)
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    return collection, model


def enough_context(message: str, structured: dict[str, Any] | None) -> bool:
    if structured and any(value not in ("", None, [], {}) for value in structured.values()):
        return True
    terms = ("soil", "rain", "water", "crop", "farm", "land", "region", "forest",
             "biodiversity", "organic carbon", "climate", "pasture")
    return any(term in message.lower() for term in terms)


def retrieve(query: str, structured: dict[str, Any] | None, top_k: int = 6) -> list[dict[str, str]]:
    collection, model = resources()
    enriched = query
    if structured:
        enriched += "\nStructured land data: " + json.dumps(structured, sort_keys=True)
    vector = model.encode([enriched], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=vector, n_results=top_k, include=["documents", "metadatas"])
    documents = result.get("documents", [[]])[0]
    metadata = result.get("metadatas", [[]])[0]
    return [{"text": text, "source": str((meta or {}).get("source", "Unknown source"))}
            for text, meta in zip(documents, metadata)]


def _prompt(message: str, structured: dict[str, Any] | None, context: list[dict[str, str]]) -> str:
    passages = "\n\n".join(f"[{item['source']}]\n{item['text']}" for item in context)
    data = json.dumps(structured, sort_keys=True) if structured else "(none)"
    return f"""Retrieved passages:
{passages}

Conversation memory:
{memory.as_prompt()}

Structured input:
{data}

Current user message:
{message}"""


def stream_answer(message: str, structured: dict[str, Any] | None = None) -> tuple[Iterator[str], list[str]]:
    if not enough_context(message, structured):
        return iter(["Please share at least your soil condition (such as organic carbon), rainfall or water pattern, and land-use/crop type so I can give an evidence-backed recommendation."]), []
    context = retrieve(message, structured)
    prompt = _prompt(message, structured, context)
    sources = list(dict.fromkeys(item["source"] for item in context))

    def generate() -> Iterator[str]:
        groq_key = os.getenv("GROQ_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if groq_key:
            try:
                client = Groq(api_key=groq_key)
                response = client.chat.completions.create(
                    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),  # or "opt/ossaa-30b" qwen/qwen3.8-27b
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                    temperature=0.2, stream=True,
                )
                for item in response:
                    token = item.choices[0].delta.content
                    if token:
                        yield token
                return
            except Exception as groq_error:
                if not gemini_key:
                    raise RuntimeError("Groq failed and no Gemini key is configured.") from groq_error
        if not gemini_key:
            raise RuntimeError("Set GROQ_API_KEY or GEMINI_API_KEY/GOOGLE_API_KEY in .env.")
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        client = genai.Client(api_key=gemini_key)
        try:
            response = client.models.generate_content_stream(
                model=model_name,
                contents=prompt,
                config={"system_instruction": SYSTEM, "temperature": 0.2},
            )
        except genai_errors.ServerError as gemini_error:
            raise RuntimeError(
                f"Gemini is temporarily unavailable (503) for model '{model_name}'. "
                "Retry in a few minutes, or configure GROQ_API_KEY or another "
                "available GEMINI_MODEL in .env."
            ) from gemini_error
        except genai_errors.ClientError as gemini_error:
            raise RuntimeError(
                f"Gemini rejected model '{model_name}'. Check GEMINI_MODEL in .env "
                "and choose a model available to your API key."
            ) from gemini_error
        try:
            for item in response:
                if item.text:
                    yield item.text
        except genai_errors.ServerError as gemini_error:
            raise RuntimeError(
                f"Gemini became temporarily unavailable (503) for model '{model_name}'. "
                "Retry in a few minutes, or configure GROQ_API_KEY or another "
                "available GEMINI_MODEL in .env."
            ) from gemini_error
        except genai_errors.ClientError as gemini_error:
            raise RuntimeError(
                f"Gemini rejected model '{model_name}'. Check GEMINI_MODEL in .env "
                "and choose a model available to your API key."
            ) from gemini_error

    return generate(), sources
