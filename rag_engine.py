"""Retrieval, grounded multi-metric reasoning, and streaming provider fallback."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from groq import Groq
import requests

import memory

ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
INDEX_PATH = os.path.join(ROOT, "data", "tfidf_index.json")

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
def resources() -> Any:
    with open(INDEX_PATH, encoding="utf-8") as index_file:
        return json.load(index_file)


def enough_context(message: str, structured: dict[str, Any] | None) -> bool:
    if structured and any(value not in ("", None, [], {}) for value in structured.values()):
        return True
    terms = ("soil", "rain", "water", "crop", "farm", "land", "region", "forest",
             "biodiversity", "organic carbon", "climate", "pasture")
    return any(term in message.lower() for term in terms)


def retrieve(query: str, structured: dict[str, Any] | None, top_k: int = 6) -> list[dict[str, str]]:
    enriched = query
    if structured:
        enriched += "\nStructured land data: " + json.dumps(structured, sort_keys=True)
    index = resources()
    query_words = re.findall(r"[a-z0-9]{2,}", enriched.lower())
    counts: dict[str, int] = {}
    for word in query_words:
        if word in index["idf"]:
            counts[word] = counts.get(word, 0) + 1
    total = sum(counts.values()) or 1
    query_vector = {word: count / total * index["idf"][word] for word, count in counts.items()}
    norm = math.sqrt(sum(value * value for value in query_vector.values())) or 1
    query_vector = {word: value / norm for word, value in query_vector.items()}
    ranked = []
    for position, document_vector in enumerate(index["vectors"]):
        score = sum(query_vector.get(word, 0.0) * value for word, value in document_vector.items())
        ranked.append((score, position))
    ranked.sort(reverse=True)
    return [
        {"text": index["documents"][position], "source": index["sources"][position]}
        for score, position in ranked[:top_k]
        if score > 0
    ]


def _prompt(message: str, structured: dict[str, Any] | None, context: list[dict[str, str]], session_id: str) -> str:
    passages = "\n\n".join(f"[{item['source']}]\n{item['text']}" for item in context)
    data = json.dumps(structured, sort_keys=True) if structured else "(none)"
    return f"""Retrieved passages:
{passages}

Conversation memory:
{memory.as_prompt(session_id)}

Structured input:
{data}

Current user message:
{message}"""


def _gemini(prompt: str) -> str:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is not configured")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": key}, timeout=60,
        json={"system_instruction": {"parts": [{"text": SYSTEM}]}, "contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}},
    )
    if not response.ok:
        raise RuntimeError(f"Gemini returned HTTP {response.status_code}")
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


def _groq(prompt: str) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not configured")
    response = Groq(api_key=key).chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], temperature=0.2,
    )
    return response.choices[0].message.content or "The provider returned an empty answer."


def stream_answer(message: str, structured: dict[str, Any] | None = None, session_id: str = "default") -> tuple[Iterator[str], list[str]]:
    if not enough_context(message, structured):
        return iter(["Please share at least your soil condition (such as organic carbon), rainfall or water pattern, and land-use/crop type so I can give an evidence-backed recommendation."]), []
    context = retrieve(message, structured)
    prompt = _prompt(message, structured, context, session_id)
    sources = list(dict.fromkeys(item["source"] for item in context))

    def generate() -> Iterator[str]:
        errors = []
        for provider in (_gemini, _groq):
            try:
                answer = provider(prompt)
                for part in re.findall(r".{1,80}(?:\s+|$)", answer, flags=re.S):
                    yield part
                return
            except Exception as error:
                errors.append(f"{provider.__name__[1:]}: {error}")
        raise RuntimeError("Both text providers failed. " + " | ".join(errors))

    return generate(), sources
