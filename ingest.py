"""Build the local Chroma knowledge base from PDFs and configured HTML sources."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from pypdf import PdfReader

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
UPLOAD_DIR = ROOT / "data" / "uploads"
INDEX_PATH = ROOT / "data" / "tfidf_index.json"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunks(text: str, size: int = 650, overlap: int = 100) -> list[str]:
    words = clean(text).split()
    result: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + size]).strip()
        if chunk:
            result.append(chunk)
        if start + size >= len(words):
            break
    return result


def pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{2,}", text.lower())


def store_documents(documents: list[tuple[str, str]], *, recreate: bool = False) -> int:
    del recreate
    texts: list[str] = []
    sources: list[str] = []
    for source, text in documents:
        for chunk in chunks(text):
            texts.append(chunk)
            sources.append(source)
    document_tokens = [tokens(text) for text in texts]
    document_frequency: dict[str, int] = {}
    for words in document_tokens:
        for word in set(words):
            document_frequency[word] = document_frequency.get(word, 0) + 1
    document_count = len(texts)
    vocabulary = {
        word: math.log((1 + document_count) / (1 + frequency)) + 1
        for word, frequency in document_frequency.items()
    }
    vectors: list[dict[str, float]] = []
    for words in document_tokens:
        counts: dict[str, int] = {}
        for word in words:
            if word in vocabulary:
                counts[word] = counts.get(word, 0) + 1
        total = sum(counts.values()) or 1
        vector = {word: (count / total) * vocabulary[word] for word, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1
        vectors.append({word: value / norm for word, value in vector.items()})
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(
        json.dumps({"idf": vocabulary, "documents": texts, "sources": sources, "vectors": vectors}),
        encoding="utf-8",
    )
    return len(texts)


def configured_documents() -> list[tuple[str, str]]:
    """Load the repository corpus, runtime uploads, and optional HTML sources."""
    load_dotenv(ROOT / ".env")
    documents: list[tuple[str, str]] = []
    for path in sorted(RAW_DIR.glob("*.pdf")):
        documents.append((path.name, pdf_text(path)))
    for path in sorted(UPLOAD_DIR.glob("*.pdf")):
        documents.append((uploaded_source(path), pdf_text(path)))
    for url in urls():
        source = Path(urlparse(url).path).name or urlparse(url).netloc
        documents.append((f"{source} ({url})", html_text(url)))
    return documents


def ensure_index() -> int:
    """Build the local TF-IDF index once when it is missing or empty."""
    if INDEX_PATH.exists():
        try:
            index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
            if index.get("documents"):
                return len(index["documents"])
        except (OSError, json.JSONDecodeError):
            pass
    return store_documents(configured_documents(), recreate=True)


def ingest_pdf(path: Path, source: str | None = None) -> int:
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files can be ingested.")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("PDF files must be 20 MB or smaller.")
    text = pdf_text(path)
    if not clean(text):
        raise ValueError("The PDF does not contain extractable text.")
    return store_documents(configured_documents(), recreate=True)


def uploaded_source(path: Path) -> str:
    prefix, separator, original_name = path.name.partition("_")
    return original_name if len(prefix) == 12 and separator else path.name


def urls() -> list[str]:
    raw = os.getenv("KNOWLEDGE_URLS", "") or os.getenv("CLIMATE_CHANGE", "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def html_text(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    parser = TextExtractor()
    parser.feed(response.text)
    return " ".join(parser.parts)


def main() -> None:
    documents = configured_documents()
    stored_chunks = store_documents(documents, recreate=True)
    print(f"Processed {len(documents)} sources and stored {stored_chunks} chunks in {INDEX_PATH}.")


if __name__ == "__main__":
    main()
