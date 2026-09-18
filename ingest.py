"""Build the local Chroma knowledge base from PDFs and configured HTML sources."""

from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

# sentence-transformers only needs the PyTorch path here. Disabling optional
# TensorFlow imports avoids unrelated local TensorFlow/protobuf conflicts.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import chromadb
import requests
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
UPLOAD_DIR = ROOT / "data" / "uploads"
DB_DIR = ROOT / "chroma_db"
COLLECTION = "biodiversity_knowledge"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
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


@lru_cache(maxsize=1)
def embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def collection(*, recreate: bool = False):
    client = chromadb.PersistentClient(path=str(DB_DIR))
    existing = client.list_collections()
    existing_names = [item if isinstance(item, str) else item.name for item in existing]
    if recreate and COLLECTION in existing_names:
        client.delete_collection(COLLECTION)
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def store_documents(documents: list[tuple[str, str]], *, recreate: bool = False) -> int:
    target = collection(recreate=recreate)
    ids: list[str] = []
    texts: list[str] = []
    metadata: list[dict[str, str | int]] = []
    for source, text in documents:
        for index, chunk in enumerate(chunks(text)):
            ids.append(hashlib.sha1(f"{source}:{index}:{chunk}".encode()).hexdigest())
            texts.append(chunk)
            metadata.append({"source": source, "chunk_index": index})
    if texts:
        embeddings = embedding_model().encode(
            texts, batch_size=32, normalize_embeddings=True
        ).tolist()  # type: ignore
        target.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadata)
    return len(texts)


def ingest_pdf(path: Path, source: str | None = None) -> int:
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files can be ingested.")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("PDF files must be 20 MB or smaller.")
    text = pdf_text(path)
    if not clean(text):
        raise ValueError("The PDF does not contain extractable text.")
    return store_documents([(source or path.name, text)])


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
    load_dotenv(ROOT / ".env")
    documents: list[tuple[str, str]] = []
    for path in sorted(RAW_DIR.glob("*.pdf")):
        documents.append((path.name, pdf_text(path)))
    for path in sorted(UPLOAD_DIR.glob("*.pdf")):
        documents.append((uploaded_source(path), pdf_text(path)))
    for url in urls():
        source = Path(urlparse(url).path).name or urlparse(url).netloc
        documents.append((f"{source} ({url})", html_text(url)))

    stored_chunks = store_documents(documents, recreate=True)
    print(f"Processed {len(documents)} sources and stored {stored_chunks} chunks in {DB_DIR}.")


if __name__ == "__main__":
    main()
