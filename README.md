# Darukaa.Earth Biodiversity Intelligence

FastAPI RAG application that turns the PDFs in `data/raw/` into grounded biodiversity and land-management answers. It intentionally uses a small local TF-IDF index—not BAAI/BGE, Chroma, or a paid embedding API—so it fits Render's free-tier memory limits.

## Architecture and knowledge schema

`ingest.py` extracts text from each PDF, splits it into overlapping 650-word chunks, calculates sparse normalized TF-IDF vectors, and writes `data/tfidf_index.json`. `rag_engine.py` ranks the six most relevant chunks with cosine similarity, puts their source names and the recent session context in the model prompt, and requires scientific reasoning, metrics, sources, and a time horizon. The index schema is `idf`, `documents`, `sources`, and `vectors`; it has no external database or embedding dependency.

`app.py` provides the HTML chat UI plus `/api/chat` and `/health`. The browser retains a random session id and the server keeps its last 12 messages, enabling follow-up questions. Incomplete land context receives a targeted clarifying question.

## Models and fallback

Generation uses `gemini-2.5-flash-lite` first and falls back automatically to Groq `llama-3.3-70b-versatile`. Neither is used for embeddings. Set both API keys so a quota/provider failure does not stop the chat.

## Local setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python ingest.py
python start.py
```

Open `http://localhost:8501`. Put the environmental PDFs in `data/raw/`; the hackathon brief remains outside the answer corpus.

## Render deployment and CI/CD

Commit the project (but never `.env`) to GitHub, create a Render Blueprint from `render.yaml`, then supply `GOOGLE_API_KEY` and `GROQ_API_KEY` in Render's environment settings. Render runs `pip install -r requirements.txt`, then `python start.py`; the startup process rebuilds the local index if necessary and exposes `/health` for health checks. A Git push to the connected branch triggers deployment. For CI, run `python -m compileall app.py ingest.py rag_engine.py memory.py start.py` and a retrieval smoke test before merging.

Runtime PDF uploads are not persisted by Render free instances. Keep the required public corpus in Git, or attach a persistent disk/object storage if user uploads must survive redeploys.
