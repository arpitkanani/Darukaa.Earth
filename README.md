# Darukaa.Earth Biodiversity Intelligence Chatbot

This is a small retrieval-augmented Streamlit application for evidence-backed
land and ecosystem advice. `ingest.py` extracts the supplied PDFs, optionally
downloads configured HTML sources, creates hosted Gemini embeddings, and stores
them in persistent ChromaDB.
`rag_engine.py` retrieves six passages for every substantive answer, adds the
last six conversation messages, and streams from Groq with Gemini fallback.
The prompt requires every recommendation to connect at least three ecological
metrics and to use a fixed evidence/source/time-horizon format.

The left sidebar accepts one PDF at a time, up to 20 MB. Clicking **Ingest PDF**
stores it in `data/uploads/`, extracts its text, embeds its chunks, and upserts
them into the existing Chroma collection alongside the checked-in corpus. Only
PDF files are accepted. Uploaded PDFs are also included if you later rebuild
the index with `python ingest.py`.

## Run locally

Use Python 3.11 or newer:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `GROQ_API_KEY` and either
`GEMINI_API_KEY` or the existing `GOOGLE_API_KEY`. `KNOWLEDGE_URLS` accepts
comma-separated HTML URLs; the provided FAO URL can remain in that setting.
Gemini embeddings use the hosted `gemini-embedding-001` model, so the
application does not download or load a local sentence-transformers model.
Groq is tried first when `GROQ_API_KEY` is present. If Gemini returns a
temporary `503 UNAVAILABLE` response because of provider demand, retry later or
set `GEMINI_MODEL` to another model available to your API key (the default is
`gemini-2.5-flash`). The app now
displays an actionable error instead of exposing the provider traceback.
The hackathon brief PDF is documentation for the challenge and is intentionally
not added to the scientific answer corpus.

Build or rebuild the local index (this clears and recreates the collection):

```bash
python ingest.py
streamlit run app.py
```

Run `python ingest.py` once after upgrading from the BGE version. The Gemini
index uses a different vector size and is stored in a new collection, so the
old BGE collection cannot be queried with the new embeddings. The rebuild also
removes the old BGE collection from the local Chroma database.

The first ingestion calls the Gemini Embeddings API and may take a few minutes
depending on API quota and network speed. `chroma_db/` is generated locally and
ignored by Git. Streamlit keeps conversation memory only in the current browser
session.

## Chunking strategy

`ingest.py` normalizes whitespace, splits each source into word-based chunks of
up to 650 words, and uses 100 words of overlap between neighboring chunks
(a 550-word stride). The same strategy is used for checked-in PDFs, configured
HTML sources, and sidebar-uploaded PDFs. Retrieval embeds documents and user
queries with Gemini's hosted `gemini-embedding-001` model using 768-dimensional
vectors, then returns the six closest chunks using cosine similarity. The
Chroma collection is named
`biodiversity_knowledge_gemini` because changing embedding dimensions requires a
fresh index.

## JSON question examples

The **Structured input** panel accepts a JSON object with a required
`question` field and optional land context. Click **Ask JSON question** to
submit it directly; you do not need to copy the question into the chat box.

```json
{
  "question": "How can I improve soil health while protecting biodiversity?",
  "soil_organic_carbon": 0.3,
  "rainfall": "low and seasonal",
  "water_availability": "limited",
  "crop": "wheat",
  "region": "semi-arid",
  "land_use": "intensive agriculture",
  "human_impact": "overgrazing nearby"
}
```

Additional JSON questions to test:

```json
{
  "question": "What practices can reduce erosion and improve water retention?",
  "soil_type": "sandy loam",
  "rainfall": "irregular",
  "crop": "maize",
  "region": "dry tropical",
  "land_use": "rain-fed farming"
}
```

```json
{
  "question": "What restoration plan would improve biodiversity on degraded land?",
  "soil_condition": "low organic matter",
  "rainfall": "moderate",
  "water_availability": "seasonal",
  "land_use": "overgrazed pasture",
  "region": "semi-arid",
  "human_impact": "grazing pressure and fuelwood collection"
}
```

## Render deployment

Create a Python web service with build command `pip install -r requirements.txt`
and start command `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`.
Set the API keys and `KNOWLEDGE_URLS` as Render environment variables. Run
`python ingest.py` during the build (or run it once in a persistent disk-backed
deployment); the embedding model is CPU-only and intentionally lightweight.
Because Chroma and the model are local files, use persistent storage if the
service is expected to survive redeploys without re-ingestion.

For this project, the recommended Render settings are:

- **Build command:** `pip install -r requirements.txt && python ingest.py`
- **Start command:** `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`
- **Environment variables:** `GROQ_API_KEY`, `GEMINI_API_KEY` or
  `GOOGLE_API_KEY`, `GEMINI_MODEL`, `GEMINI_EMBEDDING_MODEL`, and
  `KNOWLEDGE_URLS`
- **Persistent disk:** mount it at the project data location if uploaded PDFs
  and the Chroma index must survive redeploys.

## Submission information

The hackathon submission asks for one Word document containing the project
links and a short explanation of how the project works. Create that document
from the project details below before submitting.

Before submitting, replace these placeholders in the Word document:

- GitHub repository: `https://github.com/<your-account>/<your-repository>`
- Live demo: `https://<your-service>.onrender.com`

The document covers the architecture, database/schema, local setup, Render
deployment and CI/CD notes, and the information needed to verify the project.

## Dataset and deployment recommendation

Keep the small, non-sensitive source PDFs in the GitHub repository when their
size and licensing allow it. This makes the project reproducible: Render can
run `python ingest.py` during deployment and rebuild the Chroma index from the
same sources.

Do not commit API keys, private documents, or user-uploaded PDFs. Sidebar
uploads are runtime data stored under `data/uploads/`. On Render, use a
persistent disk for `data/` if those uploads must remain available after a
restart. For larger or private datasets, store the files in private object
storage and download them during a controlled ingestion step instead of
putting them in GitHub. Keep `.env` local and configure secrets through Render
environment variables.

## CI/CD

The repository can be connected to Render so a push to the configured branch
triggers a new deployment. Each deployment installs the dependencies, calls Gemini to rebuild the local
knowledge index, and starts Streamlit. For a production setup, add a
CI check that runs Python compilation and a smoke test before Render is
allowed to deploy.
