# DevContext_AI

Streamlit app + ingestion pipeline that clones a GitHub repo, indexes supported files into a persisted Chroma vector DB, and answers questions grounded in the retrieved code context.

## Requirements

- Python 3.10+
- An OpenAI API key

## Setup (Windows PowerShell)

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file (or set an environment variable):

```bash
copy .env.example .env
```

Then edit `.env` and set:

- `OPENAI_API_KEY`

## Run the app

```bash
streamlit run app.py
```

In the sidebar, paste a GitHub repo URL and click **Process Repo**. The Chroma DB is persisted to `./chroma_db` by default.

## Run ingestion from CLI (optional)

```bash
python ingestor.py https://github.com/org/repo.git
```

## Project layout

- `app.py`: Streamlit UI + chat/retrieval chain
- `ingestor.py`: clone/load/split/embed/persist pipeline
- `data/chroma/`: reserved location for vector DB storage (optional)
- `tests/`: test suite (placeholder)
