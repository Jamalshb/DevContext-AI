import os

try:
    import streamlit as st
except Exception:
    st = None


def get_secret(key: str, default=None):
    # أولًا: Streamlit secrets
    try:
        if st is not None and hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    # ثانيًا: Environment variables
    return os.getenv(key, default)


MODEL_PROVIDER = str(get_secret("MODEL_PROVIDER", "ollama")).lower()

CHROMA_BASE_DIR = "data/chroma"
TEMP_REPO_DIR = ".temp_repo"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

SUPPORTED_EXTENSIONS = [".py", ".js", ".ts", ".md"]

IGNORED_DIRECTORIES = {
    ".git",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "venv",
    ".venv",
}

# Ollama local
OLLAMA_CHAT_MODEL = get_secret("OLLAMA_CHAT_MODEL", "llama3.2:1b")
OLLAMA_EMBED_MODEL = get_secret("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = get_secret("OLLAMA_BASE_URL", "http://localhost:11434")

# OpenAI cloud
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = get_secret("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = get_secret("OPENAI_EMBED_MODEL", "text-embedding-3-small")