"""Project settings. Every value can be overridden with an environment variable (or .env)."""
import os

from dotenv import load_dotenv

load_dotenv()

# --- Ollama (free, local) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# --- Storage ---
CHROMA_BASE_DIR = os.getenv("CHROMA_BASE_DIR", "data/chroma")

# --- Chunking / retrieval ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "6"))
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(300_000)))

SUPPORTED_EXTENSIONS = [
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".md",
    ".html", ".css", ".json", ".yml", ".yaml", ".toml", ".txt",
]

IGNORED_DIRECTORIES = {
    ".git", "__pycache__", "node_modules", "dist", "build",
    "venv", ".venv", ".idea", ".vscode", "target", ".next",
}
