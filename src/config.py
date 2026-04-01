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

OLLAMA_CHAT_MODEL = "llama3.2:1b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"