"""Free local models via Ollama."""
import json
import urllib.request

from src.config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL, OLLAMA_EMBED_MODEL


def get_embeddings(model: str = OLLAMA_EMBED_MODEL):
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=model, base_url=OLLAMA_BASE_URL)


def get_llm(model: str = OLLAMA_CHAT_MODEL, temperature: float = 0.1):
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=temperature)


def list_ollama_models(timeout: float = 2.0) -> list[str]:
    """Return installed model names, or raise ConnectionError if Ollama is not running."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout) as r:
            data = json.load(r)
    except Exception as e:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. Is it running? (run: ollama serve)"
        ) from e
    return [m.get("name", "") for m in data.get("models", [])]


def has_model(installed: list[str], wanted: str) -> bool:
    """'llama3.2' matches 'llama3.2:latest'."""
    return any(n == wanted or n.split(":")[0] == wanted for n in installed)
