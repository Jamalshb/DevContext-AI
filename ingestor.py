"""
ingestor.py

Backend ingestion pipeline for an AI Developer Onboarding Agent.

What it does:
- Clones a GitHub repo to a temporary folder (GitPython)
- Loads .py/.js/.ts/.md files (LangChain GenericLoader + LanguageParser)
- Splits documents into chunks (RecursiveCharacterTextSplitter)
- Embeds chunks (OpenAIEmbeddings) and persists to Chroma at ./chroma_db

Environment:
- Expects OPENAI_API_KEY in your environment or a local .env file.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".py", ".js", ".ts", ".md")
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_PERSIST_DIRECTORY = "./chroma_db"


class IngestionError(RuntimeError):
    """Raised when the ingestion pipeline fails."""


@dataclass(frozen=True)
class IngestConfig:
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY


def _require_openai_key() -> None:
    """
    Ensures OPENAI_API_KEY is available after loading dotenv.

    Note: We load dotenv here (and also in __main__) so ingest_repo()
    works when imported and called programmatically.
    """

    if load_dotenv is None:
        raise IngestionError(
            "python-dotenv is not installed. Install it with: pip install python-dotenv"
        )

    load_dotenv(override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise IngestionError(
            "OPENAI_API_KEY is not set. Add it to your environment or a .env file."
        )


def clone_repo_to_temp(repo_url: str) -> Tuple[Path, tempfile.TemporaryDirectory]:
    """
    Clone the provided GitHub repository URL into a temporary directory.

    Returns:
        (repo_path, temp_dir_handle)

    Important:
        Keep the returned TemporaryDirectory handle alive for as long as you
        need the cloned repository on disk.
    """

    if not repo_url or not isinstance(repo_url, str):
        raise IngestionError("repo_url must be a non-empty string.")

    tmp = tempfile.TemporaryDirectory(prefix="repo_ingest_")
    repo_path = Path(tmp.name) / "repo"

    try:
        try:
            from git import Repo  # GitPython
        except Exception as e:
            raise IngestionError(
                "GitPython is not installed. Install it with: pip install GitPython"
            ) from e

        Repo.clone_from(repo_url, str(repo_path))
    except Exception as e:
        # Ensure temp dir is cleaned up if clone fails.
        try:
            tmp.cleanup()
        except Exception:
            pass
        raise IngestionError(f"Failed to clone repository: {e}") from e

    return repo_path, tmp


def _safe_rmtree(path: Path) -> None:
    """Best-effort directory removal."""

    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _langchain_imports():
    """
    LangChain has changed module paths across versions.
    This helper resolves the required symbols with fallbacks.
    """

    # GenericLoader
    try:
        from langchain_community.document_loaders.generic import GenericLoader  # type: ignore
    except Exception:  # pragma: no cover
        from langchain.document_loaders.generic import GenericLoader  # type: ignore

    # LanguageParser
    try:
        from langchain_community.document_loaders.parsers import LanguageParser  # type: ignore
    except Exception:  # pragma: no cover
        from langchain.document_loaders.parsers import LanguageParser  # type: ignore

    # RecursiveCharacterTextSplitter + Language enum (optional)
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
    except Exception:  # pragma: no cover
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

    try:
        from langchain.text_splitter import Language  # type: ignore
    except Exception:  # pragma: no cover
        Language = None  # type: ignore

    # OpenAIEmbeddings
    try:
        from langchain_openai import OpenAIEmbeddings  # type: ignore
    except Exception:  # pragma: no cover
        from langchain.embeddings.openai import OpenAIEmbeddings  # type: ignore

    # Chroma
    try:
        from langchain_community.vectorstores import Chroma  # type: ignore
    except Exception:  # pragma: no cover
        from langchain.vectorstores import Chroma  # type: ignore

    return GenericLoader, LanguageParser, RecursiveCharacterTextSplitter, Language, OpenAIEmbeddings, Chroma


def load_repo_documents(repo_path: Path) -> List["Document"]:
    """
    Load supported source files from repo_path as LangChain Documents.

    Uses GenericLoader + LanguageParser per file-type so code-aware parsing
    can extract useful structure where supported.
    """

    if not repo_path.exists():
        raise IngestionError(f"Repository path does not exist: {repo_path}")

    GenericLoader, LanguageParser, _, Language, _, _ = _langchain_imports()

    loaders = []

    # Prefer Language enum when available; otherwise, fall back to LanguageParser defaults.
    def _parser_for(ext: str):
        if Language is None:
            return LanguageParser()
        mapping = {
            ".py": getattr(Language, "PYTHON", None),
            ".js": getattr(Language, "JS", None) or getattr(Language, "JAVASCRIPT", None),
            ".ts": getattr(Language, "TS", None) or getattr(Language, "TYPESCRIPT", None),
            ".md": getattr(Language, "MARKDOWN", None),
        }
        lang = mapping.get(ext)
        return LanguageParser(language=lang) if lang is not None else LanguageParser()

    for ext in SUPPORTED_EXTENSIONS:
        loaders.append(
            GenericLoader.from_filesystem(
                str(repo_path),
                glob=f"**/*{ext}",
                suffixes=[ext],
                parser=_parser_for(ext),
            )
        )

    documents: List["Document"] = []
    for loader in loaders:
        try:
            documents.extend(loader.load())
        except Exception as e:
            raise IngestionError(f"Failed while loading documents: {e}") from e

    if not documents:
        raise IngestionError(
            f"No documents found with extensions: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    return documents


def split_documents(
    documents: Sequence["Document"],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List["Document"]:
    """Split loaded documents into chunks using RecursiveCharacterTextSplitter."""

    if chunk_size <= 0:
        raise IngestionError("chunk_size must be > 0.")
    if chunk_overlap < 0:
        raise IngestionError("chunk_overlap must be >= 0.")
    if chunk_overlap >= chunk_size:
        raise IngestionError("chunk_overlap must be < chunk_size.")

    _, _, RecursiveCharacterTextSplitter, _, _, _ = _langchain_imports()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    try:
        return splitter.split_documents(list(documents))
    except Exception as e:
        raise IngestionError(f"Failed while splitting documents: {e}") from e


def embed_and_persist(
    chunks: Sequence["Document"],
    *,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
) -> int:
    """
    Embed chunks and persist them to Chroma.

    Returns:
        The number of chunks embedded/persisted.
    """

    if not chunks:
        raise IngestionError("No chunks to embed.")

    _require_openai_key()

    _, _, _, _, OpenAIEmbeddings, Chroma = _langchain_imports()

    try:
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        raise IngestionError(f"Failed to initialize OpenAIEmbeddings: {e}") from e

    try:
        vectordb = Chroma.from_documents(
            documents=list(chunks),
            embedding=embeddings,
            persist_directory=persist_directory,
        )
        # Some versions persist automatically; calling persist() is safe when present.
        if hasattr(vectordb, "persist"):
            vectordb.persist()
    except Exception as e:
        raise IngestionError(f"Failed while creating/persisting Chroma DB: {e}") from e

    return len(chunks)


def ingest_repo(repo_url: str, config: Optional[IngestConfig] = None) -> str:
    """
    End-to-end ingestion pipeline.

    Steps:
    1) Clone repo into a temp directory
    2) Load .py/.js/.ts/.md files using GenericLoader + LanguageParser
    3) Split into chunks (1000/200 by default)
    4) Embed and persist chunks into Chroma at ./chroma_db

    Returns:
        A success message string, or raises IngestionError on failure.
    """

    cfg = config or IngestConfig()

    repo_path = None
    tmp_handle: Optional[tempfile.TemporaryDirectory] = None

    try:
        repo_path, tmp_handle = clone_repo_to_temp(repo_url)
        documents = load_repo_documents(repo_path)
        chunks = split_documents(
            documents,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        count = embed_and_persist(
            chunks,
            persist_directory=cfg.persist_directory,
        )
        return (
            f"Success: ingested {count} chunks from {repo_url} "
            f"into Chroma persist_directory='{cfg.persist_directory}'."
        )
    except IngestionError:
        raise
    except Exception as e:
        raise IngestionError(f"Unexpected ingestion failure: {e}") from e
    finally:
        # Ensure temp directory is cleaned up.
        try:
            if tmp_handle is not None:
                tmp_handle.cleanup()
        except Exception:
            # Best-effort cleanup; don't mask ingestion outcome.
            pass


def _print_usage() -> None:
    print("Usage: python ingestor.py <github_repo_url>")
    print("Example: python ingestor.py https://github.com/user/repo.git")


def main(argv: Sequence[str]) -> int:
    if load_dotenv is None:
        print(
            "Error: python-dotenv is not installed. Install it with: pip install python-dotenv",
            file=sys.stderr,
        )
        return 1

    load_dotenv(override=False)

    if len(argv) != 2:
        _print_usage()
        return 2

    repo_url = argv[1].strip()
    try:
        msg = ingest_repo(repo_url)
        print(msg)
        return 0
    except IngestionError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

