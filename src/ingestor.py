from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from git import Repo
from langchain_core.documents import Document
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_BASE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    SUPPORTED_EXTENSIONS,
)

DEFAULT_PERSIST_DIRECTORY = CHROMA_BASE_DIR


class IngestionError(RuntimeError):
    pass


@dataclass(frozen=True)
class IngestConfig:
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY


def clone_repo_to_temp(repo_url: str) -> Tuple[Path, tempfile.TemporaryDirectory]:
    if not repo_url or not isinstance(repo_url, str):
        raise IngestionError("repo_url must be a non-empty string.")

    tmp = tempfile.TemporaryDirectory(prefix="repo_ingest_")
    repo_path = Path(tmp.name) / "repo"

    try:
        Repo.clone_from(repo_url, str(repo_path))
    except Exception as e:
        try:
            tmp.cleanup()
        except Exception:
            pass
        raise IngestionError(f"Failed to clone repository: {e}") from e

    return repo_path, tmp


def load_repo_documents(repo_path: Path) -> List[Document]:
    if not repo_path.exists():
        raise IngestionError(f"Repository path does not exist: {repo_path}")

    loaders = []

    for ext in SUPPORTED_EXTENSIONS:
        loaders.append(
            GenericLoader.from_filesystem(
                str(repo_path),
                glob=f"**/*{ext}",
                suffixes=[ext],
                parser=LanguageParser(),
            )
        )

    documents: List[Document] = []

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
    documents: Sequence[Document],
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    try:
        return splitter.split_documents(list(documents))
    except Exception as e:
        raise IngestionError(f"Failed while splitting documents: {e}") from e


def _build_embeddings():
    return FakeEmbeddings(size=1536)


def embed_and_persist(
    chunks: Sequence[Document],
    *,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
) -> int:
    if not chunks:
        raise IngestionError("No chunks to embed.")

    try:
        embeddings = _build_embeddings()

        vectordb = Chroma.from_documents(
            documents=list(chunks),
            embedding=embeddings,
            persist_directory=persist_directory,
        )

        if hasattr(vectordb, "persist"):
            vectordb.persist()

    except Exception as e:
        raise IngestionError(f"Failed while creating Chroma DB: {e}") from e

    return len(chunks)


def ingest_repo(repo_url: str, config: Optional[IngestConfig] = None) -> str:
    cfg = config or IngestConfig()
    tmp_handle: Optional[tempfile.TemporaryDirectory] = None

    try:
        repo_path, tmp_handle = clone_repo_to_temp(repo_url)

        documents = load_repo_documents(repo_path)

        chunks = split_documents(
            documents,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )

        embed_and_persist(
            chunks,
            persist_directory=cfg.persist_directory,
        )

        return cfg.persist_directory

    except IngestionError:
        raise
    except Exception as e:
        raise IngestionError(f"Unexpected ingestion failure: {e}") from e

    finally:
        try:
            if tmp_handle is not None:
                tmp_handle.cleanup()
        except Exception:
            pass