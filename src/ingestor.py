"""Clone a GitHub repo, read its source files, split them into chunks and store them in Chroma."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from git import Repo
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_BASE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    IGNORED_DIRECTORIES,
    MAX_FILE_BYTES,
    SUPPORTED_EXTENSIONS,
)
from src.utils import normalize_repo_url, repo_slug

_LANGUAGE_BY_EXT = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".md": Language.MARKDOWN,
    ".html": Language.HTML,
}

ProgressFn = Callable[[str], None]


class IngestionError(RuntimeError):
    pass


@dataclass(frozen=True)
class IngestConfig:
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    base_directory: str = CHROMA_BASE_DIR


@dataclass(frozen=True)
class IngestResult:
    persist_directory: str
    collection_name: str
    num_files: int
    num_chunks: int


def clone_repo(repo_url: str, dest: Path) -> Path:
    try:
        Repo.clone_from(normalize_repo_url(repo_url), str(dest), depth=1)
    except ValueError as e:
        raise IngestionError(str(e)) from e
    except Exception as e:
        raise IngestionError(f"Failed to clone repository: {e}") from e
    return dest


def load_repo_documents(repo_path: Path) -> List[Document]:
    if not repo_path.exists():
        raise IngestionError(f"Repository path does not exist: {repo_path}")

    documents: List[Document] = []
    for path in sorted(repo_path.rglob("*")):
        rel = path.relative_to(repo_path)
        if any(part in IGNORED_DIRECTORIES for part in rel.parts):
            continue
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if not text.strip():
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": rel.as_posix(), "extension": path.suffix.lower()},
            )
        )

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
    """Split each file with a splitter that understands its language (functions, classes...)."""
    chunks: List[Document] = []
    for doc in documents:
        lang = _LANGUAGE_BY_EXT.get(doc.metadata.get("extension", ""))
        if lang is not None:
            splitter = RecursiveCharacterTextSplitter.from_language(
                lang, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        for chunk in splitter.split_documents([doc]):
            # Put the file path in the text so the model always knows where code comes from.
            chunk.page_content = f"File: {chunk.metadata['source']}\n\n{chunk.page_content}"
            chunks.append(chunk)
    return chunks


def embed_and_persist(
    chunks: Sequence[Document],
    *,
    embeddings,
    persist_directory: str,
    collection_name: str,
) -> int:
    if not chunks:
        raise IngestionError("No chunks to embed.")
    try:
        store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
        store.delete_collection()  # re-indexing replaces old data instead of duplicating it
        Chroma.from_documents(
            documents=list(chunks),
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
    except Exception as e:
        raise IngestionError(f"Failed while creating embeddings / Chroma DB: {e}") from e
    return len(chunks)


def ingest_repo(
    repo_url: str,
    config: Optional[IngestConfig] = None,
    *,
    embeddings=None,
    progress: Optional[ProgressFn] = None,
) -> IngestResult:
    cfg = config or IngestConfig()
    say = progress or (lambda _msg: None)
    if embeddings is None:
        from src.llm import get_embeddings

        embeddings = get_embeddings()

    try:
        slug = repo_slug(repo_url)
    except ValueError as e:
        raise IngestionError(str(e)) from e
    persist_directory = str(Path(cfg.base_directory) / slug)

    with tempfile.TemporaryDirectory(prefix="repo_ingest_") as tmp:
        say("Cloning repository...")
        repo_path = clone_repo(repo_url, Path(tmp) / "repo")
        say("Reading files...")
        documents = load_repo_documents(repo_path)

    say(f"Splitting {len(documents)} files...")
    chunks = split_documents(
        documents, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
    )
    say(f"Creating embeddings for {len(chunks)} chunks (this can take a while)...")
    embed_and_persist(
        chunks,
        embeddings=embeddings,
        persist_directory=persist_directory,
        collection_name=slug,
    )
    return IngestResult(persist_directory, slug, len(documents), len(chunks))
