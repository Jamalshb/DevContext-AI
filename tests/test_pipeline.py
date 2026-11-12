"""Tests run fully offline: fake embeddings + fake LLM, no Ollama or internet needed."""
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from src import ingestor
from src.ingestor import IngestConfig, IngestionError, load_repo_documents, split_documents
from src.llm import has_model
from src.qa_chain import build_messages, build_qa_chain
from src.utils import extract_repo_name, normalize_repo_url, repo_slug


def make_repo(root: Path) -> Path:
    (root / "src").mkdir(parents=True)
    (root / "node_modules" / "lib").mkdir(parents=True)
    (root / "README.md").write_text("# Demo\nA demo project.")
    (root / "src" / "auth.py").write_text("def login(user, password):\n    return user == 'admin'\n")
    (root / "node_modules" / "lib" / "x.js").write_text("ignored()")
    (root / "image.png").write_bytes(b"\x89PNG")
    return root


# ---------- utils ----------
@pytest.mark.parametrize("url,expected", [
    ("https://github.com/Owner/My-Repo", "my-repo"),
    ("https://github.com/owner/repo.git", "repo"),
    ("owner/repo", "repo"),
])
def test_extract_repo_name(url, expected):
    assert extract_repo_name(url) == expected


def test_repo_slug_is_unique_per_owner():
    assert repo_slug("https://github.com/a/repo") != repo_slug("https://github.com/b/repo")
    assert repo_slug("owner/repo.git") == "owner__repo"


@pytest.mark.parametrize("bad", ["", "not a url", "https://github.com/onlyowner"])
def test_normalize_rejects_bad_urls(bad):
    with pytest.raises(ValueError):
        normalize_repo_url(bad)


def test_has_model():
    assert has_model(["llama3.2:latest"], "llama3.2")
    assert not has_model(["mistral:latest"], "llama3.2")


# ---------- ingestion ----------
def test_load_skips_ignored_and_unsupported(tmp_path):
    docs = load_repo_documents(make_repo(tmp_path))
    assert sorted(d.metadata["source"] for d in docs) == ["README.md", "src/auth.py"]


def test_load_empty_repo_raises(tmp_path):
    with pytest.raises(IngestionError):
        load_repo_documents(tmp_path)


def test_split_adds_file_path(tmp_path):
    chunks = split_documents(load_repo_documents(make_repo(tmp_path)), chunk_size=200, chunk_overlap=0)
    assert all(c.page_content.startswith("File: ") for c in chunks)


def test_ingest_and_ask_end_to_end(tmp_path, monkeypatch):
    src_repo = make_repo(tmp_path / "source")

    def fake_clone(_url, dest):
        import shutil
        shutil.copytree(src_repo, dest)
        return dest

    monkeypatch.setattr(ingestor, "clone_repo", fake_clone)
    emb = DeterministicFakeEmbedding(size=64)
    cfg = IngestConfig(base_directory=str(tmp_path / "db"))

    result = ingestor.ingest_repo("owner/demo", cfg, embeddings=emb)
    assert result.num_files == 2 and result.num_chunks >= 2

    # Re-indexing must replace, not duplicate.
    again = ingestor.ingest_repo("owner/demo", cfg, embeddings=emb)
    qa = build_qa_chain(again.persist_directory, again.collection_name,
                        embeddings=emb, llm=FakeListChatModel(responses=["login is in src/auth.py"]), k=10)
    assert qa.store._collection.count() == result.num_chunks

    out = qa.ask("Where is login?")
    assert out["answer"] == "login is in src/auth.py"
    assert {d.metadata["source"] for d in out["source_documents"]} == {"README.md", "src/auth.py"}


def test_build_messages_includes_context_and_history():
    from langchain_core.documents import Document

    msgs = build_messages("q2", [Document(page_content="File: a.py\n\nCODE")], [("q1", "a1")])
    assert "CODE" in msgs[0].content
    assert [m.content for m in msgs[1:]] == ["q1", "a1", "q2"]
