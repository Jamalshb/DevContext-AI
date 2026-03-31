"""
app.py

Streamlit frontend for DevContext: AI Onboarding Agent.

Features:
- Sidebar repo ingestion using ingest_repo() from ingestor.py
- ChatGPT-like chat UI (st.chat_message + st.chat_input)
- Context-grounded Q&A over persisted Chroma DB (./chroma_db) using LangChain
  ConversationalRetrievalChain + ChatOpenAI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from ingestor import DEFAULT_PERSIST_DIRECTORY, IngestionError, ingest_repo


@dataclass(frozen=True)
class AppConfig:
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY
    model_name: str = "gpt-4o-mini"  # fallback handled at runtime if unavailable
    k: int = 4


def _require_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or a .env file."
        )


def _langchain_imports():
    """
    Resolve LangChain imports with fallbacks across versions.
    """

    # Embeddings
    try:
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # type: ignore
    except Exception:  # pragma: no cover
        from langchain.embeddings.openai import OpenAIEmbeddings  # type: ignore
        from langchain.chat_models import ChatOpenAI  # type: ignore

    # Vector store
    try:
        from langchain_community.vectorstores import Chroma  # type: ignore
    except Exception:  # pragma: no cover
        from langchain.vectorstores import Chroma  # type: ignore

    # Chain
    try:
        from langchain.chains import ConversationalRetrievalChain  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to import ConversationalRetrievalChain: {e}") from e

    # Prompt
    try:
        from langchain_core.prompts import PromptTemplate  # type: ignore
    except Exception:  # pragma: no cover
        from langchain.prompts import PromptTemplate  # type: ignore

    return OpenAIEmbeddings, ChatOpenAI, Chroma, ConversationalRetrievalChain, PromptTemplate


def _get_chain(cfg: AppConfig):
    """
    Lazily constructs (and caches) the retrieval chain against the persisted Chroma DB.
    """

    if "qa_chain" in st.session_state and st.session_state["qa_chain"] is not None:
        return st.session_state["qa_chain"]

    _require_openai_key()

    OpenAIEmbeddings, ChatOpenAI, Chroma, ConversationalRetrievalChain, PromptTemplate = (
        _langchain_imports()
    )

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=cfg.persist_directory, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": cfg.k})

    qa_prompt = PromptTemplate.from_template(
        """You are DevContext, an AI Developer Onboarding Agent.
Answer the user's question using ONLY the provided codebase context.

Rules:
- If the context does not contain the answer, say: "I don't know based on the indexed repository."
- Do not invent APIs, files, or behavior that is not present in context.
- When helpful, reference filenames/paths mentioned in the context.

Context:
{context}

Question:
{question}

Answer:"""
    )

    llm = ChatOpenAI(model=cfg.model_name, temperature=0)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )

    st.session_state["qa_chain"] = chain
    return chain


def _init_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("chat_history", [])  # List[Tuple[str, str]]
    st.session_state.setdefault("repo_status", None)  # Optional[str]
    st.session_state.setdefault("qa_chain", None)


def _set_dark_theme_css() -> None:
    st.markdown(
        """
<style>
/* App background + typography */
.stApp {
  background: radial-gradient(1200px 800px at 10% 0%, #111827 0%, #0B1220 35%, #050A14 100%);
  color: #E5E7EB;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0B1220 0%, #060B14 100%);
  border-right: 1px solid rgba(255, 255, 255, 0.06);
}

/* Chat bubbles */
div[data-testid="stChatMessage"] {
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(17, 24, 39, 0.55);
  border-radius: 14px;
}

/* Inputs */
div[data-baseweb="input"] input, textarea {
  background-color: rgba(17, 24, 39, 0.65) !important;
  color: #E5E7EB !important;
  border-color: rgba(255, 255, 255, 0.12) !important;
}

/* Buttons */
button[kind="primary"] {
  background: linear-gradient(90deg, #2563EB 0%, #7C3AED 100%) !important;
  border: none !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(cfg: AppConfig) -> None:
    with st.sidebar:
        st.markdown("### DevContext Controls")
        repo_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/org/repo.git")

        col1, col2 = st.columns([1, 1])
        with col1:
            process = st.button("Process Repo", type="primary", use_container_width=True)
        with col2:
            clear = st.button("Clear Chat", use_container_width=True)

        if clear:
            st.session_state["messages"] = []
            st.session_state["chat_history"] = []
            st.session_state["repo_status"] = None

        if process:
            if not repo_url.strip():
                st.error("Please enter a GitHub repository URL.")
                return

            with st.spinner("Cloning and indexing repository..."):
                try:
                    msg = ingest_repo(repo_url.strip())
                    st.session_state["repo_status"] = msg
                    # Invalidate chain so next question reloads the updated DB
                    st.session_state["qa_chain"] = None
                    st.success(msg)
                except IngestionError as e:
                    st.session_state["repo_status"] = None
                    st.error(str(e))
                except Exception as e:
                    st.session_state["repo_status"] = None
                    st.error(f"Unexpected error: {e}")

        st.markdown("---")
        st.caption(f"Chroma DB: `{cfg.persist_directory}`")
        st.caption(f"Model: `{cfg.model_name}`")


def _render_header() -> None:
    st.markdown("### DevContext: AI Onboarding Agent")
    st.caption("Ask questions about the indexed repository. Answers are grounded strictly in retrieved code context.")


def _render_chat(cfg: AppConfig) -> None:
    # Display prior messages
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_question = st.chat_input("Ask a question about the repo…")
    if not user_question:
        return

    # Add user message
    st.session_state["messages"].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Answer
    with st.chat_message("assistant"):
        with st.spinner("Searching the codebase context..."):
            try:
                chain = _get_chain(cfg)

                result: Dict[str, Any] = chain(
                    {
                        "question": user_question,
                        "chat_history": st.session_state["chat_history"],
                    }
                )

                answer = (result.get("answer") or "").strip()
                if not answer:
                    answer = "I don't know based on the indexed repository."

                st.markdown(answer)
                st.session_state["messages"].append({"role": "assistant", "content": answer})
                st.session_state["chat_history"].append((user_question, answer))
            except Exception as e:
                err = f"Error while answering: {e}"
                st.error(err)
                st.session_state["messages"].append({"role": "assistant", "content": err})


def main() -> None:
    cfg = AppConfig()

    st.set_page_config(
        page_title="DevContext: AI Onboarding Agent",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_session_state()
    _set_dark_theme_css()
    _render_sidebar(cfg)
    _render_header()
    _render_chat(cfg)


if __name__ == "__main__":
    main()

