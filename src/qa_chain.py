"""Retrieval-Augmented Generation: find relevant code chunks, then let the LLM answer from them."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config import RETRIEVAL_K

SYSTEM_PROMPT = """You are DevContext-AI, an expert software engineer who explains codebases.
Answer the user's question using ONLY the repository context below.
- Mention the file paths you used.
- Use code blocks when showing code.
- If the context does not contain the answer, say so honestly instead of guessing.
- Answer in the same language the user asked in.

Repository context:
{context}"""

ChatHistory = Sequence[Tuple[str, str]]


def format_context(docs: Sequence[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


def build_messages(question: str, docs: Sequence[Document], chat_history: ChatHistory = ()):
    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=format_context(docs)))]
    for q, a in list(chat_history)[-4:]:  # keep only recent turns so the prompt stays small
        messages += [HumanMessage(content=q), AIMessage(content=a)]
    messages.append(HumanMessage(content=question))
    return messages


class RepoQA:
    def __init__(self, persist_directory: str, collection_name: str, *, embeddings=None, llm=None, k: int = RETRIEVAL_K):
        if embeddings is None or llm is None:
            from src.llm import get_embeddings, get_llm

            embeddings = embeddings or get_embeddings()
            llm = llm or get_llm()
        self.llm = llm
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
        self.retriever = self.store.as_retriever(search_kwargs={"k": k})

    def retrieve(self, question: str) -> List[Document]:
        return self.retriever.invoke(question)

    def ask(self, question: str, chat_history: ChatHistory = ()) -> dict:
        docs = self.retrieve(question)
        if not docs:
            return {"answer": "No relevant code was found in the repository.", "source_documents": []}
        response = self.llm.invoke(build_messages(question, docs, chat_history))
        return {"answer": response.content, "source_documents": docs}

    def stream(self, question: str, docs: Sequence[Document], chat_history: ChatHistory = ()):
        """Yield the answer piece by piece (for a live typing effect in the UI)."""
        for chunk in self.llm.stream(build_messages(question, docs, chat_history)):
            if chunk.content:
                yield chunk.content


def build_qa_chain(persist_directory: str, collection_name: str, *, embeddings=None, llm=None, k: int = RETRIEVAL_K) -> RepoQA:
    return RepoQA(persist_directory, collection_name, embeddings=embeddings, llm=llm, k=k)
