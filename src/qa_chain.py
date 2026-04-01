from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL, OLLAMA_EMBED_MODEL


def build_qa_chain(persist_directory: str):
    # embeddings
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    # vector store
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    # retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    # LLM
    llm = ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    def ask_question(payload: dict):
        question = payload["question"]
        chat_history = payload.get("chat_history", [])

        # retrieve documents
        docs = retriever.invoke(question)

        # 🔥 خلي README أولًا
        docs = sorted(
            docs,
            key=lambda d: "readme" not in d.metadata.get("source", "").lower()
        )

        # build context
        context = "\n\n".join(doc.page_content for doc in docs)

        # chat history
        history_text = "\n".join(
            [f"User: {q}\nAssistant: {a}" for q, a in chat_history]
        )

        # prompt
        prompt = f"""
You are an expert AI assistant that analyzes GitHub repositories.

IMPORTANT:
- This project uses Ollama locally (NOT OpenAI)
- Always explain the project clearly
- Even if context is partial, try to infer the purpose

Chat history:
{history_text}

Context:
{context}

Question:
{question}
"""

        # get response
        response = llm.invoke(prompt)

        return {
            "answer": response.content,
            "source_documents": docs,
        }

    return ask_question