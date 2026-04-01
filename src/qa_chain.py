from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL, OLLAMA_EMBED_MODEL


def build_qa_chain(persist_directory: str):
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    def ask_question(payload: dict):
        question = payload["question"]
        chat_history = payload.get("chat_history", [])

        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        history_text = "\n".join(
            [f"User: {q}\nAssistant: {a}" for q, a in chat_history]
        )

        prompt = f"""
You are a helpful AI assistant for understanding a GitHub repository.

Use only the provided context to answer.
If the answer is not in the context, say you do not know.

Chat history:
{history_text}

Context:
{context}

Question:
{question}
"""

        response = llm.invoke(prompt)

        return {
            "answer": response.content,
            "source_documents": docs,
        }

    return ask_question