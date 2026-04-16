from langchain_community.vectorstores import Chroma

from src.config import (
    MODEL_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBED_MODEL,
)

if MODEL_PROVIDER == "openai":
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
else:
    from langchain_ollama import ChatOllama, OllamaEmbeddings


def build_qa_chain(persist_directory: str):
    if MODEL_PROVIDER == "openai":
        embeddings = OpenAIEmbeddings(
            model=OPENAI_EMBED_MODEL,
            api_key=OPENAI_API_KEY,
        )
        llm = ChatOpenAI(
            model=OPENAI_CHAT_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0,
        )
    else:
        embeddings = OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        llm = ChatOllama(
            model=OLLAMA_CHAT_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
        )

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    def ask_question(payload: dict):
        question = payload["question"]
        chat_history = payload.get("chat_history", [])

        docs = retriever.invoke(question)

        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
You are an expert AI assistant that analyzes GitHub repositories.

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