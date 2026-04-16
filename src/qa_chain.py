from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma


def build_qa_chain(persist_directory: str):
    embeddings = FakeEmbeddings(size=1536)

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    def ask_question(payload: dict):
        question = payload["question"]

        docs = retriever.invoke(question)

        docs = sorted(
            docs,
            key=lambda d: "readme" not in d.metadata.get("source", "").lower()
        )

        if not docs:
            return {
                "answer": "No relevant documents were found in the repository.",
                "source_documents": [],
            }

        context = "\n\n".join(doc.page_content for doc in docs[:3])

        answer = (
            "Demo Mode (Free Version)\n\n"
            "This deployed version works without paid AI APIs.\n"
            "It retrieves the most relevant repository content and shows a summary based on it.\n\n"
            f"Question: {question}\n\n"
            "Relevant repository content:\n"
            f"{context[:1200]}"
        )

        return {
            "answer": answer,
            "source_documents": docs,
        }

    return ask_question