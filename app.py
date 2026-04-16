import streamlit as st

from src.ingestor import ingest_repo
from src.qa_chain import build_qa_chain
from src.utils import extract_repo_name


st.set_page_config(page_title="DevContext-AI", layout="wide")
st.title("DevContext-AI")
st.write("Ask questions about any GitHub repository.")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_repo" not in st.session_state:
    st.session_state.current_repo = None


st.sidebar.header("Repository Setup")
repo_url = st.sidebar.text_input("GitHub Repository URL")

if st.sidebar.button("Index Repository"):
    if not repo_url.strip():
        st.sidebar.error("Please enter a repository URL.")
    else:
        try:
            with st.spinner("Indexing repository..."):
                persist_directory = ingest_repo(repo_url)
                st.session_state.qa_chain = build_qa_chain(persist_directory)
                st.session_state.current_repo = extract_repo_name(repo_url)

            st.sidebar.success("Repository indexed successfully.")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")


if st.session_state.current_repo:
    st.info(f"Current repository: {st.session_state.current_repo}")


question = st.text_input("Ask a question about the repository")

if st.button("Ask"):
    if not st.session_state.qa_chain:
        st.error("Please index a repository first.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        try:
            result = st.session_state.qa_chain(
                {
                    "question": question,
                    "chat_history": st.session_state.chat_history,
                }
            )

            answer = result["answer"]
            sources = result.get("source_documents", [])

            st.session_state.chat_history.append((question, answer))

            st.subheader("Answer")
            st.write(answer)

            if sources:
                st.subheader("Sources")
                shown = set()
                for doc in sources:
                    source = doc.metadata.get("source", "Unknown")
                    if source not in shown:
                        st.write(f"- {source}")
                        shown.add(source)

        except Exception as e:
            st.error(f"Error: {str(e)}")