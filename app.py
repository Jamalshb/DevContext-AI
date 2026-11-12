import streamlit as st

from src.config import OLLAMA_CHAT_MODEL, OLLAMA_EMBED_MODEL
from src.ingestor import IngestionError, ingest_repo
from src.llm import get_embeddings, get_llm, has_model, list_ollama_models
from src.qa_chain import build_qa_chain
from src.utils import extract_repo_name

st.set_page_config(page_title="DevContext-AI", page_icon="🧠", layout="wide")

for key, default in {"qa": None, "messages": [], "current_repo": None}.items():
    st.session_state.setdefault(key, default)

# ---------------- Sidebar ----------------
st.sidebar.title("🧠 DevContext-AI")
st.sidebar.caption("100% free & local — powered by Ollama")

ollama_ok = False
installed: list[str] = []
try:
    installed = list_ollama_models()
    ollama_ok = True
    st.sidebar.success("Ollama is running ✅")
except ConnectionError as e:
    st.sidebar.error(str(e))
    st.sidebar.markdown("Install it from **ollama.com**, then run `ollama serve`.")

chat_models = [m for m in installed if "embed" not in m] or [OLLAMA_CHAT_MODEL]
default_idx = next((i for i, m in enumerate(chat_models) if has_model([m], OLLAMA_CHAT_MODEL)), 0)
chat_model = st.sidebar.selectbox("Chat model", chat_models, index=default_idx)

if ollama_ok:
    missing = [m for m in (OLLAMA_EMBED_MODEL, chat_model) if not has_model(installed, m)]
    if missing:
        st.sidebar.warning("Missing models. Run:\n\n" + "\n\n".join(f"`ollama pull {m}`" for m in missing))

st.sidebar.divider()
st.sidebar.header("Repository")
repo_url = st.sidebar.text_input("GitHub URL", placeholder="https://github.com/owner/repo")

if st.sidebar.button("Index repository", type="primary", disabled=not ollama_ok, use_container_width=True):
    try:
        with st.status("Indexing repository...", expanded=True) as status:
            embeddings = get_embeddings()
            result = ingest_repo(repo_url, embeddings=embeddings, progress=status.write)
            st.session_state.qa = build_qa_chain(
                result.persist_directory,
                result.collection_name,
                embeddings=embeddings,
                llm=get_llm(chat_model),
            )
            st.session_state.current_repo = extract_repo_name(repo_url)
            st.session_state.messages = []
            status.update(
                label=f"Indexed {result.num_files} files ({result.num_chunks} chunks) ✅",
                state="complete",
            )
    except (IngestionError, ValueError) as e:
        st.sidebar.error(str(e))
    except Exception as e:
        st.sidebar.error(f"Unexpected error: {e}")

# Switching the chat model doesn't require re-indexing.
if st.session_state.qa is not None and getattr(st.session_state.qa.llm, "model", chat_model) != chat_model:
    st.session_state.qa.llm = get_llm(chat_model)

if st.session_state.messages and st.sidebar.button("Clear chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# ---------------- Main chat ----------------
st.title("Ask questions about any GitHub repository")
if st.session_state.current_repo:
    st.info(f"📦 Current repository: **{st.session_state.current_repo}**")
else:
    st.write("1. Start Ollama  2. Paste a GitHub URL in the sidebar  3. Click **Index repository**  4. Ask!")


def show_sources(sources):
    if sources:
        with st.expander("📄 Sources"):
            for s in dict.fromkeys(sources):
                st.markdown(f"- `{s}`")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg.get("sources"))

question = st.chat_input("e.g. How does authentication work in this project?", disabled=st.session_state.qa is None)

if question:
    qa = st.session_state.qa
    history = [
        (st.session_state.messages[i]["content"], st.session_state.messages[i + 1]["content"])
        for i in range(0, len(st.session_state.messages) - 1, 2)
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            docs = qa.retrieve(question)
            if docs:
                answer = st.write_stream(qa.stream(question, docs, history))
            else:
                answer = "No relevant code was found in the repository."
                st.markdown(answer)
            sources = [d.metadata.get("source", "Unknown") for d in docs]
            show_sources(sources)
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        except Exception as e:
            st.session_state.messages.pop()  # keep user/assistant pairs aligned
            st.error(f"Error: {e}")
