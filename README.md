# DevContext-AI 🧠

**Chat with any GitHub repository — 100% free and local.**
DevContext-AI clones a repository, indexes its code in a vector database, and lets you ask questions about it in natural language (English, Arabic, …). Answers come from a local LLM running on your machine via [Ollama](https://ollama.com) — no API keys, no cost, your code never leaves your computer.

## ✨ Features
- 🔍 Index any public GitHub repo (`https://github.com/owner/repo` or just `owner/repo`)
- 🧩 Language-aware chunking (Python, JS/TS, Java, Markdown, HTML…) so functions and classes stay together
- 🧠 Real semantic search with local embeddings (`nomic-embed-text`)
- 💬 Streaming chat answers with **source files** for every answer
- 🗂️ Separate vector DB per repository; re-indexing replaces old data
- 🔁 Switch the chat model from the sidebar without re-indexing
- ✅ Offline test suite (no Ollama needed to run the tests)

## 🏗️ How it works (RAG)
```
GitHub repo ──clone──▶ read files ──split──▶ chunks ──embed (Ollama)──▶ ChromaDB
                                                                          │
Question ──embed──▶ similarity search ──top-k chunks──▶ LLM (Ollama) ──▶ answer + sources
```

| File | Role |
|---|---|
| `app.py` | Streamlit chat UI |
| `src/ingestor.py` | Clone → load → split → embed → store |
| `src/qa_chain.py` | Retrieval + prompt + LLM answer (streaming) |
| `src/llm.py` | Ollama embeddings / chat model / health check |
| `src/config.py` | Settings (overridable with `.env`) |
| `src/utils.py` | URL parsing & validation |

## ⚙️ Setup
**Requirements:** Python 3.10+, Git, [Ollama](https://ollama.com/download).

```bash
# 1. Get the free models (one time)
ollama pull llama3.2
ollama pull nomic-embed-text

# 2. Install Python dependencies
python -m venv .venv
source .venv/bin/activate        # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```
Open http://localhost:8501, paste a repo URL, click **Index repository**, and start asking.

> 💡 Weak computer? Use a smaller model: `ollama pull llama3.2:1b` and pick it in the sidebar.
> Stronger computer? Try `qwen2.5-coder:7b` for better code answers.

### Configuration (optional)
Copy `.env.example` to `.env` to change models, Ollama URL, chunk sizes, etc.

## 🧪 Tests
```bash
pytest -q
```
The tests use fake embeddings and a fake LLM, so they run in under a second without Ollama or internet.

---

## 🇸🇦 بالعربي
**DevContext-AI** أداة مجانية تعمل على جهازك: تعطيها رابط مستودع GitHub، فتقرأ الكود وتخزنه في قاعدة بيانات متجهات (ChromaDB)، ثم تسألها بالعربي أو الإنجليزي عن أي شيء في الكود، فتجيبك مع ذكر الملفات التي اعتمدت عليها. تستخدم تقنية **RAG** ونماذج مجانية عبر **Ollama**، بدون أي مفتاح API.

**التشغيل:** ثبّت Ollama ← `ollama pull llama3.2` و `ollama pull nomic-embed-text` ← `pip install -r requirements.txt` ← `streamlit run app.py`.

## 🛠️ Tech stack
Python · Streamlit · LangChain · ChromaDB · Ollama · GitPython · pytest
