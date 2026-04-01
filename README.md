# DevContext-AI 🚀

DevContext-AI is a local AI-powered tool that lets you **ask questions about any GitHub repository** using vector search and LLMs — fully running on your machine (no OpenAI required).

---

## ✨ Features

- 🔍 Clone and analyze any GitHub repository
- 🧠 Local embeddings using **Ollama**
- 💬 Conversational Q&A over code
- 🗄️ Persistent vector database (ChromaDB)
- ⚡ Streamlit UI for easy interaction
- 💸 100% free (no API keys required)

---

## 🧠 How it works

1. Clone a GitHub repository
2. Load and parse supported files
3. Split code into chunks
4. Generate embeddings (Ollama)
5. Store in Chroma vector DB
6. Ask questions using local LLM

---

## 📁 Supported Files

- `.py`
- `.js`
- `.ts`
- `.md`

---

## ⚙️ Requirements

- Python 3.10+
- Ollama installed

---

## 📦 Installation

```bash
pip install -r requirements.txt
