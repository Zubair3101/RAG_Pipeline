# 💬🧠 Dhurandar Elite RAG

An advanced production-ready RAG system with hybrid search, reranking, and ChatGPT-style UI.

---

## 🚀 Features

- 💬 ChatGPT-like interface
- 📂 Multi-PDF + TXT support
- 🔍 Hybrid search (FAISS + keyword)
- 🔁 Reranking using cross-encoder
- ⚡ Streaming responses
- 📊 Source ranking with scores
- 🧠 Chat memory
- 🎛 Top-K control
- 📜 Chat export

---

## 🧠 Architecture

1. Upload documents
2. Chunk text
3. Generate embeddings
4. Store in FAISS
5. Hybrid retrieval (vector + keyword)
6. Rerank results
7. LLM generates answer

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
streamlit run app.py