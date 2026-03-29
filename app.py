import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import os
from PyPDF2 import PdfReader

# ---------------------------
# 🔐 API KEY
# ---------------------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------
# 📄 PDF TEXT EXTRACTION
# ---------------------------
def extract_text(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        return file.read().decode("utf-8")

# ---------------------------
# ✂️ CHUNKING
# ---------------------------
def chunk_text(text, size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start+size]
        chunks.append(chunk)
        start += size - overlap
    return chunks

# ---------------------------
# ⚡ PROCESS DOCS
# ---------------------------
@st.cache_resource
def process(files):
    data = []

    for file in files:
        text = extract_text(file)
        chunks = chunk_text(text)

        for c in chunks:
            data.append({"text": c, "source": file.name})

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [d["text"] for d in data]
    embeddings = embed_model.encode(texts)

    emb = np.array(embeddings).astype("float32")
    faiss.normalize_L2(emb)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return data, embed_model, index, reranker

# ---------------------------
# 🔍 HYBRID SEARCH
# ---------------------------
def hybrid_search(query, data, model, index, top_k=5):
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb, top_k)

    results = []
    for idx, score in zip(I[0], D[0]):
        chunk = data[idx]
        keyword_score = query.lower() in chunk["text"].lower()
        combined_score = float(score) + (0.1 if keyword_score else 0)

        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "score": combined_score
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)

# ---------------------------
# 🔁 RERANK
# ---------------------------
def rerank(query, results, reranker, top_k=3):
    pairs = [[query, r["text"]] for r in results]
    scores = reranker.predict(pairs)

    for i, r in enumerate(results):
        r["rerank_score"] = scores[i]

    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

# ---------------------------
# 🤖 STREAMING RAG
# ---------------------------
def rag(query, history, data, model, index, reranker, top_k):
    results = hybrid_search(query, data, model, index, top_k)
    results = rerank(query, results, reranker)

    context = "\n".join([r["text"] for r in results])

    messages = [{"role": "system", "content": "Answer only from context."}]
    messages += history[-5:]

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}"
    })

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        stream=True
    )

    return response, results

# ---------------------------
# 🎨 UI
# ---------------------------
st.set_page_config(page_title="Dhurandar Elite RAG", layout="wide")
st.title("💬🧠 Dhurandar Elite RAG")

uploaded_files = st.file_uploader(
    "Upload files", type=["pdf", "txt"], accept_multiple_files=True
)

top_k = st.slider("Top-K Chunks", 1, 10, 3)

if uploaded_files:
    data, model, index, reranker = process(uploaded_files)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "history" not in st.session_state:
        st.session_state.history = []

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask anything...")

    if query:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.history.append({"role": "user", "content": query})

        stream, results = rag(
            query,
            st.session_state.history,
            data,
            model,
            index,
            reranker,
            top_k
        )

        with st.chat_message("assistant"):
            response_text = ""
            placeholder = st.empty()

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
                    placeholder.markdown(response_text)

            with st.expander("📄 Sources"):
                for r in results:
                    st.write(f"**{r['source']} (score: {round(r['rerank_score'], 3)})**")
                    st.write(r["text"])

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.session_state.history.append({"role": "assistant", "content": response_text})

# ---------------------------
# 📜 EXPORT CHAT
# ---------------------------
if st.session_state.get("messages"):
    chat_text = "\n".join([m["content"] for m in st.session_state.messages])
    st.download_button("Download Chat", chat_text, "chat.txt")