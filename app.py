import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import os
import fitz  # PyMuPDF

# ---------------------------
# 🔐 API KEY
# ---------------------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------
# 📄 EXTRACT TEXT + STORE DOC
# ---------------------------
def extract_text(file):
    if file.type == "application/pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text, doc
    else:
        return file.read().decode("utf-8"), None

# ---------------------------
# ✂️ CHUNKING
# ---------------------------
def chunk_text(text, size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return chunks

# ---------------------------
# ⚡ PROCESS FILES
# ---------------------------
@st.cache_resource
def process(files):
    data = []
    docs = {}

    for file in files:
        text, doc = extract_text(file)
        docs[file.name] = doc

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

    return data, embed_model, index, reranker, docs

# ---------------------------
# 🔍 SEARCH + RERANK
# ---------------------------
def retrieve(query, data, model, index, reranker, top_k):
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb, 10)

    candidates = []
    for idx in I[0]:
        candidates.append(data[idx])

    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    for i, c in enumerate(candidates):
        c["score"] = scores[i]

    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]

# ---------------------------
# 🤖 RAG + STREAMING
# ---------------------------
def rag(query, history, results, mode):
    context = "\n".join([r["text"] for r in results])

    if mode == "Summarize":
        user_prompt = f"Summarize this:\n{context}"
    else:
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    messages = [{"role": "system", "content": "Answer only from context."}]
    messages += history[-5:]
    messages.append({"role": "user", "content": user_prompt})

    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        stream=True
    )

    return stream

# ---------------------------
# 🎨 UI
# ---------------------------
st.set_page_config(layout="wide")
st.title("💬🧠 Dhurandar PRO++")

# Sidebar controls
mode = st.sidebar.radio("Mode", ["Q&A", "Summarize"])
top_k = st.sidebar.slider("Top-K", 1, 10, 3)

uploaded_files = st.file_uploader(
    "Upload PDFs or TXT",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# Layout
col1, col2 = st.columns([2, 1])

if uploaded_files:
    data, model, index, reranker, docs = process(uploaded_files)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "history" not in st.session_state:
        st.session_state.history = []

    with col1:
        # Chat UI
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        query = st.chat_input("Ask something...")

        if query:
            st.chat_message("user").markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.history.append({"role": "user", "content": query})

            results = retrieve(query, data, model, index, reranker, top_k)
            stream = rag(query, st.session_state.history, results, mode)

            with st.chat_message("assistant"):
                text = ""
                placeholder = st.empty()

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        text += chunk.choices[0].delta.content
                        placeholder.markdown(text)

                # Sources
                with st.expander("📄 Sources"):
                    for r in results:
                        st.write(f"**{r['source']}**")
                        st.write(r["text"])

            st.session_state.messages.append({"role": "assistant", "content": text})
            st.session_state.history.append({"role": "assistant", "content": text})

    # ---------------------------
    # 📄 DOCUMENT PREVIEW + HIGHLIGHT
    # ---------------------------
    with col2:
        st.subheader("📄 Document Preview")

        for file_name, doc in docs.items():
            if doc:
                st.write(f"### {file_name}")

                page = doc[0]
                text = page.get_text()

                if st.session_state.get("messages"):
                    last_answer = st.session_state.messages[-1]["content"]

                    # highlight simple match
                    for inst in page.search_for(last_answer[:50]):
                        page.add_highlight_annot(inst)

                pix = page.get_pixmap()
                st.image(pix.tobytes())