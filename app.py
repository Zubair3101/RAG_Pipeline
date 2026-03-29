import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
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
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ---------------------------
# ✂️ BETTER CHUNKING
# ---------------------------
def chunk_text(text, chunk_size=500, overlap=100):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# ---------------------------
# ⚡ PROCESS DOCUMENTS
# ---------------------------
@st.cache_resource
def process_documents(files):
    all_chunks = []

    for file in files:
        if file.type == "application/pdf":
            text = extract_text_from_pdf(file)
            source = file.name
        else:
            text = file.read().decode("utf-8")
            source = file.name

        chunks = chunk_text(text)

        for chunk in chunks:
            all_chunks.append({"text": chunk, "source": source})

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts)

    embedding_matrix = np.array(embeddings).astype("float32")

    # ✅ COSINE SIMILARITY
    faiss.normalize_L2(embedding_matrix)
    dimension = embedding_matrix.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embedding_matrix)

    return all_chunks, model, index

# ---------------------------
# 🔍 SEARCH
# ---------------------------
def search_faiss(query, model, index, chunks, top_k=3):
    query_vector = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, top_k)

    results = [chunks[i] for i in indices[0]]
    return results

# ---------------------------
# 🤖 RAG
# ---------------------------
def rag_query(query, model, index, chunks, history):
    results = search_faiss(query, model, index, chunks)

    context = "\n".join([r["text"] for r in results])

    messages = [
        {
            "role": "system",
            "content": "Answer ONLY from the given context. If not found, say 'I don't know'."
        }
    ]

    # ✅ CHAT MEMORY
    for msg in history[-5:]:
        messages.append(msg)

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}"
    })

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
    )

    return response.choices[0].message.content, results

# ---------------------------
# 🎨 UI
# ---------------------------
st.set_page_config(page_title="Dhurandar RAG Pro", layout="wide")
st.title("💬🧠 Dhurandar RAG Pro")

# ---------------------------
# 📂 MULTI FILE UPLOAD
# ---------------------------
uploaded_files = st.file_uploader(
    "Upload PDFs or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success("✅ Files uploaded!")

    chunks, model, index = process_documents(uploaded_files)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    query = st.chat_input("Ask something about your documents...")

    if query:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        answer, results = rag_query(
            query, model, index, chunks, st.session_state.messages
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

            # ✅ SOURCE DISPLAY
            with st.expander("📄 Sources"):
                for r in results:
                    st.write(f"**Source:** {r['source']}")
                    st.write(r["text"])

        st.session_state.messages.append({"role": "assistant", "content": answer})