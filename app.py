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
# ✂️ CHUNKING
# ---------------------------
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# ---------------------------
# ⚡ PROCESS DOCUMENT
# ---------------------------
@st.cache_resource
def process_document(text):
    chunks = chunk_text(text)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    embedding_matrix = np.array(embeddings).astype("float32")
    dimension = embedding_matrix.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)

    return chunks, model, index

# ---------------------------
# 🔍 SEARCH
# ---------------------------
def search_faiss(query, model, index, chunks, top_k=3):
    query_vector = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, top_k)
    return [chunks[i] for i in indices[0]]

# ---------------------------
# 🤖 RAG
# ---------------------------
def rag_query(query, model, index, chunks):
    results = search_faiss(query, model, index, chunks)

    context = "\n".join(results)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "Answer ONLY from the given context. If not found, say 'I don't know'."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ],
    )

    return response.choices[0].message.content, results

# ---------------------------
# 🎨 UI CONFIG
# ---------------------------
st.set_page_config(page_title="Dhurandar Chat RAG", layout="wide")
st.title("💬 Dhurandar Chat RAG")

# ---------------------------
# 📂 FILE UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8")

    st.success("✅ File processed!")

    chunks, model, index = process_document(text)

    # ---------------------------
    # 💬 CHAT HISTORY
    # ---------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---------------------------
    # 💬 CHAT INPUT
    # ---------------------------
    query = st.chat_input("Ask something about your document...")

    if query:
        # Show user message
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Get AI response
        answer, retrieved_chunks = rag_query(query, model, index, chunks)

        # Show assistant message
        with st.chat_message("assistant"):
            st.markdown(answer)

            # Expandable context (VERY COOL)
            with st.expander("📄 View Retrieved Context"):
                for i, chunk in enumerate(retrieved_chunks):
                    st.write(f"**Chunk {i+1}:** {chunk}")

        st.session_state.messages.append({"role": "assistant", "content": answer})