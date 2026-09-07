import streamlit as st
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from dotenv import load_dotenv
import os
import re

load_dotenv()

st.set_page_config(page_title="Course Notes RAG Chatbot", page_icon="📘")
st.title("📘 Course Notes Assistant")
st.caption("Ask questions - answers are grounded only in the actual course material.")

DOCS_FOLDER = "documents"

@st.cache_resource
def load_resources():
    client_ai = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="multi_docs")
    return client_ai, model, client, collection

client_ai, model, chroma_client, collection = load_resources()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def is_useful_chunk(chunk, min_length=50):
    stripped = chunk.strip()
    if len(stripped) < min_length:
        return False
    lines = [l.strip() for l in stripped.split("\n") if l.strip()]
    if not lines:
        return False
    footer_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{4}.*Department of Computer Engineering\s*\d*", re.IGNORECASE)
    footer_lines = [l for l in lines if footer_pattern.search(l)]
    if len(footer_lines) >= 3 and len(footer_lines) / len(lines) > 0.4:
        return False
    return True

def add_document(uploaded_file):
    os.makedirs(DOCS_FOLDER, exist_ok=True)
    filepath = os.path.join(DOCS_FOLDER, uploaded_file.name)

    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    existing = collection.get()
    already_processed = set(meta["source"] for meta in existing["metadatas"]) if existing["metadatas"] else set()
    if uploaded_file.name in already_processed:
        return 0, "already_exists"

    text = extract_text(filepath)
    raw_chunks = splitter.split_text(text)
    useful_chunks = [c for c in raw_chunks if is_useful_chunk(c)]

    if not useful_chunks:
        return 0, "no_content"

    embeddings = model.encode(useful_chunks).tolist()
    ids = [f"{uploaded_file.name}_chunk_{i}" for i in range(len(useful_chunks))]
    metadatas = [{"source": uploaded_file.name} for _ in useful_chunks]

    collection.add(ids=ids, embeddings=embeddings, documents=useful_chunks, metadatas=metadatas)
    return len(useful_chunks), "success"

def ask(question, n_results=3):
    query_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    retrieved_chunks = results['documents'][0]
    retrieved_sources = results['metadatas'][0]

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    response = client_ai.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content, retrieved_chunks, retrieved_sources


# --- Sidebar: upload new material ---
with st.sidebar:
    st.header("📤 Add Study Material")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file is not None:
        if st.button("Add to knowledge base"):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                count, status = add_document(uploaded_file)

            if status == "success":
                st.success(f"Added {count} chunks from {uploaded_file.name}")
            elif status == "already_exists":
                st.info(f"{uploaded_file.name} is already in the knowledge base.")
            elif status == "no_content":
                st.warning("No usable text found in this PDF.")

    st.divider()
    total_chunks = collection.count()
    st.caption(f"📊 Knowledge base: {total_chunks} chunks stored")


# --- Main chat interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your lecture notes...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching notes..."):
            answer, sources, metadata = ask(question)
        st.markdown(answer)

        with st.expander(f"📚 Sources ({len(sources)} chunks used)"):
            for i, (chunk, meta) in enumerate(zip(sources, metadata), 1):
                st.markdown(f"**[{i}] From: {meta['source']}**")
                st.text(chunk.strip())
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})