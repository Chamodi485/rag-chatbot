import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

DOCS_FOLDER = "documents"

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="multi_docs")

# Step 1: find out which files are ALREADY in the database
existing = collection.get()  # pulls everything currently stored
already_processed = set(meta["source"] for meta in existing["metadatas"])

print(f"Already in database: {already_processed}")

# Step 2: only process files not already processed
new_chunks = []
new_metadata = []
new_ids = []

for filename in os.listdir(DOCS_FOLDER):
    if not filename.endswith(".pdf"):
        continue
    if filename in already_processed:
        print(f"Skipping (already added): {filename}")
        continue

    filepath = os.path.join(DOCS_FOLDER, filename)
    print(f"Processing NEW file: {filename}")

    text = extract_text(filepath)
    chunks = splitter.split_text(text)

    for i, chunk in enumerate(chunks):
        new_chunks.append(chunk)
        new_metadata.append({"source": filename})
        # ID includes filename so it can NEVER collide with another file's chunks
        new_ids.append(f"{filename}_chunk_{i}")

if not new_chunks:
    print("\nNo new files to add. Database is already up to date.")
else:
    embeddings = model.encode(new_chunks).tolist()
    collection.add(
        ids=new_ids,
        embeddings=embeddings,
        documents=new_chunks,
        metadatas=new_metadata
    )
    print(f"\nAdded {len(new_chunks)} new chunks to the database.")

print(f"Total chunks in database now: {collection.count()}")