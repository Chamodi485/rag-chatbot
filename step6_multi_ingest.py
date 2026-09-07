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

def is_useful_chunk(chunk, min_length=50):
    """
    Filters out chunks that are mostly extraction noise —
    e.g. repeated slide footers like 'Department of Computer
    Engineering 2 3 4 5...' with little to no real content.
    """
    stripped = chunk.strip()
    if len(stripped) < min_length:
        return False

    # Count how many lines are just short, repeated boilerplate
    lines = [l.strip() for l in stripped.split("\n") if l.strip()]
    if not lines:
        return False

    # If most lines are very short (likely footer/page-number junk),
    # treat the whole chunk as low value
    short_lines = [l for l in lines if len(l) < 40]
    if len(short_lines) / len(lines) > 0.7:
        return False

    return True

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")

# Delete the old collection so we rebuild cleanly (removes noisy chunks from before)
try:
    client.delete_collection(name="multi_docs")
    print("Cleared old 'multi_docs' collection.")
except Exception:
    print("No existing collection to clear (first run).")

collection = client.get_or_create_collection(name="multi_docs")

all_chunks = []
all_metadata = []
skipped_count = 0

for filename in os.listdir(DOCS_FOLDER):
    if not filename.endswith(".pdf"):
        continue

    filepath = os.path.join(DOCS_FOLDER, filename)
    print(f"Processing: {filename}")

    text = extract_text(filepath)
    raw_chunks = splitter.split_text(text)

    for chunk in raw_chunks:
        if is_useful_chunk(chunk):
            all_chunks.append(chunk)
            all_metadata.append({"source": filename})
        else:
            skipped_count += 1

print(f"\nTotal useful chunks kept: {len(all_chunks)}")
print(f"Noisy/low-value chunks filtered out: {skipped_count}")

embeddings = model.encode(all_chunks).tolist()
ids = [f"{meta['source']}_chunk_{i}" for i, meta in enumerate(all_metadata)]

collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=all_chunks,
    metadatas=all_metadata
)

print(f"Stored {len(all_chunks)} chunks into Chroma.")