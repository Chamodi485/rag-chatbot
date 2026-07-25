import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Load model and process document
model = SentenceTransformer('all-MiniLM-L6-v2')
text = extract_text("document.pdf")
chunks = chunk_text(text)

# Set up Chroma (persistent = saved to disk, not just memory)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_document")

# Add chunks with their embeddings to the database
ids = [f"chunk_{i}" for i in range(len(chunks))]
embeddings = model.encode(chunks).tolist()

collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=chunks
)

print(f"Stored {len(chunks)} chunks in Chroma.")

# Quick test: search for something
results = collection.query(
    query_texts=["What is feature selection?"],
    n_results=2
)

print("\n--- Top 2 matching chunks for the test query ---")
for doc in results['documents'][0]:
    print(doc[:200])
    print("---")