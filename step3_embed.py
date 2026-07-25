from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

print("Script started")
from sentence_transformers import SentenceTransformer
print("Import successful")



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

# Load the embedding model (downloads once, then caches locally)
model = SentenceTransformer('all-MiniLM-L6-v2')

text = extract_text("document.pdf")
chunks = chunk_text(text)

# Convert all chunks into embeddings (vectors)
embeddings = model.encode(chunks)

print(f"Number of chunks: {len(chunks)}")
print(f"Shape of embeddings: {embeddings.shape}")
print(f"First embedding (first 10 numbers only): {embeddings[0][:10]}")