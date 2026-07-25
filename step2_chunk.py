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
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap   # move forward, but overlap a bit
    return chunks

text = extract_text("document.pdf")
chunks = chunk_text(text)

print(f"Total chunks created: {len(chunks)}")
print("--- First chunk ---")
print(chunks[0])
print("\n--- Second chunk (notice the overlap at the start) ---")
print(chunks[1])