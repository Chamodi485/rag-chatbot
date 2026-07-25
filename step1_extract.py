from pypdf import PdfReader

reader = PdfReader("document.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

print(f"Total characters extracted: {len(text)}")
print("--- First 500 characters ---")
print(text[:500])