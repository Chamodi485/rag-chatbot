import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()
client_ai = Groq(api_key=os.getenv("GROQ_API_KEY"))

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="my_document")

def ask(question, n_results=3):
    query_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    retrieved_chunks = results['documents'][0]

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    response = client_ai.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content, retrieved_chunks

question = "What is feature selection and why is it used?"
answer, sources = ask(question)

print(f"Question: {question}\n")
print(f"Answer: {answer}\n")
print("--- Sources used ---")
for i, chunk in enumerate(sources):
    print(f"[{i+1}] {chunk[:150]}...")
