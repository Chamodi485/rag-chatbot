import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()
client_ai = Groq(api_key=os.getenv("GROQ_API_KEY"))

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="multi_docs")  # the new multi-doc collection

def ask(question, n_results=3):
    query_embedding = model.encode([question]).tolist()

    # Notice: we now also request 'metadatas' so we get the source filename back
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    retrieved_chunks = results['documents'][0]
    retrieved_sources = results['metadatas'][0]  # list of {"source": filename} dicts

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


print("Multi-document RAG Chatbot ready. Type your question, or 'exit' to quit.\n")

while True:
    question = input("You: ")

    if question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    answer, sources, metadata = ask(question)

    print(f"\nBot: {answer}\n")
    print("--- Sources used ---")
    for i, (chunk, meta) in enumerate(zip(sources, metadata), 1):
        print(f"[{i}] From: {meta['source']}")
        print(f"    {chunk.strip()}...")
    print()