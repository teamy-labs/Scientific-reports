import os
from google import genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("⚠️ GEMINI_API_KEY is missing! Check your .env file.")

# Configure the new Gemini Client
client = genai.Client(api_key=api_key)

print("-> Loading Local Embedding Model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

vector_db = None
text_chunks = []

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_vector_db(full_text):
    global vector_db, text_chunks
    print("-> Chunking and Embedding Data...")
    
    text_chunks = chunk_text(full_text)
    embeddings = embedder.encode(text_chunks)
    
    dimension = embeddings.shape[1]
    vector_db = faiss.IndexFlatL2(dimension)
    vector_db.add(np.array(embeddings))
    print(f"-> Vector DB built with {len(text_chunks)} chunks.")

def generate_answer(query):
    global vector_db, text_chunks
    if vector_db is None:
        return "System Error: Please upload a PDF first."

    # 1. Search / Retrieval (Find the top 4 most relevant chunks)
    print(f"-> Searching FAISS for: {query}")
    query_vector = embedder.encode([query])
    distances, indices = vector_db.search(np.array(query_vector), k=4)
    
    # 2. Data Integration
    retrieved_chunks = [text_chunks[idx] for idx in indices[0]]
    context = "\n\n--- NEXT EXCERPT ---\n\n".join(retrieved_chunks)
    
    # 3. Gemini LLM Synthesis using the new SDK
    prompt = f"""
    You are an expert scientific researcher and interpreter. 
    Your goal is to answer the user's question accurately based strictly on the provided context extracted from a research paper.
    
    Rules:
    1. Do not use outside knowledge. If the answer is not in the context, say "I cannot find the answer to this in the uploaded document."
    2. Be clear, concise, and professional.
    3. If applicable, use bullet points to break down complex ideas.

    Context from Document:
    {context}

    User Question: {query}
    
    Expert Answer:
    """
    
    print("-> Asking Gemini for the final synthesized answer...")
    try:
        # Use gemini-2.5-flash for speed and reliability
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "Sorry, I encountered an error while communicating with the AI brain."