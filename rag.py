
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
# Load embedding model locally
model = SentenceTransformer("all-MiniLM-L6-v2")


# STEP 1: Extract text from PDF
def extract_text_from_pdf(pdf_file):

    reader = PdfReader(pdf_file)
    extracted_text = ""

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:
            extracted_text += f"\n{text}\n"

    return extracted_text


# STEP 2: Split text into chunks

import re

def split_into_chunks(text, chunk_size=700, overlap=100):

    # Remove unnecessary extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        # Check if adding sentence exceeds chunk size
        if len(current_chunk) + len(sentence) > chunk_size:

            if current_chunk:
                chunks.append(current_chunk.strip())

                # Keep some previous context
                current_chunk = current_chunk[-overlap:]

        current_chunk += " " + sentence

    # Add remaining text
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# STEP 3: Create embeddings and FAISS database

def create_vector_database(chunks):

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    # Inner Product on normalized vectors = Cosine Similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)
    print("First document vector norm:",
      np.linalg.norm(embeddings[0]))

    print("FAISS index type:",
        type(index))
    print("First vector norm:", np.linalg.norm(embeddings[0]))
    return index


def search_similar_chunks(question, chunks, index, top_k=3):

    # Generate normalized embedding for the question
    # normalize_embeddings=True
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = np.array(
        question_embedding,
        dtype="float32"
    )

    # Search using cosine similarity
    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )
    print("Question vector norm:",
      np.linalg.norm(question_embedding))

    print("FAISS index type:",
        type(index))

    print("FAISS index total vectors:",
        index.ntotal)
    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx != -1:
            results.append({
                "text": chunks[idx],
                "score": float(score)
            })

    return results

# STEP 4: Generate AI Answer using Groq

def generate_answer(question, retrieved_chunks):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("Groq API key not found!")

    client = Groq(api_key=api_key)

    context = "\n\n".join(
        chunk["text"] for chunk in retrieved_chunks
    )

    prompt = f"""
    Answer the question using ONLY the context given below.

    If the answer is not available in the context,
    clearly say that the information is not available
    in the uploaded notes.

    Context:
    {context}

    Question:
    {question}

    Answer in simple and clear language.
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are NoteMind, an AI assistant that answers questions from uploaded study notes."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=500
    )

    return response.choices[0].message.content