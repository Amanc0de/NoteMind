
from pypdf import PdfReader
from fastembed import TextEmbedding
import faiss
import numpy as np
import os
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Load lightweight embedding model using ONNX
model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# STEP 1: Extract text from PDF
def extract_text_from_pdf(pdf_file):

    reader = PdfReader(pdf_file)
    extracted_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            extracted_text += f"\n{text}\n"

    return extracted_text


# STEP 2: Split text into chunks
def split_into_chunks(text, chunk_size=700, overlap=100):

    text = re.sub(r'\s+', ' ', text).strip()

    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        if len(current_chunk) + len(sentence) > chunk_size:

            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = current_chunk[-overlap:]

        current_chunk += " " + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# STEP 3: Create embeddings and FAISS database
def create_vector_database(chunks):

    embeddings = list(
        model.embed(chunks, batch_size=8)
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print("FAISS index type:", type(index))
    print("Total vectors:", index.ntotal)
    print("First vector norm:", np.linalg.norm(embeddings[0]))

    return index


# STEP 4: Search similar chunks
def search_similar_chunks(question, chunks, index, top_k=3):

    question_embedding = list(
        model.embed([question], batch_size=1)
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    faiss.normalize_L2(question_embedding)

    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx != -1:
            results.append({
                "text": chunks[idx],
                "score": float(score)
            })

    return results


# STEP 5: Generate AI Answer using Groq
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
