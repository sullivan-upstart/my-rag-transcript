# """
# Stubbed embedding layer – ZERO-vector placeholder.

# This lets you:
# • Upload / chunk transcripts
# • Store vectors in Chroma (schema intact)
# • Prototype UI, filters, and query plumbing

# Swap back to a real model later by:
#   - uncommenting the SentenceTransformer block
#   - deleting the dummy-vector lines
# """
# from typing import List
# import numpy as np
# import chromadb
# from chromadb.config import Settings

# # ── If you’re ready for real embeddings later, uncomment: ────────────
# # from sentence_transformers import SentenceTransformer
# # MODEL_PATH = "./models/all-MiniLM-L6-v2"
# # model = SentenceTransformer(MODEL_PATH)
# # VECTOR_SIZE = model.get_sentence_embedding_dimension()
# # ─────────────────────────────────────────────────────────────────────

# VECTOR_SIZE = 384  # MiniLM size; adjust if you choose a different model

# client = chromadb.Client(
#     Settings(persist_directory="./chroma_db", anonymized_telemetry=False)
# )
# collection = client.get_or_create_collection("transcripts")


# # ─── utility: simple sliding-window splitter ─────────────────────────
# def _chunk_text(text: str, size: int = 1000, overlap: int = 200) -> List[str]:
#     chunks, i = [], 0
#     while i < len(text):
#         chunks.append(text[i : i + size])
#         i += size - overlap
#     return chunks


# # ─── public API used by main.py ──────────────────────────────────────
# def embed_transcript(text: str, metadata: dict | None = None) -> None:
#     """
#     • Splits transcript
#     • Stores ZERO vectors (shape = VECTOR_SIZE) in Chroma
#     """
#     metadata = metadata or {}
#     chunks = _chunk_text(text)
#     dummy_vecs = np.zeros((len(chunks), VECTOR_SIZE)).tolist()

#     ids = [f"{metadata.get('filename', 'file')}_{i}" for i in range(len(chunks))]
#     collection.add(
#         ids=ids,
#         documents=chunks,
#         embeddings=dummy_vecs,
#         metadatas=[metadata] * len(chunks),
#     )


# def query_transcripts(question: str, k: int = 4) -> str:
#     """
#     Returns a stub answer until real embeddings + LLM are wired up.
#     You can still test UI→/query endpoint flow.
#     """
#     return (
#         "[stub] Embedding model not integrated yet.\n"
#         "Once ready, this function will retrieve relevant chunks "
#         "from Chroma and call an LLM to generate an answer."
#     )


"""
Hash-TF-IDF placeholder · keeps Chroma happy until real embeddings arrive.
Swap-in later: comment out the vectorizer + hash lines and enable SentenceTransformer.
"""
from typing import List

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
import chromadb
from chromadb.config import Settings
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

# ── real model (future) ──────────────────────────────────────────────
# from sentence_transformers import SentenceTransformer
# MODEL_PATH = "./models/all-MiniLM-L6-v2"
# model = SentenceTransformer(MODEL_PATH)
# VECTOR_SIZE = model.get_sentence_embedding_dimension()

# ── TEMP: hashing-tfidf placeholder ──────────────────────────────────
# Pick a power of 2 (8k / 16k / 32k). 16 384 ≈ MiniLM×43 in size but compressible.
VECTOR_SIZE = 16_384
_vectorizer = HashingVectorizer(
    n_features=VECTOR_SIZE,
    norm="l2",
    alternate_sign=False,   # keep values non-negative
    stop_words="english",
)

# ── Chroma client ────────────────────────────────────────────────────
client = chromadb.Client(
    Settings(persist_directory="./chroma_db", anonymized_telemetry=False)
)
collection = client.get_or_create_collection("transcripts")


def _chunk(text: str, size: int = 1000, overlap: int = 200) -> List[str]:
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + size])
        i += size - overlap
    return out


def embed_transcript(text: str, metadata: dict | None = None) -> None:
    """
    • Split transcript
    • Produce Hash-TF-IDF vectors
    • Store in Chroma
    """
    metadata = metadata or {}
    chunks = _chunk(text)
    # HashingVectorizer expects an *iterable of str*
    mat = _vectorizer.transform(chunks)        # sparse matrix
    vecs = mat.toarray().tolist()              # dense lists (Chroma wants list[list])
    ids = [f"{metadata.get('filename','file')}_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=vecs,
        metadatas=[metadata] * len(chunks),
    )


def query_transcripts(question: str, k: int = 4) -> str:
    """
    Retrieve relevant chunks and generate an answer via OpenAI ChatCompletion.
    """
    q_vec = _vectorizer.transform([question]).toarray().tolist()
    res = collection.query(query_embeddings=q_vec, n_results=k)
    docs = res["documents"][0]
    context = "\n---\n".join(docs)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Answer questions about the transcripts using the given context.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )

    return completion["choices"][0]["message"]["content"].strip()
