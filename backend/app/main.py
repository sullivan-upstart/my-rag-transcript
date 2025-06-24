"""FastAPI entry point for RAG Transcript QA."""
from pathlib import Path
import uuid

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from .embeddings import embed_transcript, query_transcripts

app = FastAPI(title="RAG-Transcript-QA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Save transcript, embed & store."""
    name = f"{uuid.uuid4()}_{file.filename}"
    path = UPLOAD_DIR / name
    text = (await file.read()).decode("utf-8", errors="ignore")
    path.write_text(text)
    embed_transcript(text, {"filename": name})
    return {"status": "ok", "filename": name}

@app.post("/query")
async def query(question: str = Form(...)):
    """Return answer based on retrieved chunks (mock for now)."""
    return {"answer": query_transcripts(question)}
