"""FastAPI entry point for RAG Transcript QA."""
from pathlib import Path
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Temporarily comment out to test startup
# from .embeddings import embed_transcript, query_transcripts
from .parsing import parse_transcript

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
    """Save transcript, clean text, embed & store."""
    # Ensure a unique filename so repeated uploads do not collide on disk.
    name = f"{uuid.uuid4()}_{file.filename}"
    path = UPLOAD_DIR / name

    # ``UploadFile.read`` yields ``bytes``; decode to ``str``. ``errors='ignore'``
    # avoids failures on odd characters that occasionally appear in transcripts.
    try:
        raw = (await file.read()).decode("utf-8", errors="ignore")

        # Normalize the transcript (strip numeric indices / timestamps) so that
        # embedding operates on pure text instead of subtitle artefacts.
        text = parse_transcript(raw)

        # Persist the original file for debugging/auditing purposes.
        path.write_text(raw)

        # Temporarily comment out embedding
        # embed_transcript(text, {"filename": name})
    except Exception as exc:  # pragma: no cover - defensive
        if path.exists():
            path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return {"status": "ok", "filename": name}

@app.post("/query")
async def query(question: str = Form(...)):
    """Return answer based on retrieved chunks (mock for now)."""
    # Temporarily return mock response
    # return {"answer": query_transcripts(question)}
    return {"answer": f"Mock response to: {question}"}
