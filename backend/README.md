# Backend Usage

This folder contains the FastAPI service used for uploading transcripts and querying the vector database.

## Local development

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt scikit-learn
uvicorn app.main:app --reload --port 8000
```

The extra `scikit-learn` install step is required for the temporary Hash-TF-IDF embeddings in `app/embeddings.py`.
