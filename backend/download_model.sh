#!/usr/bin/env bash
set -e
export HF_HUB_DISABLE_SSL_VERIFY=1
python - <<'PY'
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model cached at:", m.cache_folder)
PY
unset HF_HUB_DISABLE_SSL_VERIFY
echo "Done."
