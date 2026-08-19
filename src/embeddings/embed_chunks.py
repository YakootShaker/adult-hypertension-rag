"""
embed_chunks.py

Reads data/processed/validated_chunks.json, embeds every chunk using
Google Gemini text-embedding-004 via the google-genai SDK, and writes
to data/processed/embeddings.json.

Usage:
    python src/embeddings/embed_chunks.py

Requires:
    GEMINI_API_KEY in .env
    pip install google-genai python-dotenv
"""

import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

# ============================================================
# Configuration
# ============================================================

VALIDATED_CHUNKS_PATH = "data/processed/validated_chunks.json"
OUTPUT_PATH           = "data/processed/embeddings.json"

EMBEDDING_MODEL = "gemini-embedding-001"  # 768-dim, 2048-token context window
TASK_TYPE       = "RETRIEVAL_DOCUMENT"  # optimised for RAG document indexing

REQUEST_DELAY_SEC = 0.1   # 100 ms between requests (safe for free tier)
MAX_RETRIES       = 3
RETRY_DELAY_SEC   = 5.0

# ============================================================
# Setup
# ============================================================

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

# ============================================================
# Helpers
# ============================================================

def embed_text(text: str, title: str = "") -> list:
    """Embed a single text with retry logic."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=TASK_TYPE,
                    title=title if title else None,
                ),
            )
            return response.embeddings[0].values
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"    [retry {attempt}/{MAX_RETRIES}] {exc} -- waiting {RETRY_DELAY_SEC}s")
            time.sleep(RETRY_DELAY_SEC)

# ============================================================
# Main
# ============================================================

def main():
    chunks_path = Path(VALIDATED_CHUNKS_PATH)
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Not found: {VALIDATED_CHUNKS_PATH}. Run chunking pipeline first."
        )

    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    if isinstance(chunks, dict) and "chunks" in chunks:
        chunks = chunks["chunks"]

    print("=" * 70)
    print("EMBEDDING PIPELINE -- adult-hypertension-rag")
    print("=" * 70)
    print(f"Model:      {EMBEDDING_MODEL}")
    print(f"Task type:  {TASK_TYPE}")
    print(f"Chunks:     {len(chunks)}")
    print()

    results = []
    start_time = time.time()

    for i, chunk in enumerate(chunks, start=1):
        cid       = chunk.get("chunk_id", f"chunk_{i}")
        text      = chunk.get("text", "")
        sec_title = chunk.get("section_title", "")

        print(
            f"  [{i:02d}/{len(chunks)}] {cid} "
            f"({chunk.get('token_count', '?')} tokens) ... ",
            end="", flush=True,
        )

        embedding = embed_text(text, title=sec_title)

        results.append({
            "chunk_id":        cid,
            "document_id":     chunk.get("document_id"),
            "document_name":   chunk.get("document_name"),
            "section_id":      chunk.get("section_id"),
            "section_title":   sec_title,
            "page_start":      chunk.get("page_start"),
            "page_end":        chunk.get("page_end"),
            "token_count":     chunk.get("token_count"),
            "text":            text,
            "embedding":       embedding,
            "embedding_model": EMBEDDING_MODEL,
        })

        print(f"done ({len(embedding)} dims)")

        if i < len(chunks):
            time.sleep(REQUEST_DELAY_SEC)

    elapsed = time.time() - start_time

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 70)
    print(f"Chunks embedded:  {len(results)}")
    print(f"Embedding dims:   {len(results[0]['embedding']) if results else 'N/A'}")
    print(f"Time elapsed:     {elapsed:.1f}s")
    print(f"Output:           {out_path}")


if __name__ == "__main__":
    main()
