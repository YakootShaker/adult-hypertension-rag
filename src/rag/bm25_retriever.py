"""
bm25_retriever.py

Lexical / Sparse search engine using BM25Okapi for clinical guideline chunks.
Preserves medical terminology, numeric BP ranges (e.g. 140/90, 130-139), and abbreviations.
"""

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

DEFAULT_CHUNKS_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "validated_chunks.json"


def tokenize_clinical_text(text: str) -> list[str]:
    """Tokenize text preserving clinical abbreviations, alphanumeric terms, and numbers.

    Extracts terms such as '140/90', '130-139', 'acei', 'ccb', 'thiazide-like', etc.
    """
    if not text:
        return []
    # Replace slashes and hyphens inside numbers or terms smoothly
    tokens = re.findall(r"\b[a-zA-Z0-9]+(?:[-/][a-zA-Z0-9]+)*\b", text.lower())
    return [t for t in tokens if len(t) > 1 or t.isdigit()]


class BM25Retriever:
    """In-memory BM25 retrieval engine over processed guideline chunks."""

    def __init__(self, chunks_path: str | Path = DEFAULT_CHUNKS_PATH, chunks: list[dict[str, Any]] | None = None):
        if chunks is not None:
            self.chunks = chunks
        else:
            chunks_path = Path(chunks_path)
            if not chunks_path.exists():
                raise FileNotFoundError(f"Chunks file not found at: {chunks_path}")
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)

        self.corpus = [
            f"{c.get('section_title', '')} {c.get('text', '')}"
            for c in self.chunks
        ]
        self.tokenized_corpus = [tokenize_clinical_text(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.chunk_id_map = {c["chunk_id"]: c for c in self.chunks}

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Perform BM25 search for a given clinical query."""
        tokenized_query = tokenize_clinical_text(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            chunk = dict(self.chunks[idx])
            chunk["bm25_score"] = score
            results.append(chunk)

        return results

    def reload(self, chunks_path: str | Path = DEFAULT_CHUNKS_PATH):
        """Reload chunks and rebuild BM25 index."""
        chunks_path = Path(chunks_path)
        if not chunks_path.exists():
            return
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.corpus = [
            f"{c.get('section_title', '')} {c.get('text', '')}"
            for c in self.chunks
        ]
        self.tokenized_corpus = [tokenize_clinical_text(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.chunk_id_map = {c["chunk_id"]: c for c in self.chunks}

