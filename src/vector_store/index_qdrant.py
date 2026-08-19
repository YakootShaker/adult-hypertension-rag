"""
index_qdrant.py

Reads data/processed/embeddings.json and indexes all chunks into a
persistent local Qdrant collection.

Usage:
    python src/vector_store/index_qdrant.py

Output:
    data/qdrant/  — on-disk Qdrant collection (persisted between runs)

Collection spec:
    name:     hypertension_guidelines
    vectors:  3072-dim, Cosine distance
    payload:  chunk_id, document_id, document_name, section_id,
              section_title, page_start, page_end, token_count, text
"""

import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

# ============================================================
# Configuration
# ============================================================

EMBEDDINGS_PATH  = "data/processed/embeddings.json"
QDRANT_PATH      = "data/qdrant"          # local on-disk persistence
COLLECTION_NAME  = "hypertension_guidelines"
VECTOR_SIZE      = 3072                   # gemini-embedding-001 output dims
DISTANCE         = Distance.COSINE        # best for semantic similarity

# ============================================================
# Helpers
# ============================================================

def chunk_id_to_uuid(chunk_id: str) -> str:
    """
    Qdrant point IDs must be unsigned integers or UUIDs.
    We deterministically derive a UUID from the chunk_id string so
    re-indexing the same chunk always produces the same point ID,
    making upserts idempotent.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


# ============================================================
# Main
# ============================================================

def main():
    # Load embeddings
    embeddings_path = Path(EMBEDDINGS_PATH)
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Not found: {EMBEDDINGS_PATH}. Run embed_chunks.py first."
        )

    with embeddings_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    print("=" * 70)
    print("QDRANT INDEXING -- adult-hypertension-rag")
    print("=" * 70)
    print(f"Collection:    {COLLECTION_NAME}")
    print(f"Vector size:   {VECTOR_SIZE}")
    print(f"Distance:      {DISTANCE}")
    print(f"Records:       {len(records)}")
    print(f"Storage path:  {QDRANT_PATH}")
    print()

    # Connect to local on-disk Qdrant
    Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=QDRANT_PATH)

    # (Re)create collection — drop existing if present so re-runs are clean
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"  Dropping existing collection '{COLLECTION_NAME}' ...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE),
    )
    print(f"  Created collection '{COLLECTION_NAME}'")
    print()

    # Build points
    points = []
    for record in records:
        cid = record.get("chunk_id", "")
        points.append(
            PointStruct(
                id=chunk_id_to_uuid(cid),
                vector=record["embedding"],
                payload={
                    "chunk_id":      cid,
                    "document_id":   record.get("document_id"),
                    "document_name": record.get("document_name"),
                    "section_id":    record.get("section_id"),
                    "section_title": record.get("section_title"),
                    "page_start":    record.get("page_start"),
                    "page_end":      record.get("page_end"),
                    "token_count":   record.get("token_count"),
                    "text":          record.get("text"),
                },
            )
        )

    # Upsert all points in one batch
    print(f"  Indexing {len(points)} points ...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    # Verify
    info = client.get_collection(COLLECTION_NAME)
    indexed = info.points_count

    print(f"  Done. Points in collection: {indexed}")
    print()
    print("=" * 70)
    print("INDEXING COMPLETE")
    print("=" * 70)
    print(f"Collection '{COLLECTION_NAME}' is ready for retrieval.")
    print(f"Qdrant data stored at: {Path(QDRANT_PATH).resolve()}")


if __name__ == "__main__":
    main()
