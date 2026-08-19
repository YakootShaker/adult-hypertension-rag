"""
ablation_chunking.py

Module 2: Chunk Size & Overlap Ablation Experiment.
Compares 3 chunking configurations:
  1. Small:    Target 300 tokens, Overlap 40 tokens
  2. Balanced: Target 600 tokens, Overlap 80 tokens (Current)
  3. Large:    Target 900 tokens, Overlap 150 tokens

Evaluates:
  - Total chunks generated
  - Min/Max/Avg token count per chunk
  - Hit Rate@k and Precision@k on evaluation/test_set.json
Outputs:
  - evaluation/chunking_ablation_results.json
"""

import json
import os
import sys
from pathlib import Path
import tiktoken

from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

# Add src to path to import chunking logic
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chunking.chunker import (
    detect_sections,
    build_sections,
    split_into_chunks,
    get_section_pages,
    get_section_text,
    build_page_token_map,
    SectionCursor,
    map_chunk_to_pages,
    count_tokens,
    EXCLUDED_SECTIONS,
)

TEST_SET_PATH   = "evaluation/test_set.json"
PAGES_PATH      = "data/processed/cleaned_pages.json"
OUTPUT_JSON     = "evaluation/chunking_ablation_results.json"
QDRANT_EVAL_DIR = "data/qdrant_ablation"

CONFIGURATIONS = [
    {
        "config_id": "cfg_small",
        "name": "Small Chunks (300/40)",
        "target_tokens": 300,
        "max_tokens": 450,
        "min_tokens": 200,
        "overlap_tokens": 40,
    },
    {
        "config_id": "cfg_balanced",
        "name": "Balanced Chunks (600/80) - Current",
        "target_tokens": 600,
        "max_tokens": 900,
        "min_tokens": 400,
        "overlap_tokens": 80,
    },
    {
        "config_id": "cfg_large",
        "name": "Large Chunks (900/150)",
        "target_tokens": 900,
        "max_tokens": 1350,
        "min_tokens": 600,
        "overlap_tokens": 150,
    },
]


def generate_chunks_for_config(sections, cfg):
    chunks = []
    for section in sections:
        if section["section_title"].upper() in EXCLUDED_SECTIONS:
            continue

        section_pages = get_section_pages(section)
        section_text  = get_section_text(section)
        
        section_chunks = split_into_chunks(
            section_text,
            target_tokens=cfg["target_tokens"],
            max_tokens=cfg["max_tokens"],
            min_tokens=cfg["min_tokens"],
            overlap_tokens=cfg["overlap_tokens"]
        )

        page_token_map = build_page_token_map(section_pages)
        cursor = SectionCursor(page_token_map, overlap_tokens=cfg["overlap_tokens"])

        for idx, chunk_text in enumerate(section_chunks, 1):
            page_start, page_end = map_chunk_to_pages(cursor, chunk_text)
            token_cnt = count_tokens(chunk_text)
            chunks.append({
                "chunk_id": f"who_001_{section['section_id']}_c{idx:03d}",
                "section_id": section["section_id"],
                "section_title": section["section_title"],
                "page_start": page_start,
                "page_end": page_end,
                "token_count": token_cnt,
                "text": chunk_text,
            })
    return chunks


def embed_text_with_retry(genai_client, text: str, title: str = "", task_type: str = "RETRIEVAL_DOCUMENT"):
    import time
    for attempt in range(5):
        try:
            resp = genai_client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type, title=title if title else None),
            )
            return resp.embeddings[0].values
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"    [Rate limit 429] Backing off 6s before retry {attempt+1}/5...")
                time.sleep(6.0)
            else:
                time.sleep(2.0)
    raise RuntimeError(f"Failed to embed after 5 retries for text: {text[:50]}")


def evaluate_config(cfg, chunks, test_set, client, genai_client):
    import time
    import uuid
    col_name = f"col_{cfg['config_id']}"
    
    # Create / Recreate Qdrant collection
    existing = [c.name for c in client.get_collections().collections]
    if col_name in existing:
        client.delete_collection(col_name)

    client.create_collection(
        collection_name=col_name,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )

    # Check if we can reuse pre-computed embeddings for balanced config
    cached_embeddings = {}
    if cfg["config_id"] == "cfg_balanced" and os.path.exists("data/processed/embeddings.json"):
        with open("data/processed/embeddings.json", "r", encoding="utf-8") as f:
            records = json.load(f)
            cached_embeddings = {r["chunk_id"]: r["embedding"] for r in records}

    points = []
    print(f"  Embedding & indexing {len(chunks)} chunks for {cfg['name']}...")
    for i, c in enumerate(chunks, 1):
        cid = c["chunk_id"]
        if cid in cached_embeddings:
            emb = cached_embeddings[cid]
        else:
            emb = embed_text_with_retry(genai_client, c["text"], title=c["section_title"], task_type="RETRIEVAL_DOCUMENT")
            time.sleep(0.35)  # Safe free tier pacing

        uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{cfg['config_id']}_{cid}"))
        points.append(
            PointStruct(
                id=uid,
                vector=emb,
                payload={
                    "chunk_id": cid,
                    "section_id": c["section_id"],
                    "section_title": c["section_title"],
                    "page_start": c["page_start"],
                    "page_end": c["page_end"],
                    "token_count": c["token_count"],
                    "text": c["text"],
                }
            )
        )
    client.upsert(collection_name=col_name, points=points)

    # Query evaluation
    hit_5 = 0
    precision_5_sum = 0.0
    k = 5

    for tc in test_set:
        query = tc["query"]
        expected_sec = tc.get("expected_section_id")

        q_vec = embed_text_with_retry(genai_client, query, task_type="RETRIEVAL_QUERY")
        time.sleep(0.2)

        hits = client.query_points(
            collection_name=col_name,
            query=q_vec,
            limit=k,
            with_payload=True,
        ).points

        retrieved_secs = [h.payload.get("section_id") for h in hits]
        matches = sum(1 for sid in retrieved_secs if sid == expected_sec)
        
        if matches > 0:
            hit_5 += 1
        precision_5_sum += (matches / k)

    n = len(test_set)
    tokens_list = [c["token_count"] for c in chunks]

    return {
        "config_id": cfg["config_id"],
        "name": cfg["name"],
        "target_tokens": cfg["target_tokens"],
        "overlap_tokens": cfg["overlap_tokens"],
        "total_chunks": len(chunks),
        "min_tokens": min(tokens_list) if tokens_list else 0,
        "max_tokens": max(tokens_list) if tokens_list else 0,
        "avg_tokens": int(sum(tokens_list) / max(1, len(tokens_list))),
        "hit_rate_at_5": round(hit_5 / n, 4),
        "precision_at_5": round(precision_5_sum / n, 4),
    }


def main():
    print("=" * 70)
    print("DAY 2: CHUNK SIZE & OVERLAP ABLATION EXPERIMENT")
    print("=" * 70)

    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    # Detect sections
    headings = detect_sections()
    sections = build_sections(headings)

    qdrant_client = QdrantClient(path=QDRANT_EVAL_DIR)
    api_key = os.getenv("GEMINI_API_KEY")
    genai_client = genai.Client(api_key=api_key)

    results = []
    for cfg in CONFIGURATIONS:
        print(f"\nEvaluating: {cfg['name']}...")
        chunks = generate_chunks_for_config(sections, cfg)
        print(f"  Generated {len(chunks)} chunks.")
        res = evaluate_config(cfg, chunks, test_set, qdrant_client, genai_client)
        results.append(res)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("CHUNK SIZE & OVERLAP ABLATION RESULTS")
    print("=" * 70)
    print(f"{'Configuration':<28} | {'Chunks':<7} | {'Avg Tok':<8} | {'Hit@5':<8} | {'Prec@5':<8}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<28} | {r['total_chunks']:<7} | {r['avg_tokens']:<8} | {r['hit_rate_at_5']:<8.2%} | {r['precision_at_5']:<8.2%}")
    print("=" * 70)
    print(f"Saved results to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
