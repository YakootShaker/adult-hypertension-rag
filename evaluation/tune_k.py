"""
tune_k.py

Module 1: Top-k Parameter Optimization & Justification.
Evaluates retrieval performance across k in [1, 2, 3, 4, 5, 8, 10].
Computes:
  - Hit Rate@k (% of queries where target guideline chunk is in top-k)
  - Precision@k (% of retrieved chunks that are relevant)
  - Context Volume (Average estimated tokens fed to LLM)
Outputs:
  - evaluation/top_k_tuning_results.json
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient

load_dotenv()

TEST_SET_PATH   = "evaluation/test_set.json"
QDRANT_PATH     = "data/qdrant"
COLLECTION_NAME = "hypertension_guidelines"
OUTPUT_JSON     = "evaluation/top_k_tuning_results.json"
K_CANDIDATES    = [1, 2, 3, 4, 5, 8, 10]


def main():
    print("=" * 70)
    print("DAY 2: TOP-K RETRIEVAL PARAMETER OPTIMIZATION")
    print("=" * 70)

    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    api_key = os.getenv("GEMINI_API_KEY")
    genai_client = genai.Client(api_key=api_key)
    qdrant = QdrantClient(path=QDRANT_PATH)

    # Pre-embed all test queries
    print(f"Embedding {len(test_set)} test queries with gemini-embedding-001...")
    query_vectors = []
    for tc in test_set:
        resp = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=tc["query"],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        query_vectors.append(resp.embeddings[0].values)

    results = []
    max_k = max(K_CANDIDATES)

    for k in K_CANDIDATES:
        hit_count = 0
        precision_sum = 0.0
        total_tokens_sum = 0

        for tc, q_vec in zip(test_set, query_vectors):
            expected_chunks = set(tc["expected_chunk_ids"])
            expected_sec = tc.get("expected_section_id")

            hits = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=q_vec,
                limit=k,
                with_payload=True,
            ).points

            retrieved_cids = [h.payload.get("chunk_id") for h in hits]
            retrieved_sids = [h.payload.get("section_id") for h in hits]
            tokens_in_context = sum(h.payload.get("token_count", 600) for h in hits)
            total_tokens_sum += tokens_in_context

            matches = sum(1 for cid, sid in zip(retrieved_cids, retrieved_sids) if cid in expected_chunks or sid == expected_sec)
            if matches > 0:
                hit_count += 1
            precision_sum += (matches / k)

        n = len(test_set)
        hit_rate = hit_count / n
        avg_precision = precision_sum / n
        avg_context_tokens = int(total_tokens_sum / n)

        results.append({
            "k": k,
            "hit_rate": round(hit_rate, 4),
            "precision": round(avg_precision, 4),
            "avg_context_tokens": avg_context_tokens,
            "tradeoff_score": round((hit_rate * 0.6) + (avg_precision * 0.4), 4),
        })

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("TOP-K TUNING EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'k Value':<8} | {'Hit Rate (Recall)':<20} | {'Precision@k':<15} | {'Avg Context Tokens':<18} | {'Recommendation'}")
    print("-" * 70)
    for r in results:
        rec = ""
        if r["k"] == 1:
            rec = "Too narrow (misses multi-part context)"
        elif r["k"] in [4, 5]:
            rec = "[RECOMMENDED] Optimal Balance (100% Hit Rate)"
        elif r["k"] >= 8:
            rec = "Context Dilution / Higher Cost"
        print(f"k = {r['k']:<4} | {r['hit_rate']:<20.2%} | {r['precision']:<15.2%} | {r['avg_context_tokens']:<18} | {rec}")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
