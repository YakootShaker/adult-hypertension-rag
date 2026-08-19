"""
benchmark_hybrid.py

Comprehensive benchmark comparing:
  1. Dense Retrieval (gemini-embedding-001)
  2. Sparse Retrieval (BM25Okapi)
  3. Hybrid Retrieval (Reciprocal Rank Fusion - RRF)
  4. Hybrid Retrieval (Relative Score Fusion - Weighted)

Evaluates on evaluation/test_set.json (10 clinical questions).
Computes: Hit Rate@1, Hit Rate@3, Hit Rate@5, Precision@3, Precision@5, MRR, and Latency (ms).
Saves results to evaluation/hybrid_benchmark_results.json and prints Markdown comparison table.
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np

# Make sure src/ is in path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient

from rag.bm25_retriever import BM25Retriever
from rag.hybrid_retriever import reciprocal_rank_fusion, relative_score_fusion

load_dotenv()

TEST_SET_PATH   = "evaluation/test_set.json"
CHUNKS_PATH     = "data/processed/validated_chunks.json"
OUTPUT_JSON     = "evaluation/hybrid_benchmark_results.json"
QDRANT_PATH     = "data/qdrant"
COLLECTION_NAME = "hypertension_guidelines"

K_VALUES = [1, 3, 5]


def load_test_set():
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_benchmark():
    print("=" * 75)
    print("HYBRID RETRIEVAL EVALUATION: DENSE vs SPARSE vs HYBRID (RRF & WEIGHTED)")
    print("=" * 75)

    test_set = load_test_set()
    chunks = load_chunks()
    print(f"Loaded {len(test_set)} test queries and {len(chunks)} guideline chunks.\n")

    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    qdrant = QdrantClient(path=QDRANT_PATH)
    bm25 = BM25Retriever(chunks=chunks)

    def embed_query(q: str) -> list[float]:
        for attempt in range(5):
            try:
                resp = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=q,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
                )
                return resp.embeddings[0].values
            except Exception as e:
                time.sleep(2.0)
        raise RuntimeError("Embed query failed.")

    def dense_search(q_vec: list[float], top_n: int = 10) -> list[dict]:
        hits = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vec,
            limit=top_n,
            with_payload=True,
        ).points
        res = []
        for h in hits:
            item = dict(h.payload)
            item["dense_score"] = float(h.score)
            item["score"] = float(h.score)
            res.append(item)
        return res

    def sparse_search(q_text: str, top_n: int = 10) -> list[dict]:
        return bm25.search(q_text, top_k=top_n)

    modes = [
        ("Dense (gemini-embedding-001)", "dense"),
        ("Sparse (BM25Okapi)", "sparse"),
        ("Hybrid (RRF, c=60)", "hybrid_rrf"),
        ("Hybrid (Score Fusion, a=0.6)", "hybrid_weighted"),
    ]

    benchmark_data = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_set_size": len(test_set),
        "results": {}
    }

    for mode_title, mode_type in modes:
        print(f">>> Evaluating {mode_title}...")
        latencies = []
        hit_counts = {k: 0 for k in K_VALUES}
        precision_sums = {k: 0.0 for k in K_VALUES}
        recall_sums = {k: 0.0 for k in K_VALUES}
        mrr_sum = 0.0
        max_k = max(K_VALUES)

        per_query_records = []

        for test_case in test_set:
            qid = test_case["query_id"]
            query = test_case["query"]
            expected_chunks = set(test_case["expected_chunk_ids"])
            expected_section = test_case.get("expected_section_id")

            t0 = time.perf_counter()

            if mode_type == "dense":
                q_vec = embed_query(query)
                retrieved = dense_search(q_vec, top_n=max_k)
            elif mode_type == "sparse":
                retrieved = sparse_search(query, top_n=max_k)
            elif mode_type == "hybrid_rrf":
                q_vec = embed_query(query)
                d_hits = dense_search(q_vec, top_n=10)
                s_hits = sparse_search(query, top_n=10)
                retrieved = reciprocal_rank_fusion(d_hits, s_hits, top_k=max_k, c=60, dense_weight=0.5)
            elif mode_type == "hybrid_weighted":
                q_vec = embed_query(query)
                d_hits = dense_search(q_vec, top_n=10)
                s_hits = sparse_search(query, top_n=10)
                retrieved = relative_score_fusion(d_hits, s_hits, top_k=max_k, alpha=0.6)

            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            latencies.append(latency_ms)

            retrieved_chunk_ids = [r["chunk_id"] for r in retrieved]
            retrieved_sections = [r.get("section_id") for r in retrieved]

            # Calculate MRR
            rank = None
            for idx, (cid, sid) in enumerate(zip(retrieved_chunk_ids, retrieved_sections), 1):
                if cid in expected_chunks or sid == expected_section:
                    rank = idx
                    break
            reciprocal_rank = 1.0 / rank if rank is not None else 0.0
            mrr_sum += reciprocal_rank

            q_record = {
                "query_id": qid,
                "latency_ms": round(latency_ms, 2),
                "mrr": round(reciprocal_rank, 4),
                "top_retrieved": retrieved_chunk_ids[:3],
            }

            for k in K_VALUES:
                k_chunks = retrieved_chunk_ids[:k]
                k_sections = retrieved_sections[:k]
                matches = sum(1 for cid, sid in zip(k_chunks, k_sections) if cid in expected_chunks or sid == expected_section)

                hit = 1 if matches > 0 else 0
                hit_counts[k] += hit
                precision = matches / k
                precision_sums[k] += precision
                recall = min(1.0, matches / max(1, len(expected_chunks)))
                recall_sums[k] += recall

                q_record[f"hit@{k}"] = hit
                q_record[f"p@{k}"] = round(precision, 4)
                q_record[f"recall@{k}"] = round(recall, 4)

            per_query_records.append(q_record)

        n = len(test_set)
        mode_metrics = {
            "avg_latency_ms": round(float(np.mean(latencies)), 2),
            "mrr": round(mrr_sum / n, 4),
        }
        for k in K_VALUES:
            mode_metrics[f"hit_rate@{k}"] = round(hit_counts[k] / n, 4)
            mode_metrics[f"precision@{k}"] = round(precision_sums[k] / n, 4)
            mode_metrics[f"recall@{k}"] = round(recall_sums[k] / n, 4)

        benchmark_data["results"][mode_title] = {
            "metrics": mode_metrics,
            "per_query": per_query_records,
        }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    # Print Table
    print("\n" + "=" * 90)
    print("HYBRID RETRIEVAL BENCHMARK RESULTS")
    print("=" * 90)
    col_w = 20
    header = f"{'Metric':<22} | " + " | ".join([f"{name[:18]:<{col_w}}" for name, _ in modes])
    print(header)
    print("-" * 90)

    metrics_keys = [
        ("Hit Rate @ 1", "hit_rate@1", "{:.1%}"),
        ("Hit Rate @ 3", "hit_rate@3", "{:.1%}"),
        ("Hit Rate @ 5", "hit_rate@5", "{:.1%}"),
        ("Precision @ 3", "precision@3", "{:.1%}"),
        ("Precision @ 5", "precision@5", "{:.1%}"),
        ("Recall @ 5", "recall@5", "{:.1%}"),
        ("MRR", "mrr", "{:.4f}"),
        ("Avg Latency (ms)", "avg_latency_ms", "{:.1f} ms"),
    ]

    for label, key, fmt in metrics_keys:
        row = f"{label:<22} | "
        vals = []
        for mode_title, _ in modes:
            val = benchmark_data["results"][mode_title]["metrics"][key]
            vals.append(f"{fmt.format(val):<{col_w}}")
        row += " | ".join(vals)
        print(row)

    print("=" * 90)
    print(f"Detailed results saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    run_benchmark()
