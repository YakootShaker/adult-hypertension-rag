"""
benchmark_models.py

Head-to-head retrieval evaluation comparing:
  1. Google gemini-embedding-001 (3072 dims, Cloud API)
  2. sentence-transformers/all-MiniLM-L6-v2 (384 dims, Local CPU)

Evaluates on evaluation/test_set.json (10 clinical questions).
Computes: Hit Rate@k, Precision@k, Recall@k, MRR, and Latency (ms).
Outputs: evaluation/model_benchmark_results.json & Markdown summary.
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np

from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

TEST_SET_PATH   = "evaluation/test_set.json"
CHUNKS_PATH     = "data/processed/validated_chunks.json"
OUTPUT_JSON     = "evaluation/model_benchmark_results.json"
QDRANT_EVAL_DIR = "data/qdrant_eval"

K_VALUES = [1, 3, 5]


def load_test_set():
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Model Evaluator Base Class
# ============================================================

class ModelEvaluator:
    def __init__(self, name: str, vector_size: int, client: QdrantClient):
        self.name = name
        self.vector_size = vector_size
        self.collection_name = f"eval_{name.replace('/', '_').replace('-', '_')}"
        self.client = client

    def index_chunks(self, chunks: list):
        raise NotImplementedError

    def embed_query(self, query: str) -> list[float]:
        raise NotImplementedError

    def evaluate(self, test_set: list) -> dict:
        results = {
            "model_name": self.name,
            "vector_size": self.vector_size,
            "per_query": [],
            "metrics": {}
        }

        latencies = []
        hit_counts = {k: 0 for k in K_VALUES}
        precision_sums = {k: 0.0 for k in K_VALUES}
        recall_sums = {k: 0.0 for k in K_VALUES}
        mrr_sum = 0.0

        max_k = max(K_VALUES)

        for test_case in test_set:
            qid = test_case["query_id"]
            query = test_case["query"]
            expected_chunks = set(test_case["expected_chunk_ids"])
            expected_section = test_case.get("expected_section_id")

            # Measure latency
            t0 = time.perf_counter()
            q_vec = self.embed_query(query)
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=q_vec,
                limit=max_k,
                with_payload=True,
            ).points
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            latencies.append(latency_ms)

            retrieved_chunk_ids = [h.payload["chunk_id"] for h in hits]
            retrieved_sections = [h.payload.get("section_id") for h in hits]

            # Calculate MRR (First match by exact chunk OR target section)
            rank = None
            for idx, (cid, sid) in enumerate(zip(retrieved_chunk_ids, retrieved_sections), 1):
                if cid in expected_chunks or sid == expected_section:
                    rank = idx
                    break
            reciprocal_rank = 1.0 / rank if rank is not None else 0.0
            mrr_sum += reciprocal_rank

            query_record = {
                "query_id": qid,
                "latency_ms": round(latency_ms, 2),
                "mrr": round(reciprocal_rank, 4),
                "top_retrieved": retrieved_chunk_ids[:3],
            }

            for k in K_VALUES:
                k_chunks = retrieved_chunk_ids[:k]
                k_sections = retrieved_sections[:k]
                # Relevant if exact chunk matched OR chunk belongs to expected section
                matches = sum(1 for cid, sid in zip(k_chunks, k_sections) if cid in expected_chunks or sid == expected_section)
                
                # Hit rate: at least 1 match
                hit = 1 if matches > 0 else 0
                hit_counts[k] += hit
                
                # Precision@k: matches / k
                precision = matches / k
                precision_sums[k] += precision
                
                # Recall@k: matches / total expected chunks (bounded by 1.0)
                recall = min(1.0, matches / max(1, len(expected_chunks)))
                recall_sums[k] += recall

                query_record[f"p@{k}"] = round(precision, 4)
                query_record[f"recall@{k}"] = round(recall, 4)
                query_record[f"hit@{k}"] = hit

            results["per_query"].append(query_record)

        n = len(test_set)
        results["metrics"] = {
            "avg_latency_ms": round(float(np.mean(latencies)), 2),
            "mrr": round(mrr_sum / n, 4),
        }
        for k in K_VALUES:
            results["metrics"][f"hit_rate@{k}"] = round(hit_counts[k] / n, 4)
            results["metrics"][f"precision@{k}"] = round(precision_sums[k] / n, 4)
            results["metrics"][f"recall@{k}"] = round(recall_sums[k] / n, 4)

        return results


# ============================================================
# Gemini Evaluator
# ============================================================

class GeminiEvaluator(ModelEvaluator):
    def __init__(self, client: QdrantClient):
        super().__init__(name="gemini-embedding-001", vector_size=3072, client=client)
        api_key = os.getenv("GEMINI_API_KEY")
        self.genai_client = genai.Client(api_key=api_key)

    def index_chunks(self, chunks: list):
        print(f"Indexing {len(chunks)} chunks into Gemini Qdrant collection...")
        # Check existing embeddings in data/processed/embeddings.json to avoid redundant API calls
        if os.path.exists("data/processed/embeddings.json"):
            with open("data/processed/embeddings.json", "r", encoding="utf-8") as f:
                records = json.load(f)
        else:
            raise FileNotFoundError("Run src/embeddings/embed_chunks.py first.")

        # Recreate collection
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing:
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

        import uuid
        points = []
        for r in records:
            cid = r["chunk_id"]
            uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"gemini_{cid}"))
            points.append(
                PointStruct(
                    id=uid,
                    vector=r["embedding"],
                    payload={
                        "chunk_id": cid,
                        "section_id": r.get("section_id"),
                        "section_title": r.get("section_title"),
                        "text": r.get("text"),
                    }
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)

    def embed_query(self, query: str) -> list[float]:
        import time
        for attempt in range(5):
            try:
                resp = self.genai_client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=query,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
                )
                return resp.embeddings[0].values
            except Exception as e:
                time.sleep(3.0)
        raise RuntimeError("Gemini embed query failed after retries.")


# ============================================================
# Sentence Transformers Evaluator
# ============================================================

class LocalSTEvaluator(ModelEvaluator):
    def __init__(self, client: QdrantClient, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(name=f"local-{model_name}", vector_size=384, client=client)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def index_chunks(self, chunks: list):
        print(f"Embedding & indexing {len(chunks)} chunks with {self.name}...")
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing:
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

        import uuid
        points = []
        for c, emb in zip(chunks, embeddings):
            cid = c["chunk_id"]
            uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"local_{cid}"))
            points.append(
                PointStruct(
                    id=uid,
                    vector=emb.tolist(),
                    payload={
                        "chunk_id": cid,
                        "section_id": c.get("section_id"),
                        "section_title": c.get("section_title"),
                        "text": c.get("text"),
                    }
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode([query], normalize_embeddings=True)[0].tolist()


# ============================================================
# Main Runner
# ============================================================

def main():
    print("=" * 70)
    print("DAY 2: HEAD-TO-HEAD EMBEDDING MODEL BENCHMARK")
    print("=" * 70)

    test_set = load_test_set()
    chunks = load_chunks()
    print(f"Loaded {len(test_set)} test queries across {len(chunks)} chunks.\n")

    shared_client = QdrantClient(path=QDRANT_EVAL_DIR)

    # 1. Evaluate Gemini
    print(">>> 1. Benchmarking Gemini Embedding 001 (Cloud API, 3072 dims)...")
    gemini_eval = GeminiEvaluator(shared_client)
    gemini_eval.index_chunks(chunks)
    gemini_results = gemini_eval.evaluate(test_set)
    print("Gemini evaluation complete.")

    # 2. Evaluate Local SentenceTransformer
    print("\n>>> 2. Benchmarking all-MiniLM-L6-v2 (Local CPU, 384 dims)...")
    st_eval = LocalSTEvaluator(shared_client, "all-MiniLM-L6-v2")
    st_eval.index_chunks(chunks)
    st_results = st_eval.evaluate(test_set)
    print("Local model evaluation complete.")

    # Save Results
    all_results = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_set_size": len(test_set),
        "models": {
            "gemini-embedding-001": gemini_results,
            "all-MiniLM-L6-v2": st_results,
        }
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Print Comparison Table
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Metric':<24} | {'gemini-embedding-001':<22} | {'all-MiniLM-L6-v2':<20}")
    print("-" * 70)
    print(f"{'Vector Dimensions':<24} | {'3072':<22} | {'384':<20}")
    print(f"{'Hit Rate @ 1':<24} | {gemini_results['metrics']['hit_rate@1']:<22.2%} | {st_results['metrics']['hit_rate@1']:<20.2%}")
    print(f"{'Hit Rate @ 3':<24} | {gemini_results['metrics']['hit_rate@3']:<22.2%} | {st_results['metrics']['hit_rate@3']:<20.2%}")
    print(f"{'Hit Rate @ 5':<24} | {gemini_results['metrics']['hit_rate@5']:<22.2%} | {st_results['metrics']['hit_rate@5']:<20.2%}")
    print(f"{'Precision @ 3':<24} | {gemini_results['metrics']['precision@3']:<22.2%} | {st_results['metrics']['precision@3']:<20.2%}")
    print(f"{'Precision @ 5':<24} | {gemini_results['metrics']['precision@5']:<22.2%} | {st_results['metrics']['precision@5']:<20.2%}")
    print(f"{'Mean Recip. Rank (MRR)':<24} | {gemini_results['metrics']['mrr']:<22.4f} | {st_results['metrics']['mrr']:<20.4f}")
    print(f"{'Avg Latency (ms)':<24} | {gemini_results['metrics']['avg_latency_ms']:<19.1f} ms | {st_results['metrics']['avg_latency_ms']:<17.1f} ms")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
