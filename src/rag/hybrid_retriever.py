"""
hybrid_retriever.py

Combines dense vector retrieval (Qdrant / Gemini Embeddings) with sparse lexical retrieval (BM25Okapi).
Supports:
  1. Reciprocal Rank Fusion (RRF)
  2. Relative Score Normalization (Weighted Score Fusion)
"""

from typing import Any


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    top_k: int = 5,
    c: int = 60,
    dense_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """Compute Reciprocal Rank Fusion (RRF) over dense and sparse ranked lists.

    Score(d) = dense_weight / (c + rank_dense) + (1 - dense_weight) / (c + rank_sparse)
    """
    sparse_weight = 1.0 - dense_weight
    scores: dict[str, float] = {}
    item_map: dict[str, dict[str, Any]] = {}

    # Dense ranks (1-indexed)
    for rank, item in enumerate(dense_results, 1):
        cid = item["chunk_id"]
        item_map[cid] = item
        scores[cid] = scores.get(cid, 0.0) + (dense_weight / (c + rank))

    # Sparse ranks (1-indexed)
    for rank, item in enumerate(sparse_results, 1):
        cid = item["chunk_id"]
        if cid not in item_map:
            item_map[cid] = item
        scores[cid] = scores.get(cid, 0.0) + (sparse_weight / (c + rank))

    # Sort items by fused RRF score descending
    sorted_cids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]

    fused_results = []
    for cid in sorted_cids:
        res = dict(item_map[cid])
        res["fusion_score"] = round(scores[cid], 5)
        res["score"] = res["fusion_score"]  # unified score
        fused_results.append(res)

    return fused_results


def relative_score_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    top_k: int = 5,
    alpha: float = 0.6,
) -> list[dict[str, Any]]:
    """Weighted Relative Score Fusion with Min-Max normalization.

    Score(d) = alpha * norm(dense_score) + (1 - alpha) * norm(bm25_score)
    """
    def _min_max_norm(items: list[dict[str, Any]], score_key: str) -> dict[str, float]:
        if not items:
            return {}
        scores = [it.get(score_key, 0.0) for it in items]
        s_min, s_max = min(scores), max(scores)
        denom = s_max - s_min
        if denom == 0:
            return {it["chunk_id"]: 1.0 for it in items}
        return {it["chunk_id"]: (it.get(score_key, 0.0) - s_min) / denom for it in items}

    dense_norm = _min_max_norm(dense_results, "dense_score")
    sparse_norm = _min_max_norm(sparse_results, "bm25_score")

    item_map: dict[str, dict[str, Any]] = {}
    for it in dense_results:
        item_map[it["chunk_id"]] = it
    for it in sparse_results:
        if it["chunk_id"] not in item_map:
            item_map[it["chunk_id"]] = it

    all_cids = set(dense_norm.keys()) | set(sparse_norm.keys())
    combined_scores: dict[str, float] = {}

    for cid in all_cids:
        d_s = dense_norm.get(cid, 0.0)
        s_s = sparse_norm.get(cid, 0.0)
        combined_scores[cid] = alpha * d_s + (1.0 - alpha) * s_s

    sorted_cids = sorted(combined_scores.keys(), key=lambda cid: combined_scores[cid], reverse=True)[:top_k]

    fused_results = []
    for cid in sorted_cids:
        res = dict(item_map[cid])
        res["fusion_score"] = round(combined_scores[cid], 5)
        res["score"] = res["fusion_score"]
        fused_results.append(res)

    return fused_results
