"""
rag_pipeline.py

Core RAG logic:
  1. Embed the user query (RETRIEVAL_QUERY task type)
  2. Search using Hybrid Retrieval (Dense Qdrant + Sparse BM25 via Reciprocal Rank Fusion)
  3. Build a grounded prompt and call Gemini to generate an answer
  4. Return the answer + source metadata for citation
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient

from rag.bm25_retriever import BM25Retriever
from rag.hybrid_retriever import reciprocal_rank_fusion, relative_score_fusion

# ============================================================
# Configuration
# ============================================================

QDRANT_PATH     = "data/qdrant"
COLLECTION_NAME = "hypertension_guidelines"
EMBED_MODEL     = "gemini-embedding-001"
LLM_MODEL       = "gemini-flash-latest"
TOP_K           = 5   # chunks to retrieve per query
RETRIEVAL_MODE  = "hybrid"  # "hybrid", "dense", "sparse"

SYSTEM_PROMPT = """You are an expert clinical evidence AI assistant specializing in evidence-based hypertension guidelines (including WHO, NICE, ESC, CDC, USPSTF, and indexed clinical references).

CONTEXT BOUNDARY:
1. Answer ONLY using the provided context passages below. Do not use outside medical knowledge or training data to invent recommendations.
2. CITATION RULE: Always attribute findings accurately to the specific document and organization named in the context (e.g. CDC, ESC, USPSTF, WHO, NICE). Never attribute recommendations to WHO unless the cited document is explicitly the WHO guideline.
3. If the provided context from the requested guideline does not address the question, state clearly that the targeted guideline context does not contain this specific information, and summarize what relevant topics are present.
4. Format your answer clearly with bold text, bullet points, and citations [1], [2], etc."""


# ============================================================
# Setup
# ============================================================

load_dotenv()

_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise EnvironmentError("GEMINI_API_KEY not found in .env")

_client = genai.Client(api_key=_api_key)
_qdrant_client: QdrantClient | None = None

def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
    return _qdrant_client

_bm25_retriever: BM25Retriever | None = None

def get_bm25_retriever(force_reload: bool = False) -> BM25Retriever:
    global _bm25_retriever
    chunks_path = Path(__file__).parent.parent.parent / "data" / "processed" / "validated_chunks.json"
    if _bm25_retriever is None:
        _bm25_retriever = BM25Retriever(chunks_path=chunks_path)
    elif force_reload:
        _bm25_retriever.reload(chunks_path=chunks_path)
    return _bm25_retriever


# ============================================================
# Data classes
# ============================================================

@dataclass
class Source:
    chunk_id:      str
    document_id:   str
    document_name: str
    pdf_file:      str
    section_title: str
    page_start:    int
    page_end:      int
    score:         float
    text:          str


@dataclass
class RAGResult:
    answer:  str
    sources: list[Source]


# ============================================================
# Pipeline steps
# ============================================================

def _embed_query(query: str) -> list[float]:
    """Embed the user query for retrieval (asymmetric: RETRIEVAL_QUERY)."""
    response = _client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return response.embeddings[0].values


def _format_source(item: dict, score: float) -> Source:
    doc_to_file = {
        "who_guideline_01": "WHO_guideline_01.pdf",
        "doc_001": "WHO_guideline_01.pdf",
        "nice_guideline_02": "NICE_guideline_02.pdf",
    }
    doc_id = item.get("document_id", "who_guideline_01")
    pdf_file = item.get("pdf_file") or doc_to_file.get(doc_id, f"{doc_id}.pdf")
    doc_name = item.get("document_name") or (
        "WHO Guideline for the Pharmacological Treatment of Hypertension in Adults"
        if "who" in doc_id.lower()
        else doc_id.replace("_", " ").title()
    )
    return Source(
        chunk_id      = item.get("chunk_id", ""),
        document_id   = doc_id,
        document_name = doc_name,
        pdf_file      = pdf_file,
        section_title = item.get("section_title", ""),
        page_start    = item.get("page_start", 0),
        page_end      = item.get("page_end", 0),
        score         = round(score, 4),
        text          = item.get("text", ""),
    )



def _resolve_target_doc(doc_filter: str | None) -> dict | None:
    """
    Resolve user filter string (e.g. 'cdc', 'doc_cdc_164016_ds1', 'WHO_guideline_01.pdf', 'ESC')
    to its canonical document metadata in source_registry.json.
    """
    if not doc_filter:
        return None
    flt = doc_filter.strip().lower().replace(".pdf", "").replace("@", "")
    if not flt:
        return None

    import json
    registry = []
    reg_path = Path(__file__).parent.parent.parent / "data" / "source_registry.json"
    if reg_path.exists():
        try:
            with open(reg_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception:
            registry = []

    # 1. Exact matches on document_id, pdf_file, or organization
    for doc in registry:
        doc_id = str(doc.get("document_id", "")).lower()
        pdf_file = str(Path(doc.get("local_file", "")).name).lower().replace(".pdf", "")
        org = str(doc.get("organization", "")).lower().replace(")", "").replace(".", "").strip()
        doc_name = str(doc.get("document_name", "")).lower()
        if flt == doc_id or flt == pdf_file or flt == org or flt == doc_name:
            return doc

    # 2. Token / prefix matches on org, doc_id, or pdf_file
    for doc in registry:
        doc_id = str(doc.get("document_id", "")).lower()
        pdf_file = str(Path(doc.get("local_file", "")).name).lower().replace(".pdf", "")
        org = str(doc.get("organization", "")).lower().replace(")", "").replace(".", "").strip()
        if (org and (flt == org or org in flt or flt in org)) or \
           (doc_id and (flt == doc_id or flt in doc_id or doc_id in flt)) or \
           (pdf_file and (flt == pdf_file or flt in pdf_file or pdf_file in flt)):
            return doc

    return {"document_id": flt, "pdf_file": f"{flt}.pdf", "document_name": flt}


def _doc_matches(item: dict, target_info: dict | None) -> bool:
    """Check if a retrieved chunk payload strictly matches the target document."""
    if not target_info:
        return True

    item_doc_id = str(item.get("document_id", "")).lower()
    item_pdf_file = str(item.get("pdf_file", "")).lower().replace(".pdf", "")

    target_id = str(target_info.get("document_id", "")).lower()
    target_pdf = str(Path(target_info.get("local_file", target_info.get("pdf_file", ""))).name).lower().replace(".pdf", "")
    target_org = str(target_info.get("organization", "")).lower().replace(")", "").replace(".", "").strip()

    if target_id and (target_id == item_doc_id or target_id in item_doc_id):
        return True
    if target_pdf and (target_pdf == item_pdf_file or target_pdf in item_pdf_file):
        return True
    if target_org and (target_org in item_doc_id or target_org in item_pdf_file):
        return True

    return False


def _retrieve_dense(query_vector: list[float], top_k: int = TOP_K, doc_filter: str | None = None) -> list[dict]:
    """Search Qdrant for dense vector similarity with optional document filtering."""
    target_info = _resolve_target_doc(doc_filter)
    qclient = get_qdrant_client()
    hits = qclient.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=max(top_k * 6, 40) if target_info else top_k,
        with_payload=True,
    ).points

    results = []
    for hit in hits:
        item = dict(hit.payload)
        if target_info and not _doc_matches(item, target_info):
            continue
        item["dense_score"] = float(hit.score)
        item["score"] = float(hit.score)
        results.append(item)
        if len(results) >= top_k:
            break
    return results


def _retrieve_sparse(query: str, top_k: int = TOP_K, doc_filter: str | None = None) -> list[dict]:
    """Search BM25 index for lexical keyword similarity with optional document filtering."""
    target_info = _resolve_target_doc(doc_filter)
    retriever = get_bm25_retriever()
    hits = retriever.search(query, top_k=max(top_k * 6, 40) if target_info else top_k)
    filtered = []
    for h in hits:
        if target_info and not _doc_matches(h, target_info):
            continue
        h["score"] = h.get("bm25_score", 0.0)
        filtered.append(h)
        if len(filtered) >= top_k:
            break
    return filtered


def _retrieve(
    query_vector: list[float] | None = None,
    query_text: str = "",
    top_k: int = TOP_K,
    mode: str = RETRIEVAL_MODE,
    doc_filter: str | None = None,
) -> list[Source]:
    """Hybrid, Dense, or Sparse retrieval returning unified top-k Source objects."""
    if mode == "dense":
        if query_vector is None:
            query_vector = _embed_query(query_text)
        items = _retrieve_dense(query_vector, top_k=top_k, doc_filter=doc_filter)
        return [_format_source(item, item["dense_score"]) for item in items]

    elif mode == "sparse":
        items = _retrieve_sparse(query_text, top_k=top_k, doc_filter=doc_filter)
        return [_format_source(item, item["bm25_score"]) for item in items]

    else:
        # Default: Hybrid retrieval with Reciprocal Rank Fusion
        if query_vector is None:
            query_vector = _embed_query(query_text)
        dense_hits = _retrieve_dense(query_vector, top_k=max(top_k * 2, 10), doc_filter=doc_filter)
        sparse_hits = _retrieve_sparse(query_text, top_k=max(top_k * 2, 10), doc_filter=doc_filter)
        fused_hits = reciprocal_rank_fusion(dense_hits, sparse_hits, top_k=top_k, c=60, dense_weight=0.5)
        return [_format_source(item, item["score"]) for item in fused_hits]


def _build_context(sources: list[Source]) -> str:
    """Format retrieved chunks into a numbered context block with document, section, and page metadata."""
    if not sources:
        return "[No matching context passages retrieved for this specific guideline.]"

    blocks = []
    for i, s in enumerate(sources, 1):
        page_str = f"Page {s.page_start}" if s.page_start == s.page_end else f"Pages {s.page_start}-{s.page_end}"
        section_str = f", Section: {s.section_title}" if s.section_title else ""
        header = f"[{i}] Document: {s.document_name}{section_str} | {page_str}"
        blocks.append(f"{header}\n{s.text}")
    return "\n\n---\n\n".join(blocks)


FALLBACK_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
]


def _generate(
    query: str,
    context: str,
    history: list[dict] | None = None,
    doc_filter: str | None = None,
    patient_context: dict | str | None = None,
) -> str:
    """Call Gemini with conversation history, optional guideline target, patient context, and grounded context."""
    # Build conversation turns for multi-turn memory
    history_block = ""
    if history:
        turns = []
        for turn in history[-6:]:  # Keep last 6 turns to avoid token overflow
            turns.append(f"User: {turn['query']}\nAssistant: {turn['answer']}")
        history_block = "\n\n".join(turns) + "\n\n"

    target_info = _resolve_target_doc(doc_filter)
    target_name = target_info.get("document_name", doc_filter) if target_info else doc_filter
    target_org = target_info.get("organization", "") if target_info else ""

    target_note = ""
    if doc_filter:
        target_note = (
            f"\nIMPORTANT TARGET GUIDELINE INSTRUCTION: The user has targeted the guideline: '{target_name}' (Organization: {target_org}). "
            f"You must base your answer EXCLUSIVELY on the provided passages from this specific guideline. "
            f"Do not cite or reference other organizations (such as WHO or others) unless they appear in this document's text. "
            f"If the provided context from this guideline does not contain the answer, explicitly state that this guideline does not mention it.\n"
        )

    patient_note = ""
    if patient_context:
        if isinstance(patient_context, dict):
            vitals = patient_context.get("vitals", patient_context)
            sbp = vitals.get("systolic", "Unknown")
            dbp = vitals.get("diastolic", "Unknown")
            cat = vitals.get("bp_category", "Hypertension")
            comorb = ", ".join(vitals.get("comorbidities", [])) or "None specified"
            meds = ", ".join(vitals.get("medications", [])) or "None reported"
            summary = patient_context.get("summary", "")
            patient_note = (
                f"\nPATIENT'S UPLOADED TEST RESULTS & CLINICAL REPORT:\n"
                f"- Blood Pressure: {sbp}/{dbp} mmHg ({cat})\n"
                f"- Comorbidities / Risk Factors: {comorb}\n"
                f"- Current Medications: {meds}\n"
                f"- Report Summary: {summary}\n\n"
                f"TAILORING INSTRUCTION: Specifically contextualize your answer to this patient's blood pressure level and risk factors using the guideline thresholds and evidence from context.\n"
            )
        else:
            patient_note = f"\nPATIENT'S CLINICAL REPORT CONTEXT:\n{patient_context}\n\n"

    prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"{patient_note}"
        f"{history_block}"
        f"{target_note}"
        f"QUESTION: {query}\n\n"
        f"ANSWER:"
    )
    last_err = None
    for model_name in FALLBACK_MODELS:
        for attempt in range(2):
            try:
                response = _client.models.generate_content(
                    model=model_name,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1,
                        max_output_tokens=1024,
                    ),
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                last_err = e
                import time
                time.sleep(1.0)
    raise RuntimeError(f"All LLM models failed. Last error: {last_err}")


# ============================================================
# Public entry point
# ============================================================

def answer(
    query: str,
    mode: str = RETRIEVAL_MODE,
    top_k: int = TOP_K,
    history: list[dict] | None = None,
    doc_filter: str | None = None,
    patient_context: dict | str | None = None,
) -> RAGResult:
    """Run the full RAG pipeline and return an answer with sources, optionally filtering by specific document and tailoring to patient test results."""
    # If patient context is present, enrich the query vector search slightly to retrieve relevant guideline sections (e.g. thresholds, comorbidity treatment)
    search_query = query
    if patient_context and isinstance(patient_context, dict):
        vitals = patient_context.get("vitals", {})
        cat = vitals.get("bp_category", "")
        comorb = " ".join(vitals.get("comorbidities", []))
        if cat or comorb:
            search_query = f"{query} {cat} {comorb}".strip()

    query_vec   = _embed_query(search_query)
    sources     = _retrieve(query_vector=query_vec, query_text=query, top_k=top_k, mode=mode, doc_filter=doc_filter)
    context     = _build_context(sources)
    answer_text = _generate(query, context, history=history, doc_filter=doc_filter, patient_context=patient_context)
    return RAGResult(answer=answer_text, sources=sources)


