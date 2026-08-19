"""
auto_ingest.py — Universal Automated Ingestion Pipeline for Clinical Guideline PDFs.

Supports all major medical guideline formats & typography styles:
- WHO, NICE, ACC/AHA, ESC, ISH, JNC8, KDIGO, ADA, CHEST, and Ministry of Health guidelines.
- Handles PDF native bookmarks (TOC), printed TOC pages, decimal numbering (1.1, 1.1.1),
  Roman numerals (I., IV.), clinical prefixes (Recommendation, Chapter, Section, Step),
  all-caps titles, title-case bold headings, multi-line titles, and two-column layouts.
- Dynamic running header/footer filtering and page-exact citation tracking.
- Embedding generation (Gemini) + incremental Qdrant vector indexing + BM25 keyword sync.
"""

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import pymupdf as fitz
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import tiktoken

from ingestion.cleaner import clean_text

load_dotenv()

# ============================================================
# Paths & Defaults
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
QDRANT_PATH = DATA_DIR / "qdrant"
SOURCE_REGISTRY_PATH = DATA_DIR / "source_registry.json"
CHUNKS_FILE_PATH = PROCESSED_DIR / "validated_chunks.json"

COLLECTION_NAME = "hypertension_guidelines"
EMBEDDING_MODEL = "gemini-embedding-001"
VECTOR_SIZE = 3072
DISTANCE = Distance.COSINE

TARGET_TOKENS = 600
MAX_TOKENS = 900
MIN_TOKENS = 400
OVERLAP_TOKENS = 80

TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ============================================================
# Helper Functions
# ============================================================

def encode_text(text: str) -> list[int]:
    return TOKENIZER.encode(text, disallowed_special=())


def decode_tokens(tokens: list[int]) -> str:
    return TOKENIZER.decode(tokens)


def count_tokens(text: str) -> int:
    return len(encode_text(text))


def chunk_id_to_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


# ============================================================
# Universal Heading & Taxonomy Patterns
# ============================================================

# 1. Numbered hierarchies: 1., 1.1, 1.1.1, 1.2.3.4, 1), (1)
DECIMAL_NUMBER_PATTERN = re.compile(
    r"^(?:\d+\.){1,4}\s*(?:\d+\.?)?\s+[A-Z0-9\(\"']",
    re.IGNORECASE
)

# 2. Roman numerals: I., II., III., IV., VII., etc.
ROMAN_NUMERAL_PATTERN = re.compile(
    r"^(?:[IVXLCDM]+\.)\s+[A-Z0-9]",
    re.IGNORECASE
)

# 3. Clinical guideline structural prefixes
CLINICAL_PREFIX_PATTERN = re.compile(
    r"^(?:RECOMMENDATION|RECOMMENDATIONS|SECTION|CHAPTER|PART|MODULE|STEP|STAGE|ALGORITHM|BOX|TABLE|KEY\s+MESSAGE|PICO|GRADE|CLASS)\s+(?:[0-9IVXLCDM\.\-]+|\b(?:ONE|TWO|THREE|FOUR)\b)",
    re.IGNORECASE
)

# 4. Standard clinical topics & all-caps section titles
CLINICAL_TOPIC_PATTERN = re.compile(
    r"^(?:EXECUTIVE\s+SUMMARY|INTRODUCTION|BACKGROUND|EPIDEMIOLOGY|DIAGNOSIS|INVESTIGATIONS|ASSESSMENT|"
    r"MANAGEMENT|PHARMACOLOGICAL\s+TREATMENT|NON-PHARMACOLOGICAL\s+MANAGEMENT|LIFESTYLE\s+INTERVENTIONS|"
    r"DRUG\s+THERAPY|MONITORING|BLOOD\s+PRESSURE\s+TARGETS|CARDIOVASCULAR\s+RISK|TARGET\s+ORGAN\s+DAMAGE|"
    r"SPECIAL\s+POPULATIONS|SPECIAL\s+SETTINGS|DIABETES|CHRONIC\s+KIDNEY\s+DISEASE|PREGNANCY|ELDERLY|"
    r"SECONDARY\s+HYPERTENSION|RESISTANT\s+HYPERTENSION|HYPERTENSIVE\s+EMERGENCIES|HYPERTENSIVE\s+CRISIS|"
    r"FOLLOW-UP|IMPLEMENTATION|RATIONALE\s+AND\s+IMPACT|TERMS\s+USED\s+IN\s+THIS\s+GUIDELINE|"
    r"RECOMMENDATIONS\s+FOR\s+RESEARCH|CONTEXT)$",
    re.IGNORECASE
)

# Sections to exclude from clinical RAG knowledge base
EXCLUDED_TITLES = {
    "REFERENCES", "BIBLIOGRAPHY", "ANNEXES", "ANNEX", "APPENDIX", "APPENDICES",
    "INDEX", "DISCLOSURES", "CONFLICTS OF INTEREST", "ACKNOWLEDGEMENTS",
    "LIST OF CONTRIBUTORS", "WEB ANNEX"
}

# Ignore standard noise / metadata lines
NOISE_LINE_PATTERNS = [
    re.compile(r"^https?://", re.IGNORECASE),
    re.compile(r"^www\.", re.IGNORECASE),
    re.compile(r"^published\s*:", re.IGNORECASE),
    re.compile(r"^last\s+updated\s*:", re.IGNORECASE),
    re.compile(r"^page\s+\d+", re.IGNORECASE),
    re.compile(r"^isbn\s*:", re.IGNORECASE),
    re.compile(r"^doi\s*:", re.IGNORECASE),
    re.compile(r"^©\s*\d{4}", re.IGNORECASE),
    re.compile(r"^downloaded\s+from", re.IGNORECASE),
]


def is_noise_line(line: str) -> bool:
    """Check if a line is a URL, copyright notice, or publication metadata."""
    line = line.strip()
    if not line:
        return True
    if line.isdigit() or (line.startswith("[") and line.endswith("]")):
        return True
    for pat in NOISE_LINE_PATTERNS:
        if pat.search(line):
            return True
    return False


def is_rule_based_heading(line: str) -> bool:
    """Check if line matches any standard clinical or numbered heading patterns."""
    line = line.strip()
    if len(line) < 3 or len(line) > 140 or is_noise_line(line):
        return False

    # Check decimal pattern (e.g. 1.1 Measuring blood pressure)
    if DECIMAL_NUMBER_PATTERN.match(line):
        return True

    # Check Roman numerals (e.g. II. Diagnosis of Hypertension)
    if ROMAN_NUMERAL_PATTERN.match(line):
        return True

    # Check clinical prefixes (e.g. Recommendation 1.4, Step 1 Treatment)
    if CLINICAL_PREFIX_PATTERN.match(line):
        return True

    # Check clinical standard topics (e.g. Lifestyle Interventions, Executive Summary)
    if CLINICAL_TOPIC_PATTERN.match(line):
        return True

    # Short all-caps lines without period (e.g. SPECIAL POPULATIONS)
    if line == line.upper() and len(line) >= 4 and len(line) <= 60 and not line.endswith((".", ":", ";")):
        if any(c.isalpha() for c in line):
            return True

    return False


# ============================================================
# Step 1: PDF Extraction & Cleaning
# ============================================================

def extract_and_clean_pdf(pdf_path: Path | str) -> list[dict[str, Any]]:
    """Extract and clean text from each page of the PDF."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    pages: list[dict[str, Any]] = []

    for page_num, page in enumerate(doc, start=1):
        raw_text = page.get_text("text")
        cleaned = clean_text(raw_text)
        if cleaned.strip():
            pages.append({
                "page_number": page_num,
                "text": cleaned,
            })

    doc.close()
    return pages


# ============================================================
# Step 2: Multi-Strategy Universal Section Detection
# ============================================================

def detect_document_sections(pdf_path: Path | str, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Multi-strategy section detector supporting ALL guideline styles:
    Strategy 1: PDF Native Bookmarks / Outlines (doc.get_toc())
    Strategy 2: Visual Layout & Font Hierarchy (size + weight + spacing)
    Strategy 3: Universal Clinical Pattern Recognition
    Strategy 4: Adaptive Fallback
    """
    doc = fitz.open(str(pdf_path))
    detected_headings: list[tuple[int, str]] = []
    seen_titles: set[str] = set()

    # --- Strategy 1: PDF Native Outlines / Bookmarks ---
    try:
        toc = doc.get_toc(simple=True)
        if toc and len(toc) >= 3:
            for item in toc:
                lvl, title, page_num = item[0], item[1].strip(), item[2]
                title_clean = re.sub(r"\s+", " ", title).strip(":-. ")
                title_upper = title_clean.upper()

                if (
                    title_clean
                    and page_num > 0
                    and not any(ex in title_upper for ex in EXCLUDED_TITLES)
                    and title_clean not in seen_titles
                    and len(title_clean) < 140
                ):
                    seen_titles.add(title_clean)
                    detected_headings.append((page_num, title_clean))

            if len(detected_headings) >= 4:
                doc.close()
                return _build_section_ranges(detected_headings, pages)
    except Exception:
        pass  # Fall through to visual & pattern extraction

    # --- Strategy 2: Visual Typography Hierarchy ---
    font_sizes: dict[float, int] = {}
    for page in doc:
        blocks = page.get_text("dict").get("blocks", [])
        for b in blocks:
            if "lines" in b:
                for l in b["lines"]:
                    for s in l["spans"]:
                        t = s["text"].strip()
                        if t:
                            sz = round(s["size"], 1)
                            font_sizes[sz] = font_sizes.get(sz, 0) + len(t)

    body_size = max(font_sizes.items(), key=lambda x: x[1])[0] if font_sizes else 10.0

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict").get("blocks", [])
        for b in blocks:
            if "lines" not in b:
                continue

            block_heading_lines: list[str] = []
            for l in b["lines"]:
                line_text = " ".join(s["text"].strip() for s in l["spans"] if s["text"].strip()).strip()
                if not line_text or len(line_text) < 3 or is_noise_line(line_text):
                    continue

                max_sz = max(round(s["size"], 1) for s in l["spans"])
                is_bold = any(
                    "Bold" in s.get("font", "")
                    or "bold" in s.get("font", "")
                    or "SemiBold" in s.get("font", "")
                    or "Heavy" in s.get("font", "")
                    or "Black" in s.get("font", "")
                    or "ExtraBold" in s.get("font", "")
                    for s in l["spans"]
                )

                # Heading conditions:
                # A. Larger font than body (H1/H2)
                # B. Bold with font >= body size
                # C. Matches clinical or numbered patterns
                is_heading = (
                    (max_sz >= body_size + 2.0)
                    or (is_bold and max_sz >= body_size + 0.5)
                    or (is_bold and is_rule_based_heading(line_text))
                    or is_rule_based_heading(line_text)
                )

                if is_heading:
                    block_heading_lines.append(line_text)

            if block_heading_lines:
                merged_title = " ".join(block_heading_lines)
                merged_title = re.sub(r"\s+", " ", merged_title).strip()
                title_clean = merged_title.strip(":-. ")
                title_upper = title_clean.upper()

                if (
                    title_clean
                    and len(title_clean) < 140
                    and title_clean.lower() not in {"contents", "table of contents", "your responsibility"}
                    and not any(ex in title_upper for ex in EXCLUDED_TITLES)
                    and title_clean not in seen_titles
                ):
                    seen_titles.add(title_clean)
                    detected_headings.append((page_num, title_clean))

    doc.close()

    # --- Strategy 3: Text-based line scanner fallback if font detection yielded few sections ---
    if len(detected_headings) < 3:
        for page in pages:
            p_num = page["page_number"]
            for line in page["text"].splitlines():
                line = line.strip()
                if is_rule_based_heading(line):
                    title_clean = line.strip(":-. ")
                    if title_clean not in seen_titles and len(title_clean) < 140:
                        seen_titles.add(title_clean)
                        detected_headings.append((p_num, title_clean))

    # --- Strategy 4: Adaptive Section Bucketing fallback ---
    if len(detected_headings) < 2 and pages:
        sections = []
        step = 5
        for i in range(0, len(pages), step):
            start_p = pages[i]["page_number"]
            end_p = pages[min(i + step - 1, len(pages) - 1)]["page_number"]
            sections.append({
                "section_id": f"sec_{i // step + 1:03d}",
                "section_title": f"Guideline Content (Pages {start_p}-{end_p})",
                "start_page": start_p,
                "end_page": end_p,
            })
        return sections

    return _build_section_ranges(detected_headings, pages)


def _build_section_ranges(
    detected_headings: list[tuple[int, str]], pages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Convert (page, title) tuples into continuous section ranges."""
    sections = []
    for i, (start_page, title) in enumerate(detected_headings):
        if i + 1 < len(detected_headings):
            end_page = detected_headings[i + 1][0] - 1
            if end_page < start_page:
                end_page = start_page
        else:
            end_page = pages[-1]["page_number"] if pages else start_page

        sections.append({
            "section_id": f"sec_{i + 1:03d}",
            "section_title": title,
            "start_page": start_page,
            "end_page": end_page,
        })
    return sections


# ============================================================
# Step 3: Chunking with Accurate Page Tracking
# ============================================================

class SectionCursor:
    """Tracks token positions across pages to map chunk text to exact page spans."""

    def __init__(self, page_token_map: list[tuple[int, list[int]]], overlap_tokens: int = OVERLAP_TOKENS):
        self.page_token_map = page_token_map
        self.overlap_tokens = overlap_tokens
        self.boundaries: list[tuple[int, int, int]] = []
        cursor = 0
        for page_num, page_tokens in page_token_map:
            length = len(page_tokens)
            self.boundaries.append((page_num, cursor, cursor + length - 1))
            cursor += length
        self.total_tokens = cursor
        self.position = 0

    def page_of_token(self, token_idx: int) -> int:
        for page_num, start, end in self.boundaries:
            if start <= token_idx <= end:
                return page_num
        return self.boundaries[-1][0] if self.boundaries else 0

    def resolve(self, chunk_token_count: int) -> tuple[int, int]:
        if not self.boundaries:
            return 0, 0
        chunk_start = self.position
        chunk_end = min(self.position + chunk_token_count - 1, self.total_tokens - 1)
        p_start = self.page_of_token(chunk_start)
        p_end = self.page_of_token(chunk_end)
        advance = chunk_token_count - self.overlap_tokens
        if advance <= 0:
            advance = chunk_token_count
        self.position = min(self.position + advance, self.total_tokens)
        return p_start, p_end


def split_text_into_chunks(text: str) -> list[str]:
    """Token-bounded sliding window with small-chunk absorption."""
    tokens = encode_text(text)
    if not tokens:
        return []
    if len(tokens) <= MAX_TOKENS:
        return [text.strip()]

    raw_chunks: list[list[int]] = []
    start = 0
    total = len(tokens)

    while start < total:
        end = min(start + TARGET_TOKENS, total)
        raw_chunks.append(tokens[start:end])
        if end >= total:
            break
        next_start = end - OVERLAP_TOKENS
        if next_start <= start:
            next_start = end
        start = next_start

    # Absorb small tail chunks
    i = len(raw_chunks) - 1
    while i >= 1:
        if len(raw_chunks[i]) < MIN_TOKENS:
            merged = raw_chunks[i - 1] + raw_chunks[i]
            if len(merged) <= MAX_TOKENS:
                raw_chunks[i - 1] = merged
                raw_chunks.pop(i)
        i -= 1

    return [decode_tokens(c).strip() for c in raw_chunks]


def build_document_chunks(
    pages: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    document_id: str,
    document_name: str,
    pdf_filename: str,
) -> list[dict[str, Any]]:
    """Build standardized chunks with accurate metadata."""
    chunks: list[dict[str, Any]] = []

    for sec in sections:
        sec_title = sec["section_title"]
        if any(ex in sec_title.upper() for ex in EXCLUDED_TITLES):
            continue

        sec_pages = [p for p in pages if sec["start_page"] <= p["page_number"] <= sec["end_page"]]
        if not sec_pages:
            continue

        sec_text = "\n\n".join(p["text"].strip() for p in sec_pages if p["text"].strip())
        if not sec_text.strip():
            continue

        sec_chunks = split_text_into_chunks(sec_text)
        page_token_map = [(p["page_number"], encode_text(p["text"])) for p in sec_pages]
        cursor = SectionCursor(page_token_map)

        for idx, chunk_text in enumerate(sec_chunks, start=1):
            p_start, p_end = cursor.resolve(count_tokens(chunk_text))
            chunk_id = f"{document_id}_{sec['section_id']}_c{idx:03d}"
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "document_name": document_name,
                "pdf_file": pdf_filename,
                "section_id": sec["section_id"],
                "section_title": sec_title,
                "page_start": p_start,
                "page_end": p_end,
                "token_count": count_tokens(chunk_text),
                "text": chunk_text,
            })

    return chunks


# ============================================================
# Step 4: Embedding Generation
# ============================================================

def generate_embeddings_for_chunks(
    chunks: list[dict[str, Any]],
    batch_size: int = 15,
    max_retries: int = 6,
    progress_callback=None,
    pct_start: float = 42.0,
    pct_end: float = 85.0,
) -> list[list[float]]:
    """
    Generate Gemini embeddings for chunks using batch requests and
    automatic exponential backoff for 429 (Resource Exhausted / Rate Limit) errors.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not found in .env")

    client = genai.Client(api_key=api_key)
    all_embeddings: list[list[float]] = []

    total_chunks = len(chunks)
    num_batches  = (total_chunks + batch_size - 1) // batch_size
    print(f"  -> Generating embeddings for {total_chunks} chunks (batch size = {batch_size})...")

    for batch_num, start_idx in enumerate(range(0, total_chunks, batch_size), 1):
        end_idx    = min(start_idx + batch_size, total_chunks)
        batch      = chunks[start_idx:end_idx]
        batch_texts= [c["text"] for c in batch]

        success  = False
        last_exc = None

        for attempt in range(1, max_retries + 1):
            try:
                res = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch_texts,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )
                for emb in res.embeddings:
                    all_embeddings.append(emb.values)
                success = True
                done_pct = pct_start + (batch_num / num_batches) * (pct_end - pct_start)
                msg = f"Embedding batch {batch_num}/{num_batches} — {len(all_embeddings)}/{total_chunks} chunks embedded"
                print(f"     [{msg}]")
                if progress_callback:
                    progress_callback("embed", msg, round(done_pct, 1))
                break
            except Exception as exc:
                last_exc = exc
                err_msg  = str(exc).lower()
                if "429" in err_msg or "resource_exhausted" in err_msg or "quota" in err_msg:
                    wait_sec = min(15.0 * attempt, 60.0)
                    wait_msg = f"Rate limit hit — waiting {wait_sec:.0f}s before retry (attempt {attempt}/{max_retries})"
                    print(f"     [Rate limit 429 hit] Waiting {wait_sec:.0f}s before retry (attempt {attempt}/{max_retries})...")
                    if progress_callback:
                        progress_callback("rate_limit", wait_msg, round(pct_start + (batch_num / num_batches) * (pct_end - pct_start), 1))
                    time.sleep(wait_sec)
                else:
                    wait_sec = 2.0 * attempt
                    print(f"     [API error] {exc} -> retrying in {wait_sec:.0f}s...")
                    time.sleep(wait_sec)

        if not success:
            raise RuntimeError(f"Failed to generate embeddings for batch {start_idx}-{end_idx}: {last_exc}")

        if end_idx < total_chunks:
            time.sleep(1.2)

    return all_embeddings



# ============================================================
# Step 5: Vector DB & Store Synchronization
# ============================================================

def sync_to_qdrant_and_storage(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    doc_meta: dict[str, Any],
    qdrant_client: QdrantClient | None = None,
) -> None:
    """Upsert vectors to Qdrant, persist chunks JSON, and update source registry."""
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if qdrant_client is not None:
        client = qdrant_client
    else:
        try:
            from rag.rag_pipeline import get_qdrant_client
            client = get_qdrant_client()
        except Exception:
            client = QdrantClient(path=str(QDRANT_PATH))

    # Ensure Qdrant collection exists
    existing_cols = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing_cols:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE),
        )

    # Prepare Qdrant Points
    points = []
    for chunk, emb in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=chunk_id_to_uuid(chunk["chunk_id"]),
                vector=emb,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "document_name": chunk["document_name"],
                    "pdf_file": chunk.get("pdf_file", ""),
                    "section_id": chunk["section_id"],
                    "section_title": chunk["section_title"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "token_count": chunk["token_count"],
                    "text": chunk["text"],
                },
            )
        )

    # Upsert points
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    # Update validated_chunks.json
    all_chunks = []
    if CHUNKS_FILE_PATH.exists():
        try:
            with open(CHUNKS_FILE_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                all_chunks = loaded.get("chunks", loaded) if isinstance(loaded, dict) else loaded
        except Exception:
            all_chunks = []

    # Merge avoiding duplicate chunk_ids
    existing_ids = {c.get("chunk_id") for c in all_chunks}
    for chunk in chunks:
        if chunk["chunk_id"] not in existing_ids:
            all_chunks.append(chunk)

    with open(CHUNKS_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # Update source_registry.json
    registry = []
    if SOURCE_REGISTRY_PATH.exists():
        try:
            with open(SOURCE_REGISTRY_PATH, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception:
            registry = []

    # Check if doc_id already in registry
    reg_ids = {r.get("document_id") for r in registry}
    if doc_meta["document_id"] not in reg_ids:
        registry.append(doc_meta)
        with open(SOURCE_REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)


# ============================================================
# Main Entry Point: auto_ingest_pdf
# ============================================================

def auto_ingest_pdf(
    pdf_path: str | Path,
    document_id: str | None = None,
    document_name: str | None = None,
    organization: str = "Clinical Guideline",
    qdrant_client: QdrantClient | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    """
    Complete 1-step automated ingestion for any uploaded guideline PDF:
    PDF -> Extract -> Clean -> Sections -> Chunks -> Embeddings -> Qdrant + BM25
    Calls progress_callback(stage, message, percent) at each major step.
    """
    def _emit(stage: str, msg: str, pct: float):
        print(f"  [{stage}] {msg}")
        if progress_callback:
            progress_callback(stage, msg, pct)

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    safe_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", path.stem).lower()
    doc_id   = document_id or f"doc_{safe_stem}"
    doc_name = document_name or path.stem.replace("_", " ").title()

    _emit("start", f"Starting ingestion for: {path.name}", 2.0)

    # 1. Extract & clean
    _emit("extract", "Extracting and cleaning PDF text…", 5.0)
    pages = extract_and_clean_pdf(path)
    _emit("extract", f"Extracted {len(pages)} pages successfully", 18.0)

    # 2. Section detection
    _emit("sections", "Detecting document structure and sections…", 22.0)
    sections = detect_document_sections(path, pages)
    _emit("sections", f"Found {len(sections)} sections in the document", 32.0)

    # 3. Chunking
    _emit("chunks", "Building overlapping text chunks…", 36.0)
    chunks = build_document_chunks(
        pages=pages,
        sections=sections,
        document_id=doc_id,
        document_name=doc_name,
        pdf_filename=path.name,
    )
    _emit("chunks", f"Generated {len(chunks)} chunks ready for embedding", 42.0)

    if not chunks:
        raise ValueError(f"No valid text chunks could be extracted from {path.name}")

    # 4. Generate embeddings (progress_callback handles 42->85%)
    _emit("embed", f"Starting Gemini embedding for {len(chunks)} chunks…", 42.0)
    embeddings = generate_embeddings_for_chunks(
        chunks,
        progress_callback=progress_callback,
        pct_start=42.0,
        pct_end=85.0,
    )

    # 5. Sync to Qdrant & storage
    _emit("index", "Indexing vectors into Qdrant and BM25…", 88.0)
    doc_meta = {
        "document_id":  doc_id,
        "document_name": doc_name,
        "organization": organization,
        "local_file":   str(path.relative_to(BASE_DIR) if path.is_relative_to(BASE_DIR) else path),
        "status":       "indexed",
        "total_chunks": len(chunks),
        "total_pages":  len(pages),
    }
    sync_to_qdrant_and_storage(chunks, embeddings, doc_meta, qdrant_client=qdrant_client)
    _emit("done", f"Successfully indexed {len(chunks)} chunks from {len(pages)} pages!", 100.0)

    return {
        "status":        "success",
        "document_id":  doc_id,
        "document_name": doc_name,
        "total_pages":  len(pages),
        "total_chunks": len(chunks),
        "pdf_file":     path.name,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        target_pdf = sys.argv[1]
        result = auto_ingest_pdf(target_pdf)
        print("Result:", json.dumps(result, indent=2))
    else:
        print("Usage: python auto_ingest.py <path_to_pdf>")
