import json
import re
from pathlib import Path

import tiktoken


# ============================================================
# Configuration
# ============================================================

INPUT_PATH    = "data/processed/cleaned_pages.json"
SECTIONS_PATH = "data/processed/sections.json"
CHUNKS_PATH   = "data/processed/chunks.json"

# Pages that are clearly TOC / front matter.
# We do not detect sections from these pages.
TOC_PAGES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

# Sections excluded from the RAG knowledge base.
# References and Annexes are not clinical content.
EXCLUDED_SECTIONS = {"REFERENCES", "ANNEXES"}

# Chunking configuration
TARGET_TOKENS = 600   # aim for this size
MAX_TOKENS    = 900   # hard ceiling per chunk (raised to absorb small tail chunks)
MIN_TOKENS    = 400   # below this → merge into previous chunk
OVERLAP_TOKENS = 80   # token overlap between consecutive chunks

# Tokenizer
TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ============================================================
# Known document sections
# ============================================================

MAIN_SECTIONS = {
    "EXECUTIVE SUMMARY",
    "INTRODUCTION",
    "METHOD FOR DEVELOPING THE GUIDELINE",
    "SPECIAL SETTINGS",
    "PUBLICATION, IMPLEMENTATION, EVALUATION AND RESEARCH GAPS",
    "IMPLEMENTATION TOOLS",
    "REFERENCES",
    "ANNEXES",
}


# ============================================================
# PDF running headers / labels to ignore
# ============================================================

PDF_RUNNING_HEADERS = {
    "GUIDELINE FOR THE PHARMACOLOGICAL TREATMENT OF HYPERTENSION IN ADULTS",
    "RECOMMENDATIONS",
    "RECOMMENDATION",
}


# ============================================================
# Numbered recommendation pattern
# ============================================================

NUMBERED_REC_PATTERN = re.compile(
    r"^\d+\.\s{1,4}RECOMMENDATION(?:S)?\s+ON\s+\S",
    re.IGNORECASE,
)


# ============================================================
# Load cleaned pages
# ============================================================

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    pages = json.load(f)

# Build a page-number → page dict for fast lookup
PAGE_LOOKUP: dict[int, dict] = {
    p["page_number"]: p for p in pages
}

print(f"Number of pages: {len(pages)}")


# ============================================================
# Section detection helpers
# ============================================================

def is_section_heading(line: str) -> bool:
    """
    Return True if a line is a meaningful section heading
    in the WHO hypertension guideline.
    """

    line = line.strip()

    if not line:
        return False

    # Ignore known running headers / labels
    if line.upper() in PDF_RUNNING_HEADERS:
        return False

    # Ignore the document title
    if "PHARMACOLOGICAL TREATMENT OF HYPERTENSION" in line.upper():
        return False

    # Known top-level sections
    if line.upper() in MAIN_SECTIONS:
        # Accept only actual ALL-CAPS headings (not TOC title-case entries)
        if line == line.upper():
            return True
        return False

    # Numbered recommendations
    if NUMBERED_REC_PATTERN.match(line):
        return True

    return False


def merge_continuation_lines(lines: list[str]) -> list[str]:
    """
    Merge numbered recommendation headings that were split
    across multiple lines by PDF extraction.

    Example:
        "1.  RECOMMENDATION ON BLOOD PRESSURE THRESHOLD FOR"
        "INITIATION OF PHARMACOLOGICAL TREATMENT"
    → one combined heading line.
    """

    merged = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if (
            NUMBERED_REC_PATTERN.match(line)
            and not line.endswith((".", ":"))
        ):
            if i + 1 < len(lines):
                continuation = lines[i + 1].strip()
                if (
                    continuation
                    and continuation == continuation.upper()
                    and len(continuation) < 100
                ):
                    line = line + " " + continuation
                    i += 1

        merged.append(line)
        i += 1

    return merged


# ============================================================
# Detect sections
# ============================================================

def detect_sections() -> list[tuple[int, str]]:
    """
    Detect section headings and return list of (page_number, title).
    """

    detected_headings = []
    last_heading = None

    for page in pages:
        page_num = page["page_number"]

        if page_num in TOC_PAGES:
            continue

        lines = merge_continuation_lines(page["text"].splitlines())

        for line in lines:
            if not is_section_heading(line):
                continue

            heading_key = line.upper().strip()

            if heading_key == last_heading:
                continue

            detected_headings.append((page_num, line.strip()))
            last_heading = heading_key
            print(f"[Page {page_num:2d}] -> {line.strip()}")

    return detected_headings


# ============================================================
# Build section ranges
# ============================================================

def build_sections(detected_headings: list[tuple[int, str]]) -> list[dict]:
    """
    Convert detected headings into section dicts with page ranges.
    """

    sections = []

    for i, (start_page, title) in enumerate(detected_headings):

        if i + 1 < len(detected_headings):
            end_page = detected_headings[i + 1][0] - 1
        else:
            end_page = pages[-1]["page_number"]

        sections.append(
            {
                "section_id":    f"section_{i + 1:03d}",
                "section_title": title,
                "start_page":    start_page,
                "end_page":      end_page,
            }
        )

    return sections


# ============================================================
# Save sections
# ============================================================

def save_sections(sections: list[dict]) -> None:
    output_path = Path(SECTIONS_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)


# ============================================================
# Section text helpers
# ============================================================

def get_section_pages(section: dict) -> list[dict]:
    """Return all page dicts belonging to a section."""

    return [
        p for p in pages
        if section["start_page"] <= p["page_number"] <= section["end_page"]
    ]


def _dedupe_repeated_title(text: str, title: str) -> str:
    """
    The WHO PDF reprints the current section's own heading as a running
    header at the top of subsequent pages within that same section (the
    same mechanism that caused the "RECOMMENDATIONS" leak, but scoped to
    each section's own specific title instead of a fixed global label).

    Section detection has already run by the time this is called, so it's
    safe to drop every repeat of the section's own title line EXCEPT the
    first — the first occurrence is the genuine heading and is kept for
    context; later ones are pure page-header noise with no content value.
    """
    title_upper = title.strip().upper()
    lines = text.split("\n")
    seen = False
    out = []
    for line in lines:
        if line.strip().upper() == title_upper:
            if seen:
                continue  # drop repeat
            seen = True
        out.append(line)
    return "\n".join(out)


def _truncate_at_excluded_marker(text: str) -> str:
    """
    Section boundaries are computed at whole-page granularity, but the WHO
    PDF's actual "References" heading falls partway down a page rather than
    at the top. That means the tail of the last included section on that
    page can end up containing real bibliography entries that belong to the
    (excluded) REFERENCES/ANNEXES section.

    This truncates an included section's text at the first line that
    exactly matches an excluded section's title, dropping everything from
    that point on — the same safeguard EXCLUDED_SECTIONS already provides
    at the whole-section level, applied at the line level for the one page
    where the two chapters share space.
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip().upper() in EXCLUDED_SECTIONS:
            return "\n".join(lines[:i]).rstrip()
    return text


def get_section_text(section: dict) -> str:
    """Concatenate all page text belonging to a section, then strip
    page-header repeats of the section's own title and any bleed-through
    from an excluded section sharing the same page."""

    texts = [
        p["text"].strip()
        for p in get_section_pages(section)
        if p["text"].strip()
    ]
    text = "\n\n".join(texts)
    text = _truncate_at_excluded_marker(text)
    text = _dedupe_repeated_title(text, section["section_title"])
    return text


# ============================================================
# Token utilities
# ============================================================

def encode_text(text: str) -> list[int]:
    return TOKENIZER.encode(text, disallowed_special=())


def decode_tokens(tokens: list[int]) -> str:
    return TOKENIZER.decode(tokens)


def count_tokens(text: str) -> int:
    return len(encode_text(text))


# ============================================================
# Split text into chunks (with small-chunk merging)
# ============================================================

def split_into_chunks(
    text: str,
    target_tokens: int  = TARGET_TOKENS,
    max_tokens: int     = MAX_TOKENS,
    min_tokens: int     = MIN_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[str]:
    """
    Split section text into token-bounded chunks.

    Phase 1 — sliding window with overlap.
    Phase 2 — merge ALL chunks below MIN_TOKENS into their
               predecessor (capped at MAX_TOKENS).  This runs
               in a single backward pass so multiple small
               tail chunks are all absorbed.

    Result: all emitted chunks are in the
            [MIN_TOKENS, MAX_TOKENS] range (best-effort).
    """

    tokens = encode_text(text)

    if not tokens:
        return []

    # Short section — keep as one chunk
    if len(tokens) <= max_tokens:
        return [text.strip()]

    # ─── Phase 1: sliding-window chunking ──────────────────
    raw_chunks: list[list[int]] = []
    start = 0
    total = len(tokens)

    while start < total:
        end = min(start + target_tokens, total)
        raw_chunks.append(tokens[start:end])

        if end >= total:
            break

        next_start = end - overlap_tokens
        if next_start <= start:
            next_start = end
        start = next_start

    # ─── Phase 2: merge small chunks (backward pass) ───────
    # Walk from the last chunk backward.  If a chunk is below
    # MIN_TOKENS and merging it into the previous one stays
    # within MAX_TOKENS, absorb it.
    i = len(raw_chunks) - 1
    while i >= 1:
        if len(raw_chunks[i]) < min_tokens:
            merged = raw_chunks[i - 1] + raw_chunks[i]
            if len(merged) <= max_tokens:
                raw_chunks[i - 1] = merged
                raw_chunks.pop(i)
        i -= 1

    # ─── Phase 3: decode ───────────────────────────────────
    return [decode_tokens(c).strip() for c in raw_chunks]


# ============================================================
# Page tracking  (cursor-based, no fingerprint search)
# ============================================================

def build_page_token_map(section_pages: list[dict]) -> list[tuple[int, list[int]]]:
    """
    Encode each page's text and return
    [(page_number, [token, ...]), ...]
    in page order.
    """
    return [
        (p["page_number"], encode_text(p["text"]))
        for p in section_pages
    ]


class SectionCursor:
    """
    Tracks the current position within a section's token stream
    across consecutive chunks.

    Because chunks are produced left-to-right with a fixed overlap,
    we can advance the cursor by (chunk_size - overlap) after each
    chunk instead of searching by fingerprint — which fails when the
    same token sequence appears more than once.
    """

    def __init__(
        self,
        page_token_map: list[tuple[int, list[int]]],
        overlap_tokens: int = OVERLAP_TOKENS,
    ) -> None:
        self.page_token_map = page_token_map
        self.overlap_tokens = overlap_tokens

        # Precompute cumulative page boundaries
        # boundaries[i] = first token index of page i in the section stream
        self.boundaries: list[tuple[int, int, int]] = []  # (page_num, start_idx, end_idx)
        cursor = 0
        for page_num, page_tokens in page_token_map:
            length = len(page_tokens)
            self.boundaries.append((page_num, cursor, cursor + length - 1))
            cursor += length

        self.total_tokens = cursor
        self.position = 0   # running start of the next chunk
        self.first_chunk = True

    def page_of_token(self, token_idx: int) -> int:
        """Return the page number that owns `token_idx`."""
        for page_num, start, end in self.boundaries:
            if start <= token_idx <= end:
                return page_num
        # token_idx beyond last page — return last page
        return self.boundaries[-1][0] if self.boundaries else 0

    def resolve(
        self, chunk_token_count: int
    ) -> tuple[int, int]:
        """
        Return (page_start, page_end) for the current chunk,
        then advance the cursor for the next call.
        """
        if not self.boundaries:
            return 0, 0

        chunk_start = self.position
        chunk_end   = min(self.position + chunk_token_count - 1,
                          self.total_tokens - 1)

        page_start = self.page_of_token(chunk_start)
        page_end   = self.page_of_token(chunk_end)

        # Advance: next chunk starts at (chunk_end + 1 - overlap)
        advance = chunk_token_count - self.overlap_tokens
        if advance <= 0:
            advance = chunk_token_count
        self.position = min(self.position + advance, self.total_tokens)

        return page_start, page_end


def map_chunk_to_pages(
    cursor: "SectionCursor",
    chunk_text: str,
) -> tuple[int, int]:
    """Convenience wrapper used in build_chunks."""
    chunk_token_count = count_tokens(chunk_text)
    return cursor.resolve(chunk_token_count)


# ============================================================
# Build chunks
# ============================================================

def build_chunks(sections: list[dict]) -> list[dict]:
    """
    Build RAG chunks from all included sections.

    Sections listed in EXCLUDED_SECTIONS (REFERENCES, ANNEXES)
    are skipped entirely.
    """

    chunks      = []
    chunk_serial = 0  # global serial across all sections

    for section in sections:

        # ── Exclude non-clinical sections ───────────────────
        if section["section_title"].upper() in EXCLUDED_SECTIONS:
            print(
                f"\n[SKIP] {section['section_id']} | "
                f"{section['section_title']}"
            )
            continue

        section_pages     = get_section_pages(section)
        section_text      = get_section_text(section)
        section_token_cnt = count_tokens(section_text)

        print(
            f"\n{section['section_id']} | "
            f"{section_token_cnt} tokens | "
            f"{section['section_title']}"
        )

        section_chunks = split_into_chunks(section_text)

        # Build cursor for deterministic page tracking
        page_token_map = build_page_token_map(section_pages)
        cursor = SectionCursor(page_token_map)

        for chunk_index, chunk_text in enumerate(section_chunks, start=1):

            chunk_serial += 1

            page_start, page_end = map_chunk_to_pages(cursor, chunk_text)

            token_cnt = count_tokens(chunk_text)

            chunk = {
                "chunk_id":      f"who_001_{section['section_id']}_c{chunk_index:03d}",
                "document_id":   "who_guideline_01",
                "document_name": (
                    "WHO Guideline for the Pharmacological "
                    "Treatment of Hypertension in Adults"
                ),
                "section_id":    section["section_id"],
                "section_title": section["section_title"],
                "page_start":    page_start,
                "page_end":      page_end,
                "token_count":   token_cnt,
                "text":          chunk_text,
            }

            chunks.append(chunk)

            print(
                f"    Chunk {chunk_index:03d} | "
                f"{token_cnt} tokens | "
                f"pages {page_start}-{page_end}"
            )

    return chunks


# ============================================================
# Save chunks
# ============================================================

def save_chunks(chunks: list[dict]) -> None:
    output_path = Path(CHUNKS_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:

    print("\n" + "=" * 80)
    print("STEP 1 — SECTION DETECTION")
    print("=" * 80)

    detected_headings = detect_sections()
    print(f"\nTotal section headings detected: {len(detected_headings)}")

    print("\n" + "=" * 80)
    print("STEP 2 — BUILD SECTION RANGES")
    print("=" * 80)

    sections = build_sections(detected_headings)

    for section in sections:
        tag = " [EXCLUDED]" if section["section_title"].upper() in EXCLUDED_SECTIONS else ""
        print(
            f"[{section['section_id']}] "
            f"Pages {section['start_page']}-{section['end_page']} "
            f"-> {section['section_title']}{tag}"
        )

    save_sections(sections)
    print(f"\nSaved sections to: {SECTIONS_PATH}")

    print("\n" + "=" * 80)
    print("STEP 3 — TOKEN COUNTING & CHUNKING")
    print("=" * 80)

    chunks = build_chunks(sections)

    print("\n" + "=" * 80)
    print("STEP 4 — SAVE CHUNKS")
    print("=" * 80)

    save_chunks(chunks)

    included = [c for c in chunks]
    token_counts = [c["token_count"] for c in included]

    print(f"\nTotal chunks saved:  {len(included)}")
    if token_counts:
        print(f"Min tokens/chunk:    {min(token_counts)}")
        print(f"Max tokens/chunk:    {max(token_counts)}")
        print(f"Avg tokens/chunk:    {sum(token_counts) // len(token_counts)}")
    print(f"Saved chunks to:     {CHUNKS_PATH}")
    print("\nPipeline completed successfully.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()