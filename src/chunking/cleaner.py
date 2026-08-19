import re


# ============================================================
# Known running headers / footers to strip from every page
# ============================================================

# These strings appear verbatim on every page as PDF headers
# or footers. They carry no clinical content.

RUNNING_HEADERS: set[str] = {
    # Full document title (appears on every page header/footer)
    "GUIDELINE FOR THE PHARMACOLOGICAL TREATMENT OF HYPERTENSION IN ADULTS",
    # Truncated variant that appears at a page break
    "GUIDELINE FOR THE PHARMACOLOGICAL TREATMENT OF HYPERTENSION IN",
    # Section-page-number footers like "RECOMMENDATIONS\n7" — number removed
    # separately; we only strip the naked title here when it appears exactly
    # as the document running header (all-caps, full line, no surrounding text).
    #
    # NOTE (updated): a bare "RECOMMENDATIONS" / "RECOMMENDATION" line IS safe
    # to strip here. chunker.py's own PDF_RUNNING_HEADERS set already excludes
    # these exact strings from is_section_heading(), so they are never used to
    # detect section boundaries — they were leaking straight through into
    # chunk text with zero semantic value (validate_chunks.py flagged 7/38
    # chunks ending in a dangling "RECOMMENDATIONS" line). Numbered headings
    # like "1. RECOMMENDATION ON BLOOD PRESSURE THRESHOLD..." are matched by
    # NUMBERED_REC_PATTERN separately and are NOT affected by this — only the
    # bare, standalone word is stripped.
    "RECOMMENDATIONS",
    "RECOMMENDATION",
}


def _collapse_bullets(lines: list[str]) -> list[str]:
    """
    The WHO PDF renders bullet lists as two lines per item:

        [           ← lone bracket line (the bullet glyph)
        [ item text ← text line prefixed by a bracket

    This function collapses those pairs into a single clean line:

        - item text

    It also handles implementation-remarks bullets (double bracket):

        [
        [
        Item text

    Patterns handled:
      - Lone `[` line followed by `[ text`  → `- text`
      - Lone `[` line followed by another lone `[` then text → `- text`
      - `[text`  or  `[ text` as a standalone line → `- text`
    """

    result: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # ── Pattern A: lone "[" ──────────────────────────────
        if line == "[":
            # Peek ahead, skip extra lone "[" lines
            j = i + 1
            while j < n and lines[j] == "[":
                j += 1

            if j < n and lines[j].startswith("["):
                # Next real content line is a bracket-prefixed item
                item = lines[j].lstrip("[").strip()
                if item:
                    result.append(f"- {item}")
                else:
                    # Empty bracket line — skip
                    pass
                i = j + 1
                continue
            else:
                # Lone "[" with no following bracket line — drop it
                i += 1
                continue

        # ── Pattern B: line starts with "[ " (bullet item) ──
        if line.startswith("[ ") or (line.startswith("[") and len(line) > 1):
            item = line.lstrip("[").strip()
            if item:
                result.append(f"- {item}")
            i += 1
            continue

        result.append(line)
        i += 1

    return result


def clean_text(text: str) -> str:
    """
    Clean a single PDF page's extracted text.

    Steps applied in order:
      1. Split into lines and strip whitespace.
      2. Remove empty lines.
      3. Remove running PDF headers/footers (exact match, case-insensitive).
      4. Remove standalone page numbers (bare integers).
      5. Normalize tabs to single space.
      6. Normalize multiple whitespace to single space.
      7. Collapse bullet artifacts (`[` / `[ text` patterns).
      8. Deduplicate consecutive identical lines (handles PDF duplication).
    """

    lines = [line.strip() for line in text.splitlines()]

    # ── Step 2: remove empty ────────────────────────────────
    lines = [l for l in lines if l]

    # ── Step 3: remove running headers ──────────────────────
    lines = [
        l for l in lines
        if l.upper() not in RUNNING_HEADERS
    ]

    # ── Step 4: remove standalone page numbers ───────────────
    lines = [l for l in lines if not re.fullmatch(r"\d+", l)]

    # ── Step 5 & 6: normalize whitespace ────────────────────
    lines = [re.sub(r"\s+", " ", re.sub(r"\t+", " ", l)) for l in lines]

    # ── Step 7: collapse bullet artifacts ───────────────────
    lines = _collapse_bullets(lines)

    # ── Step 8: deduplicate consecutive identical lines ──────
    deduped: list[str] = []
    for l in lines:
        if not deduped or l != deduped[-1]:
            deduped.append(l)

    # ── Step 9: paragraph-block deduplication ────────────────
    # The WHO PDF sometimes embeds the same paragraph twice within
    # a single page extraction. We detect and remove duplicate
    # *blocks* of 3+ consecutive lines (paragraphs) that appear
    # more than once on the page, keeping only the first occurrence.
    # Single lines are left intact to avoid over-filtering.

    BLOCK_SIZE = 3   # minimum lines to form a deduplicated block

    seen_blocks: set[tuple[str, ...]] = set()
    result: list[str] = []
    i = 0
    n = len(deduped)

    while i < n:
        # Try to match a block starting at position i
        remaining = n - i
        block_len = min(BLOCK_SIZE, remaining)

        block = tuple(deduped[i : i + block_len])

        if len(block) == BLOCK_SIZE and block in seen_blocks:
            # Skip this block — it's a duplicate
            i += BLOCK_SIZE
        else:
            if len(block) == BLOCK_SIZE:
                seen_blocks.add(block)
            result.append(deduped[i])
            i += 1

    return "\n".join(result)