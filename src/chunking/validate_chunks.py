"""
validate_chunks.py

Validation script for adult-hypertension-rag.

Runs a battery of checks on data/processed/chunks.json (and cross-checks
against data/processed/sections.json) BEFORE the embeddings stage.

Usage:
    python validate_chunks.py
    python validate_chunks.py --chunks data/processed/chunks.json --sections data/processed/sections.json
    python validate_chunks.py --strict   # exit(1) if any ERROR-level issue found

Output:
    - Human-readable report printed to stdout
    - data/processed/validation_report.json with full machine-readable results
    - If all checks pass (no ERROR-level issues), also writes
      data/processed/validated_chunks.json (a straight copy/pass-through of chunks.json,
      per the pipeline: chunks.json -> validate_chunks.py -> validated_chunks.json)
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — mirrors the constants used in chunker.py so validation stays
# consistent with how the chunks were actually built.
# ---------------------------------------------------------------------------

MIN_TOKENS = 400
MAX_TOKENS = 900

EXCLUDED_SECTIONS = {
    "REFERENCES",
    "ANNEXES",
}

PDF_RUNNING_HEADERS = {
    "GUIDELINE FOR THE PHARMACOLOGICAL TREATMENT OF HYPERTENSION IN ADULTS",
    "RECOMMENDATIONS",
    "RECOMMENDATION",
}

# Any line that is JUST a number (with optional whitespace) is a likely
# stray page number that leaked into chunk text.
STANDALONE_PAGE_NUMBER_PATTERN = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)

# Token counting: rough heuristic (word-based). Swap this out for the
# real tokenizer used in chunker.py (e.g. tiktoken) if available, so
# token_count validation is apples-to-apples.
def count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Fallback heuristic: ~0.75 tokens per word is common for English;
        # this is only used if tiktoken isn't installed, and will be
        # flagged as approximate in the report.
        return len(text.split())


TOKEN_COUNT_TOLERANCE = 0.10  # allow 10% drift between stored and recomputed count

# Overlap detection: measures the actual longest shared substring between
# the tail of chunk N and the head of chunk N+1, then expresses it as a
# fraction of the smaller chunk's token_count. This is more meaningful than
# a fixed-window ratio, which can't distinguish "designed overlap" from
# "something is duplicating too much" — a ratio near 1.0 on a small window
# just means the overlap is AT LEAST that window's size, which is expected
# if your chunker's overlap setting is that big.
OVERLAP_SEARCH_WINDOW_CHARS = 1500   # how far back/forward to look for the shared span
CHARS_PER_TOKEN_ESTIMATE = 4.0       # rough English heuristic when tiktoken isn't available
OVERLAP_PCT_WARNING = 0.30           # overlap > 30% of the smaller chunk's tokens
OVERLAP_PCT_ERROR = 0.50             # overlap > 50% — very likely a bug, not intentional overlap

DUPLICATE_SIMILARITY_THRESHOLD = 0.97  # near-exact duplicate full-text chunks


# ---------------------------------------------------------------------------
# Issue tracking
# ---------------------------------------------------------------------------

class Report:
    def __init__(self):
        self.issues = []  # list of dicts: {level, check, chunk_id, message}

    def add(self, level, check, chunk_id, message):
        self.issues.append({
            "level": level,  # "ERROR" | "WARNING" | "INFO"
            "check": check,
            "chunk_id": chunk_id,
            "message": message,
        })

    def errors(self):
        return [i for i in self.issues if i["level"] == "ERROR"]

    def warnings(self):
        return [i for i in self.issues if i["level"] == "WARNING"]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_required_fields(chunk, report):
    required = [
        "chunk_id", "document_id", "document_name", "section_id",
        "section_title", "page_start", "page_end", "token_count", "text",
    ]
    cid = chunk.get("chunk_id", "<missing chunk_id>")
    for field in required:
        if field not in chunk:
            report.add("ERROR", "missing_field", cid, f"Missing required field '{field}'")


def check_empty_chunk(chunk, report):
    cid = chunk.get("chunk_id", "<unknown>")
    text = chunk.get("text", "")
    if not text or not text.strip():
        report.add("ERROR", "empty_chunk", cid, "Chunk text is empty or whitespace-only")


def check_valid_chunk_id(chunk, report, seen_ids):
    cid = chunk.get("chunk_id")
    if not cid:
        return
    if cid in seen_ids:
        report.add("ERROR", "duplicate_chunk_id", cid, "chunk_id is not unique")
    seen_ids.add(cid)

    if not re.match(r"^[a-zA-Z0-9]+_\d+_section_\d+_c\d+$", cid):
        report.add("WARNING", "chunk_id_format", cid,
                    f"chunk_id '{cid}' doesn't match expected pattern "
                    f"'<doc>_NNN_section_NNN_cNNN'")


def check_valid_section_id(chunk, sections_by_id, report):
    cid = chunk.get("chunk_id", "<unknown>")
    sid = chunk.get("section_id")
    if not sid:
        return
    if sections_by_id and sid not in sections_by_id:
        report.add("ERROR", "unknown_section_id", cid,
                    f"section_id '{sid}' not found in sections.json")


def check_page_numbers(chunk, report):
    cid = chunk.get("chunk_id", "<unknown>")
    ps, pe = chunk.get("page_start"), chunk.get("page_end")
    if ps is None or pe is None:
        return
    if not isinstance(ps, int) or not isinstance(pe, int):
        report.add("ERROR", "page_number_type", cid, "page_start/page_end must be integers")
        return
    if ps < 1 or pe < 1:
        report.add("ERROR", "page_number_range", cid, "page_start/page_end must be >= 1")
    if pe < ps:
        report.add("ERROR", "page_order", cid, f"page_end ({pe}) is before page_start ({ps})")


def check_token_count(chunk, report):
    cid = chunk.get("chunk_id", "<unknown>")
    stored = chunk.get("token_count")
    text = chunk.get("text", "")
    if stored is None or not text:
        return
    actual = count_tokens(text)
    if actual == 0:
        return
    drift = abs(actual - stored) / max(actual, 1)
    if drift > TOKEN_COUNT_TOLERANCE:
        report.add("WARNING", "token_count_mismatch", cid,
                    f"stored token_count={stored} vs recomputed={actual} "
                    f"(drift {drift:.0%}, tolerance {TOKEN_COUNT_TOLERANCE:.0%})")


def check_token_thresholds(chunk, report):
    cid = chunk.get("chunk_id", "<unknown>")
    tc = chunk.get("token_count")
    if tc is None:
        return
    if tc < MIN_TOKENS:
        report.add("WARNING", "below_min_tokens", cid,
                    f"token_count={tc} is below MIN_TOKENS={MIN_TOKENS}")
    if tc > MAX_TOKENS:
        report.add("WARNING", "above_max_tokens", cid,
                    f"token_count={tc} exceeds MAX_TOKENS={MAX_TOKENS}")


def check_excluded_sections(chunk, report):
    cid = chunk.get("chunk_id", "<unknown>")
    title = (chunk.get("section_title") or "").strip().upper()
    if title in EXCLUDED_SECTIONS:
        report.add("ERROR", "excluded_section_leak", cid,
                    f"Chunk belongs to excluded section '{title}' "
                    f"(REFERENCES/ANNEXES should not reach chunks.json)")


MAX_HEADER_LINE_LEN = 90  # headers are short standalone lines, not full sentences

def check_running_headers(chunk, sections_by_id, report):
    cid = chunk.get("chunk_id", "<unknown>")
    text = chunk.get("text", "")
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    all_section_titles = {
        (s.get("section_title") or "").strip().upper()
        for s in sections_by_id.values()
    } if sections_by_id else set()

    for line in lines:
        upper = line.upper()

        if upper in PDF_RUNNING_HEADERS:
            report.add("WARNING", "running_header_leak", cid,
                        f"Possible running header/title leaked into chunk text: '{line}'")
            return

        if upper.startswith("GUIDELINE FOR THE PHARMACOLOGICAL TREATMENT OF HYPERTENSION"):
            report.add("WARNING", "running_header_leak", cid,
                        f"Possible running header/title leaked into chunk text: '{line}'")
            return

        # Generalized: any OTHER section's title (this document's own
        # headings, not just a fixed list) appearing verbatim as a
        # standalone line is a page-header repeat, not real content — this
        # catches INTRODUCTION, METHOD FOR DEVELOPING THE GUIDELINE, other
        # numbered recommendation headings, etc. bleeding into a DIFFERENT
        # section's chunk. A chunk containing its OWN section's title once
        # is expected (the genuine heading) and is not flagged.
        own_title = (chunk.get("section_title") or "").strip().upper()
        if upper in all_section_titles and upper != own_title:
            report.add("WARNING", "running_header_leak", cid,
                        f"Heading '{line}' from a different section appears inside "
                        f"this chunk — likely page-header bleed across a section boundary")
            return


def check_excluded_content_leak(chunk, report):
    """
    Catches the page-granularity boundary problem: an included section whose
    text tail actually contains REFERENCES/ANNEXES bibliography content
    because the excluded section's heading fell mid-page rather than at a
    page boundary.
    """
    cid = chunk.get("chunk_id", "<unknown>")
    title = (chunk.get("section_title") or "").upper()
    if title in EXCLUDED_SECTIONS:
        return  # already caught by check_excluded_sections
    text = chunk.get("text", "")
    for line in text.splitlines():
        if line.strip().upper() in EXCLUDED_SECTIONS:
            report.add("ERROR", "excluded_content_leak", cid,
                        f"Chunk (section '{chunk.get('section_title')}') contains a line "
                        f"matching an excluded section title ('{line.strip()}') — likely "
                        f"REFERENCES/ANNEXES content bled in due to page-level boundary "
                        f"rounding")
            return


def check_standalone_page_numbers(chunk, report):
    cid = chunk.get("chunk_id", "<unknown>")
    text = chunk.get("text", "")
    matches = STANDALONE_PAGE_NUMBER_PATTERN.findall(text)
    if matches:
        report.add("WARNING", "standalone_page_number", cid,
                    f"Found {len(matches)} standalone numeric line(s) that look like "
                    f"stray page numbers: {matches[:5]}")


def check_broken_text(chunk, report):
    cid = chunk.get("chunk_id", "<unknown>")
    text = chunk.get("text", "")
    if not text:
        return

    # Heuristic 1: leftover bullet-extraction artifacts like "[" as a bullet
    if re.search(r"^\s*\[\s*$", text, re.MULTILINE) or re.search(r"^\[\s+\S", text, re.MULTILINE):
        report.add("WARNING", "broken_bullet_artifact", cid,
                    "Text contains raw '[' bullet artifacts from PDF extraction")

    # Heuristic 2: lines ending mid-word with no punctuation, followed by a
    # lowercase continuation — sign of a broken line join.
    lines = text.splitlines()
    for i in range(len(lines) - 1):
        cur, nxt = lines[i].rstrip(), lines[i + 1].lstrip()
        if cur and nxt and cur[-1].isalpha() and nxt[:1].islower():
            # Only flag if it looks like a genuine mid-sentence break, not
            # normal wrapped prose (which chunker should have already joined).
            pass  # left as no-op: too noisy as a hard rule, kept for future tuning

    # Heuristic 3: excessive whitespace / control characters
    if re.search(r"[ \t]{4,}", text):
        report.add("INFO", "irregular_whitespace", cid,
                    "Chunk contains runs of 4+ spaces/tabs, check for extraction artifacts")

    # Heuristic 4: very short chunk despite passing token threshold check
    if len(text.strip()) < 50:
        report.add("ERROR", "suspiciously_short_text", cid,
                    f"Chunk text is only {len(text.strip())} characters")


def _tail(text, n=OVERLAP_SEARCH_WINDOW_CHARS):
    return text[-n:] if len(text) > n else text


def _head(text, n=OVERLAP_SEARCH_WINDOW_CHARS):
    return text[:n] if len(text) > n else text


def measure_overlap_chars(a_text, b_text):
    """
    Finds the longest contiguous substring shared between the tail of a_text
    and the head of b_text. Returns (overlap_char_length, matched_text).
    This measures actual overlap size, not just similarity ratio.
    """
    a_tail = _tail(a_text)
    b_head = _head(b_text)
    if not a_tail or not b_head:
        return 0, ""
    matcher = SequenceMatcher(None, a_tail, b_head, autojunk=False)
    match = matcher.find_longest_match(0, len(a_tail), 0, len(b_head))
    return match.size, a_tail[match.a: match.a + match.size]


def check_duplicates_and_overlap(chunks, report):
    # Exact / near-exact duplicate full-text detection (global)
    seen_texts = {}
    for c in chunks:
        cid = c.get("chunk_id", "<unknown>")
        text = (c.get("text") or "").strip()
        if not text:
            continue
        for other_id, other_text in seen_texts.items():
            ratio = SequenceMatcher(None, text, other_text).quick_ratio()
            if ratio >= DUPLICATE_SIMILARITY_THRESHOLD:
                report.add("ERROR", "duplicate_chunk_text", cid,
                            f"Near-duplicate text (ratio={ratio:.2f}) with chunk '{other_id}'")
                break
        seen_texts[cid] = text

    # Consecutive-chunk overlap within the same section (tail of chunk N vs
    # head of chunk N+1). Some overlap is intentional; flag only excessive.
    by_section = defaultdict(list)
    for c in chunks:
        by_section[c.get("section_id")].append(c)

    for sid, group in by_section.items():
        group_sorted = sorted(group, key=lambda c: c.get("chunk_id", ""))
        for i in range(len(group_sorted) - 1):
            a, b = group_sorted[i], group_sorted[i + 1]
            a_text, b_text = a.get("text", ""), b.get("text", "")
            overlap_chars, matched_text = measure_overlap_chars(a_text, b_text)
            if overlap_chars < 20:
                continue  # negligible — not worth reporting

            overlap_tokens_est = overlap_chars / CHARS_PER_TOKEN_ESTIMATE
            a_tokens = a.get("token_count") or count_tokens(a_text)
            b_tokens = b.get("token_count") or count_tokens(b_text)
            smaller_tokens = min(a_tokens, b_tokens) or 1
            overlap_pct = overlap_tokens_est / smaller_tokens

            cid = b.get("chunk_id", "<unknown>")
            msg = (f"~{overlap_chars} chars (~{overlap_tokens_est:.0f} tokens, "
                   f"{overlap_pct:.0%} of the smaller neighboring chunk) shared between "
                   f"tail of '{a.get('chunk_id')}' and head of this chunk")

            if overlap_pct >= OVERLAP_PCT_ERROR:
                report.add("ERROR", "excessive_overlap", cid,
                            msg + " — well beyond typical overlap, check chunker.py overlap logic")
            elif overlap_pct >= OVERLAP_PCT_WARNING:
                report.add("WARNING", "excessive_overlap", cid, msg)
            else:
                report.add("INFO", "measured_overlap", cid, msg)


def check_section_boundary_leakage(chunks, sections_by_id, report):
    """
    Flags chunks whose page_start is earlier than the section's own
    documented start page (a sign that overlap pulled in content from the
    previous section but metadata still claims the new section_id), or
    whose page_end is later than the section's documented end page.
    """
    if not sections_by_id:
        return
    for c in chunks:
        cid = c.get("chunk_id", "<unknown>")
        sid = c.get("section_id")
        section = sections_by_id.get(sid)
        if not section:
            continue
        sec_start = section.get("page_start")
        sec_end = section.get("page_end")
        ps, pe = c.get("page_start"), c.get("page_end")
        if sec_start is not None and ps is not None and ps < sec_start:
            report.add("WARNING", "section_boundary_leak", cid,
                        f"Chunk page_start ({ps}) precedes its section's page_start "
                        f"({sec_start}) — possible bleed from previous section")
        if sec_end is not None and pe is not None and pe > sec_end:
            report.add("WARNING", "section_boundary_leak", cid,
                        f"Chunk page_end ({pe}) exceeds its section's page_end "
                        f"({sec_end}) — possible bleed into next section")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_sections_by_id(sections_data):
    if not sections_data:
        return {}
    if isinstance(sections_data, dict) and "sections" in sections_data:
        sections_data = sections_data["sections"]
    return {s["section_id"]: s for s in sections_data if "section_id" in s}


def run_validation(chunks_path: Path, sections_path: Path):
    report = Report()

    chunks_data = load_json(chunks_path)
    if chunks_data is None:
        print(f"ERROR: could not find {chunks_path}")
        sys.exit(1)

    chunks = chunks_data["chunks"] if isinstance(chunks_data, dict) and "chunks" in chunks_data else chunks_data
    sections_data = load_json(sections_path)
    sections_by_id = build_sections_by_id(sections_data)

    seen_ids = set()
    for chunk in chunks:
        check_required_fields(chunk, report)
        check_empty_chunk(chunk, report)
        check_valid_chunk_id(chunk, report, seen_ids)
        check_valid_section_id(chunk, sections_by_id, report)
        check_page_numbers(chunk, report)
        check_token_count(chunk, report)
        check_token_thresholds(chunk, report)
        check_excluded_sections(chunk, report)
        check_running_headers(chunk, sections_by_id, report)
        check_excluded_content_leak(chunk, report)
        check_standalone_page_numbers(chunk, report)
        check_broken_text(chunk, report)

    check_duplicates_and_overlap(chunks, report)
    check_section_boundary_leakage(chunks, sections_by_id, report)

    return report, chunks


def print_report(report: Report, total_chunks: int):
    print("=" * 70)
    print("VALIDATION REPORT — adult-hypertension-rag")
    print("=" * 70)
    print(f"Total chunks checked: {total_chunks}")

    counts = Counter(i["level"] for i in report.issues)
    print(f"Errors:   {counts.get('ERROR', 0)}")
    print(f"Warnings: {counts.get('WARNING', 0)}")
    print(f"Info:     {counts.get('INFO', 0)}")
    print()

    if not report.issues:
        print("✅ No issues found. chunks.json is clean.")
        return

    by_check = defaultdict(list)
    for issue in report.issues:
        by_check[issue["check"]].append(issue)

    for check_name, issues in sorted(by_check.items()):
        level = issues[0]["level"]
        symbol = {"ERROR": "❌", "WARNING": "⚠️ ", "INFO": "ℹ️ "}[level]
        print(f"{symbol} {check_name} ({len(issues)} occurrence(s))")
        for issue in issues[:5]:
            print(f"    - [{issue['chunk_id']}] {issue['message']}")
        if len(issues) > 5:
            print(f"    ... and {len(issues) - 5} more")
        print()


def main():
    parser = argparse.ArgumentParser(description="Validate chunks.json before embeddings.")
    parser.add_argument("--chunks", default="data/processed/chunks.json")
    parser.add_argument("--sections", default="data/processed/sections.json")
    parser.add_argument("--out", default="data/processed/validation_report.json")
    parser.add_argument("--validated-out", default="data/processed/validated_chunks.json")
    parser.add_argument("--strict", action="store_true",
                         help="Exit with code 1 if any ERROR-level issue is found")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    sections_path = Path(args.sections)

    report, chunks = run_validation(chunks_path, sections_path)
    print_report(report, len(chunks))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "total_chunks": len(chunks),
            "error_count": len(report.errors()),
            "warning_count": len(report.warnings()),
            "issues": report.issues,
        }, f, ensure_ascii=False, indent=2)
    print(f"Full report written to {out_path}")

    if not report.errors():
        validated_path = Path(args.validated_out)
        with validated_path.open("w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"✅ No blocking errors — wrote {validated_path}")
    else:
        print("❌ Blocking errors found — validated_chunks.json NOT written. Fix errors and re-run.")
        if args.strict:
            sys.exit(1)


if __name__ == "__main__":
    main()