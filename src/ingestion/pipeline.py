import sys
import fitz
import json
from pathlib import Path

# Make sure the src/ root is on the path so we can import cleaner
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.cleaner import clean_text


# ============================================================
# Configuration
# ============================================================

PDF_PATH    = "data/raw/guidelines/WHO_guideline_01.pdf"
OUTPUT_PATH = "data/processed/cleaned_pages.json"


# ============================================================
# PDF processing
# ============================================================

def process_pdf(pdf_path: str) -> list[dict]:
    """
    Extract and clean text from every page of the PDF.
    Returns a list of dicts:  {page_number, text}
    """

    doc = fitz.open(pdf_path)
    pages = []

    for page_number, page in enumerate(doc, start=1):
        raw_text    = page.get_text("text")
        cleaned_text = clean_text(raw_text)

        pages.append({
            "page_number": page_number,
            "text":        cleaned_text,
        })

    return pages


# ============================================================
# Run and save
# ============================================================

pages = process_pdf(PDF_PATH)

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(pages, f, ensure_ascii=False, indent=2)

print(f"Processed {len(pages)} pages.")
print(f"Saved to:  {OUTPUT_PATH}")