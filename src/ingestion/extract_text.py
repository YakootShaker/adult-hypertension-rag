import fitz
import json
from pathlib import Path


PDF_PATH = "data/raw/guidelines/WHO_guideline_01.pdf"
OUTPUT_PATH = "data/processed/extracted_pages.json"


def extract_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):

        text = page.get_text("text")

        pages.append({
            "page_number": page_number,
            "text": text
        })

    return pages


pages = extract_pdf(PDF_PATH)

Path(OUTPUT_PATH).parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        pages,
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"Extracted {len(pages)} pages")
print(f"Saved to: {OUTPUT_PATH}")