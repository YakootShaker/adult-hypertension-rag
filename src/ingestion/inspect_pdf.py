import fitz

PDF_PATH = "data/raw/guidelines/WHO_guideline_01.pdf"

doc = fitz.open(PDF_PATH)

print(f"Number of pages: {len(doc)}")

for page_number, page in enumerate(doc[:2], start=1):

    text = page.get_text()

    print("\n" + "=" * 80)
    print(f"PAGE {page_number}")
    print("=" * 80)

    print(text[:2000])