"""
Patient Blood Pressure Test Report Parser
Extracts text and clinical entities (BP readings, stage, comorbidities, medications, labs) from patient PDFs.
"""

import os
import json
import fitz  # PyMuPDF
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
_api_key = os.getenv("GEMINI_API_KEY")
_client = genai.Client(api_key=_api_key) if _api_key else None


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract raw text from patient test report PDF."""
    doc = fitz.open(str(pdf_path))
    pages_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:
            pages_text.append(f"--- Page {page_num + 1} ---\n{text}")
    doc.close()
    return "\n\n".join(pages_text)


def parse_patient_report(pdf_path: str | Path) -> dict:
    """
    Parse a patient BP test report / lab result PDF and return structured clinical parameters and summary.
    """
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        raise ValueError("Could not extract any readable text from the uploaded PDF.")

    if _client is None:
        raise EnvironmentError("GEMINI_API_KEY is not configured.")

    prompt = f"""You are a specialized Clinical Data Extraction Assistant. 
Analyze the following patient Blood Pressure test report / lab sheet and extract the clinical parameters into clean JSON.

REPORT TEXT:
\"\"\"
{raw_text[:12000]}
\"\"\"

Extract the following JSON schema:
{{
  "patient_name": "string (or Anonymous if not provided)",
  "age": "number or null",
  "gender": "string or null",
  "systolic": "average or latest systolic blood pressure (number e.g. 150)",
  "diastolic": "average or latest diastolic blood pressure (number e.g. 95)",
  "heart_rate": "number or null",
  "bp_category": "e.g. Normal, Elevated, Stage 1 Hypertension, Stage 2 Hypertension, Isolated Systolic Hypertension, Hypertensive Crisis",
  "comorbidities": ["list of comorbidities, e.g. Type 2 Diabetes, Chronic Kidney Disease, CVD, Stroke, Smoking, Dyslipidemia"],
  "medications": ["list of current medications mentioned"],
  "lab_values": {{"key": "value" (e.g. eGFR, Creatinine, Potassium, Cholesterol if present)}},
  "clinical_summary": "A concise 2-3 sentence clinical summary describing the patient's blood pressure status, hypertension stage, and relevant risk factors."
}}

Return ONLY valid JSON. Do not include markdown code block formatting.
"""

    models_to_try = [
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-3.5-flash-lite",
    ]

    last_err = None
    for model_name in models_to_try:
        try:
            response = _client.models.generate_content(
                model=model_name,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
                contents=prompt,
            )
            data = json.loads(response.text.strip())
            return {
                "raw_text": raw_text[:4000],
                "vitals": data,
                "summary": data.get("clinical_summary", "Patient Blood Pressure Report analyzed.")
            }
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed to parse patient report with LLM: {last_err}")
