# 🩺 Adult Hypertension Clinical RAG & Decision Support System

An enterprise-grade, evidence-grounded **Clinical Decision Support (CDS)** and **Retrieval-Augmented Generation (RAG)** platform specialized in adult hypertension diagnosis, risk stratification, and pharmacological management according to **WHO, ESC (2024), NICE, CDC, and USPSTF** guidelines.

---

## 🌟 Key Features

* **🔬 Multi-Guideline Hybrid Retrieval Engine**:
  * Dual dense (Qdrant semantic vector search via `text-embedding-004`) + lexical (BM25) reciprocal rank fusion (RRF).
  * Strict guideline isolation & `@` mention targeting (`@WHO`, `@ESC`, `@CDC`, `@NICE`, `@USPSTF`).
* **📖 Interactive PDF Viewer & Continuous Sub-Pixel Highlighting**:
  * Real-time document preview with synchronized character-offset text mapping.
  * Direct citation badges `[1]`, `[2]` that jump to and highlight cited evidence passages across PDF pages.
* **🩺 Patient Blood Pressure Test Report Ingestion & Personalized CDS**:
  * Upload 24-hour ABPM reports, clinic vitals sheets, or hospital lab test reports (PDF).
  * Automated clinical entity extraction: Systolic/Diastolic BP, Heart Rate, Hypertension Stage, Comorbidities (Diabetes, CKD, CVD), Medications, and Lab values (eGFR, Creatinine, HbA1c, Potassium).
  * Personalized treatment recommendation synthesis tailored directly to individual patient vitals and risk profiles.
* **🔐 User Authentication & Persistent History**:
  * SQLite database (`data/users.db`) with PBKDF2/SHA-256 password hashing.
  * Multi-turn conversational memory saved and synchronized per user.
* **⚡ Guideline Auto-Ingestion Pipeline**:
  * Upload new guideline PDFs with real-time Server-Sent Events (SSE) progress tracking.

---

## 🏗️ Architecture

```text
adult-hypertension-rag/
│
├── app.py                      # Flask backend & REST/SSE API
├── requirements.txt            # Python dependencies
├── .env.example                # Template for environment configuration
├── .gitignore                  # Git ignore rules
│
├── src/
│   ├── auth/                   # SQLite authentication & user persistence
│   │   └── auth_db.py
│   ├── chunking/               # Section-aware & sliding window chunkers
│   │   ├── hierarchical_chunker.py
│   │   └── section_chunker.py
│   ├── clinical/               # Patient test report parser & entity extractor
│   │   └── patient_parser.py
│   ├── embeddings/             # Gemini embedding client & rate limiter
│   │   └── gemini_embedder.py
│   ├── ingestion/              # PDF extractor, validator & auto-ingestion pipeline
│   │   ├── auto_ingest.py
│   │   ├── pdf_extractor.py
│   │   └── validator.py
│   ├── rag/                    # Hybrid retrieval, prompt synthesis & guideline isolation
│   │   ├── bm25_retriever.py
│   │   └── rag_pipeline.py
│   └── vector_store/           # Qdrant vector database collection manager
│       └── qdrant_store.py
│
├── templates/
│   └── index.html              # Medical UI with PDF viewer, chat, auth & CDS portal
│
└── data/
    ├── source_registry.json    # Canonical registry of indexed guidelines
    └── raw/
        └── guidelines/         # Raw guideline PDF documents (WHO, ESC, CDC, etc.)
```

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/<your-username>/adult-hypertension-rag.git
cd adult-hypertension-rag

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```ini
GEMINI_API_KEY=your_gemini_api_key_here
FLASK_SECRET_KEY=your_secure_flask_secret_key
PORT=5000
```

### 3. Run the Application

```bash
python app.py
```

Open your browser at **`http://localhost:5000`**.

---

## 📋 Clinical Usage Guide

1. **Target a Guideline**: Type `@` in the query box to select a specific clinical authority (e.g. `@ESC`, `@WHO`, `@CDC`).
2. **Upload Patient Test Results**: Click **`🩺 Upload BP Test Results`** in the sidebar or top bar to upload an ABPM or lab vitals PDF.
3. **Personalized Inquiry**: Ask questions like:
   * *"What is the recommended initial drug therapy and target BP goal for this patient?"*
   * *"When should combination therapy be initiated in Stage 2 hypertension with diabetes?"*
4. **Inspect Evidence**: Click any citation badge `[1]` to open the PDF viewer with exact passage highlighting.

---

## 📜 License & Medical Disclaimer

This tool is designed for research, academic, and clinical decision support purposes. Treatment decisions should always be made by a licensed healthcare professional.
