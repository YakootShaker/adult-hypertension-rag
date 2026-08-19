"""
app.py — Flask backend for the Hypertension RAG Assistant.

Features:
- Multi-guideline RAG with dense (Qdrant) and lexical (BM25) hybrid retrieval
- User Authentication & Personalized chat history persistence (SQLite database)
- Patient Blood Pressure Test Report ingestion & personalized clinical tailoring
- Guideline auto-ingestion with real-time SSE progress streaming
"""

import json
import os
import queue
import sys
import threading
import uuid
from pathlib import Path

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
    session,
    stream_with_context,
)
from werkzeug.utils import secure_filename

from auth.auth_db import (
    create_user,
    delete_patient_report,
    delete_user_chat,
    get_latest_patient_report,
    get_user_by_id,
    get_user_chats,
    init_db,
    save_patient_report,
    save_user_chat,
    verify_user,
)
from clinical.patient_parser import parse_patient_report
from ingestion.auto_ingest import auto_ingest_pdf
from rag.rag_pipeline import RAGResult, answer, get_bm25_retriever, get_qdrant_client

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "adult-hypertension-rag-secret-2026-auth")

# Directories
PDF_DIR = Path(__file__).parent / "data" / "raw" / "guidelines"
PDF_DIR.mkdir(parents=True, exist_ok=True)
PATIENT_UPLOADS_DIR = Path(__file__).parent / "data" / "raw" / "patient_reports"
PATIENT_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_PATH = Path(__file__).parent / "data" / "source_registry.json"

# Initialize database
init_db()

# Active upload jobs: upload_id -> Queue of progress events
_upload_jobs: dict[str, queue.Queue] = {}


# ────────────────────────────────────────────────
# Static routes
# ────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/pdf/<path:filename>")
def serve_pdf(filename: str):
    return send_from_directory(PDF_DIR, filename, mimetype="application/pdf")


# ────────────────────────────────────────────────
# User Authentication API
# ────────────────────────────────────────────────

@app.post("/api/auth/register")
def register():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    try:
        user = create_user(name, email, password)
        session["user_id"] = user["id"]
        return jsonify({"user": user, "message": "Registration successful."}), 201
    except ValueError as val_err:
        return jsonify({"error": str(val_err)}), 400
    except Exception as exc:
        return jsonify({"error": f"Failed to register: {str(exc)}"}), 500


@app.post("/api/auth/login")
def login():
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    user = verify_user(email, password)
    if not user:
        return jsonify({"error": "Invalid email or password."}), 401

    session["user_id"] = user["id"]
    return jsonify({"user": user, "message": "Login successful."})


@app.post("/api/auth/logout")
def logout():
    session.clear()
    return jsonify({"status": "logged_out", "message": "Successfully logged out."})


@app.get("/api/auth/me")
def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"user": None})

    user = get_user_by_id(user_id)
    return jsonify({"user": user})


# ────────────────────────────────────────────────
# User Chat Persistence API
# ────────────────────────────────────────────────

@app.get("/api/user/chats")
def list_user_chats():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"chats": {}})
    chats = get_user_chats(user_id)
    return jsonify({"chats": chats})


@app.post("/api/user/chats")
def sync_user_chat():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"status": "unauthenticated"}), 200

    data = request.get_json(silent=True) or {}
    chat_id = data.get("id")
    title = data.get("title", "New conversation")
    if not chat_id:
        return jsonify({"error": "Chat ID is required."}), 400

    save_user_chat(user_id, chat_id, title, data)
    return jsonify({"status": "saved", "chat_id": chat_id})


@app.delete("/api/user/chats/<chat_id>")
def remove_user_chat(chat_id: str):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"status": "unauthenticated"}), 200

    delete_user_chat(user_id, chat_id)
    return jsonify({"status": "deleted", "chat_id": chat_id})


# ────────────────────────────────────────────────
# Patient Blood Pressure Test Report Upload & Profile
# ────────────────────────────────────────────────

@app.post("/api/patient/upload-report")
def upload_patient_report():
    """Extract clinical entities & vitals from an uploaded patient BP test result PDF."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF test reports are supported."}), 400

    filename = secure_filename(file.filename) or f"patient_report_{uuid.uuid4().hex[:8]}.pdf"
    save_path = PATIENT_UPLOADS_DIR / f"{uuid.uuid4().hex[:6]}_{filename}"
    file.save(str(save_path))

    try:
        parsed_result = parse_patient_report(save_path)
        vitals = parsed_result.get("vitals", {})
        summary = parsed_result.get("summary", "")

        user_id = session.get("user_id")
        saved_record = None
        if user_id:
            saved_record = save_patient_report(user_id, filename, vitals, summary)

        # Also keep in session for instant use
        session["patient_profile"] = {
            "filename": filename,
            "vitals": vitals,
            "summary": summary,
        }

        return jsonify({
            "status": "success",
            "filename": filename,
            "vitals": vitals,
            "summary": summary,
            "saved": bool(user_id),
        })
    except Exception as exc:
        return jsonify({"error": f"Failed to parse test report: {str(exc)}"}), 500


@app.get("/api/patient/profile")
def get_patient_profile():
    user_id = session.get("user_id")
    if user_id:
        report = get_latest_patient_report(user_id)
        if report:
            return jsonify({"profile": report})

    session_profile = session.get("patient_profile")
    return jsonify({"profile": session_profile})


@app.delete("/api/patient/profile")
def clear_patient_profile():
    user_id = session.get("user_id")
    if user_id:
        delete_patient_report(user_id)
    session.pop("patient_profile", None)
    return jsonify({"status": "cleared"})


# ────────────────────────────────────────────────
# Guidelines registry
# ────────────────────────────────────────────────

@app.get("/api/guidelines")
def list_guidelines():
    """Return all currently indexed and available guideline documents."""
    guidelines = []
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                guidelines = json.load(f)
        except Exception:
            guidelines = []

    available_files = [f.name for f in PDF_DIR.glob("*.pdf")]
    return jsonify({"guidelines": guidelines, "pdf_files": available_files})


# ────────────────────────────────────────────────
# Upload — async ingestion with SSE progress
# ────────────────────────────────────────────────

@app.post("/api/upload")
def upload_pdf():
    """Save the uploaded PDF and start the ingestion pipeline in a background thread."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    filename = secure_filename(file.filename)
    if not filename:
        filename = f"guideline_{uuid.uuid4().hex[:8]}.pdf"

    save_path = PDF_DIR / filename
    file.save(str(save_path))

    org = (request.form.get("organization") or "Clinical Guideline").strip()
    doc_name = (request.form.get("document_name") or filename.replace("_", " ").replace(".pdf", "")).strip()

    upload_id = str(uuid.uuid4())
    prog_queue = queue.Queue()
    _upload_jobs[upload_id] = prog_queue

    def _run():
        def _cb(stage: str, msg: str, pct: float):
            prog_queue.put({"stage": stage, "message": msg, "percent": pct})

        try:
            result = auto_ingest_pdf(
                pdf_path=save_path,
                document_name=doc_name,
                organization=org,
                qdrant_client=get_qdrant_client(),
                progress_callback=_cb,
            )
            get_bm25_retriever(force_reload=True)
            prog_queue.put({
                "stage": "complete",
                "message": f"✓ Ingestion complete — {result['total_chunks']} chunks indexed",
                "percent": 100,
                "details": result,
            })
        except Exception as exc:
            prog_queue.put({
                "stage": "error",
                "message": f"Ingestion failed: {str(exc)}",
                "percent": -1,
            })
        finally:
            prog_queue.put(None)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"upload_id": upload_id, "filename": filename})


@app.get("/api/upload-stream/<upload_id>")
def upload_stream(upload_id: str):
    """Server-Sent Events stream for real-time ingestion progress."""
    prog_queue = _upload_jobs.get(upload_id)
    if prog_queue is None:
        return jsonify({"error": "Unknown upload_id"}), 404

    @stream_with_context
    def _generate():
        try:
            while True:
                try:
                    event = prog_queue.get(timeout=60)
                except queue.Empty:
                    yield "event: heartbeat\ndata: {}\n\n"
                    continue

                if event is None:
                    yield "event: close\ndata: {}\n\n"
                    break

                yield f"data: {json.dumps(event)}\n\n"

                if event.get("stage") in ("complete", "error"):
                    break
        finally:
            _upload_jobs.pop(upload_id, None)

    return Response(
        _generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ────────────────────────────────────────────────
# RAG Query
# ────────────────────────────────────────────────

@app.post("/api/query")
def query():
    data = request.get_json(silent=True) or {}
    q = (data.get("query") or "").strip()
    history = data.get("history") or []
    doc_filter = (data.get("doc_filter") or data.get("document_filter") or "").strip() or None
    patient_context = data.get("patient_context")

    if not q:
        return jsonify({"error": "Query cannot be empty."}), 400

    # If doc_filter wasn't explicitly passed, check if the query contains @mention
    if not doc_filter and "@" in q:
        import re
        mention_match = re.search(r'@(?:\"([^\"]+)\"|([\w\.\-]+))', q)
        if mention_match:
            doc_filter = mention_match.group(1) or mention_match.group(2)
            cleaned_q = re.sub(r'@(?:\"[^\"]+\"|[\w\.\-]+)', '', q).strip()
            if cleaned_q:
                q = cleaned_q

    # Auto-attach active patient report if available and not explicitly provided
    if not patient_context:
        user_id = session.get("user_id")
        if user_id:
            patient_context = get_latest_patient_report(user_id)
        else:
            patient_context = session.get("patient_profile")

    try:
        result: RAGResult = answer(
            q,
            history=history if history else None,
            doc_filter=doc_filter,
            patient_context=patient_context,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "answer": result.answer,
        "doc_filter": doc_filter,
        "patient_context_applied": bool(patient_context),
        "sources": [
            {
                "chunk_id": s.chunk_id,
                "document_id": s.document_id,
                "pdf_file": s.pdf_file,
                "section_title": s.section_title,
                "page_start": s.page_start,
                "page_end": s.page_end,
                "score": s.score,
                "text": s.text,
            }
            for s in result.sources
        ],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
