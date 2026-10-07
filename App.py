"""
StudyPilot AI — App.py
Single-file Flask backend for the Smart Study Generator Agent.

IMPORTANT: All documents, subjects, quiz results and study plans are held
in-memory.  Data is lost when the server restarts unless IBM Cloud storage
(Cloudant + COS) is connected via .env.  See README section at the bottom
of this file for deployment notes.

For multi-user / public deployment you MUST add authentication and
per-user data isolation before exposing this to the internet.
"""

import os, io, re, json, math, uuid, hashlib, datetime, logging, traceback
import urllib.request, urllib.error
from functools import lru_cache

from flask import (
    Flask, request, jsonify, send_file, send_from_directory,
    render_template_string
)
from dotenv import load_dotenv

# ── optional document parsing ──────────────────────────────────────────────
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ── optional BM25 retrieval ────────────────────────────────────────────────
try:
    from rank_bm25 import BM25Okapi
    BM25_SUPPORT = True
except ImportError:
    BM25_SUPPORT = False

# ── IBM watsonx SDK (optional — only used when LLM_BACKEND=watsonx) ────────
try:
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    WATSONX_SDK = True
except ImportError:
    try:
        from ibm_watson_machine_learning import APIClient
        from ibm_watson_machine_learning.foundation_models import Model as ModelInference
        WATSONX_SDK = True
    except ImportError:
        WATSONX_SDK = False

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("studypilot")

app = Flask(__name__, static_folder=None)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

# ═══════════════════════════════════════════════════════════════════════════
# IN-MEMORY STORE
# ═══════════════════════════════════════════════════════════════════════════
subjects: dict = {}        # id -> subject dict
documents: dict = {}       # id -> document dict  (text + chunks)
doc_index: dict = {}       # subject_id -> list of chunk dicts
quiz_attempts: list = []   # all quiz attempts
activity_log: list = []    # recent activity items (capped at 50)
flashcard_sets: dict = {}  # id -> flashcard set dict

def _log_activity(action: str, detail: str = ""):
    activity_log.insert(0, {
        "id": str(uuid.uuid4()),
        "action": action,
        "detail": detail,
        "ts": datetime.datetime.utcnow().isoformat()
    })
    if len(activity_log) > 50:
        activity_log.pop()

# ═══════════════════════════════════════════════════════════════════════════
# LLM CLIENT — supports two backends, selected by LLM_BACKEND in .env
#
#  LLM_BACKEND=openai_compat  (default)
#    Uses any OpenAI-compatible /chat/completions endpoint.
#    Set WATSONX_URL to the base URL (e.g. https://api.groq.com/openai/v1).
#    IBM_CLOUD_API_KEY is used as the Bearer token.
#    WATSONX_PROJECT_ID is not required.
#    Detected automatically when IBM_CLOUD_API_KEY starts with "gsk_".
#
#  LLM_BACKEND=watsonx
#    Uses the ibm-watsonx-ai SDK.
#    Requires IBM_CLOUD_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL.
# ═══════════════════════════════════════════════════════════════════════════

def _detect_backend() -> str:
    """Auto-detect backend from key prefix if LLM_BACKEND not set."""
    explicit = os.getenv("LLM_BACKEND", "").strip().lower()
    if explicit in ("openai_compat", "watsonx"):
        return explicit
    api_key = os.getenv("IBM_CLOUD_API_KEY", "")
    # gsk_ = Groq; sk- = OpenAI-style; both use openai_compat
    if api_key.startswith("gsk_") or api_key.startswith("sk-"):
        return "openai_compat"
    return "watsonx"


def _call_openai_compat(prompt: str, max_tokens: int) -> tuple[str, str | None]:
    """Call any OpenAI-compatible /chat/completions endpoint via requests."""
    import requests as _requests
    api_key  = os.getenv("IBM_CLOUD_API_KEY", "")
    base_url = os.getenv("WATSONX_URL", "https://api.groq.com/openai/v1").rstrip("/")
    model    = os.getenv("GRANITE_MODEL_ID", "openai/gpt-oss-120b")

    if not api_key or api_key.startswith("your_"):
        return "", "IBM_CLOUD_API_KEY not configured"

    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "User-Agent":    "StudyPilotAI/1.0",
    }
    try:
        resp = _requests.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
            return text.strip(), None
        body = resp.text[:300]
        log.warning("LLM HTTP %s: %s", resp.status_code, body)
        if resp.status_code == 401:
            return "", "API key rejected (401). Check IBM_CLOUD_API_KEY value."
        if resp.status_code == 403:
            # Cloudflare 1010 = bot block; actual 403 = permission denied
            if "1010" in body or "cloudflare" in body.lower():
                return "", "Request blocked by network filter (Cloudflare 1010). Retry or check network."
            return "", "Permission denied (403). The key may lack access to this model."
        if resp.status_code == 404:
            return "", f"Model not found (404): {model}. Check GRANITE_MODEL_ID."
        if resp.status_code == 429:
            return "", "Rate limit reached (429). Please wait and retry."
        return "", f"LLM error {resp.status_code}: {body[:120]}"
    except Exception as exc:
        log.warning("LLM call failed: %s", exc)
        return "", f"Network error calling LLM: {exc}"


def _call_watsonx(prompt: str, max_tokens: int) -> tuple[str, str | None]:
    """Call IBM watsonx.ai via the ibm-watsonx-ai SDK."""
    api_key = os.getenv("IBM_CLOUD_API_KEY", "")
    url     = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    proj_id = os.getenv("WATSONX_PROJECT_ID", "")
    model   = os.getenv("GRANITE_MODEL_ID", "ibm/granite-3-8b-instruct")

    if not api_key or api_key.startswith("your_"):
        return "", "IBM_CLOUD_API_KEY not configured"
    if not proj_id or proj_id.startswith("your_") or proj_id == "not_used_for_groq":
        return "", "WATSONX_PROJECT_ID not configured (required for watsonx backend)"
    if not WATSONX_SDK:
        return "", "ibm-watsonx-ai SDK not installed (pip install ibm-watsonx-ai)"
    try:
        creds  = Credentials(url=url, api_key=api_key)
        client = ModelInference(
            model_id=model, credentials=creds, project_id=proj_id,
            params={"max_new_tokens": max_tokens, "temperature": 0.3, "repetition_penalty": 1.1}
        )
        resp = client.generate_text(prompt=prompt)
        if isinstance(resp, str):
            return resp.strip(), None
        if isinstance(resp, dict):
            return resp.get("results", [{}])[0].get("generated_text", "").strip(), None
        return str(resp).strip(), None
    except Exception as exc:
        log.warning("watsonx call failed: %s", exc)
        err_msg = str(exc)
        if "401" in err_msg or "403" in err_msg:
            return "", "Authentication failed. Check IBM_CLOUD_API_KEY and WATSONX_PROJECT_ID."
        if "404" in err_msg:
            return "", f"Model not found: {model}. Check GRANITE_MODEL_ID."
        if "429" in err_msg:
            return "", "Rate limit reached. Please wait and retry."
        return "", f"watsonx error: {exc}"


def _call_granite(prompt: str, max_tokens: int = 1024) -> tuple[str, str | None]:
    """Unified LLM call — routes to the correct backend automatically."""
    backend = _detect_backend()
    if backend == "openai_compat":
        return _call_openai_compat(prompt, max_tokens)
    return _call_watsonx(prompt, max_tokens)

# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENT PROCESSING
# ═══════════════════════════════════════════════════════════════════════════
CHUNK_SIZE   = 400   # words per chunk
CHUNK_OVERLAP = 60   # word overlap between chunks

def _extract_text_pdf(data: bytes) -> list[dict]:
    """Return list of {page, text} dicts."""
    if not PDF_SUPPORT:
        raise RuntimeError("pdfplumber not installed")
    pages = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for i, pg in enumerate(pdf.pages, 1):
            t = pg.extract_text() or ""
            pages.append({"page": i, "text": t.strip()})
    return pages

def _extract_text_txt(data: bytes) -> list[dict]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    # split into ~50-line virtual pages
    lines = text.splitlines()
    pages, chunk_lines = [], 50
    for i in range(0, max(len(lines), 1), chunk_lines):
        pages.append({"page": i // chunk_lines + 1,
                      "text": "\n".join(lines[i:i + chunk_lines]).strip()})
    return pages


def _chunk_pages(pages: list[dict], doc_id: str, doc_name: str) -> list[dict]:
    chunks = []
    buffer_words: list[str] = []
    buffer_pages: list[int] = []

    def flush(idx):
        if not buffer_words:
            return
        chunks.append({
            "chunk_id": f"{doc_id}_{idx}",
            "doc_id": doc_id,
            "doc_name": doc_name,
            "pages": sorted(set(buffer_pages)),
            "text": " ".join(buffer_words),
            "tokens": buffer_words.copy(),
        })

    ci = 0
    for pg in pages:
        words = pg["text"].split()
        for w in words:
            buffer_words.append(w)
            buffer_pages.append(pg["page"])
            if len(buffer_words) >= CHUNK_SIZE:
                flush(ci)
                ci += 1
                # keep overlap
                buffer_words = buffer_words[-CHUNK_OVERLAP:]
                buffer_pages = buffer_pages[-CHUNK_OVERLAP:]
    flush(ci)
    return chunks


def _index_chunks(subject_id: str, chunks: list[dict]):
    if subject_id not in doc_index:
        doc_index[subject_id] = []
    doc_index[subject_id].extend(chunks)


def _remove_doc_chunks(subject_id: str, doc_id: str):
    if subject_id in doc_index:
        doc_index[subject_id] = [c for c in doc_index[subject_id]
                                  if c["doc_id"] != doc_id]

# ═══════════════════════════════════════════════════════════════════════════
# RETRIEVAL  (BM25 or simple keyword)
# ═══════════════════════════════════════════════════════════════════════════

def _retrieve(subject_id: str, query: str, top_k: int = 5) -> list[dict]:
    pool = doc_index.get(subject_id, [])
    if not pool:
        return []

    q_words = [w.lower() for w in query.split() if len(w) > 2]

    if BM25_SUPPORT and len(pool) > 0:
        corpus = [c["tokens"] for c in pool]
        bm25 = BM25Okapi([[t.lower() for t in toks] for toks in corpus])
        scores = bm25.get_scores(q_words)
        ranked = sorted(zip(scores, pool), key=lambda x: -x[0])
        return [c for _, c in ranked[:top_k] if _ > 0] or pool[:top_k]

    # fallback: simple term-frequency keyword search
    def score(chunk):
        txt = chunk["text"].lower()
        return sum(txt.count(w) for w in q_words)

    ranked = sorted(pool, key=score, reverse=True)
    return ranked[:top_k]


def _context_block(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        pages = ", ".join(str(p) for p in c["pages"])
        parts.append(f"[Source: {c['doc_name']}, p.{pages}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)

# ═══════════════════════════════════════════════════════════════════════════
# ADAPTIVE PLANNER  (deterministic logic)
# ═══════════════════════════════════════════════════════════════════════════

def _build_study_plan(subject_id: str) -> dict:
    subj = subjects.get(subject_id)
    if not subj:
        return {"error": "Subject not found"}

    exam_date_str = subj.get("exam_date", "")
    daily_minutes = int(subj.get("daily_minutes", 60))
    topics = subj.get("topics", [])

    try:
        exam_dt = datetime.date.fromisoformat(exam_date_str)
    except Exception:
        exam_dt = datetime.date.today() + datetime.timedelta(days=30)

    today = datetime.date.today()
    days_left = max((exam_dt - today).days, 1)
    total_minutes = days_left * daily_minutes

    # topic weights: weak topics (low quiz score) get more time
    topic_scores = {}
    for attempt in quiz_attempts:
        if attempt["subject_id"] == subject_id:
            for t, s in attempt.get("topic_scores", {}).items():
                if t not in topic_scores:
                    topic_scores[t] = []
                topic_scores[t].append(s)

    avg_scores = {t: sum(v)/len(v) for t, v in topic_scores.items()}

    # auto-extract topics from doc chunks if none set
    if not topics:
        chunks = doc_index.get(subject_id, [])
        seen, topics = set(), []
        for c in chunks:
            words = c["text"].split()
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if bigram.istitle() and bigram not in seen and len(bigram) < 40:
                    seen.add(bigram)
                    topics.append(bigram)
                    if len(topics) >= 10:
                        break
            if len(topics) >= 10:
                break
        if not topics:
            topics = ["Review uploaded materials", "Practice problems",
                      "Self-assessment", "Revision"]

    # compute per-topic weight (lower score → higher weight)
    weights = []
    for t in topics:
        score = avg_scores.get(t, 0.5)
        w = 1.0 + (1.0 - score)   # 1.0–2.0
        weights.append(w)
    total_weight = sum(weights) or 1

    # allocate minutes
    allocated = [round(w / total_weight * total_minutes) for w in weights]

    # daily schedule
    schedule_days = []
    topic_iter = iter(zip(topics, allocated))
    cur_topic, cur_mins = None, 0
    try:
        cur_topic, cur_mins = next(topic_iter)
    except StopIteration:
        pass

    for d in range(days_left):
        dt = today + datetime.timedelta(days=d)
        day_tasks = []
        remaining_day = daily_minutes
        while remaining_day > 0 and cur_topic:
            use = min(remaining_day, cur_mins)
            day_tasks.append({
                "topic": cur_topic,
                "minutes": use,
                "priority": "high" if avg_scores.get(cur_topic, 1) < 0.6 else "normal"
            })
            remaining_day -= use
            cur_mins -= use
            if cur_mins <= 0:
                try:
                    cur_topic, cur_mins = next(topic_iter)
                except StopIteration:
                    cur_topic = None
                    break
        schedule_days.append({
            "date": dt.isoformat(),
            "tasks": day_tasks,
            "total_minutes": daily_minutes - remaining_day
        })

    shortfall = None
    if cur_topic:
        shortfall = (f"Not enough study time before the exam. "
                     f"Consider increasing daily minutes or covering fewer topics.")

    return {
        "subject_id": subject_id,
        "exam_date": exam_dt.isoformat(),
        "days_left": days_left,
        "daily_minutes": daily_minutes,
        "topics": topics,
        "topic_scores": avg_scores,
        "schedule": schedule_days,
        "shortfall": shortfall,
        "generated_at": datetime.datetime.utcnow().isoformat()
    }

# ═══════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_file("Index.html")

@app.route("/Style.css")
def stylesheet():
    return send_file("Style.css", mimetype="text/css")

# ── health ─────────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    api_key = os.getenv("IBM_CLOUD_API_KEY", "")
    ai_ok   = bool(api_key and not api_key.startswith("your_"))
    backend = _detect_backend()
    return jsonify({
        "status": "ok",
        "pdf_support": PDF_SUPPORT,
        "bm25_support": BM25_SUPPORT,
        "watsonx_sdk": WATSONX_SDK,
        "ai_configured": ai_ok,
        "llm_backend": backend,
        "model": os.getenv("GRANITE_MODEL_ID", "not set"),
        "endpoint": os.getenv("WATSONX_URL", "not set"),
        "note": "In-memory store — data resets on server restart"
    })

# ── subjects ───────────────────────────────────────────────────────────────
@app.route("/api/subjects", methods=["GET"])
def list_subjects():
    return jsonify(list(subjects.values()))

@app.route("/api/subjects", methods=["POST"])
def create_subject():
    data = request.get_json(force=True)
    sid = str(uuid.uuid4())
    subj = {
        "id": sid,
        "name": data.get("name", "Untitled").strip()[:100],
        "exam_date": data.get("exam_date", ""),
        "daily_minutes": int(data.get("daily_minutes", 60)),
        "topics": data.get("topics", []),
        "created_at": datetime.datetime.utcnow().isoformat()
    }
    subjects[sid] = subj
    _log_activity("Subject created", subj["name"])
    return jsonify(subj), 201

@app.route("/api/subjects/<sid>", methods=["PUT"])
def update_subject(sid):
    if sid not in subjects:
        return jsonify({"error": "Not found"}), 404
    data = request.get_json(force=True)
    s = subjects[sid]
    s["name"]          = data.get("name", s["name"]).strip()[:100]
    s["exam_date"]     = data.get("exam_date", s["exam_date"])
    s["daily_minutes"] = int(data.get("daily_minutes", s["daily_minutes"]))
    s["topics"]        = data.get("topics", s["topics"])
    _log_activity("Subject updated", s["name"])
    return jsonify(s)

@app.route("/api/subjects/<sid>", methods=["DELETE"])
def delete_subject(sid):
    if sid not in subjects:
        return jsonify({"error": "Not found"}), 404
    name = subjects[sid]["name"]
    del subjects[sid]
    # remove documents
    to_del = [did for did, d in documents.items() if d["subject_id"] == sid]
    for did in to_del:
        del documents[did]
    doc_index.pop(sid, None)
    _log_activity("Subject deleted", name)
    return jsonify({"ok": True})

# ── documents ──────────────────────────────────────────────────────────────
@app.route("/api/subjects/<sid>/documents", methods=["GET"])
def list_documents(sid):
    docs = [d for d in documents.values() if d["subject_id"] == sid]
    # don't send full text back in list
    return jsonify([{k: v for k, v in d.items() if k != "pages"} for d in docs])

@app.route("/api/subjects/<sid>/documents", methods=["POST"])
def upload_document(sid):
    if sid not in subjects:
        return jsonify({"error": "Subject not found"}), 404
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    fname = f.filename or "upload"
    ext = fname.rsplit(".", 1)[-1].lower()
    if ext not in ("pdf", "txt"):
        return jsonify({"error": "Only PDF and TXT files are accepted"}), 415

    raw = f.read()
    if len(raw) == 0:
        return jsonify({"error": "File is empty"}), 400

    doc_type = request.form.get("doc_type", "notes")
    did = str(uuid.uuid4())

    try:
        if ext == "pdf":
            if not PDF_SUPPORT:
                return jsonify({"error": "pdfplumber not installed — PDF upload unavailable"}), 500
            pages = _extract_text_pdf(raw)
        else:
            pages = _extract_text_txt(raw)
    except Exception as exc:
        log.error("Extract error: %s", exc)
        return jsonify({"error": f"Could not extract text: {exc}"}), 500

    total_text = " ".join(p["text"] for p in pages)
    word_count = len(total_text.split())

    if word_count < 5:
        return jsonify({
            "error": ("No extractable text found. "
                      "This appears to be a scanned/image-only PDF. "
                      "Please use a text-based PDF or a TXT file."),
            "scanned": True
        }), 422

    chunks = _chunk_pages(pages, did, fname)
    _index_chunks(sid, chunks)

    doc = {
        "id": did,
        "subject_id": sid,
        "name": fname,
        "doc_type": doc_type,
        "ext": ext,
        "word_count": word_count,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "pages": pages,          # kept in memory for preview
        "status": "indexed",
        "uploaded_at": datetime.datetime.utcnow().isoformat()
    }
    documents[did] = doc
    _log_activity("Document uploaded", fname)
    return jsonify({k: v for k, v in doc.items() if k != "pages"}), 201

@app.route("/api/documents/<did>/preview", methods=["GET"])
def preview_document(did):
    if did not in documents:
        return jsonify({"error": "Not found"}), 404
    doc = documents[did]
    pages = doc.get("pages", [])
    # return first 3 pages
    return jsonify({"name": doc["name"], "pages": pages[:3]})

@app.route("/api/documents/<did>", methods=["DELETE"])
def delete_document(did):
    if did not in documents:
        return jsonify({"error": "Not found"}), 404
    doc = documents[did]
    sid = doc["subject_id"]
    name = doc["name"]
    _remove_doc_chunks(sid, did)
    del documents[did]
    _log_activity("Document deleted", name)
    return jsonify({"ok": True})

# ── chat / Q&A ─────────────────────────────────────────────────────────────
@app.route("/api/subjects/<sid>/chat", methods=["POST"])
def chat(sid):
    if sid not in subjects:
        return jsonify({"error": "Subject not found"}), 404
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()[:2000]
    if not question:
        return jsonify({"error": "Question is required"}), 400

    chunks = _retrieve(sid, question, top_k=5)
    if not chunks:
        return jsonify({
            "answer": ("No study materials have been uploaded for this subject yet. "
                       "Please upload at least one PDF or TXT file first."),
            "citations": [],
            "grounded": False
        })

    context = _context_block(chunks)
    subj_name = subjects[sid]["name"]
    prompt = f"""You are a helpful study assistant for the subject "{subj_name}".
Answer the student's question using ONLY the provided source passages.
If the passages do not contain enough information to answer, say clearly:
"The uploaded materials do not cover this topic directly."
Do not fabricate information. Cite the document name and page number.

SOURCE PASSAGES:
{context}

STUDENT QUESTION: {question}

ANSWER (be clear, concise, and educational):"""

    answer, err = _call_granite(prompt)
    if err:
        return jsonify({"error": err}), 503

    citations = [{"doc": c["doc_name"], "pages": c["pages"]} for c in chunks]
    _log_activity("Question asked", question[:60])
    return jsonify({"answer": answer, "citations": citations, "grounded": True})

# ── summaries ──────────────────────────────────────────────────────────────
@app.route("/api/subjects/<sid>/summary", methods=["POST"])
def generate_summary(sid):
    if sid not in subjects:
        return jsonify({"error": "Subject not found"}), 404
    data = request.get_json(force=True)
    topic   = (data.get("topic") or "all content").strip()
    style   = data.get("style", "short")  # short | detailed | keyterms

    chunks = _retrieve(sid, topic, top_k=8)
    if not chunks:
        return jsonify({"error": "No content found for this topic"}), 404

    context = _context_block(chunks)
    subj_name = subjects[sid]["name"]

    if style == "short":
        instruction = "Write a concise summary (150–200 words) of the key points."
    elif style == "detailed":
        instruction = "Write comprehensive revision notes covering all major concepts."
    else:  # keyterms
        instruction = ("List the most important key terms and their definitions "
                       "as a markdown definition list.")

    prompt = f"""You are a study assistant summarizing content from "{subj_name}".
{instruction}
Use ONLY information from the passages below. Do not add outside knowledge.

PASSAGES:
{context}

TOPIC: {topic}

OUTPUT:"""

    text, err = _call_granite(prompt, max_tokens=800)
    if err:
        return jsonify({"error": err}), 503

    result = {
        "id": str(uuid.uuid4()),
        "subject_id": sid,
        "topic": topic,
        "style": style,
        "content": text,
        "sources": [{"doc": c["doc_name"], "pages": c["pages"]} for c in chunks],
        "created_at": datetime.datetime.utcnow().isoformat()
    }
    _log_activity("Summary generated", f"{style} — {topic[:40]}")
    return jsonify(result)

# ── flashcards ─────────────────────────────────────────────────────────────
@app.route("/api/subjects/<sid>/flashcards", methods=["POST"])
def generate_flashcards(sid):
    if sid not in subjects:
        return jsonify({"error": "Subject not found"}), 404
    data   = request.get_json(force=True)
    topic  = (data.get("topic") or "key concepts").strip()
    count  = min(int(data.get("count", 10)), 20)

    chunks = _retrieve(sid, topic, top_k=8)
    if not chunks:
        return jsonify({"error": "No content found for this topic"}), 404

    context = _context_block(chunks)
    prompt = f"""Create exactly {count} flashcards from the study material below.
Return ONLY a valid JSON array. Each element: {{"q":"question","a":"answer"}}
Use concise, factual question/answer pairs from the passages only.

PASSAGES:
{context}

TOPIC: {topic}

JSON ARRAY:"""

    raw, err = _call_granite(prompt, max_tokens=1200)
    if err:
        return jsonify({"error": err}), 503

    cards = _parse_json_array(raw)
    if cards is None:
        return jsonify({"error": "Could not parse flashcards. Try again."}), 500

    fid = str(uuid.uuid4())
    fset = {
        "id": fid,
        "subject_id": sid,
        "topic": topic,
        "cards": cards,
        "created_at": datetime.datetime.utcnow().isoformat()
    }
    flashcard_sets[fid] = fset
    _log_activity("Flashcards generated", f"{len(cards)} cards — {topic[:40]}")
    return jsonify(fset)

@app.route("/api/subjects/<sid>/flashcards", methods=["GET"])
def list_flashcards(sid):
    sets = [v for v in flashcard_sets.values() if v["subject_id"] == sid]
    return jsonify(sets)

# ── quizzes ────────────────────────────────────────────────────────────────
# answer keys are stored server-side, never sent until submission
_pending_quizzes: dict = {}   # quiz_id -> {questions with answers}

@app.route("/api/subjects/<sid>/quiz/generate", methods=["POST"])
def generate_quiz(sid):
    if sid not in subjects:
        return jsonify({"error": "Subject not found"}), 404
    data       = request.get_json(force=True)
    topic      = (data.get("topic") or "general").strip()
    difficulty = data.get("difficulty", "medium")
    num_q      = min(int(data.get("num_questions", 5)), 10)

    chunks = _retrieve(sid, topic, top_k=10)
    if not chunks:
        return jsonify({"error": "No content found for this topic"}), 404

    context = _context_block(chunks)
    prompt = f"""Create exactly {num_q} multiple-choice quiz questions about "{topic}" at {difficulty} difficulty.
Base ALL questions strictly on the passages below.
Return ONLY a valid JSON array. Each element MUST have:
{{"q":"question text","options":["A) ...","B) ...","C) ...","D) ..."],"answer":"A","explanation":"...","source":"{{}}"}}
- "answer" must be the letter of the correct option (A/B/C/D).
- "explanation" must cite the source document.

PASSAGES:
{context}

JSON ARRAY:"""

    raw, err = _call_granite(prompt, max_tokens=2000)
    if err:
        return jsonify({"error": err}), 503

    questions = _parse_json_array(raw)
    if not questions:
        return jsonify({"error": "Could not generate valid quiz. Try again."}), 500

    # validate structure
    valid = []
    for q in questions:
        if (isinstance(q, dict) and q.get("q") and
                isinstance(q.get("options"), list) and len(q["options"]) == 4 and
                q.get("answer") in ("A", "B", "C", "D")):
            valid.append(q)

    if not valid:
        return jsonify({"error": "Generated questions did not pass validation. Try again."}), 500

    qid = str(uuid.uuid4())
    _pending_quizzes[qid] = {
        "subject_id": sid,
        "topic": topic,
        "difficulty": difficulty,
        "questions": valid   # includes answer keys — never sent to client here
    }

    # send questions WITHOUT answer keys
    client_qs = [{"idx": i, "q": q["q"], "options": q["options"]}
                 for i, q in enumerate(valid)]
    return jsonify({"quiz_id": qid, "topic": topic, "questions": client_qs})

@app.route("/api/quiz/<qid>/submit", methods=["POST"])
def submit_quiz(qid):
    if qid not in _pending_quizzes:
        return jsonify({"error": "Quiz not found or already submitted"}), 404
    quiz = _pending_quizzes.pop(qid)
    data = request.get_json(force=True)
    answers = data.get("answers", {})   # {str(idx): "A"|"B"|"C"|"D"}

    questions  = quiz["questions"]
    results    = []
    correct    = 0
    topic_hits = {}

    for i, q in enumerate(questions):
        user_ans = answers.get(str(i), "")
        is_correct = user_ans.upper() == q["answer"]
        if is_correct:
            correct += 1
        t = quiz["topic"]
        if t not in topic_hits:
            topic_hits[t] = {"correct": 0, "total": 0}
        topic_hits[t]["total"]  += 1
        topic_hits[t]["correct"] += int(is_correct)

        results.append({
            "idx": i,
            "q": q["q"],
            "options": q["options"],
            "user_answer": user_ans,
            "correct_answer": q["answer"],
            "is_correct": is_correct,
            "explanation": q.get("explanation", "")
        })

    score_pct = round(correct / len(questions) * 100) if questions else 0
    topic_scores = {t: v["correct"] / v["total"]
                    for t, v in topic_hits.items()}

    attempt = {
        "id": str(uuid.uuid4()),
        "subject_id": quiz["subject_id"],
        "topic": quiz["topic"],
        "difficulty": quiz["difficulty"],
        "score": score_pct,
        "correct": correct,
        "total": len(questions),
        "topic_scores": topic_scores,
        "results": results,
        "submitted_at": datetime.datetime.utcnow().isoformat()
    }
    quiz_attempts.append(attempt)
    _log_activity("Quiz completed",
                  f"{quiz['topic']} — {score_pct}% ({correct}/{len(questions)})")
    return jsonify(attempt)

@app.route("/api/subjects/<sid>/quiz/history", methods=["GET"])
def quiz_history(sid):
    attempts = [a for a in quiz_attempts if a["subject_id"] == sid]
    # don't include full results list in history
    return jsonify([{k: v for k, v in a.items() if k != "results"}
                    for a in attempts])

# ── study planner ──────────────────────────────────────────────────────────
@app.route("/api/subjects/<sid>/plan", methods=["GET"])
def get_plan(sid):
    plan = _build_study_plan(sid)
    if "error" in plan:
        return jsonify(plan), 404
    return jsonify(plan)

# ── analytics ──────────────────────────────────────────────────────────────
@app.route("/api/subjects/<sid>/analytics", methods=["GET"])
def analytics(sid):
    attempts = [a for a in quiz_attempts if a["subject_id"] == sid]
    if not attempts:
        return jsonify({"attempts": [], "topic_performance": {},
                        "note": "No quiz data yet."})

    topic_perf: dict = {}
    scores_over_time = []
    for a in attempts:
        scores_over_time.append({
            "date": a["submitted_at"][:10],
            "score": a["score"],
            "topic": a["topic"]
        })
        for t, s in a.get("topic_scores", {}).items():
            if t not in topic_perf:
                topic_perf[t] = []
            topic_perf[t].append(s)

    avg_perf = {t: round(sum(v)/len(v)*100, 1)
                for t, v in topic_perf.items()}
    weak = [t for t, s in avg_perf.items() if s < 60]

    return jsonify({
        "attempts": scores_over_time,
        "topic_performance": avg_perf,
        "weak_topics": weak,
        "total_attempts": len(attempts),
        "average_score": round(sum(a["score"] for a in attempts)/len(attempts), 1)
    })

# ── dashboard ──────────────────────────────────────────────────────────────
@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    today = datetime.date.today().isoformat()
    upcoming = []
    for s in subjects.values():
        ed = s.get("exam_date", "")
        if ed >= today:
            upcoming.append({"subject": s["name"], "exam_date": ed,
                              "days_left": (datetime.date.fromisoformat(ed)
                                            - datetime.date.today()).days})
    upcoming.sort(key=lambda x: x["exam_date"])

    doc_count = len(documents)
    last_attempts = sorted(quiz_attempts, key=lambda a: a["submitted_at"],
                           reverse=True)[:5]

    # today tasks from all plans
    today_tasks = []
    for sid in subjects:
        plan = _build_study_plan(sid)
        for day in plan.get("schedule", []):
            if day["date"] == today:
                for t in day["tasks"]:
                    today_tasks.append({
                        "subject": subjects[sid]["name"],
                        "topic": t["topic"],
                        "minutes": t["minutes"],
                        "priority": t["priority"]
                    })

    return jsonify({
        "subjects": len(subjects),
        "documents": doc_count,
        "upcoming_exams": upcoming,
        "today_tasks": today_tasks,
        "recent_quizzes": [{k: v for k, v in a.items() if k != "results"}
                           for a in last_attempts],
        "recent_activity": activity_log[:10]
    })

# ── export ─────────────────────────────────────────────────────────────────
@app.route("/api/export/summary", methods=["POST"])
def export_summary():
    data = request.get_json(force=True)
    content = data.get("content", "")
    fname   = data.get("filename", "summary.md")
    buf = io.BytesIO(content.encode("utf-8"))
    buf.seek(0)
    return send_file(buf, mimetype="text/markdown",
                     as_attachment=True, download_name=fname)

@app.route("/api/export/flashcards", methods=["POST"])
def export_flashcards():
    data  = request.get_json(force=True)
    cards = data.get("cards", [])
    lines = ["# Flashcards\n"]
    for i, c in enumerate(cards, 1):
        lines.append(f"## Card {i}\n**Q:** {c.get('q','')}\n\n**A:** {c.get('a','')}\n")
    content = "\n".join(lines)
    buf = io.BytesIO(content.encode("utf-8"))
    buf.seek(0)
    return send_file(buf, mimetype="text/markdown",
                     as_attachment=True, download_name="flashcards.md")

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _parse_json_array(text: str):
    """Extract first JSON array from LLM output, return list or None."""
    # try direct parse first
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    # find bracketed content
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None

# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    log.info("StudyPilot AI starting on http://localhost:%d", port)
    log.info("PDF support   : %s", PDF_SUPPORT)
    log.info("BM25 retrieval: %s", BM25_SUPPORT)
    log.info("watsonx SDK   : %s", WATSONX_SDK)
    ai_ok = bool(os.getenv("IBM_CLOUD_API_KEY") and
                 not os.getenv("IBM_CLOUD_API_KEY", "").startswith("your_"))
    log.info("AI configured : %s  (model: %s)", ai_ok,
             os.getenv("GRANITE_MODEL_ID", "not set"))
    if not ai_ok:
        log.warning("AI features will return errors until credentials are set in .env")
    app.run(host="0.0.0.0", port=port, debug=False)
