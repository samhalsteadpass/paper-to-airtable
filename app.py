"""
app_v4.py  –  Past Paper → Airtable  (manual capture + AI assignment)
=====================================================================
Flow:
  1. Upload PDFs
  2. Extract questions + mark scheme (GPT, parallel)
  3. Draw boxes on page to capture visuals (click two corners)
     → GPT immediately suggests question assignment
  4. Review / override assignments
  5. Sync to Airtable (images via Cloudinary)

requirements.txt:
    streamlit streamlit-image-coordinates openai requests pymupdf pillow pandas

Secrets:
    OPENAI_API_KEY          = "sk-..."
    AIRTABLE_TOKEN          = "pat..."
    AIRTABLE_BASE_ID        = "app..."
    CLOUDINARY_CLOUD_NAME   = "my-cloud"
    CLOUDINARY_UPLOAD_PRESET = "my-preset"
"""

import io
import json
import re
import base64
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import fitz
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw
from openai import OpenAI
from streamlit_image_coordinates import streamlit_image_coordinates

# ── Config ────────────────────────────────────────────────────────────────
TEXT_MODEL        = "gpt-4.1-mini"
VISION_MODEL      = "gpt-4.1"
MAX_OUTPUT_TOKENS = 8000
CHUNK_PAGES       = 2
MAX_WORKERS       = 3
MAX_RETRIES       = 4
BASE_BACKOFF      = 2
IMAGE_MAX_SIZE    = (1200, 1200)
JPEG_QUALITY      = 70
RENDER_DPI        = 150
EXTRACT_DPI       = 300
CANVAS_MAX_WIDTH  = 900

AT_API    = "https://api.airtable.com/v0"
AT_META   = "https://api.airtable.com/v0/meta"
CLOUDINARY_API = "https://api.cloudinary.com/v1_1"

AT_FIELDS = [
    ("Question Number",          "singleLineText"),
    ("Question Text",            "multilineText"),
    ("Mark Allocation",          "number"),
    ("Topic",                    "singleLineText"),
    ("Subtopic",                 "singleLineText"),
    ("Mark Scheme Answer",       "multilineText"),
    ("Image Description",        "multilineText"),
    ("Has Images",               "checkbox"),
    ("Images",                   "multipleAttachments"),
    ("Paper Name",               "singleLineText"),
    ("Exam Type",                "singleLineText"),
    ("Page Number",              "number"),
    ("Image Mapping Confidence", "singleLineText"),
    ("Image Mapping Notes",      "multilineText"),
]

SKIP_PAGE_KEYWORDS = [
    "do not write on this page",
    "additional page, if required",
    "there are no questions printed",
    "copyright information",
]

# ── Prompts ───────────────────────────────────────────────────────────────
QUESTION_PROMPT = """Extract EVERY question from this exam paper.
Return ONLY a raw JSON array. No markdown fences. No explanation.

Each element:
{
  "questionNumber": "1a",
  "questionText": "Full question including shared context",
  "markAllocation": 4,
  "topic": "Algebra",
  "subtopic": "Quadratics",
  "hasImages": false,
  "imageDescription": "Describe any diagram/graph/table. Empty string if none.",
  "pageNumber": 2
}

Rules:
- Split answerable sub-questions into separate rows.
- Attach shared context to each child question.
- markAllocation must be an integer (0 if missing).
- pageNumber = page number within this chunk only.
"""

MS_PROMPT = """Extract ALL answers from this mark scheme.
Return ONLY a raw JSON array. No markdown fences. No explanation.

Each element:
{
  "questionNumber": "1a",
  "markSchemeAnswer": "Full answer including allow/reject/key words/worked solutions"
}
"""

AI_ASSIGN_PROMPT = """You are matching a cropped exam image to the most likely question.

Return ONLY raw JSON:
{{
  "questionNumber": "2a",
  "confidence": "high",
  "notes": "Short reason"
}}

confidence: high, medium, or low.

Candidate questions (question number | page | question text):
{candidates}

Rules:
- Choose the single best match from the candidates list.
- Use visual clues in the crop: labels, axis titles, table headers, figure numbers.
- If genuinely ambiguous, pick the best candidate but set confidence to low.
- questionNumber must be exactly as shown in the candidates list.
"""

# ── Secrets ───────────────────────────────────────────────────────────────
def get_secret(key: str, fallback: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return fallback

# ── Generic helpers ───────────────────────────────────────────────────────
def clean_json(raw: str) -> str:
    raw = (raw or "").strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*",     "", raw)
    raw = re.sub(r"\s*```$",     "", raw)
    return raw.strip()

def safe_json_loads(raw: str, default: Any):
    try:
        return json.loads(clean_json(raw))
    except Exception:
        return default

def normalise_qnum(v: Any) -> str:
    return str(v or "").strip().replace(" ", "").replace("(", "").replace(")", "")

def clamp_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return default

def chunk_list(items: list, size: int) -> list:
    return [items[i:i+size] for i in range(0, len(items), size)]

def run_with_retry(fn, *args, **kwargs):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if attempt == MAX_RETRIES:
                raise
            time.sleep(BASE_BACKOFF * attempt)
    raise last_err

def oai_text(resp) -> str:
    t = getattr(resp, "output_text", None)
    if t:
        return t
    parts = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            t2 = getattr(c, "text", None)
            if t2:
                parts.append(t2)
    return "\n".join(parts)

# ── OpenAI ────────────────────────────────────────────────────────────────
def encode_pil(img: Image.Image) -> str:
    img = img.copy().convert("RGB")
    img.thumbnail(IMAGE_MAX_SIZE)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

def call_gpt(client: OpenAI, content: list, model: str,
             max_tokens: int = MAX_OUTPUT_TOKENS) -> str:
    def _call():
        return client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=max_tokens,
        )
    return oai_text(run_with_retry(_call))

# ── PDF helpers ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_question_pages(pdf_bytes: bytes) -> list[int]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i in range(len(doc)):
        pn = i + 1
        if pn == 1:
            continue
        text = doc[i].get_text().lower()
        if not any(kw in text for kw in SKIP_PAGE_KEYWORDS):
            pages.append(pn)
    doc.close()
    return pages

@st.cache_data(show_spinner=False)
def render_page_cached(pdf_bytes: bytes, page_num: int,
                        dpi: int = RENDER_DPI) -> bytes:
    """Returns PNG bytes of a rendered page. Cached so repeated renders are free."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = doc[page_num - 1].get_pixmap(matrix=mat, alpha=False)
    doc.close()
    return pix.tobytes("png")

def split_pdf(pdf_bytes: bytes, chunk_size: int) -> list[bytes]:
    doc, chunks = fitz.open(stream=pdf_bytes, filetype="pdf"), []
    for start in range(0, len(doc), chunk_size):
        w = fitz.open()
        w.insert_pdf(doc, from_page=start,
                     to_page=min(start + chunk_size, len(doc)) - 1)
        buf = io.BytesIO(); w.save(buf); w.close()
        chunks.append(buf.getvalue())
    doc.close()
    return chunks

def pdf_chunk_pils(pdf_bytes: bytes, dpi: int = RENDER_DPI) -> list[Image.Image]:
    doc, out = fitz.open(stream=pdf_bytes, filetype="pdf"), []
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
    doc.close()
    return out

def crop_from_rel(pdf_bytes: bytes, page_num: int,
                   rel: dict, dpi: int = EXTRACT_DPI) -> bytes:
    """Crop a region (relative coords 0-1) from a page at high DPI."""
    doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_num - 1]
    pr   = page.rect
    rect = fitz.Rect(
        pr.x0 + rel["x"] * pr.width,
        pr.y0 + rel["y"] * pr.height,
        pr.x0 + (rel["x"] + rel["w"]) * pr.width,
        pr.y0 + (rel["y"] + rel["h"]) * pr.height,
    )
    rect = fitz.Rect(
        max(pr.x0, rect.x0 - 4), max(pr.y0, rect.y0 - 4),
        min(pr.x1, rect.x1 + 4), min(pr.y1, rect.y1 + 4),
    )
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    doc.close()
    return pix.tobytes("png")

# ── Box store ─────────────────────────────────────────────────────────────
def boxes() -> dict[int, list[dict]]:
    return st.session_state.setdefault("boxes", {})

def page_boxes(pn: int) -> list[dict]:
    return boxes().get(pn, [])

def set_page_boxes(pn: int, b: list[dict]):
    boxes()[pn] = b

def all_boxes() -> list[dict]:
    store = boxes()
    return [b for pn in sorted(store) for b in store[pn]]

def reindex(pn: int):
    pb = page_boxes(pn)
    for i, b in enumerate(pb, 1):
        b["idx"]  = i
        b["name"] = f"p{pn}_box{i}.png"
    set_page_boxes(pn, pb)

def add_box(pdf_bytes: bytes, pn: int, rel: dict,
            qnum: str = "", notes: str = "") -> dict:
    pb     = page_boxes(pn)
    idx    = len(pb) + 1
    data   = crop_from_rel(pdf_bytes, pn, rel)
    pil    = Image.open(io.BytesIO(data))
    b = {
        "page":          pn,
        "idx":           idx,
        "name":          f"p{pn}_box{idx}.png",
        "rel":           rel,
        "data":          data,
        "width":         pil.width,
        "height":        pil.height,
        "questionNumber": qnum,
        "ai_qnum":       "",
        "ai_conf":       "",
        "ai_notes":      "",
        "notes":         notes or "manual",
    }
    pb.append(b)
    set_page_boxes(pn, pb)
    return b

# ── Extraction ────────────────────────────────────────────────────────────
def extract_questions(client: OpenAI,
                       pdf_bytes: bytes) -> tuple[list[dict], list[str]]:
    chunks = split_pdf(pdf_bytes, CHUNK_PAGES)
    logs: list[str] = []

    def process(i, chunk):
        offset  = i * CHUNK_PAGES
        pils    = pdf_chunk_pils(chunk)
        content = [{"type": "input_text", "text": QUESTION_PROMPT}]
        for p in pils:
            content.append({"type": "input_image",
                             "image_url": f"data:image/jpeg;base64,{encode_pil(p)}"})
        rows = safe_json_loads(call_gpt(client, content, TEXT_MODEL), [])
        fixed = [{
            "questionNumber":   normalise_qnum(r.get("questionNumber",   "")),
            "questionText":     str(r.get("questionText",     "") or ""),
            "markAllocation":   clamp_int(r.get("markAllocation", 0), 0),
            "topic":            str(r.get("topic",            "") or ""),
            "subtopic":         str(r.get("subtopic",         "") or ""),
            "hasImages":        bool(r.get("hasImages",       False)),
            "imageDescription": str(r.get("imageDescription", "") or ""),
            "pageNumber":       clamp_int(r.get("pageNumber", 1), 1) + offset,
        } for r in rows]
        return i, fixed, f"Chunk {i+1}/{len(chunks)}: {len(fixed)} questions"

    ordered: dict[int, list] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, rows, msg in [f.result() for f in
                              as_completed([ex.submit(process, i, c)
                                            for i, c in enumerate(chunks)])]:
            ordered[i] = rows; logs.append(msg)

    out = []
    for i in range(len(chunks)):
        out.extend(ordered.get(i, []))
    return out, logs

def extract_markscheme(client: OpenAI,
                        pdf_bytes: bytes) -> tuple[dict[str, str], list[str]]:
    chunks = split_pdf(pdf_bytes, CHUNK_PAGES)
    logs: list[str] = []

    def process(i, chunk):
        pils    = pdf_chunk_pils(chunk)
        content = [{"type": "input_text", "text": MS_PROMPT}]
        for p in pils:
            content.append({"type": "input_image",
                             "image_url": f"data:image/jpeg;base64,{encode_pil(p)}"})
        rows  = safe_json_loads(call_gpt(client, content, TEXT_MODEL), [])
        local = {normalise_qnum(r.get("questionNumber", "")): str(r.get("markSchemeAnswer", "") or "")
                 for r in rows if r.get("questionNumber") and r.get("markSchemeAnswer")}
        return i, local, f"MS chunk {i+1}/{len(chunks)}: {len(local)} entries"

    ordered: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, local, msg in [f.result() for f in
                               as_completed([ex.submit(process, i, c)
                                             for i, c in enumerate(chunks)])]:
            ordered[i] = local; logs.append(msg)

    ms: dict[str, str] = {}
    for i in range(len(chunks)):
        ms.update(ordered.get(i, {}))
    return ms, logs

# ── AI image assignment ───────────────────────────────────────────────────
def candidates_for_page(records: list[dict], pn: int) -> list[dict]:
    """Return questions on same page + adjacent pages, max 25."""
    same    = [r for r in records if clamp_int(r.get("pageNumber")) == pn]
    adj     = [r for r in records if clamp_int(r.get("pageNumber")) in {pn-1, pn+1}]
    seen    = {normalise_qnum(r.get("questionNumber", "")) for r in same}
    cands   = list(same)
    for r in adj:
        qn = normalise_qnum(r.get("questionNumber", ""))
        if qn and qn not in seen:
            cands.append(r); seen.add(qn)
    return (cands or records)[:25]

def ai_assign(client: OpenAI, box: dict,
               records: list[dict]) -> dict:
    cands  = candidates_for_page(records, box["page"])
    cblock = "\n".join(
        f"- {normalise_qnum(r.get('questionNumber',''))} | "
        f"p{clamp_int(r.get('pageNumber',0))} | "
        f"{re.sub(chr(10), ' ', str(r.get('questionText','') or ''))[:200]}"
        for r in cands
    )
    prompt = AI_ASSIGN_PROMPT.format(candidates=cblock)
    img    = Image.open(io.BytesIO(box["data"])).convert("RGB")
    content = [
        {"type": "input_text",  "text": prompt},
        {"type": "input_image",
         "image_url": f"data:image/jpeg;base64,{encode_pil(img)}"},
    ]
    parsed = safe_json_loads(call_gpt(client, content, VISION_MODEL,
                                       max_tokens=300), {})

    qn   = normalise_qnum(parsed.get("questionNumber", ""))
    conf = str(parsed.get("confidence", "") or "").strip().lower()
    note = str(parsed.get("notes", "") or "").strip()

    valid = {normalise_qnum(r.get("questionNumber", "")) for r in cands}
    if qn not in valid:
        qn = ""; conf = "low"
        note = (note + " " if note else "") + "[outside candidate set]"
    if conf not in {"high", "medium", "low"}:
        conf = "low"

    return {"questionNumber": qn, "confidence": conf, "notes": note or "—"}

# ── Drawing overlay ───────────────────────────────────────────────────────
def draw_overlay(display: Image.Image, pb: list[dict],
                  highlight_qnum: str = "") -> Image.Image:
    img  = display.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for b in pb:
        rel     = b["rel"]
        x0, y0  = rel["x"] * w, rel["y"] * h
        x1, y1  = x0 + rel["w"] * w, y0 + rel["h"] * h
        qn      = normalise_qnum(b.get("questionNumber", ""))
        color   = "#e74c3c" if (highlight_qnum and qn == highlight_qnum) else "#e67e22"
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = str(b.get("idx", ""))
        if label:
            draw.rectangle([x0+1, y0-18, x0+26, y0-2], fill=color)
            draw.text((x0+5, y0-17), label, fill="white")
    return img

# ── Airtable + Cloudinary ─────────────────────────────────────────────────
def at_headers(t):
    return {"Authorization": f"Bearer {t}", "Content-Type": "application/json"}

def ensure_table(token, base_id, table):
    try:
        r = requests.get(f"{AT_META}/bases/{base_id}/tables",
                         headers=at_headers(token), timeout=60)
        if r.status_code == 401:
            st.info("Skipping auto table creation (needs schema.bases:write).")
            return
        r.raise_for_status()
        if table in [t["name"] for t in r.json().get("tables", [])]:
            return
        fields = []
        for name, ftype in AT_FIELDS:
            if ftype == "number":
                fields.append({"name": name, "type": "number", "options": {"precision": 0}})
            elif ftype == "checkbox":
                fields.append({"name": name, "type": "checkbox",
                               "options": {"icon": "check", "color": "greenBright"}})
            elif ftype == "multipleAttachments":
                fields.append({"name": name, "type": "multipleAttachments"})
            else:
                fields.append({"name": name, "type": ftype})
        r2 = requests.post(f"{AT_META}/bases/{base_id}/tables",
                           headers=at_headers(token),
                           json={"name": table, "fields": fields}, timeout=60)
        if not r2.ok:
            st.warning(f"Could not auto-create table ({r2.status_code}).")
    except Exception as e:
        st.warning(f"Table check skipped: {e}")

def get_existing_fields(token, base_id, table) -> set[str]:
    resp = requests.get(f"{AT_META}/bases/{base_id}/tables",
                        headers=at_headers(token), timeout=60)
    resp.raise_for_status()
    for t in resp.json().get("tables", []):
        if t["name"] == table:
            return {f["name"] for f in t.get("fields", [])}
    return set()

def upload_cloudinary(cloud: str, preset: str, img: dict) -> str | None:
    pid  = img["name"].rsplit(".", 1)[0].replace(".", "_")
    resp = requests.post(
        f"{CLOUDINARY_API}/{cloud}/image/upload",
        data={"upload_preset": preset, "public_id": pid},
        files={"file": (img["name"], img["data"], "image/png")},
        timeout=120,
    )
    return resp.json().get("secure_url") if resp.ok else None

def push_airtable(token, base_id, table, records) -> list[dict]:
    ensure_table(token, base_id, table)
    url     = f"{AT_API}/{base_id}/{requests.utils.quote(table, safe='')}"
    created = []
    for batch in chunk_list(records, 10):
        resp = requests.post(url, headers=at_headers(token),
                             json={"records": batch}, timeout=60)
        if not resp.ok:
            raise RuntimeError(f"Airtable {resp.status_code}: {resp.text[:400]}")
        created.extend(resp.json().get("records", []))
    return created

# ── Streamlit UI ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Past Paper → Airtable",
                   page_icon="📄", layout="wide")
st.title("📄 Past Paper → Airtable")
st.caption("Draw boxes to capture visuals · AI suggests question assignment · Sync to Airtable")

OPENAI_KEY       = get_secret("OPENAI_API_KEY")
AT_TOKEN         = get_secret("AIRTABLE_TOKEN")
AT_BASE          = get_secret("AIRTABLE_BASE_ID")
CLD_CLOUD        = get_secret("CLOUDINARY_CLOUD_NAME")
CLD_PRESET       = get_secret("CLOUDINARY_UPLOAD_PRESET")

with st.sidebar:
    st.header("⚙️ Configuration")
    for key, label, ph in [
        ("OPENAI_API_KEY",           "OpenAI API key",           "sk-..."),
        ("AIRTABLE_TOKEN",           "Airtable token",           "patXXX"),
        ("AIRTABLE_BASE_ID",         "Airtable Base ID",         "appXXX"),
        ("CLOUDINARY_CLOUD_NAME",    "Cloudinary cloud name",    "my-cloud"),
        ("CLOUDINARY_UPLOAD_PRESET", "Cloudinary upload preset", "my-preset"),
    ]:
        val = get_secret(key)
        if val:
            st.success(f"✓ {label} loaded")
        else:
            locals()[key.lower()] = st.text_input(label, type="password", placeholder=ph)

    AT_TABLE         = st.text_input("Table name", value="Questions")
    AUTO_ASSIGN      = st.checkbox("Auto-assign high-confidence AI suggestions", value=True)
    st.divider()
    st.markdown("**Required Airtable fields**")
    for name, ftype in AT_FIELDS:
        st.markdown(f"- `{name}` — {ftype}")

# ── 1. Upload ──────────────────────────────────────────────────────────────
st.subheader("1 · Upload PDFs")
c1, c2 = st.columns(2)
with c1:
    paper_name = st.text_input("Paper name", placeholder="AQA Maths P1 2024")
    exam_type  = st.text_input("Exam type",  placeholder="GCSE / A-Level / IB HL")
    paper_file = st.file_uploader("Past paper PDF *", type="pdf")
with c2:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    ms_file = st.file_uploader("Mark scheme PDF (optional)", type="pdf")

if st.button("Load PDF", disabled=not paper_file):
    paper_file.seek(0)
    pdf = paper_file.read()
    st.session_state["pdf"]   = pdf
    st.session_state["pages"] = get_question_pages(pdf)
    st.session_state.pop("records", None)
    st.session_state["boxes"] = {}
    st.success(f"Loaded — {len(st.session_state['pages'])} question pages found.")

# ── 2. Extract ─────────────────────────────────────────────────────────────
if "pdf" in st.session_state:
    st.subheader("2 · Extract questions + mark scheme")

    if st.button("✨ Extract", type="primary",
                 disabled=not (paper_name and exam_type and OPENAI_KEY)):
        pdf      = st.session_state["pdf"]
        ms_bytes = ms_file.read() if ms_file else None
        client   = OpenAI(api_key=OPENAI_KEY)

        with st.status("Extracting…", expanded=True) as status:
            st.write("🤖 Questions…")
            questions, ql = extract_questions(client, pdf)
            for l in ql: st.write(f"   {l}")

            ms_map: dict[str, str] = {}
            if ms_bytes:
                st.write("🤖 Mark scheme…")
                ms_map, ml = extract_markscheme(client, ms_bytes)
                for l in ml: st.write(f"   {l}")

            records = []
            for q in questions:
                qn = normalise_qnum(q.get("questionNumber", ""))
                records.append({
                    "questionNumber":         qn,
                    "questionText":           q.get("questionText",     ""),
                    "markAllocation":         clamp_int(q.get("markAllocation", 0)),
                    "topic":                  q.get("topic",            ""),
                    "subtopic":               q.get("subtopic",         ""),
                    "markSchemeAnswer":        ms_map.get(qn,           ""),
                    "imageDescription":        q.get("imageDescription", ""),
                    "hasImages":               bool(q.get("hasImages",  False)),
                    "pageNumber":              clamp_int(q.get("pageNumber", 1), 1),
                    "paperName":               paper_name,
                    "examType":                exam_type,
                    "imageMappingConfidence":  "",
                    "imageMappingNotes":       "",
                    "images":                  [],
                })

            st.session_state["records"] = records
            status.update(label=f"✅ {len(records)} questions extracted", state="complete")

# ── 3. Capture ─────────────────────────────────────────────────────────────
if "pdf" in st.session_state:
    st.subheader("3 · Capture visuals")
    st.caption("Click two corners on the page to draw a box. AI suggests the question assignment automatically.")

    pdf     = st.session_state["pdf"]
    pages   = st.session_state.get("pages", [])
    records = st.session_state.get("records", [])

    if not pages:
        st.info("Load a PDF first.")
    else:
        left, right = st.columns([1, 2])

        with left:
            sel_page = st.selectbox("Page", pages,
                                    index=pages.index(
                                        st.session_state.get("sel_page", pages[0]))
                                    if st.session_state.get("sel_page") in pages else 0)
            st.session_state["sel_page"] = sel_page

            # Assignment mode
            mode = st.radio("Assignment", ["AI suggest", "Manual"],
                            horizontal=True)
            manual_qnum = ""
            if mode == "Manual" and records:
                qnums = list(dict.fromkeys(
                    normalise_qnum(r.get("questionNumber", "")) for r in records))
                manual_qnum = st.selectbox("Question", qnums)

            notes_input = st.text_input("Notes", placeholder="e.g. diagram, table")

            # Current page boxes
            pb = page_boxes(sel_page)
            st.markdown(f"**{len(pb)} box(es) on this page**")
            for b in pb:
                qn    = b.get("questionNumber", "")
                ai_qn = b.get("ai_qnum", "")
                ai_cf = b.get("ai_conf", "")
                cap   = (f"→ Q{qn}" if qn
                         else f"→ AI: Q{ai_qn} ({ai_cf})" if ai_qn
                         else "unassigned")
                st.image(b["data"],
                         caption=f"Box {b['idx']} {cap}",
                         use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Undo last", disabled=not pb):
                    pb2 = pb[:-1]
                    set_page_boxes(sel_page, pb2)
                    reindex(sel_page)
                    st.session_state[f"clicks_{sel_page}"] = []
                    st.rerun()
            with col_b:
                if st.button("Clear page", disabled=not pb):
                    set_page_boxes(sel_page, [])
                    st.session_state[f"clicks_{sel_page}"] = []
                    st.rerun()

        with right:
            page_png = render_page_cached(pdf, sel_page, dpi=RENDER_DPI)
            page_pil = Image.open(io.BytesIO(page_png)).convert("RGB")
            pw, ph   = page_pil.size
            scale    = min(1.0, CANVAS_MAX_WIDTH / pw)
            dw, dh   = int(pw * scale), int(ph * scale)
            disp_pil = page_pil.resize((dw, dh))

            pb        = page_boxes(sel_page)
            highlight = normalise_qnum(manual_qnum) if mode == "Manual" else ""
            overlay   = draw_overlay(disp_pil, pb, highlight)

            ckey = f"clicks_{sel_page}"
            if ckey not in st.session_state:
                st.session_state[ckey] = []

            click = streamlit_image_coordinates(
                overlay,
                key=f"canvas_{sel_page}_{len(pb)}_{mode}_{manual_qnum}",
            )

            if click:
                cx, cy   = int(click["x"]), int(click["y"])
                clicks   = st.session_state[ckey]
                pt       = (cx, cy)

                if not clicks or clicks[-1] != pt:
                    clicks.append(pt)
                if len(clicks) > 2:
                    clicks = clicks[-2:]
                st.session_state[ckey] = clicks

                if len(clicks) == 2:
                    (x1, y1), (x2, y2) = clicks
                    bw = abs(x2 - x1); bh = abs(y2 - y1)

                    if bw > 5 and bh > 5:
                        rel = {
                            "x": min(x1, x2) / dw,
                            "y": min(y1, y2) / dh,
                            "w": bw / dw,
                            "h": bh / dh,
                        }
                        qn_for_box = normalise_qnum(manual_qnum) if mode == "Manual" else ""
                        box = add_box(pdf, sel_page, rel, qnum=qn_for_box,
                                       notes=notes_input.strip() or "manual")

                        # AI assign
                        if mode == "AI suggest" and OPENAI_KEY and records:
                            try:
                                client = OpenAI(api_key=OPENAI_KEY)
                                result = ai_assign(client, box, records)
                                pb2    = page_boxes(sel_page)
                                for b in pb2:
                                    if b["idx"] == box["idx"]:
                                        b["ai_qnum"]  = result["questionNumber"]
                                        b["ai_conf"]  = result["confidence"]
                                        b["ai_notes"] = result["notes"]
                                        if AUTO_ASSIGN and result["questionNumber"] and result["confidence"] == "high":
                                            b["questionNumber"] = result["questionNumber"]
                                set_page_boxes(sel_page, pb2)
                                st.toast(
                                    f"AI: Q{result['questionNumber'] or 'none'} "
                                    f"({result['confidence']})")
                            except Exception as e:
                                st.warning(f"AI assignment failed: {e}")

                        st.session_state[ckey] = []
                        st.rerun()
                    else:
                        st.warning("Box too small — try again.")
                        st.session_state[ckey] = []
                else:
                    st.caption(f"First corner set at {clicks[0]}. Click the second corner.")

# ── 4. Bulk AI assign ──────────────────────────────────────────────────────
if st.session_state.get("records") and any(all_boxes()):
    st.subheader("4 · Bulk AI assign unassigned boxes")
    ab         = all_boxes()
    unassigned = [b for b in ab if not normalise_qnum(b.get("questionNumber", ""))]
    st.write(f"Unassigned: **{len(unassigned)}** of {len(ab)} boxes")

    if st.button("AI assign all unassigned", disabled=not (unassigned and OPENAI_KEY)):
        client  = OpenAI(api_key=OPENAI_KEY)
        records = st.session_state["records"]
        done = failed = 0

        with st.status("Assigning…", expanded=True) as status:
            store = boxes()
            for pn in sorted(store):
                pb = store[pn]
                for b in pb:
                    if normalise_qnum(b.get("questionNumber", "")):
                        continue
                    try:
                        r = ai_assign(client, b, records)
                        b["ai_qnum"]  = r["questionNumber"]
                        b["ai_conf"]  = r["confidence"]
                        b["ai_notes"] = r["notes"]
                        if AUTO_ASSIGN and r["questionNumber"] and r["confidence"] == "high":
                            b["questionNumber"] = r["questionNumber"]
                        done += 1
                        st.write(f"p{pn} box {b['idx']}: "
                                 f"Q{r['questionNumber'] or 'none'} ({r['confidence']})")
                    except Exception as e:
                        failed += 1
                        st.write(f"p{pn} box {b['idx']}: failed — {e}")
                set_page_boxes(pn, pb)
            status.update(label=f"✅ {done} done · {failed} failed", state="complete")

# ── 5. Review assignments ──────────────────────────────────────────────────
if any(all_boxes()):
    st.subheader("5 · Review image assignments")
    ab = all_boxes()

    rows = [{
        "Page":            b["page"],
        "Box":             b["idx"],
        "Name":            b["name"],
        "Final Q #":       b.get("questionNumber",  ""),
        "AI suggested Q":  b.get("ai_qnum",         ""),
        "AI confidence":   b.get("ai_conf",          ""),
        "AI notes":        b.get("ai_notes",         ""),
        "Notes":           b.get("notes",            ""),
    } for b in ab]

    edited = st.data_editor(pd.DataFrame(rows),
                             use_container_width=True,
                             num_rows="fixed", height=400)

    if st.button("Save assignments"):
        store = boxes()
        update_map: dict[tuple, dict] = {
            (int(row["Page"]), int(row["Box"])): row
            for _, row in edited.iterrows()
        }
        for pn in store:
            for b in store[pn]:
                key = (b["page"], b["idx"])
                if key in update_map:
                    row = update_map[key]
                    b["questionNumber"] = normalise_qnum(row["Final Q #"])
                    b["ai_qnum"]        = normalise_qnum(row["AI suggested Q"])
                    b["ai_conf"]        = str(row["AI confidence"] or "").lower()
                    b["ai_notes"]       = str(row["AI notes"]      or "")
                    b["notes"]          = str(row["Notes"]         or "manual")
            set_page_boxes(pn, store[pn])
        st.success("Assignments saved.")
        st.rerun()

# ── Merge images into records (runs silently before review) ───────────────
if "records" in st.session_state:
    records    = st.session_state["records"]
    ab         = all_boxes()
    q_imgs:  dict[str, list[str]] = {}
    q_notes: dict[str, list[str]] = {}
    for b in ab:
        qn = normalise_qnum(b.get("questionNumber", ""))
        if not qn:
            continue
        q_imgs .setdefault(qn, []).append(b["name"])
        ai_part = (f" | AI {b['ai_qnum']} ({b['ai_conf']})"
                   if b.get("ai_qnum") else "")
        q_notes.setdefault(qn, []).append(f"{b['name']}: {b.get('notes','')}{ai_part}")

    for r in records:
        qn = normalise_qnum(r.get("questionNumber", ""))
        imgs  = q_imgs .get(qn, [])
        notes = q_notes.get(qn, [])
        r["images"]                = imgs
        r["hasImages"]             = bool(imgs) or r.get("hasImages", False)
        r["imageMappingConfidence"] = "manual+ai" if any("AI " in n for n in notes) else ("manual" if imgs else "")
        r["imageMappingNotes"]     = "\n".join(notes)

    st.session_state["records"] = records

# ── 6. Review records ──────────────────────────────────────────────────────
if "records" in st.session_state:
    records = st.session_state["records"]
    ab      = all_boxes()

    st.subheader("6 · Review & edit")

    df = pd.DataFrame([{
        "Q #":           r["questionNumber"],
        "Question Text": r["questionText"],
        "Marks":         r["markAllocation"],
        "Topic":         r["topic"],
        "Subtopic":      r["subtopic"],
        "Mark Scheme":   r["markSchemeAnswer"],
        "Image Desc.":   r["imageDescription"],
        "Has Images":    r["hasImages"],
        "Images":        ", ".join(r.get("images", [])),
        "Conf.":         r.get("imageMappingConfidence", ""),
        "Page":          r.get("pageNumber", 1),
    } for r in records])

    edited_df = st.data_editor(df, use_container_width=True,
                                num_rows="dynamic", height=420)

    for i, row in edited_df.iterrows():
        if i < len(records):
            records[i].update({
                "questionNumber":    normalise_qnum(row["Q #"]),
                "questionText":      row["Question Text"],
                "markAllocation":    clamp_int(row["Marks"], 0),
                "topic":             row["Topic"],
                "subtopic":          row["Subtopic"],
                "markSchemeAnswer":  row["Mark Scheme"],
                "imageDescription":  row["Image Desc."],
                "hasImages":         bool(row["Has Images"]),
                "pageNumber":        clamp_int(row["Page"], 1),
                "images":            [x.strip() for x in str(row["Images"] or "").split(",") if x.strip()],
            })

    if ab:
        with st.expander(f"🖼 Captured visuals ({len(ab)})"):
            cols = st.columns(4)
            for i, b in enumerate(ab):
                with cols[i % 4]:
                    qn = b.get("questionNumber", "")
                    st.image(b["data"],
                             caption=f"{b['name']} → Q{qn or '?'}",
                             use_container_width=True)

# ── 7. Export / Sync ───────────────────────────────────────────────────────
if "records" in st.session_state:
    records = st.session_state["records"]
    ab      = all_boxes()

    st.subheader("7 · Export / Sync")
    dl_col, sync_col = st.columns([1, 2])

    with dl_col:
        st.download_button(
            "⬇ Download JSON",
            data=json.dumps(records, indent=2, ensure_ascii=False).encode(),
            file_name=f"{paper_name or 'questions'}.json",
            mime="application/json",
        )
        if ab:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                for b in ab:
                    zf.writestr(b["name"], b["data"])
            st.download_button(
                "⬇ Download visuals (.zip)",
                data=buf.getvalue(),
                file_name=f"{paper_name or 'paper'}_visuals.zip",
                mime="application/zip",
            )

    with sync_col:
        if not (AT_TOKEN and AT_BASE):
            st.warning("Add Airtable credentials to sync.")
        elif not AT_TOKEN.startswith("pat"):
            st.error("❌ Token should start with `pat`.")
        elif not AT_BASE.startswith("app"):
            st.error("❌ Base ID should start with `app`.")
        else:
            tp = AT_TOKEN[:8] + "..." + AT_TOKEN[-4:]
            st.caption(f"Token: `{tp}` | Base: `{AT_BASE}` | Table: `{AT_TABLE}`")

            if st.button("🚀 Sync to Airtable", type="primary"):
                log_lines: list[str] = []
                def log(m): log_lines.append(m)

                img_url_map: dict[str, str] = {}
                if ab and CLD_CLOUD and CLD_PRESET:
                    log(f"Uploading {len(ab)} visuals to Cloudinary…")
                    def _upload(b):
                        return b["name"], upload_cloudinary(CLD_CLOUD, CLD_PRESET, b)
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                        for name, url in [f.result() for f in
                                           as_completed([ex.submit(_upload, b)
                                                         for b in ab])]:
                            if url:
                                img_url_map[name] = url
                                log(f"  ✅ {name}")
                            else:
                                log(f"  ❌ {name} failed")
                elif ab:
                    log("⚠️ Cloudinary not configured — images will not be attached.")

                try:
                    existing = get_existing_fields(AT_TOKEN, AT_BASE, AT_TABLE)
                    payload  = []
                    for r in records:
                        urls   = [img_url_map[n] for n in r.get("images", [])
                                  if n in img_url_map]
                        fields = {
                            "Question Number":          r.get("questionNumber",         ""),
                            "Question Text":            r.get("questionText",           ""),
                            "Mark Allocation":          clamp_int(r.get("markAllocation", 0)),
                            "Topic":                    r.get("topic",                  ""),
                            "Subtopic":                 r.get("subtopic",               ""),
                            "Mark Scheme Answer":       r.get("markSchemeAnswer",       ""),
                            "Image Description":        r.get("imageDescription",       ""),
                            "Has Images":               bool(urls or r.get("hasImages", False)),
                            "Images":                   [{"url": u} for u in urls],
                            "Paper Name":               r.get("paperName",   paper_name),
                            "Exam Type":                r.get("examType",    exam_type),
                            "Page Number":              clamp_int(r.get("pageNumber", 1), 1),
                            "Image Mapping Confidence": r.get("imageMappingConfidence", ""),
                            "Image Mapping Notes":      r.get("imageMappingNotes",      ""),
                        }
                        fields = {k: v for k, v in fields.items() if k in existing}
                        payload.append({"fields": fields})

                    log(f"Pushing {len(payload)} records…")
                    created = push_airtable(AT_TOKEN, AT_BASE, AT_TABLE, payload)
                    log(f"✅ {len(created)} records synced!")
                except Exception as e:
                    log(f"❌ Sync failed: {e}")

                st.text("\n".join(log_lines))
                st.markdown(f"[Open in Airtable →](https://airtable.com/{AT_BASE})")
