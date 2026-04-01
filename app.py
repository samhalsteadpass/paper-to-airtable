"""
app.py  –  Past Paper → Airtable  (OpenAI version, Streamlit Cloud)
====================================================================
Secrets (.streamlit/secrets.toml or Streamlit Cloud dashboard):
    OPENAI_API_KEY   = "sk-..."
    AIRTABLE_TOKEN   = "pat..."
    AIRTABLE_BASE_ID = "app..."
    IMGBB_API_KEY    = "..."

requirements.txt:
    streamlit openai requests pymupdf pillow pandas
"""

import io
import json
import re
import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import fitz
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────
TEXT_MODEL        = "gpt-4.1-mini"
VISION_MODEL      = "gpt-4.1"
MAX_OUTPUT_TOKENS = 8000
CHUNK_PAGES       = 2
MAX_WORKERS       = 3
MAX_RETRIES       = 4
BASE_BACKOFF      = 2
MAX_IMAGES_PER_REQUEST = 2
IMAGE_MAX_SIZE    = (1200, 1200)
JPEG_QUALITY      = 70
RENDER_DPI        = 150
VISION_DPI        = 170

AT_API    = "https://api.airtable.com/v0"
AT_META   = "https://api.airtable.com/v0/meta"
IMGBB_API = "https://api.imgbb.com/1/upload"

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

QUESTION_PROMPT = """Extract EVERY question from this exam paper.
Return ONLY a raw JSON array. No markdown fences. No explanation.

Each element must be:
{
  "questionNumber": "1a",
  "questionText": "Full question including shared context and source text needed to answer it",
  "markAllocation": 4,
  "topic": "Cell Biology",
  "subtopic": "Microscopy",
  "hasImages": false,
  "imageDescription": "Describe any diagram, graph, table or chart used by this question. Empty string if none.",
  "pageNumber": 2
}

Rules:
- Split sub-questions into separate rows when they are separately answerable.
- Keep shared context attached to each relevant child question.
- markAllocation must be an integer. Use 0 if missing.
- Preserve question wording as closely as possible.
- pageNumber must be the page number within only the chunk you were shown.
"""

MS_PROMPT = """Extract ALL answers from this mark scheme.
Return ONLY a raw JSON array. No markdown fences. No explanation.

Each element must be:
{
  "questionNumber": "1a",
  "markSchemeAnswer": "Full acceptable answer, notes, working, allow/reject guidance and key words"
}
"""

VISUAL_DETECT_PROMPT = """This is page {page_num} of an exam paper.

Do two things at once:
1. Identify every visual element a student needs to answer a question.
   Include: diagrams, graphs, grids, tables, formula boxes, number lines, geometric shapes, charts.
   Exclude: answer lines, page borders, headers, footers, barcodes, page numbers, "do not write" text.

2. For each visual, identify which question number it belongs to by reading the page layout.

Return ONLY a raw JSON array. No markdown. No explanation.
Each element:
{{
  "label": "centimetre grid",
  "questionNumber": "3a",
  "confidence": "high",
  "notes": "Grid appears directly under Q3a",
  "x": 0.25,
  "y": 0.10,
  "w": 0.30,
  "h": 0.25
}}

x, y = top-left corner as fraction of page width/height (0.0 to 1.0)
w, h = width and height as fraction of page width/height
confidence: high, medium, or low
If no question can be assigned use questionNumber = "none"
If there are no relevant visuals on this page return an empty array: []
"""

# Pages to skip
SKIP_PAGE_KEYWORDS = [
    "do not write on this page",
    "additional page, if required",
    "there are no questions printed",
    "copyright information",
]

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

def normalise_qnum(value: Any) -> str:
    value = str(value or "").strip()
    return value.replace(" ", "").replace("(", "").replace(")", "")

def clamp_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
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

def openai_response_text(resp) -> str:
    text = getattr(resp, "output_text", None)
    if text:
        return text
    parts = []
    for item in getattr(resp, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            t = getattr(content, "text", None)
            if t:
                parts.append(t)
    return "\n".join(parts)

# ── OpenAI helpers ────────────────────────────────────────────────────────
def encode_image(img_pil: Image.Image) -> str:
    img_pil = img_pil.copy().convert("RGB")
    img_pil.thumbnail(IMAGE_MAX_SIZE)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

def call_gpt_vision(client: OpenAI, images: list[Image.Image], prompt: str,
                    model: str, max_images: int = MAX_IMAGES_PER_REQUEST,
                    max_output_tokens: int = MAX_OUTPUT_TOKENS) -> str:
    images  = images[:max_images]
    content = [{"type": "input_text", "text": prompt}]
    for img in images:
        content.append({
            "type":      "input_image",
            "image_url": f"data:image/jpeg;base64,{encode_image(img)}",
        })
    def _call():
        return client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=max_output_tokens,
        )
    return openai_response_text(run_with_retry(_call))

# ── PDF helpers ───────────────────────────────────────────────────────────
def render_page_as_pil(pdf_bytes: bytes, page_num: int, dpi: int = RENDER_DPI) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = doc[page_num - 1].get_pixmap(matrix=mat, alpha=False)
    doc.close()
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

def split_pdf(pdf_bytes: bytes, chunk_size: int) -> list[bytes]:
    doc, chunks = fitz.open(stream=pdf_bytes, filetype="pdf"), []
    for start in range(0, len(doc), chunk_size):
        w = fitz.open()
        w.insert_pdf(doc, from_page=start, to_page=min(start + chunk_size, len(doc)) - 1)
        buf = io.BytesIO()
        w.save(buf); w.close()
        chunks.append(buf.getvalue())
    doc.close()
    return chunks

def pdf_chunk_to_pil_images(pdf_bytes: bytes, dpi: int = RENDER_DPI) -> list[Image.Image]:
    doc, out = fitz.open(stream=pdf_bytes, filetype="pdf"), []
    mat = fitz.Matrix(dpi/72, dpi/72)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
    doc.close()
    return out

def is_question_page(pdf_bytes: bytes, page_num: int) -> bool:
    """Return False for cover, blank, and trailing answer/copyright pages."""
    if page_num == 1:
        return False
    doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = doc[page_num - 1].get_text().lower()
    doc.close()
    for kw in SKIP_PAGE_KEYWORDS:
        if kw in text:
            return False
    return True

# ── GPT-Vision visual extraction ──────────────────────────────────────────
def extract_visuals_from_page(client: OpenAI, pdf_bytes: bytes,
                               page_num: int) -> list[dict]:
    """Ask GPT-4V to locate all question-relevant visuals on a page, then crop them."""
    page_pil       = render_page_as_pil(pdf_bytes, page_num, dpi=VISION_DPI)
    page_w, page_h = page_pil.size

    content = [
        {"type": "input_text",
         "text": VISUAL_DETECT_PROMPT.format(page_num=page_num)},
        {"type": "input_image",
         "image_url": f"data:image/jpeg;base64,{encode_image(page_pil)}"},
    ]
    def _call():
        return client.responses.create(
            model=VISION_MODEL,
            input=[{"role": "user", "content": content}],
            max_output_tokens=2000,
        )
    raw  = openai_response_text(run_with_retry(_call))
    rows = safe_json_loads(raw, [])

    visuals = []
    for idx, row in enumerate(rows, 1):
        try:
            x = float(row["x"]); y = float(row["y"])
            w = float(row["w"]); h = float(row["h"])

            # Add small padding, clamp to page
            pad = 0.01
            x1  = max(0.0, x - pad);     y1 = max(0.0, y - pad)
            x2  = min(1.0, x + w + pad); y2 = min(1.0, y + h + pad)

            # Skip full-page or tiny boxes
            if (x2 - x1) > 0.95 and (y2 - y1) > 0.85:
                continue
            if (x2 - x1) < 0.03 or (y2 - y1) < 0.03:
                continue

            crop = page_pil.crop((
                int(x1 * page_w), int(y1 * page_h),
                int(x2 * page_w), int(y2 * page_h),
            ))
            buf = io.BytesIO()
            crop.save(buf, format="PNG")

            visuals.append({
                "page":   page_num,
                "name":   f"p{page_num}_v{idx}.png",
                "kind":   str(row.get("label", f"visual{idx}")),
                "data":   buf.getvalue(),
                "width":  crop.width,
                "height": crop.height,
            })
        except Exception:
            continue
    return visuals

@st.cache_data(show_spinner=False)
def extract_visual_regions(client_key: str, pdf_bytes: bytes) -> list[dict]:
    """Run GPT-4V visual extraction on all question pages in parallel."""
    client = OpenAI(api_key=client_key)
    doc    = fitz.open(stream=pdf_bytes, filetype="pdf")
    total  = len(doc)
    doc.close()

    question_pages = [p for p in range(1, total + 1)
                      if is_question_page(pdf_bytes, p)]

    results: dict[int, list] = {}

    def process(page_num):
        try:
            return page_num, extract_visuals_from_page(client, pdf_bytes, page_num)
        except Exception:
            return page_num, []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for pn, visuals in [f.result() for f in
                            as_completed([ex.submit(process, p)
                                          for p in question_pages])]:
            results[pn] = visuals

    all_visuals = []
    for p in sorted(results):
        all_visuals.extend(results[p])
    return all_visuals

# ── Parallel question extraction ──────────────────────────────────────────
def extract_questions_parallel(client: OpenAI, pdf_bytes: bytes) -> tuple[list[dict], list[str]]:
    pdf_chunks = split_pdf(pdf_bytes, CHUNK_PAGES)
    logs: list[str] = []

    def process_one(index: int, chunk_bytes: bytes):
        offset = index * CHUNK_PAGES
        images = pdf_chunk_to_pil_images(chunk_bytes)
        raw    = call_gpt_vision(client, images, QUESTION_PROMPT, model=TEXT_MODEL)
        rows   = safe_json_loads(raw, [])
        fixed  = []
        for row in rows:
            fixed.append({
                "questionNumber":   normalise_qnum(row.get("questionNumber",   "")),
                "questionText":     str(row.get("questionText",     "") or ""),
                "markAllocation":   clamp_int(row.get("markAllocation", 0), 0),
                "topic":            str(row.get("topic",            "") or ""),
                "subtopic":         str(row.get("subtopic",         "") or ""),
                "hasImages":        bool(row.get("hasImages",       False)),
                "imageDescription": str(row.get("imageDescription", "") or ""),
                "pageNumber":       clamp_int(row.get("pageNumber", 1), 1) + offset,
            })
        return index, fixed, f"Chunk {index+1}/{len(pdf_chunks)}: {len(fixed)} questions"

    ordered: dict[int, list] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_one, i, c) for i, c in enumerate(pdf_chunks)]
        for f in as_completed(futures):
            i, rows, msg = f.result()
            ordered[i] = rows; logs.append(msg)

    collected = []
    for i in range(len(pdf_chunks)):
        collected.extend(ordered.get(i, []))
    return collected, logs

def extract_markscheme_parallel(client: OpenAI, pdf_bytes: bytes) -> tuple[dict[str, str], list[str]]:
    pdf_chunks = split_pdf(pdf_bytes, CHUNK_PAGES)
    logs: list[str] = []

    def process_one(index: int, chunk_bytes: bytes):
        images = pdf_chunk_to_pil_images(chunk_bytes)
        raw    = call_gpt_vision(client, images, MS_PROMPT, model=TEXT_MODEL)
        rows   = safe_json_loads(raw, [])
        local  = {}
        for row in rows:
            qn  = normalise_qnum(row.get("questionNumber",   ""))
            ans = str(row.get("markSchemeAnswer", "") or "")
            if qn and ans:
                local[qn] = ans
        return index, local, f"MS chunk {index+1}/{len(pdf_chunks)}: {len(local)} entries"

    ordered: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_one, i, c) for i, c in enumerate(pdf_chunks)]
        for f in as_completed(futures):
            i, local, msg = f.result()
            ordered[i] = local; logs.append(msg)

    ms_map: dict[str, str] = {}
    for i in range(len(pdf_chunks)):
        ms_map.update(ordered.get(i, {}))
    return ms_map, logs

# ── Image → question mapping ──────────────────────────────────────────────
def map_images_for_page(client: OpenAI, pdf_bytes: bytes,
                        page_num: int, page_imgs: list[dict]) -> dict[str, dict]:
    page_pil = render_page_as_pil(pdf_bytes, page_num, dpi=VISION_DPI)
    img_list = "\n".join(
        f"- {img['name']} ({img['width']}x{img['height']}, kind={img.get('kind','?')})"
        for img in page_imgs
    )
    prompt = f"""This is page {page_num} of an exam paper.

The following visuals were extracted from this page:
{img_list}

For each visual, identify which question number it belongs to by reading the page layout.
Return ONLY a JSON array:
[{{"imageName": "p{page_num}_v1.png", "questionNumber": "3b", "confidence": "high", "notes": "Table is beside Q3b"}}]

confidence: high, medium, or low.
If no question can be assigned use questionNumber = "none".
"""
    raw  = call_gpt_vision(client, [page_pil], prompt, model=VISION_MODEL,
                           max_images=1, max_output_tokens=2500)
    rows = safe_json_loads(raw, [])
    result: dict[str, dict] = {}
    for row in rows:
        name = str(row.get("imageName",     "") or "").strip()
        qnum = normalise_qnum(row.get("questionNumber", "none"))
        conf = str(row.get("confidence",    "low") or "low").strip().lower()
        note = str(row.get("notes",         "")   or "").strip()
        if name:
            result[name] = {
                "questionNumber": qnum or "none",
                "confidence":     conf if conf in {"high","medium","low"} else "low",
                "notes":          note,
                "source":         "vision",
            }
    return result

def build_page_q_index(records: list[dict]) -> dict[int, list[str]]:
    index: dict[int, list[str]] = {}
    for r in records:
        page = clamp_int(r.get("pageNumber", 0))
        qn   = normalise_qnum(r.get("questionNumber", ""))
        if page and qn:
            index.setdefault(page, [])
            if qn not in index[page]:
                index[page].append(qn)
    return index

@st.cache_data(show_spinner=False)
def map_images_to_questions(pdf_bytes: bytes, openai_key: str,
                             images: list[dict], records: list[dict]) -> dict[str, dict]:
    client      = OpenAI(api_key=openai_key)
    page_groups: dict[int, list] = {}
    for img in images:
        page_groups.setdefault(img["page"], []).append(img)

    mapping: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(map_images_for_page, client, pdf_bytes, pn, imgs)
                   for pn, imgs in page_groups.items()]
        for f in as_completed(futures):
            mapping.update(f.result())

    # Fallback: assign unmapped images to last question on their page
    pq_index = build_page_q_index(records)
    for img in images:
        name = img["name"]
        if name not in mapping or mapping[name].get("questionNumber") in {"", "none"}:
            qs = pq_index.get(img["page"], [])
            mapping[name] = {
                "questionNumber": qs[-1] if qs else "none",
                "confidence":     "low",
                "notes":          "Fallback: last question on page",
                "source":         "fallback",
            }
    return mapping

# ── Airtable helpers ──────────────────────────────────────────────────────
def at_headers(token: str):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def ensure_table(token: str, base_id: str, table: str):
    try:
        r = requests.get(f"{AT_META}/bases/{base_id}/tables",
                         headers=at_headers(token), timeout=60)
        if r.status_code == 401:
            st.info("Skipping auto table creation (token lacks schema.bases:write).")
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
            st.warning(f"Could not auto-create table ({r2.status_code}). Create it manually.")
    except Exception as e:
        st.warning(f"Table check skipped: {e}")

def get_existing_fields(token: str, base_id: str, table: str) -> set[str]:
    resp = requests.get(f"{AT_META}/bases/{base_id}/tables",
                        headers=at_headers(token), timeout=60)
    resp.raise_for_status()
    for t in resp.json().get("tables", []):
        if t["name"] == table:
            return {f["name"] for f in t.get("fields", [])}
    return set()

def upload_to_imgbb(api_key: str, img: dict) -> str | None:
    b64  = base64.standard_b64encode(img["data"]).decode()
    resp = requests.post(IMGBB_API,
                         data={"key": api_key, "name": img["name"], "image": b64},
                         timeout=60)
    if resp.ok:
        return resp.json()["data"]["url"]
    return None

def create_airtable_records(token: str, base_id: str, table: str,
                             records: list[dict]) -> list[dict]:
    ensure_table(token, base_id, table)
    url     = f"{AT_API}/{base_id}/{requests.utils.quote(table, safe='')}"
    created = []
    for batch in chunk_list(records, 10):
        resp = requests.post(url, headers=at_headers(token),
                             json={"records": batch}, timeout=60)
        if not resp.ok:
            raise RuntimeError(f"Airtable {resp.status_code}: {resp.text[:500]}")
        created.extend(resp.json().get("records", []))
    return created

# ── Streamlit UI ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Past Paper → Airtable", page_icon="📄", layout="wide")
st.title("📄 Past Paper → Airtable")
st.caption("Upload exam PDFs, extract questions with GPT, map visuals, review, then sync to Airtable.")

OPENAI_KEY = get_secret("OPENAI_API_KEY")
AT_TOKEN   = get_secret("AIRTABLE_TOKEN")
AT_BASE    = get_secret("AIRTABLE_BASE_ID")
IMGBB_KEY  = get_secret("IMGBB_API_KEY")

with st.sidebar:
    st.header("⚙️ Configuration")
    if not OPENAI_KEY:
        OPENAI_KEY = st.text_input("OpenAI API key", type="password", placeholder="sk-...")
    else:
        st.success("✓ OpenAI key loaded")
    if not AT_TOKEN:
        AT_TOKEN = st.text_input("Airtable Token", type="password", placeholder="patXXXXXX")
    else:
        st.success("✓ Airtable token loaded")
    if not AT_BASE:
        AT_BASE = st.text_input("Airtable Base ID", placeholder="appXXXXXX")
    else:
        st.success("✓ Airtable Base ID loaded")
    if not IMGBB_KEY:
        IMGBB_KEY = st.text_input("imgbb API key", type="password", placeholder="imgbb.com/api")
    else:
        st.success("✓ imgbb key loaded")
    AT_TABLE = st.text_input("Table name", value="Questions")
    st.divider()
    st.markdown("**Required Airtable fields**")
    for name, ftype in AT_FIELDS:
        st.markdown(f"- `{name}` — {ftype}")

# ── Step 1: Upload ────────────────────────────────────────────────────────
st.subheader("1 · Upload PDFs")
col1, col2 = st.columns(2)
with col1:
    paper_name = st.text_input("Paper name", placeholder="AQA Biology P1 2023")
    exam_type  = st.text_input("Exam type",  placeholder="GCSE / A-Level / IB HL")
    paper_file = st.file_uploader("Past paper PDF *", type="pdf")
with col2:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    ms_file = st.file_uploader("Mark scheme PDF (optional)", type="pdf")

# ── Step 2: Extract ───────────────────────────────────────────────────────
st.subheader("2 · Extract with GPT")
if st.button("✨ Extract Questions", type="primary",
             disabled=not (paper_file and paper_name and exam_type and OPENAI_KEY)):
    paper_bytes = paper_file.read()
    ms_bytes    = ms_file.read() if ms_file else None
    client      = OpenAI(api_key=OPENAI_KEY)

    with st.status("Extracting…", expanded=True) as status:
        st.write("📎 Extracting visuals with GPT-4V…")
        extract_visual_regions.clear()
        images, image_map = extract_visual_regions(OPENAI_KEY, paper_bytes)
        mapped = sum(1 for v in image_map.values()
                     if v.get("questionNumber") not in {"none", ""})
        st.write(f"   Found {len(images)} visuals across "
                 f"{len(set(i['page'] for i in images))} pages "
                 f"({mapped} mapped to questions)")

        st.write("🤖 Extracting questions (parallel)…")
        questions, q_logs = extract_questions_parallel(client, paper_bytes)
        for line in q_logs:
            st.write(f"   {line}")

        ms_map: dict[str, str] = {}
        if ms_bytes:
            st.write("🧠 Extracting mark scheme (parallel)…")
            ms_map, ms_logs = extract_markscheme_parallel(client, ms_bytes)
            for line in ms_logs:
                st.write(f"   {line}")

        records = []
        for q in questions:
            qnum = normalise_qnum(q.get("questionNumber", ""))
            records.append({
                "questionNumber":          qnum,
                "questionText":            q.get("questionText",     ""),
                "markAllocation":          clamp_int(q.get("markAllocation", 0)),
                "topic":                   q.get("topic",            ""),
                "subtopic":                q.get("subtopic",         ""),
                "markSchemeAnswer":        ms_map.get(qnum,          ""),
                "imageDescription":        q.get("imageDescription", ""),
                "hasImages":               bool(q.get("hasImages",   False)),
                "pageNumber":              clamp_int(q.get("pageNumber", 1), 1),
                "paperName":               paper_name,
                "examType":                exam_type,
                "imageMappingConfidence":  "",
                "imageMappingNotes":       "",
                "images":                  [],
            })

        if images:
            # Apply fallback mapping for any visuals GPT didn't assign
            pq_index: dict[int, list[str]] = {}
            for r in records:
                page = clamp_int(r.get("pageNumber", 0))
                qn   = normalise_qnum(r.get("questionNumber", ""))
                if page and qn:
                    pq_index.setdefault(page, [])
                    if qn not in pq_index[page]:
                        pq_index[page].append(qn)

            for img in images:
                name = img["name"]
                if name not in image_map or image_map[name].get("questionNumber") in {"", "none"}:
                    qs = pq_index.get(img["page"], [])
                    image_map[name] = {
                        "questionNumber": qs[-1] if qs else "none",
                        "confidence":     "low",
                        "notes":          "Fallback: last question on page",
                        "source":         "fallback",
                    }

            q_to_imgs  : dict[str, list[str]] = {}
            q_to_conf  : dict[str, list[str]] = {}
            q_to_notes : dict[str, list[str]] = {}
            for name, meta in image_map.items():
                qn = normalise_qnum(meta.get("questionNumber", "none"))
                if qn and qn != "none":
                    q_to_imgs .setdefault(qn, []).append(name)
                    q_to_conf .setdefault(qn, []).append(meta.get("confidence", "low"))
                    q_to_notes.setdefault(qn, []).append(
                        f"{name}: {meta.get('notes','')} [{meta.get('source','vision')}]")

            for r in records:
                qn    = r["questionNumber"]
                imgs  = q_to_imgs .get(qn, [])
                confs = q_to_conf .get(qn, [])
                notes = q_to_notes.get(qn, [])
                if imgs:
                    r["hasImages"]              = True
                    r["images"]                 = imgs
                    r["imageMappingConfidence"] = ("high" if "high" in confs
                                                   else "medium" if "medium" in confs
                                                   else "low")
                    r["imageMappingNotes"]      = "\n".join(notes)
        else:
            image_map = {}

        st.session_state["records"]   = records
        st.session_state["images"]    = images
        st.session_state["image_map"] = image_map
        status.update(label=f"✅ Done — {len(records)} questions extracted", state="complete")

# ── Step 3: Review ────────────────────────────────────────────────────────
if "records" in st.session_state:
    records = st.session_state["records"]
    images  = st.session_state.get("images", [])

    st.subheader("3 · Review & edit")
    st.caption("Edit any cell before syncing.")

    df = pd.DataFrame([{
        "Q #":                      r["questionNumber"],
        "Question Text":            r["questionText"],
        "Marks":                    r["markAllocation"],
        "Topic":                    r["topic"],
        "Subtopic":                 r["subtopic"],
        "Mark Scheme":              r["markSchemeAnswer"],
        "Image Desc.":              r["imageDescription"],
        "Has Images":               r["hasImages"],
        "Image Names":              ", ".join(r.get("images", [])),
        "Image Mapping Confidence": r.get("imageMappingConfidence", ""),
        "Image Mapping Notes":      r.get("imageMappingNotes",      ""),
        "Page Number":              r.get("pageNumber", 1),
    } for r in records])

    edited = st.data_editor(df, use_container_width=True, num_rows="dynamic", height=460)

    col_map = {
        "Q #":                      "questionNumber",
        "Question Text":            "questionText",
        "Marks":                    "markAllocation",
        "Topic":                    "topic",
        "Subtopic":                 "subtopic",
        "Mark Scheme":              "markSchemeAnswer",
        "Image Desc.":              "imageDescription",
        "Has Images":               "hasImages",
        "Image Mapping Confidence": "imageMappingConfidence",
        "Image Mapping Notes":      "imageMappingNotes",
        "Page Number":              "pageNumber",
    }
    for i, row in edited.iterrows():
        if i < len(records):
            for col, key in col_map.items():
                records[i][key] = row[col]
            records[i]["questionNumber"] = normalise_qnum(row["Q #"])
            records[i]["markAllocation"] = clamp_int(row["Marks"], 0)
            records[i]["pageNumber"]     = clamp_int(row["Page Number"], 1)
            raw_names = str(row.get("Image Names") or "")
            records[i]["images"] = [x.strip() for x in raw_names.split(",") if x.strip()]

    if images:
        with st.expander(f"🖼 Extracted visuals ({len(images)})"):
            cols = st.columns(4)
            for idx, img in enumerate(images):
                with cols[idx % 4]:
                    st.image(img["data"],
                             caption=f"{img['name']} — {img.get('kind','?')}",
                             use_container_width=True)

    # ── Step 4: Export / Sync ─────────────────────────────────────────────
    st.subheader("4 · Export / Sync")
    dl_col, sync_col = st.columns([1, 2])

    with dl_col:
        st.download_button(
            "⬇ Download JSON",
            data=json.dumps(records, indent=2, ensure_ascii=False).encode(),
            file_name=f"{paper_name or 'questions'}.json",
            mime="application/json",
        )
        if images:
            import zipfile
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                for img in images:
                    zf.writestr(img["name"], img["data"])
            st.download_button(
                "⬇ Download Visuals (.zip)",
                data=buf.getvalue(),
                file_name=f"{paper_name or 'visuals'}_images.zip",
                mime="application/zip",
            )

    with sync_col:
        if not (AT_TOKEN and AT_BASE):
            st.warning("Add your Airtable token and Base ID in the sidebar to sync.")
        elif not AT_TOKEN.startswith("pat"):
            st.error("❌ Token should start with `pat`.")
        elif not AT_BASE.startswith("app"):
            st.error("❌ Base ID should start with `app`.")
        else:
            token_preview = AT_TOKEN[:8] + "..." + AT_TOKEN[-4:]
            st.caption(f"Token: `{token_preview}` | Base: `{AT_BASE}` | Table: `{AT_TABLE}`")

            if st.button("🚀 Sync to Airtable", type="primary"):
                _records = st.session_state.get("records", [])
                _images  = st.session_state.get("images",  [])
                _imgbb   = get_secret("IMGBB_API_KEY")
                log_lines: list[str] = []
                def log(msg): log_lines.append(msg)

                # Upload images to imgbb in parallel
                img_url_map: dict[str, str] = {}
                if _images and _imgbb:
                    log(f"Uploading {len(_images)} visuals to imgbb…")
                    def upload_one(img):
                        return img["name"], upload_to_imgbb(_imgbb, img)
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                        for name, url in [f.result() for f in as_completed(
                                [ex.submit(upload_one, img) for img in _images])]:
                            if url:
                                img_url_map[name] = url
                                log(f"  ✅ {name}")
                            else:
                                log(f"  ❌ {name} failed")
                elif _images and not _imgbb:
                    log("⚠️ IMGBB_API_KEY missing — visuals will not be attached.")

                try:
                    existing = get_existing_fields(AT_TOKEN, AT_BASE, AT_TABLE)
                    payload  = []
                    for r in _records:
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

                    log(f"Pushing {len(payload)} records to Airtable…")
                    created = create_airtable_records(AT_TOKEN, AT_BASE, AT_TABLE, payload)
                    log(f"✅ {len(created)} records synced!")
                except Exception as e:
                    log(f"❌ Sync failed: {e}")

                st.text("\n".join(log_lines))
                st.markdown(f"[Open in Airtable →](https://airtable.com/{AT_BASE})")
