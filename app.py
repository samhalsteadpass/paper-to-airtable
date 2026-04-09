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
from PIL import Image, ImageDraw
from openai import OpenAI

try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    st_canvas = None

# ── Config ────────────────────────────────────────────────────────────────
TEXT_MODEL = "gpt-4.1-mini"
MAX_OUTPUT_TOKENS = 8000
CHUNK_PAGES = 2
MAX_WORKERS = 3
MAX_RETRIES = 4
BASE_BACKOFF = 2
IMAGE_MAX_SIZE = (1200, 1200)
JPEG_QUALITY = 70
RENDER_DPI = 150
EXTRACT_DPI = 300
CANVAS_MAX_WIDTH = 950

AT_API = "https://api.airtable.com/v0"
AT_META = "https://api.airtable.com/v0/meta"
CLOUDINARY_API = "https://api.cloudinary.com/v1_1"

AT_FIELDS = [
    ("Question Number", "singleLineText"),
    ("Question Text", "multilineText"),
    ("Mark Allocation", "number"),
    ("Topic", "singleLineText"),
    ("Subtopic", "singleLineText"),
    ("Mark Scheme Answer", "multilineText"),
    ("Image Description", "multilineText"),
    ("Has Images", "checkbox"),
    ("Images", "multipleAttachments"),
    ("Paper Name", "singleLineText"),
    ("Exam Type", "singleLineText"),
    ("Page Number", "number"),
    ("Image Mapping Confidence", "singleLineText"),
    ("Image Mapping Notes", "multilineText"),
]

SKIP_PAGE_KEYWORDS = [
    "do not write on this page",
    "additional page, if required",
    "there are no questions printed",
    "copyright information",
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
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()

def safe_json_loads(raw: str, default: Any):
    try:
        return json.loads(clean_json(raw))
    except Exception:
        return default

def normalise_qnum(value: Any) -> str:
    return str(value or "").strip().replace(" ", "").replace("(", "").replace(")", "")

def clamp_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default

def chunk_list(items: list, size: int) -> list:
    return [items[i:i + size] for i in range(0, len(items), size)]

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
        for c in getattr(item, "content", []) or []:
            t = getattr(c, "text", None)
            if t:
                parts.append(t)
    return "\n".join(parts)

# ── OpenAI helpers ────────────────────────────────────────────────────────
def encode_pil(img: Image.Image) -> str:
    img = img.copy().convert("RGB")
    img.thumbnail(IMAGE_MAX_SIZE)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

def call_gpt(client: OpenAI, content: list, model: str,
             max_output_tokens: int = MAX_OUTPUT_TOKENS) -> str:
    def _call():
        return client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=max_output_tokens,
        )
    return openai_response_text(run_with_retry(_call))

# ── PDF helpers ───────────────────────────────────────────────────────────
def render_page_pil(pdf_bytes: bytes, page_num: int, dpi: int = RENDER_DPI) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = doc[page_num - 1].get_pixmap(matrix=mat, alpha=False)
    doc.close()
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

def split_pdf(pdf_bytes: bytes, chunk_size: int) -> list[bytes]:
    doc, chunks = fitz.open(stream=pdf_bytes, filetype="pdf"), []
    for start in range(0, len(doc), chunk_size):
        w = fitz.open()
        w.insert_pdf(doc, from_page=start, to_page=min(start + chunk_size, len(doc)) - 1)
        buf = io.BytesIO()
        w.save(buf)
        w.close()
        chunks.append(buf.getvalue())
    doc.close()
    return chunks

def pdf_chunk_to_pils(pdf_bytes: bytes, dpi: int = RENDER_DPI) -> list[Image.Image]:
    doc, out = fitz.open(stream=pdf_bytes, filetype="pdf"), []
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
    doc.close()
    return out

def is_question_page(pdf_bytes: bytes, page_num: int) -> bool:
    if page_num == 1:
        return False
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = doc[page_num - 1].get_text().lower()
    doc.close()
    return not any(kw in text for kw in SKIP_PAGE_KEYWORDS)

# ── Geometry / crop helpers ───────────────────────────────────────────────
def crop_rect(page: fitz.Page, rect: fitz.Rect, dpi: int = EXTRACT_DPI) -> bytes:
    pr = page.rect
    rect = fitz.Rect(
        max(pr.x0, rect.x0),
        max(pr.y0, rect.y0),
        min(pr.x1, rect.x1),
        min(pr.y1, rect.y1),
    )
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    return pix.tobytes("png")

def rel_to_rect(rel_bbox: dict, page_rect: fitz.Rect) -> fitz.Rect:
    x0 = page_rect.x0 + rel_bbox["x"] * page_rect.width
    y0 = page_rect.y0 + rel_bbox["y"] * page_rect.height
    x1 = x0 + rel_bbox["w"] * page_rect.width
    y1 = y0 + rel_bbox["h"] * page_rect.height
    return fitz.Rect(x0, y0, x1, y1)

def crop_from_rel_bbox(pdf_bytes: bytes, page_num: int, rel_bbox: dict, dpi: int = EXTRACT_DPI) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_num - 1]
    rect = rel_to_rect(rel_bbox, page.rect)
    data = crop_rect(page, rect, dpi=dpi)
    doc.close()
    return data

def render_page_for_display(pdf_bytes: bytes, page_num: int, max_width: int = CANVAS_MAX_WIDTH, dpi: int = RENDER_DPI):
    pil = render_page_pil(pdf_bytes, page_num, dpi=dpi)
    w, h = pil.size
    display_w = min(max_width, w)
    scale = display_w / w
    display_h = int(h * scale)
    display_pil = pil.resize((display_w, display_h))
    return pil, display_pil, scale

def draw_boxes_on_preview(image: Image.Image, boxes: list[dict], selected_qnum: str = "") -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for box in boxes:
        rel = box["rel_bbox"]
        x0 = rel["x"] * img.width
        y0 = rel["y"] * img.height
        x1 = x0 + rel["w"] * img.width
        y1 = y0 + rel["h"] * img.height

        qn = normalise_qnum(box.get("questionNumber", ""))
        outline = "red" if not selected_qnum or qn == selected_qnum else "orange"
        draw.rectangle([x0, y0, x1, y1], outline=outline, width=3)

        label = str(box.get("boxIndex", ""))
        if label:
            tx0 = x0 + 2
            ty0 = max(0, y0 - 18)
            tx1 = tx0 + 28
            ty1 = ty0 + 16
            draw.rectangle([tx0, ty0, tx1, ty1], fill=outline)
            draw.text((tx0 + 4, ty0 + 1), label, fill="white")

    return img

def extract_latest_rect_from_canvas(canvas_result) -> dict | None:
    if not canvas_result or not getattr(canvas_result, "json_data", None):
        return None

    objects = canvas_result.json_data.get("objects", []) or []
    rects = [obj for obj in objects if obj.get("type") == "rect"]
    if not rects:
        return None

    obj = rects[-1]

    left = float(obj.get("left", 0))
    top = float(obj.get("top", 0))
    width = float(obj.get("width", 0)) * float(obj.get("scaleX", 1))
    height = float(obj.get("height", 0)) * float(obj.get("scaleY", 1))

    if width <= 0 or height <= 0:
        return None

    return {
        "left": left,
        "top": top,
        "width": width,
        "height": height,
    }

# ── Manual box store ──────────────────────────────────────────────────────
def get_manual_boxes() -> dict[int, list[dict]]:
    return st.session_state.setdefault("manual_boxes_by_page", {})

def get_boxes_for_page(page_num: int) -> list[dict]:
    return get_manual_boxes().get(page_num, [])

def set_boxes_for_page(page_num: int, boxes: list[dict]):
    store = get_manual_boxes()
    store[page_num] = boxes

def rebuild_page_boxes_from_rel(pdf_bytes: bytes, page_num: int, rel_boxes: list[dict], existing_boxes: list[dict] | None = None) -> list[dict]:
    existing_boxes = existing_boxes or []
    rebuilt = []

    for i, rel in enumerate(rel_boxes, 1):
        img_bytes = crop_from_rel_bbox(pdf_bytes, page_num, rel, dpi=EXTRACT_DPI)
        pil = Image.open(io.BytesIO(img_bytes))

        prev_qnum = ""
        prev_notes = "Manual box"
        if i - 1 < len(existing_boxes):
            prev_qnum = existing_boxes[i - 1].get("questionNumber", "")
            prev_notes = existing_boxes[i - 1].get("notes", "Manual box")

        rebuilt.append({
            "page": page_num,
            "name": f"p{page_num}_box{i}.png",
            "source": "manual_box",
            "rel_bbox": rel,
            "data": img_bytes,
            "width": pil.width,
            "height": pil.height,
            "questionNumber": prev_qnum,
            "kind": "",
            "confidence": "",
            "notes": prev_notes,
            "boxIndex": i,
        })
    return rebuilt

def get_all_boxes() -> list[dict]:
    store = get_manual_boxes()
    all_boxes = []
    for page in sorted(store):
        all_boxes.extend(store[page])
    return all_boxes

def save_box_for_selected_question(
    pdf_bytes: bytes,
    page_num: int,
    rel: dict,
    selected_qnum: str,
    notes: str = "Manual box",
):
    page_boxes = get_boxes_for_page(page_num)
    rel_boxes = [b["rel_bbox"] for b in page_boxes] + [rel]

    rebuilt = rebuild_page_boxes_from_rel(
        pdf_bytes=pdf_bytes,
        page_num=page_num,
        rel_boxes=rel_boxes,
        existing_boxes=page_boxes,
    )

    if rebuilt:
        rebuilt[-1]["questionNumber"] = normalise_qnum(selected_qnum)
        rebuilt[-1]["notes"] = notes.strip() or "Manual box"

    set_boxes_for_page(page_num, rebuilt)

# ── Question + mark scheme extraction ─────────────────────────────────────
def extract_questions_parallel(client: OpenAI, pdf_bytes: bytes) -> tuple[list[dict], list[str]]:
    chunks = split_pdf(pdf_bytes, CHUNK_PAGES)
    logs: list[str] = []

    def process(i: int, chunk: bytes):
        offset = i * CHUNK_PAGES
        pils = pdf_chunk_to_pils(chunk)
        content = [{"type": "input_text", "text": QUESTION_PROMPT}]
        for p in pils:
            content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{encode_pil(p)}"})
        raw = call_gpt(client, content, TEXT_MODEL)
        rows = safe_json_loads(raw, [])
        fixed = [{
            "questionNumber": normalise_qnum(r.get("questionNumber", "")),
            "questionText": str(r.get("questionText", "") or ""),
            "markAllocation": clamp_int(r.get("markAllocation", 0), 0),
            "topic": str(r.get("topic", "") or ""),
            "subtopic": str(r.get("subtopic", "") or ""),
            "hasImages": bool(r.get("hasImages", False)),
            "imageDescription": str(r.get("imageDescription", "") or ""),
            "pageNumber": clamp_int(r.get("pageNumber", 1), 1) + offset,
        } for r in rows]
        return i, fixed, f"Chunk {i+1}/{len(chunks)}: {len(fixed)} questions"

    ordered: dict[int, list] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process, i, c) for i, c in enumerate(chunks)]
        for i, rows, msg in [f.result() for f in as_completed(futures)]:
            ordered[i] = rows
            logs.append(msg)

    collected = []
    for i in range(len(chunks)):
        collected.extend(ordered.get(i, []))
    return collected, logs

def extract_markscheme_parallel(client: OpenAI, pdf_bytes: bytes) -> tuple[dict[str, str], list[str]]:
    chunks = split_pdf(pdf_bytes, CHUNK_PAGES)
    logs: list[str] = []

    def process(i: int, chunk: bytes):
        pils = pdf_chunk_to_pils(chunk)
        content = [{"type": "input_text", "text": MS_PROMPT}]
        for p in pils:
            content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{encode_pil(p)}"})
        raw = call_gpt(client, content, TEXT_MODEL)
        rows = safe_json_loads(raw, [])
        local = {
            normalise_qnum(r.get("questionNumber", "")): str(r.get("markSchemeAnswer", "") or "")
            for r in rows
            if r.get("questionNumber") and r.get("markSchemeAnswer")
        }
        return i, local, f"MS chunk {i+1}/{len(chunks)}: {len(local)} entries"

    ordered: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process, i, c) for i, c in enumerate(chunks)]
        for i, local, msg in [f.result() for f in as_completed(futures)]:
            ordered[i] = local
            logs.append(msg)

    ms_map: dict[str, str] = {}
    for i in range(len(chunks)):
        ms_map.update(ordered.get(i, {}))
    return ms_map, logs

# ── Airtable helpers ──────────────────────────────────────────────────────
def at_headers(token: str):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def ensure_table(token: str, base_id: str, table: str):
    try:
        r = requests.get(f"{AT_META}/bases/{base_id}/tables", headers=at_headers(token), timeout=60)
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
                fields.append({"name": name, "type": "checkbox", "options": {"icon": "check", "color": "greenBright"}})
            elif ftype == "multipleAttachments":
                fields.append({"name": name, "type": "multipleAttachments"})
            else:
                fields.append({"name": name, "type": ftype})
        r2 = requests.post(
            f"{AT_META}/bases/{base_id}/tables",
            headers=at_headers(token),
            json={"name": table, "fields": fields},
            timeout=60,
        )
        if not r2.ok:
            st.warning(f"Could not auto-create table ({r2.status_code}).")
    except Exception as e:
        st.warning(f"Table check skipped: {e}")

def get_existing_fields(token: str, base_id: str, table: str) -> set[str]:
    resp = requests.get(f"{AT_META}/bases/{base_id}/tables", headers=at_headers(token), timeout=60)
    resp.raise_for_status()
    for t in resp.json().get("tables", []):
        if t["name"] == table:
            return {f["name"] for f in t.get("fields", [])}
    return set()

def upload_to_cloudinary(cloud_name: str, upload_preset: str, img: dict) -> str | None:
    public_id = img["name"].rsplit(".", 1)[0].replace(".", "_")
    resp = requests.post(
        f"{CLOUDINARY_API}/{cloud_name}/image/upload",
        data={"upload_preset": upload_preset, "public_id": public_id},
        files={"file": (img["name"], img["data"], "image/png")},
        timeout=120,
    )
    if resp.ok:
        return resp.json().get("secure_url")
    return None

def create_airtable_records(token: str, base_id: str, table: str, records: list[dict]) -> list[dict]:
    ensure_table(token, base_id, table)
    url = f"{AT_API}/{base_id}/{requests.utils.quote(table, safe='')}"
    created = []
    for batch in chunk_list(records, 10):
        resp = requests.post(url, headers=at_headers(token), json={"records": batch}, timeout=60)
        if not resp.ok:
            raise RuntimeError(f"Airtable {resp.status_code}: {resp.text[:500]}")
        created.extend(resp.json().get("records", []))
    return created

# ── UI ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Past Paper → Airtable (Rectangle capture)", page_icon="📄", layout="wide")
st.title("📄 Past Paper → Airtable")
st.caption("Extract questions, then draw rectangles to attach graphs, diagrams and tables to the correct question.")

OPENAI_KEY = get_secret("OPENAI_API_KEY")
AT_TOKEN = get_secret("AIRTABLE_TOKEN")
AT_BASE = get_secret("AIRTABLE_BASE_ID")
CLOUDINARY_CLOUD = get_secret("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_PRESET = get_secret("CLOUDINARY_UPLOAD_PRESET")

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
    if not CLOUDINARY_CLOUD:
        CLOUDINARY_CLOUD = st.text_input("Cloudinary Cloud Name", placeholder="my-cloud")
    else:
        st.success("✓ Cloudinary cloud name loaded")
    if not CLOUDINARY_PRESET:
        CLOUDINARY_PRESET = st.text_input("Cloudinary Upload Preset", placeholder="my-preset")
    else:
        st.success("✓ Cloudinary upload preset loaded")

    AT_TABLE = st.text_input("Table name", value="Questions")
    USE_AI_QUESTION_EXTRACTION = st.checkbox("Use AI for question extraction", value=True)
    USE_AI_MS_EXTRACTION = st.checkbox("Use AI for mark scheme extraction", value=True)

# ── Step 1: Upload ────────────────────────────────────────────────────────
st.subheader("1 · Upload PDFs")
col1, col2 = st.columns(2)
with col1:
    paper_name = st.text_input("Paper name", placeholder="AQA Maths P1 2024")
    exam_type = st.text_input("Exam type", placeholder="GCSE / A-Level / IB HL")
    paper_file = st.file_uploader("Past paper PDF *", type="pdf")
with col2:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    ms_file = st.file_uploader("Mark scheme PDF (optional)", type="pdf")

if st.button("Load PDF", disabled=not paper_file):
    paper_file.seek(0)
    paper_bytes = paper_file.read()
    st.session_state["paper_bytes"] = paper_bytes

    doc = fitz.open(stream=paper_bytes, filetype="pdf")
    pages = [p + 1 for p in range(len(doc)) if is_question_page(paper_bytes, p + 1)]
    doc.close()

    st.session_state["question_pages"] = pages
    st.session_state["selected_page"] = pages[0] if pages else 1
    st.session_state["manual_boxes_by_page"] = {}
    st.session_state["last_saved_rect_sig"] = ""
    st.success(f"Loaded PDF with {len(pages)} question pages.")

# ── Step 2: Extract questions / mark scheme ──────────────────────────────
if "paper_bytes" in st.session_state:
    st.subheader("2 · Extract questions and mark scheme")

    if st.button("Run extraction", type="primary", disabled=not paper_name or not exam_type):
        paper_bytes = st.session_state["paper_bytes"]
        ms_bytes = ms_file.read() if ms_file else None
        client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

        all_boxes = get_all_boxes()

        with st.status("Processing…", expanded=True) as status:
            questions = []
            q_logs = []

            if USE_AI_QUESTION_EXTRACTION:
                if not OPENAI_KEY:
                    st.error("OpenAI key is required for AI question extraction.")
                    st.stop()
                st.write("🤖 Extracting questions…")
                questions, q_logs = extract_questions_parallel(client, paper_bytes)
                for line in q_logs:
                    st.write(f"   {line}")
            else:
                st.warning("Question extraction is turned off.")

            ms_map = {}
            if ms_bytes and USE_AI_MS_EXTRACTION:
                if not OPENAI_KEY:
                    st.error("OpenAI key is required for AI mark scheme extraction.")
                    st.stop()
                st.write("🧠 Extracting mark scheme…")
                ms_map, ms_logs = extract_markscheme_parallel(client, ms_bytes)
                for line in ms_logs:
                    st.write(f"   {line}")

            records = []
            if questions:
                for q in questions:
                    qnum = normalise_qnum(q.get("questionNumber", ""))
                    records.append({
                        "questionNumber": qnum,
                        "questionText": q.get("questionText", ""),
                        "markAllocation": clamp_int(q.get("markAllocation", 0)),
                        "topic": q.get("topic", ""),
                        "subtopic": q.get("subtopic", ""),
                        "markSchemeAnswer": ms_map.get(qnum, ""),
                        "imageDescription": q.get("imageDescription", ""),
                        "hasImages": bool(q.get("hasImages", False)),
                        "pageNumber": clamp_int(q.get("pageNumber", 1), 1),
                        "paperName": paper_name,
                        "examType": exam_type,
                        "imageMappingConfidence": "",
                        "imageMappingNotes": "",
                        "images": [],
                    })

                q_to_imgs = {}
                q_to_notes = {}
                for b in all_boxes:
                    qn = normalise_qnum(b.get("questionNumber", ""))
                    if qn:
                        q_to_imgs.setdefault(qn, []).append(b["name"])
                        q_to_notes.setdefault(qn, []).append(f"{b['name']}: {b.get('notes', '')}")

                for r in records:
                    qn = r["questionNumber"]
                    imgs = q_to_imgs.get(qn, [])
                    notes = q_to_notes.get(qn, [])
                    if imgs:
                        r["hasImages"] = True
                        r["images"] = imgs
                        r["imageMappingConfidence"] = "manual"
                        r["imageMappingNotes"] = "\n".join(notes)

            st.session_state["records"] = records
            st.session_state["images"] = all_boxes

            if records:
                extracted_qnums = [
                    normalise_qnum(r.get("questionNumber", ""))
                    for r in records
                    if normalise_qnum(r.get("questionNumber", ""))
                ]
                extracted_qnums = list(dict.fromkeys(extracted_qnums))
                if extracted_qnums and "selected_question_for_capture" not in st.session_state:
                    st.session_state["selected_question_for_capture"] = extracted_qnums[0]

            status.update(
                label=f"✅ Done — {len(records)} questions · {len(all_boxes)} visuals",
                state="complete"
            )

# ── Step 3: Rectangle capture ─────────────────────────────────────────────
if "paper_bytes" in st.session_state:
    st.subheader("3 · Capture images by question")
    st.caption("Select a question, draw a rectangle over the graph or diagram, then save the latest rectangle.")

    if st_canvas is None:
        st.error("Install streamlit-drawable-canvas-fix to use rectangle drawing.")
        st.stop()

    paper_bytes = st.session_state["paper_bytes"]
    pages = st.session_state.get("question_pages", [])

    extracted_qnums = []
    if "records" in st.session_state:
        extracted_qnums = [
            normalise_qnum(r.get("questionNumber", ""))
            for r in st.session_state.get("records", [])
            if normalise_qnum(r.get("questionNumber", ""))
        ]
        extracted_qnums = list(dict.fromkeys(extracted_qnums))

    left, right = st.columns([1, 2])

    with left:
        st.markdown("### Selected question")

        if extracted_qnums:
            selected_question_for_capture = st.selectbox(
                "Question",
                options=extracted_qnums,
                key="selected_question_for_capture",
            )
        else:
            selected_question_for_capture = st.text_input(
                "Question number",
                key="selected_question_for_capture_manual",
                placeholder="e.g. 2 or 2a"
            )

        selected_qnum_norm = normalise_qnum(selected_question_for_capture)

        capture_notes = st.text_input(
            "Notes for new captures",
            key="capture_notes",
            placeholder="e.g. graph, diagram, table"
        )

        if pages:
            selected_page = st.selectbox(
                "Page",
                pages,
                index=max(0, pages.index(st.session_state.get("selected_page", pages[0])))
                if st.session_state.get("selected_page") in pages else 0,
            )
            st.session_state["selected_page"] = selected_page
        else:
            selected_page = 1
            st.info("No question pages found.")

        all_boxes = get_all_boxes()
        question_boxes = [
            b for b in all_boxes
            if normalise_qnum(b.get("questionNumber", "")) == selected_qnum_norm
        ]

        st.markdown(f"**Images linked to Q{selected_qnum_norm or '?'}:** {len(question_boxes)}")

        for box in question_boxes:
            caption = f"{box['name']} (p{box['page']})"
            st.image(box["data"], caption=caption, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button(
                "Undo last for this question",
                disabled=not question_boxes,
                key="undo_last_for_question"
            ):
                store = get_manual_boxes()
                removed = False
                for page_num in sorted(store.keys(), reverse=True):
                    page_boxes = store[page_num]
                    for idx in range(len(page_boxes) - 1, -1, -1):
                        if normalise_qnum(page_boxes[idx].get("questionNumber", "")) == selected_qnum_norm:
                            del page_boxes[idx]
                            for i, b in enumerate(page_boxes, 1):
                                b["boxIndex"] = i
                                b["name"] = f"p{page_num}_box{i}.png"
                            set_boxes_for_page(page_num, page_boxes)
                            removed = True
                            break
                    if removed:
                        break
                st.session_state["last_saved_rect_sig"] = ""
                st.rerun()

        with c2:
            if st.button(
                "Clear this question",
                disabled=not question_boxes,
                key="clear_this_question"
            ):
                store = get_manual_boxes()
                for page_num, page_boxes in store.items():
                    filtered = [
                        b for b in page_boxes
                        if normalise_qnum(b.get("questionNumber", "")) != selected_qnum_norm
                    ]
                    for i, b in enumerate(filtered, 1):
                        b["boxIndex"] = i
                        b["name"] = f"p{page_num}_box{i}.png"
                    set_boxes_for_page(page_num, filtered)
                st.session_state["last_saved_rect_sig"] = ""
                st.rerun()

    with right:
        if pages:
            _, display_pil, _ = render_page_for_display(
                paper_bytes,
                selected_page,
                max_width=CANVAS_MAX_WIDTH,
                dpi=RENDER_DPI
            )
            display_w, display_h = display_pil.size

            page_boxes = get_boxes_for_page(selected_page)
            preview_img = draw_boxes_on_preview(display_pil, page_boxes, selected_qnum_norm)

            st.markdown(f"### Page {selected_page}")
            st.markdown(f"**Current target question:** `{selected_qnum_norm or 'None'}`")

            canvas_bg = preview_img.convert("RGBA")

            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.15)",
                stroke_width=2,
                stroke_color="#ff0000",
                background_color="rgba(0,0,0,0)",
                background_image=canvas_bg,
                update_streamlit=True,
                height=canvas_bg.height,
                width=canvas_bg.width,
                drawing_mode="rect",
                display_toolbar=True,
                key=f"canvas_{selected_page}_{selected_qnum_norm}",
            )

            latest_rect = extract_latest_rect_from_canvas(canvas_result)

            if latest_rect:
                st.info(
                    f"Latest rectangle: x={int(latest_rect['left'])}, y={int(latest_rect['top'])}, "
                    f"w={int(latest_rect['width'])}, h={int(latest_rect['height'])}"
                )

            c3, c4 = st.columns(2)

            with c3:
                can_save = latest_rect is not None and bool(selected_qnum_norm)
                if st.button("Save latest rectangle", disabled=not can_save, key="save_latest_rectangle"):
                    rect = latest_rect
                    width = rect["width"]
                    height = rect["height"]

                    if width > 5 and height > 5:
                        rel = {
                            "x": rect["left"] / display_w,
                            "y": rect["top"] / display_h,
                            "w": rect["width"] / display_w,
                            "h": rect["height"] / display_h,
                        }

                        rect_sig = (
                            f"{selected_page}|{selected_qnum_norm}|"
                            f"{round(rel['x'], 4)}|{round(rel['y'], 4)}|"
                            f"{round(rel['w'], 4)}|{round(rel['h'], 4)}"
                        )

                        if rect_sig != st.session_state.get("last_saved_rect_sig", ""):
                            save_box_for_selected_question(
                                pdf_bytes=paper_bytes,
                                page_num=selected_page,
                                rel=rel,
                                selected_qnum=selected_qnum_norm,
                                notes=capture_notes or "Manual box",
                            )
                            st.session_state["last_saved_rect_sig"] = rect_sig
                            st.success("Rectangle saved to selected question.")
                            st.rerun()
                        else:
                            st.warning("That rectangle was already saved.")
                    else:
                        st.warning("Rectangle is too small.")

            with c4:
                st.write(f"Canvas size: {display_w} × {display_h}")

            st.caption("Draw one rectangle at a time. After saving, the page refreshes and your crop appears in the question list on the left.")

# ── Step 4: Review image assignments ──────────────────────────────────────
if "paper_bytes" in st.session_state:
    st.subheader("4 · Review image assignments")
    st.caption("Review or edit image-to-question assignments.")

    all_boxes = get_all_boxes()
    st.write(f"Current saved visual crops: **{len(all_boxes)}**")

    if all_boxes:
        rows = []
        for b in all_boxes:
            rows.append({
                "Page": b["page"],
                "Box #": b["boxIndex"],
                "Image Name": b["name"],
                "Question Number": b.get("questionNumber", ""),
                "Notes": b.get("notes", ""),
            })

        map_df = pd.DataFrame(rows)
        edited_map_df = st.data_editor(map_df, use_container_width=True, num_rows="fixed", height=360)

        if st.button("Save image assignments"):
            by_page = {}
            for _, row in edited_map_df.iterrows():
                page = int(row["Page"])
                box_index = int(row["Box #"])
                qnum = normalise_qnum(row["Question Number"])
                notes = str(row["Notes"] or "")

                by_page.setdefault(page, {})
                by_page[page][box_index] = {"questionNumber": qnum, "notes": notes}

            store = get_manual_boxes()
            for page, page_boxes in store.items():
                for box in page_boxes:
                    idx = box["boxIndex"]
                    if page in by_page and idx in by_page[page]:
                        box["questionNumber"] = by_page[page][idx]["questionNumber"]
                        box["notes"] = by_page[page][idx]["notes"] or "Manual box"

            st.success("Assignments saved.")
            st.rerun()

        with st.expander("Preview all crops"):
            by_page_preview = {}
            for b in all_boxes:
                by_page_preview.setdefault(b["page"], []).append(b)
            for pg in sorted(by_page_preview):
                st.markdown(f"**Page {pg}**")
                cols = st.columns(4)
                for i, box in enumerate(by_page_preview[pg]):
                    with cols[i % 4]:
                        qnum = box.get("questionNumber", "")
                        caption = f"{box['name']}" if not qnum else f"{box['name']} → Q{qnum}"
                        st.image(box["data"], caption=caption, use_container_width=True)

# ── Step 5: Merge images into extracted records ───────────────────────────
if "records" in st.session_state:
    records = st.session_state.get("records", [])
    all_boxes = get_all_boxes()

    q_to_imgs = {}
    q_to_notes = {}
    for b in all_boxes:
        qn = normalise_qnum(b.get("questionNumber", ""))
        if qn:
            q_to_imgs.setdefault(qn, []).append(b["name"])
            q_to_notes.setdefault(qn, []).append(f"{b['name']}: {b.get('notes', '')}")

    for r in records:
        qn = normalise_qnum(r.get("questionNumber", ""))
        imgs = q_to_imgs.get(qn, [])
        notes = q_to_notes.get(qn, [])
        r["images"] = imgs
        if imgs:
            r["hasImages"] = True
            r["imageMappingConfidence"] = "manual"
            r["imageMappingNotes"] = "\n".join(notes)
        else:
            r["images"] = []
            r["imageMappingConfidence"] = ""
            r["imageMappingNotes"] = ""

    st.session_state["records"] = records
    st.session_state["images"] = all_boxes

# ── Step 6: Review extracted records ──────────────────────────────────────
if "records" in st.session_state or "images" in st.session_state:
    records = st.session_state.get("records", [])
    images = st.session_state.get("images", [])

    st.subheader("6 · Review & edit")

    if records:
        df = pd.DataFrame([{
            "Q #": r["questionNumber"],
            "Question Text": r["questionText"],
            "Marks": r["markAllocation"],
            "Topic": r["topic"],
            "Subtopic": r["subtopic"],
            "Mark Scheme": r["markSchemeAnswer"],
            "Image Desc.": r["imageDescription"],
            "Has Images": r["hasImages"],
            "Image Names": ", ".join(r.get("images", [])),
            "Image Mapping Confidence": r.get("imageMappingConfidence", ""),
            "Image Mapping Notes": r.get("imageMappingNotes", ""),
            "Page Number": r.get("pageNumber", 1),
        } for r in records])

        edited = st.data_editor(df, use_container_width=True, num_rows="dynamic", height=460)

        col_map = {
            "Q #": "questionNumber",
            "Question Text": "questionText",
            "Marks": "markAllocation",
            "Topic": "topic",
            "Subtopic": "subtopic",
            "Mark Scheme": "markSchemeAnswer",
            "Image Desc.": "imageDescription",
            "Has Images": "hasImages",
            "Image Mapping Confidence": "imageMappingConfidence",
            "Image Mapping Notes": "imageMappingNotes",
            "Page Number": "pageNumber",
        }
        for i, row in edited.iterrows():
            if i < len(records):
                for col, key in col_map.items():
                    records[i][key] = row[col]
                records[i]["questionNumber"] = normalise_qnum(row["Q #"])
                records[i]["markAllocation"] = clamp_int(row["Marks"], 0)
                records[i]["pageNumber"] = clamp_int(row["Page Number"], 1)
                raw_names = str(row.get("Image Names") or "")
                records[i]["images"] = [x.strip() for x in raw_names.split(",") if x.strip()]
    else:
        st.info("No question records created yet. You can still export the images.")

    if images:
        with st.expander(f"🖼 Visuals ({len(images)})"):
            cols = st.columns(4)
            for idx, img in enumerate(images):
                with cols[idx % 4]:
                    qn = img.get("questionNumber", "")
                    cap = img["name"] if not qn else f"{img['name']} → Q{qn}"
                    st.image(img["data"], caption=cap, use_container_width=True)

# ── Step 7: Export / Sync ────────────────────────────────────────────────
if "images" in st.session_state or "records" in st.session_state:
    records = st.session_state.get("records", [])
    images = st.session_state.get("images", [])

    st.subheader("7 · Export / Sync")
    dl_col, sync_col = st.columns([1, 2])

    with dl_col:
        st.download_button(
            "⬇ Download records JSON",
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
                "⬇ Download visual crops (.zip)",
                data=buf.getvalue(),
                file_name=f"{paper_name or 'visuals'}_images.zip",
                mime="application/zip",
            )

    with sync_col:
        if not (AT_TOKEN and AT_BASE):
            st.warning("Add your Airtable token and Base ID to sync.")
        elif not AT_TOKEN.startswith("pat"):
            st.error("❌ Token should start with `pat`.")
        elif not AT_BASE.startswith("app"):
            st.error("❌ Base ID should start with `app`.")
        else:
            token_preview = AT_TOKEN[:8] + "..." + AT_TOKEN[-4:]
            st.caption(f"Token: `{token_preview}` | Base: `{AT_BASE}` | Table: `{AT_TABLE}`")

            if st.button("🚀 Sync to Airtable", type="primary"):
                log_lines = []

                def log(msg):
                    log_lines.append(msg)

                img_url_map = {}
                if images and CLOUDINARY_CLOUD and CLOUDINARY_PRESET:
                    log(f"Uploading {len(images)} images to Cloudinary…")
                    for img in images:
                        url = upload_to_cloudinary(CLOUDINARY_CLOUD, CLOUDINARY_PRESET, img)
                        if url:
                            img_url_map[img["name"]] = url
                            log(f"  ✅ {img['name']}")
                        else:
                            log(f"  ❌ {img['name']} failed")
                elif images:
                    log("⚠️ Cloudinary not configured — images skipped.")

                try:
                    existing = get_existing_fields(AT_TOKEN, AT_BASE, AT_TABLE)

                    payload = []
                    for r in records:
                        urls = [img_url_map[n] for n in r.get("images", []) if n in img_url_map]
                        fields = {
                            "Question Number": r.get("questionNumber", ""),
                            "Question Text": r.get("questionText", ""),
                            "Mark Allocation": clamp_int(r.get("markAllocation", 0)),
                            "Topic": r.get("topic", ""),
                            "Subtopic": r.get("subtopic", ""),
                            "Mark Scheme Answer": r.get("markSchemeAnswer", ""),
                            "Image Description": r.get("imageDescription", ""),
                            "Has Images": bool(urls or r.get("hasImages", False)),
                            "Images": [{"url": u} for u in urls],
                            "Paper Name": r.get("paperName", paper_name),
                            "Exam Type": r.get("examType", exam_type),
                            "Page Number": clamp_int(r.get("pageNumber", 1), 1),
                            "Image Mapping Confidence": r.get("imageMappingConfidence", ""),
                            "Image Mapping Notes": r.get("imageMappingNotes", ""),
                        }
                        fields = {k: v for k, v in fields.items() if k in existing}
                        payload.append({"fields": fields})

                    if payload:
                        log(f"Pushing {len(payload)} records…")
                        created = create_airtable_records(AT_TOKEN, AT_BASE, AT_TABLE, payload)
                        log(f"✅ {len(created)} records synced!")
                    else:
                        log("No record payload to sync. You may have only exported images.")

                except Exception as e:
                    log(f"❌ Sync failed: {e}")

                st.text("\n".join(log_lines))
                st.markdown(f"[Open in Airtable →](https://airtable.com/{AT_BASE})")
