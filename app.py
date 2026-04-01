import io
import json
import re
import base64
import time
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────
TEXT_MODEL = "gpt-4.1"
VISION_MODEL = "gpt-4.1"
MAX_OUTPUT_TOKENS = 8000
CHUNK_PAGES = 6
MAX_WORKERS = 4
MAX_RETRIES = 4
BASE_BACKOFF = 2
REQUEST_TIMEOUT = 120
RENDER_DPI = 150
VISION_DPI = 170
AT_API = "https://api.airtable.com/v0"
AT_META = "https://api.airtable.com/v0/meta"
IMGBB_API = "https://api.imgbb.com/1/upload"

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
- pageNumber must be the page on which the question appears within the chunk you were shown.
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
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def safe_json_loads(raw: str, default: Any):
    try:
        return json.loads(clean_json(raw))
    except Exception:
        return default


def normalise_question_number(value: str) -> str:
    value = str(value or "").strip()
    value = value.replace(" ", "")
    value = value.replace("(", "").replace(")", "")
    return value


def clamp_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def chunk_list(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def run_with_retry(fn, *args, **kwargs):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt == MAX_RETRIES:
                raise
            time.sleep(BASE_BACKOFF * attempt)
    raise last_error

# ── OpenAI helpers ────────────────────────────────────────────────────────
def openai_response_text(resp) -> str:
    text = getattr(resp, "output_text", None)
    if text:
        return text
    parts: list[str] = []
    for item in getattr(resp, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            maybe = getattr(content, "text", None)
            if maybe:
                parts.append(maybe)
    return "\n".join(parts)


def create_openai_response(client: OpenAI, model: str, content: list[dict], max_output_tokens: int = MAX_OUTPUT_TOKENS):
    def _call():
        return client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=max_output_tokens,
            timeout=REQUEST_TIMEOUT,
        )
    return run_with_retry(_call)

# ── PDF / image helpers ───────────────────────────────────────────────────
def render_page_as_png(pdf_bytes: bytes, page_num: int, dpi: int = RENDER_DPI) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = doc[page_num - 1].get_pixmap(matrix=mat, alpha=False)
    doc.close()
    return pix.tobytes("png")


def split_pdf(pdf_bytes: bytes, chunk_size: int) -> list[bytes]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks: list[bytes] = []
    for start in range(0, len(doc), chunk_size):
        writer = fitz.open()
        writer.insert_pdf(doc, from_page=start, to_page=min(start + chunk_size, len(doc)) - 1)
        buf = io.BytesIO()
        writer.save(buf)
        writer.close()
        chunks.append(buf.getvalue())
    doc.close()
    return chunks


def pdf_chunk_to_base64_images(pdf_bytes: bytes, dpi: int = RENDER_DPI) -> list[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out: list[str] = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out.append(base64.standard_b64encode(pix.tobytes("png")).decode())
    doc.close()
    return out


def looks_useful_image(img: Image.Image) -> tuple[bool, str]:
    width, height = img.width, img.height
    if width < 80 or height < 80:
        return False, "too small"
    aspect = width / max(height, 1)
    if aspect > 7 or aspect < 0.14:
        return False, "extreme aspect ratio"
    area = width * height
    if area < 12000:
        return False, "area too small"

    try:
        gray = img.convert("L")
        bbox = gray.getbbox()
        if not bbox:
            return False, "blank image"
    except Exception:
        pass

    return True, "ok"


def extract_images(pdf_bytes: bytes) -> list[dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: list[dict] = []
    for page_num, page in enumerate(doc, 1):
        for idx, img_info in enumerate(page.get_images(full=True), 1):
            try:
                extracted = doc.extract_image(img_info[0])
                data = extracted["image"]
                ext = extracted.get("ext", "png")
                pil = Image.open(io.BytesIO(data))
                useful, reason = looks_useful_image(pil)
                if not useful:
                    continue
                images.append({
                    "page": page_num,
                    "name": f"p{page_num}_img{idx}.{ext}",
                    "data": data,
                    "width": pil.width,
                    "height": pil.height,
                    "ext": ext,
                    "filter_reason": reason,
                })
            except Exception:
                pass
    doc.close()
    return images

# ── Extraction with parallel processing ───────────────────────────────────
from io import BytesIO
import base64

MAX_IMAGES_PER_REQUEST = 2
IMAGE_MAX_SIZE = (1200, 1200)
JPEG_QUALITY = 70

def encode_image(img_pil):
    """
    Resize + compress image before sending to GPT
    """
    img_pil = img_pil.copy()
    img_pil.thumbnail(IMAGE_MAX_SIZE)

    buffer = BytesIO()
    img_pil.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buffer.getvalue()).decode()
from io import BytesIO
import base64

MAX_IMAGES_PER_REQUEST = 2
IMAGE_MAX_SIZE = (1200, 1200)
JPEG_QUALITY = 70


def encode_image(img_pil):
    img_pil = img_pil.copy()
    img_pil.thumbnail(IMAGE_MAX_SIZE)

    buffer = BytesIO()
    img_pil.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buffer.getvalue()).decode()


def call_gpt_vision(
    client,
    images,
    prompt,
    model="gpt-4.1-mini",
    max_images=MAX_IMAGES_PER_REQUEST,
    debug_label="GPT"
):
    """
    Unified GPT vision caller (safe + reusable)
    """

    images = images[:max_images]

    content = [{"type": "input_text", "text": prompt}]
    total_size = 0

    for img in images:
        base64_img = encode_image(img)
        total_size += len(base64_img)

        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{base64_img}"
        })

    print(f"[{debug_label}] {len(images)} images | ~{round(total_size/1e6,2)} MB")

    try:
        response = client.responses.create(
            model=model,
            input=[{
                "role": "user",
                "content": content
            }]
        )

        return response.output_text

    except Exception as e:
        print(f"[{debug_label} ERROR] {e}")
        return ""

def ask_gpt_for_questions(client, images, prompt):
    return call_gpt_vision(
        client,
        images,
        prompt,
        model="gpt-4.1-mini",
        debug_label="QUESTIONS"
    )

def ask_gpt_for_markscheme(client, images, prompt):
    return call_gpt_vision(
        client,
        images,
        prompt,
        model="gpt-4.1-mini",
        debug_label="MARKSCHEME"
    )


def extract_questions_parallel(client: OpenAI, pdf_bytes: bytes, chunk_pages: int) -> tuple[list[dict], list[str]]:
    pdf_chunks = split_pdf(pdf_bytes, chunk_pages)
    raw_logs: list[str] = []
    collected: list[dict] = []

    def process_one(index: int, chunk_bytes: bytes):
        page_offset = index * chunk_pages
        chunk_images = pdf_chunk_to_base64_images(chunk_bytes)
        raw = ask_gpt_for_questions(client, chunk_images, QUESTION_PROMPT)
        rows = safe_json_loads(raw, [])
        fixed: list[dict] = []
        for row in rows:
            page_num = clamp_int(row.get("pageNumber", 1), 1) + page_offset
            fixed.append({
                "questionNumber": normalise_question_number(row.get("questionNumber", "")),
                "questionText": str(row.get("questionText", "") or ""),
                "markAllocation": clamp_int(row.get("markAllocation", 0), 0),
                "topic": str(row.get("topic", "") or ""),
                "subtopic": str(row.get("subtopic", "") or ""),
                "hasImages": bool(row.get("hasImages", False)),
                "imageDescription": str(row.get("imageDescription", "") or ""),
                "pageNumber": page_num,
            })
        return fixed, f"Chunk {index + 1}/{len(pdf_chunks)} complete, {len(fixed)} questions"

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one, i, chunk): i for i, chunk in enumerate(pdf_chunks)}
        ordered: dict[int, list[dict]] = {}
        for future in as_completed(futures):
            i = futures[future]
            rows, message = future.result()
            ordered[i] = rows
            raw_logs.append(message)

    for i in range(len(pdf_chunks)):
        collected.extend(ordered.get(i, []))

    return collected, raw_logs


def extract_markscheme_parallel(client: OpenAI, pdf_bytes: bytes, chunk_pages: int) -> tuple[dict[str, str], list[str]]:
    pdf_chunks = split_pdf(pdf_bytes, chunk_pages)
    ms_map: dict[str, str] = {}
    raw_logs: list[str] = []

    def process_one(index: int, chunk_bytes: bytes):
        chunk_images = pdf_chunk_to_base64_images(chunk_bytes)
        raw = ask_gpt_for_markscheme(client, chunk_images, MS_PROMPT)
        rows = safe_json_loads(raw, [])
        local: dict[str, str] = {}
        for row in rows:
            qn = normalise_question_number(row.get("questionNumber", ""))
            ans = str(row.get("markSchemeAnswer", "") or "")
            if qn and ans:
                local[qn] = ans
        return local, f"Mark scheme chunk {index + 1}/{len(pdf_chunks)} complete, {len(local)} entries"

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one, i, chunk): i for i, chunk in enumerate(pdf_chunks)}
        ordered: dict[int, dict[str, str]] = {}
        for future in as_completed(futures):
            i = futures[future]
            local, message = future.result()
            ordered[i] = local
            raw_logs.append(message)

    for i in range(len(pdf_chunks)):
        ms_map.update(ordered.get(i, {}))

    return ms_map, raw_logs

# ── Image mapping with confidence + fallback ──────────────────────────────
def map_images_for_page(client: OpenAI, pdf_bytes: bytes, page_num: int, page_imgs: list[dict]) -> dict[str, dict]:
    png_bytes = render_page_as_png(pdf_bytes, page_num, dpi=VISION_DPI)
    png_b64 = base64.standard_b64encode(png_bytes).decode()
    img_list = "\n".join(
        f"- {img['name']} ({img['width']}x{img['height']})" for img in page_imgs
    )

    prompt = f"""This is page {page_num} of an exam paper.

The following extracted images came from this page:
{img_list}

For each image, identify the question number it belongs to by reading the visual layout of the page.
Use the nearest clearly associated question number.
Ignore page logos, decorative images, and unrelated artwork.

Return ONLY a JSON array like this:
[
  {{
    "imageName": "p{page_num}_img1.png",
    "questionNumber": "3b",
    "confidence": "high",
    "notes": "Image sits directly under question 3b"
  }}
]

Use confidence values: high, medium, low.
If no question can be assigned, use questionNumber = "none".
"""

    content = [
        {"type": "input_image", "image_base64": png_b64},
        {"type": "input_text", "text": prompt},
    ]
    resp = create_openai_response(client, VISION_MODEL, content, max_output_tokens=2500)
    rows = safe_json_loads(openai_response_text(resp), [])

    result: dict[str, dict] = {}
    for row in rows:
        name = str(row.get("imageName", "") or "").strip()
        qnum = normalise_question_number(row.get("questionNumber", "none"))
        conf = str(row.get("confidence", "low") or "low").strip().lower()
        notes = str(row.get("notes", "") or "").strip()
        if name:
            result[name] = {
                "questionNumber": qnum if qnum else "none",
                "confidence": conf if conf in {"high", "medium", "low"} else "low",
                "notes": notes,
                "source": "vision",
            }
    return result


def build_page_question_index(records: list[dict]) -> dict[int, list[str]]:
    index: dict[int, list[str]] = {}
    for r in records:
        page = clamp_int(r.get("pageNumber", 0), 0)
        qn = normalise_question_number(r.get("questionNumber", ""))
        if page and qn:
            index.setdefault(page, []).append(qn)
    for page in index:
        seen = []
        for qn in index[page]:
            if qn not in seen:
                seen.append(qn)
        index[page] = seen
    return index


def fallback_map_images(images: list[dict], page_question_index: dict[int, list[str]], existing_map: dict[str, dict]) -> dict[str, dict]:
    for img in images:
        name = img["name"]
        if name in existing_map and existing_map[name].get("questionNumber") not in {"", "none"}:
            continue
        page_questions = page_question_index.get(img["page"], [])
        qnum = page_questions[-1] if page_questions else "none"
        existing_map[name] = {
            "questionNumber": qnum,
            "confidence": "low" if qnum != "none" else "low",
            "notes": "Fallback mapping based on page-level question list",
            "source": "fallback",
        }
    return existing_map


def map_images_to_questions_parallel(client: OpenAI, pdf_bytes: bytes, images: list[dict], records: list[dict]) -> dict[str, dict]:
    page_groups: dict[int, list[dict]] = {}
    for img in images:
        page_groups.setdefault(img["page"], []).append(img)

    mapping: dict[str, dict] = {}
    if page_groups:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(map_images_for_page, client, pdf_bytes, page_num, page_imgs): page_num
                for page_num, page_imgs in page_groups.items()
            }
            for future in as_completed(futures):
                local_map = future.result()
                mapping.update(local_map)

    page_question_index = build_page_question_index(records)
    mapping = fallback_map_images(images, page_question_index, mapping)
    return mapping

# ── Airtable helpers ──────────────────────────────────────────────────────
def at_headers(token: str):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def ensure_table(token: str, base_id: str, table: str):
    try:
        r = requests.get(f"{AT_META}/bases/{base_id}/tables", headers=at_headers(token), timeout=REQUEST_TIMEOUT)
        if r.status_code == 401:
            st.info("Skipping auto table creation because this token does not have schema.bases:write.")
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
            timeout=REQUEST_TIMEOUT,
        )
        if not r2.ok:
            st.warning(f"Could not auto-create table ({r2.status_code}). Create it manually.")
    except Exception as e:
        st.warning(f"Table check skipped: {e}")


def upload_to_imgbb(api_key: str, img: dict) -> str | None:
    b64 = base64.standard_b64encode(img["data"]).decode()
    resp = requests.post(
        IMGBB_API,
        data={"key": api_key, "name": img["name"], "image": b64},
        timeout=REQUEST_TIMEOUT,
    )
    if resp.ok:
        return resp.json()["data"]["url"]
    return None


def patch_record_images(token: str, base_id: str, table: str, record_id: str, fields: dict):
    resp = requests.patch(
        f"{AT_API}/{base_id}/{requests.utils.quote(table)}/{record_id}",
        headers=at_headers(token),
        json={"fields": fields},
        timeout=REQUEST_TIMEOUT,
    )
    if not resp.ok:
        raise RuntimeError(f"{resp.status_code} — {resp.text[:300]}")


def create_airtable_records(token: str, base_id: str, table: str, records: list[dict]) -> list[dict]:
    ensure_table(token, base_id, table)
    url = f"{AT_API}/{base_id}/{requests.utils.quote(table)}"
    created: list[dict] = []
    for batch in chunk_list(records, 10):
        resp = requests.post(url, headers=at_headers(token), json={"records": batch}, timeout=REQUEST_TIMEOUT)
        if not resp.ok:
            raise RuntimeError(f"Airtable {resp.status_code}: {resp.text[:500]}")
        created.extend(resp.json().get("records", []))
    return created

# ── Caching ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_extract_images(pdf_bytes: bytes) -> list[dict]:
    return extract_images(pdf_bytes)


@st.cache_data(show_spinner=False)
def cached_extract_questions_pdf(pdf_bytes: bytes, openai_key: str) -> tuple[list[dict], list[str]]:
    client = OpenAI(api_key=openai_key)
    return extract_questions_parallel(client, pdf_bytes, CHUNK_PAGES)


@st.cache_data(show_spinner=False)
def cached_extract_markscheme_pdf(pdf_bytes: bytes, openai_key: str) -> tuple[dict[str, str], list[str]]:
    client = OpenAI(api_key=openai_key)
    return extract_markscheme_parallel(client, pdf_bytes, CHUNK_PAGES)


@st.cache_data(show_spinner=False)
def cached_map_images(pdf_bytes: bytes, openai_key: str, images: list[dict], records: list[dict]) -> dict[str, dict]:
    client = OpenAI(api_key=openai_key)
    return map_images_to_questions_parallel(client, pdf_bytes, images, records)

# ── UI ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Past Paper → Airtable (GPT)", page_icon="📄", layout="wide")
st.title("📄 Past Paper → Airtable (GPT)")
st.caption("Upload exam PDFs, extract questions with GPT, map images, review the output, then sync to Airtable.")

OPENAI_KEY = get_secret("OPENAI_API_KEY")
AT_TOKEN = get_secret("AIRTABLE_TOKEN")
AT_BASE = get_secret("AIRTABLE_BASE_ID")
IMGBB_KEY = get_secret("IMGBB_API_KEY")

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

st.subheader("1 · Upload PDFs")
col1, col2 = st.columns(2)
with col1:
    paper_name = st.text_input("Paper name", placeholder="AQA Biology P1 2023")
    exam_type = st.text_input("Exam type", placeholder="GCSE / A-Level / IGCSE / IB HL")
    paper_file = st.file_uploader("Past paper PDF *", type="pdf")
with col2:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    ms_file = st.file_uploader("Mark scheme PDF (optional)", type="pdf")

st.subheader("2 · Extract with GPT")
extract_disabled = not (paper_file and paper_name and exam_type and OPENAI_KEY)
if st.button("✨ Extract Questions", type="primary", disabled=extract_disabled):
    paper_bytes = paper_file.read()
    ms_bytes = ms_file.read() if ms_file else None

    with st.status("Extracting…", expanded=True) as status:
        st.write("📎 Extracting useful embedded images…")
        images = cached_extract_images(paper_bytes)
        st.write(f"Found {len(images)} useful extracted images")

        st.write("🤖 Extracting questions in parallel…")
        questions, q_logs = cached_extract_questions_pdf(paper_bytes, OPENAI_KEY)
        for line in q_logs:
            st.write(line)

        ms_map: dict[str, str] = {}
        if ms_bytes:
            st.write("🧠 Extracting mark scheme in parallel…")
            ms_map, ms_logs = cached_extract_markscheme_pdf(ms_bytes, OPENAI_KEY)
            for line in ms_logs:
                st.write(line)
            st.write(f"Matched {len(ms_map)} mark scheme entries")

        records: list[dict] = []
        for q in questions:
            qnum = normalise_question_number(q.get("questionNumber", ""))
            page = clamp_int(q.get("pageNumber", 1), 1)
            records.append({
                "questionNumber": qnum,
                "questionText": q.get("questionText", ""),
                "markAllocation": clamp_int(q.get("markAllocation", 0), 0),
                "topic": q.get("topic", ""),
                "subtopic": q.get("subtopic", ""),
                "markSchemeAnswer": ms_map.get(qnum, ""),
                "imageDescription": q.get("imageDescription", ""),
                "hasImages": bool(q.get("hasImages", False)),
                "pageNumber": page,
                "paperName": paper_name,
                "examType": exam_type,
                "imageMappingConfidence": "",
                "imageMappingNotes": "",
                "images": [],
            })

        st.write("🔍 Mapping images to questions in parallel…")
        image_map = cached_map_images(paper_bytes, OPENAI_KEY, images, records) if images else {}
        mapped_count = sum(1 for v in image_map.values() if v.get("questionNumber") not in {"none", ""})
        st.write(f"Mapped {mapped_count}/{len(images)} images to question numbers")

        q_to_images: dict[str, list[str]] = {}
        q_to_conf: dict[str, list[str]] = {}
        q_to_notes: dict[str, list[str]] = {}

        for img_name, meta in image_map.items():
            qnum = normalise_question_number(meta.get("questionNumber", "none"))
            if qnum and qnum != "none":
                q_to_images.setdefault(qnum, []).append(img_name)
                q_to_conf.setdefault(qnum, []).append(meta.get("confidence", "low"))
                note = f"{img_name}: {meta.get('notes', '')} [{meta.get('source', 'vision')}]"
                q_to_notes.setdefault(qnum, []).append(note)

        for r in records:
            qn = r["questionNumber"]
            imgs = q_to_images.get(qn, [])
            confs = q_to_conf.get(qn, [])
            notes = q_to_notes.get(qn, [])
            if imgs:
                r["hasImages"] = True
                r["images"] = imgs
                if "high" in confs:
                    r["imageMappingConfidence"] = "high"
                elif "medium" in confs:
                    r["imageMappingConfidence"] = "medium"
                else:
                    r["imageMappingConfidence"] = "low"
                r["imageMappingNotes"] = "\n".join(notes)

        st.session_state["records"] = records
        st.session_state["images"] = images
        st.session_state["image_map"] = image_map
        status.update(label=f"✅ Done — {len(records)} questions extracted", state="complete")

if "records" in st.session_state:
    st.subheader("3 · Review & edit")
    st.caption("Edit anything before syncing. Image columns show the linked image names and mapping confidence.")

    records = st.session_state["records"]
    images = st.session_state.get("images", [])

    df = pd.DataFrame([
        {
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
        }
        for r in records
    ])

    edited = st.data_editor(df, use_container_width=True, num_rows="dynamic", height=460)

    for i, row in edited.iterrows():
        if i < len(records):
            records[i]["questionNumber"] = normalise_question_number(row["Q #"])
            records[i]["questionText"] = row["Question Text"]
            records[i]["markAllocation"] = clamp_int(row["Marks"], 0)
            records[i]["topic"] = row["Topic"]
            records[i]["subtopic"] = row["Subtopic"]
            records[i]["markSchemeAnswer"] = row["Mark Scheme"]
            records[i]["imageDescription"] = row["Image Desc."]
            records[i]["hasImages"] = bool(row["Has Images"])
            records[i]["imageMappingConfidence"] = row["Image Mapping Confidence"]
            records[i]["imageMappingNotes"] = row["Image Mapping Notes"]
            records[i]["pageNumber"] = clamp_int(row["Page Number"], 1)
            image_names_raw = str(row["Image Names"] or "")
            records[i]["images"] = [x.strip() for x in image_names_raw.split(",") if x.strip()]

    if images:
        with st.expander(f"🖼 Useful extracted images ({len(images)})"):
            cols = st.columns(4)
            for idx, img in enumerate(images):
                with cols[idx % 4]:
                    st.image(img["data"], caption=img["name"], use_container_width=True)

    st.subheader("4 · Export / Sync")
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.download_button(
            "⬇ Download JSON",
            data=json.dumps(records, indent=2, ensure_ascii=False).encode(),
            file_name=f"{paper_name or 'questions'}.json",
            mime="application/json",
        )

    with col_b:
        if not (AT_TOKEN and AT_BASE):
            st.warning("Add your Airtable token and Base ID in the sidebar to sync.")
        elif not AT_TOKEN.startswith("pat"):
            st.error("Token should start with pat.")
        elif not AT_BASE.startswith("app"):
            st.error("Base ID should start with app.")
        elif st.button("🚀 Sync to Airtable", type="primary"):
            _records = st.session_state.get("records", [])
            _images = st.session_state.get("images", [])
            _imgbb = IMGBB_KEY

            log_lines: list[str] = []
            def log(msg: str):
                log_lines.append(msg)

            try:
                image_name_to_url: dict[str, str] = {}
                if _images and _imgbb:
                    log("Uploading images to imgbb…")

                    def upload_one(img: dict):
                        return img["name"], upload_to_imgbb(_imgbb, img)

                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        futures = [executor.submit(upload_one, img) for img in _images]
                        for future in as_completed(futures):
                            name, url = future.result()
                            if url:
                                image_name_to_url[name] = url
                                log(f"  ✅ {name} uploaded")
                            else:
                                log(f"  ❌ {name} failed to upload")
                elif _images and not _imgbb:
                    log("⚠️ IMGBB_API_KEY missing, so no images can be attached.")
                else:
                    log("ℹ️ No useful images were extracted.")

                airtable_payload = []
                for r in _records:
                    urls = [image_name_to_url[name] for name in r.get("images", []) if name in image_name_to_url]
                    airtable_payload.append({
                        "fields": {
                            "Question Number": r.get("questionNumber", ""),
                            "Question Text": r.get("questionText", ""),
                            "Mark Allocation": clamp_int(r.get("markAllocation", 0), 0),
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
                    })

                log("Creating Airtable records…")
                created = create_airtable_records(AT_TOKEN, AT_BASE, AT_TABLE, airtable_payload)
                log(f"✅ {len(created)} records pushed to Airtable")
                st.text("\n".join(log_lines))
                st.markdown(f"[Open in Airtable →](https://airtable.com/{AT_BASE})")
            except Exception as e:
                log(f"❌ Sync failed: {e}")
                st.text("\n".join(log_lines))
