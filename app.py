"""
app_v2.py  –  Past Paper → Airtable  (OpenAI, Streamlit Cloud)
==============================================================
Architecture:
  Stage 1 — PyMuPDF extracts ALL visuals from each page (no filtering).
  Stage 2 — GPT-4V reviews each page's crops, judges relevance,
             assigns question number, all in one call per page.
  Stage 3 — Recovery pass: any hasImages question with no visual gets
             a targeted GPT crop attempt using the image description.

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
IMAGE_MAX_SIZE    = (1200, 1200)
JPEG_QUALITY      = 70
RENDER_DPI        = 150
VISION_DPI        = 170
EXTRACT_DPI       = 190

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

SKIP_PAGE_KEYWORDS = [
    "do not write on this page",
    "additional page, if required",
    "there are no questions printed",
    "copyright information",
]

# ── Prompts ───────────────────────────────────────────────────────────────
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

GPT_JUDGE_PROMPT = """You are reviewing extracted visual crops from page {page_num} of an exam paper.

For each crop, decide:
1. Is it relevant? A relevant visual is something a student NEEDS to answer a question:
   diagrams, graphs, grids, tables, formula boxes, info boxes, number lines, geometric shapes.
   NOT relevant: blank answer lines, empty boxes, page borders, barcodes, headers/footers,
   "do not write" boxes, page number boxes.

2. If relevant, which question number does it belong to?

Return ONLY a raw JSON array. No markdown. No explanation.
Each element:
{{
  "cropName": "p{page_num}_crop1.png",
  "relevant": true,
  "questionNumber": "7a",
  "confidence": "high",
  "label": "pizza toppings completion table",
  "notes": "Table with SM entry beside Q7a"
}}

confidence: high, medium, or low
If relevant but question cannot be determined use questionNumber = "none"
If not relevant, still include the entry with relevant = false and questionNumber = "none"
"""

FIND_MISSING_PROMPT = """This is page {page_num} of an exam paper.

Question {qnum} is marked as having an associated visual, but none was found automatically.

Question text: {question_text}
Expected visual description: {image_desc}

Find the visual that belongs to this question on the page.
Return ONLY a raw JSON object. No markdown. No explanation.
{{
  "found": true,
  "x": 0.25,
  "y": 0.10,
  "w": 0.40,
  "h": 0.30,
  "label": "centimetre grid",
  "confidence": "high",
  "notes": "Grid appears directly under Q3a"
}}

If you cannot find any visual for this question return:
{{"found": false}}

x, y = top-left corner as fraction of page width/height (0.0-1.0)
w, h = width/height as fraction of page dimensions
Be generous with the bounding box.
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

def encode_bytes(data: bytes) -> str:
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    pil.thumbnail(IMAGE_MAX_SIZE)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
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
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = doc[page_num-1].get_pixmap(matrix=mat, alpha=False)
    doc.close()
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

def split_pdf(pdf_bytes: bytes, chunk_size: int) -> list[bytes]:
    doc, chunks = fitz.open(stream=pdf_bytes, filetype="pdf"), []
    for start in range(0, len(doc), chunk_size):
        w = fitz.open()
        w.insert_pdf(doc, from_page=start,
                     to_page=min(start + chunk_size, len(doc)) - 1)
        buf = io.BytesIO()
        w.save(buf); w.close()
        chunks.append(buf.getvalue())
    doc.close()
    return chunks

def pdf_chunk_to_pils(pdf_bytes: bytes, dpi: int = RENDER_DPI) -> list[Image.Image]:
    doc, out = fitz.open(stream=pdf_bytes, filetype="pdf"), []
    mat = fitz.Matrix(dpi/72, dpi/72)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
    doc.close()
    return out

def is_question_page(pdf_bytes: bytes, page_num: int) -> bool:
    if page_num == 1:
        return False
    doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = doc[page_num-1].get_text().lower()
    doc.close()
    return not any(kw in text for kw in SKIP_PAGE_KEYWORDS)

# ── Stage 1: PyMuPDF — extract ALL visuals, zero filtering ───────────────
def rects_overlap(a: fitz.Rect, b: fitz.Rect, pad: float = 4.0) -> bool:
    return (a.x0 - pad < b.x1 and a.x1 + pad > b.x0 and
            a.y0 - pad < b.y1 and a.y1 + pad > b.y0)

def merge_rects(rects: list[fitz.Rect],
                pad: float = 6.0,
                max_h_gap: float = 30.0) -> list[fitz.Rect]:
    """
    Merge overlapping rects. Rects with horizontal gap > max_h_gap
    are NOT merged — keeps side-by-side boxes separate.
    """
    rects, merged = list(rects), []
    while rects:
        cur, changed = rects.pop(0), True
        while changed:
            changed, remaining = False, []
            for r in rects:
                h_gap = max(r.x0 - cur.x1, cur.x0 - r.x1, 0.0)
                if h_gap <= max_h_gap and rects_overlap(cur, r, pad):
                    cur |= r; changed = True
                else:
                    remaining.append(r)
            rects = remaining
        merged.append(cur)
    return merged

def render_clip(page: fitz.Page, rect: fitz.Rect, dpi: int = EXTRACT_DPI) -> bytes:
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    return pix.tobytes("png")

def extract_all_crops(pdf_bytes: bytes) -> list[dict]:
    """
    Extract every possible visual candidate from every question page.
    Order: raster images → find_tables() → path regions (skip if overlaps a table).
    No relevance filtering — GPT judges that in Stage 2.
    """
    doc       = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_crops: list[dict] = []

    for page_num, page in enumerate(doc, 1):
        if not is_question_page(pdf_bytes, page_num):
            continue

        pr = page.rect

        # ── 1. Embedded raster images ──────────────────────────────────
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)["blocks"]
        for idx, block in enumerate(blocks, 1):
            if block.get("type") != 1:
                continue
            try:
                img_bytes = block.get("image")
                if not img_bytes:
                    continue
                pil = Image.open(io.BytesIO(img_bytes))
                w, h = pil.size
                if w < 40 or h < 40:
                    continue
                all_crops.append({
                    "page":   page_num,
                    "name":   f"p{page_num}_raster{idx}.png",
                    "source": "raster",
                    "data":   img_bytes,
                    "width":  w,
                    "height": h,
                    "bbox":   fitz.Rect(block.get("bbox", [0,0,0,0])),
                })
            except Exception:
                pass

        # ── 2. Tables via find_tables() — semantic, pixel-perfect ──────
        table_rects: list[fitz.Rect] = []
        try:
            for idx, table in enumerate(page.find_tables().tables, 1):
                rect = fitz.Rect(table.bbox)
                # Expand slightly to catch full border
                rect = fitz.Rect(rect.x0 - 3, rect.y0 - 3,
                                 rect.x1 + 3, rect.y1 + 3)
                table_rects.append(rect)
                if rect.width < 30 or rect.height < 30:
                    continue
                data = render_clip(page, rect)
                pil  = Image.open(io.BytesIO(data))
                w, h = pil.size
                if w < 20 or h < 20:
                    continue
                all_crops.append({
                    "page":   page_num,
                    "name":   f"p{page_num}_table{idx}.png",
                    "source": "table",
                    "data":   data,
                    "width":  w,
                    "height": h,
                    "bbox":   rect,
                })
        except Exception:
            pass

        # ── 3. Path regions — skip anything that overlaps a table ──────
        path_rects: list[fitz.Rect] = []
        for path in page.get_drawings():
            r = path.get("rect")
            if not r:
                continue
            r = fitz.Rect(r)
            if r.width < 10 or r.height < 10:
                continue
            # Skip if this path is inside or overlaps a table we already have
            if any(rects_overlap(r, tr, pad=6) for tr in table_rects):
                continue
            path_rects.append(r)

        for idx, rect in enumerate(merge_rects(path_rects), 1):
            # Skip obvious full-page borders
            if rect.width > pr.width * 0.93 and rect.height > pr.height * 0.83:
                continue
            if rect.width < 15 or rect.height < 15:
                continue
            # Skip if overlaps a table bbox (belt-and-braces)
            if any(rects_overlap(rect, tr, pad=6) for tr in table_rects):
                continue
            try:
                data = render_clip(page, rect)
                pil  = Image.open(io.BytesIO(data))
                w, h = pil.size
                if w < 20 or h < 20:
                    continue
                all_crops.append({
                    "page":   page_num,
                    "name":   f"p{page_num}_path{idx}.png",
                    "source": "path",
                    "data":   data,
                    "width":  w,
                    "height": h,
                    "bbox":   rect,
                })
            except Exception:
                pass

    doc.close()
    return all_crops

# ── Stage 2: GPT judges relevance + assigns question number ───────────────
def judge_page_crops(client: OpenAI, pdf_bytes: bytes,
                     page_num: int, crops: list[dict]) -> list[dict]:
    """
    Send the rendered page + all crops to GPT-4V.
    GPT decides: relevant? which question?
    Returns only the relevant crops with mapping metadata.
    """
    if not crops:
        return []

    page_pil = render_page_pil(pdf_bytes, page_num, dpi=VISION_DPI)

    content: list[dict] = [
        {"type": "input_text",
         "text": GPT_JUDGE_PROMPT.format(page_num=page_num)},
        {"type": "input_image",
         "image_url": f"data:image/jpeg;base64,{encode_pil(page_pil)}"},
    ]
    for crop in crops:
        content.append({"type": "input_text",  "text": f"Crop: {crop['name']}"})
        content.append({"type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{encode_bytes(crop['data'])}"})

    raw  = call_gpt(client, content, VISION_MODEL, max_output_tokens=3000)
    rows = safe_json_loads(raw, [])

    judgements: dict[str, dict] = {
        str(r.get("cropName", "") or "").strip(): r
        for r in rows
    }

    kept: list[dict] = []
    for crop in crops:
        j = judgements.get(crop["name"], {})
        if not j.get("relevant", False):
            continue
        qnum = normalise_qnum(j.get("questionNumber", "none"))
        kept.append({
            **crop,
            "kind":           str(j.get("label",      "visual")),
            "questionNumber": qnum or "none",
            "confidence":     str(j.get("confidence", "low")).lower(),
            "notes":          str(j.get("notes",      "")),
        })
    return kept

@st.cache_data(show_spinner=False)
def extract_and_judge_visuals(openai_key: str,
                               pdf_bytes: bytes) -> tuple[list[dict], dict[str, dict]]:
    """
    Stage 1: PyMuPDF extracts all crops.
    Stage 2: GPT judges each page's crops in parallel.
    Returns (visuals, image_map).
    """
    client    = OpenAI(api_key=openai_key)
    all_crops = extract_all_crops(pdf_bytes)

    by_page: dict[int, list[dict]] = {}
    for crop in all_crops:
        by_page.setdefault(crop["page"], []).append(crop)

    results: dict[int, list] = {}

    def process_page(page_num: int, crops: list[dict]):
        judged = judge_page_crops(client, pdf_bytes, page_num, crops)
        return page_num, judged

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for pn, judged in [f.result() for f in
                            as_completed([ex.submit(process_page, pn, crops)
                                          for pn, crops in by_page.items()])]:
            results[pn] = judged

    kept_visuals: list[dict]       = []
    full_mapping: dict[str, dict]  = {}
    for pn in sorted(results):
        for v in results[pn]:
            kept_visuals.append(v)
            full_mapping[v["name"]] = {
                "questionNumber": v.get("questionNumber", "none"),
                "confidence":     v.get("confidence",     "low"),
                "notes":          v.get("notes",          ""),
                "source":         "gpt_judge",
            }

    return kept_visuals, full_mapping

# ── Stage 3: Recovery — find visuals for hasImages questions with none ────
def recover_missing_visual(client: OpenAI, pdf_bytes: bytes,
                            record: dict) -> dict | None:
    page_num = clamp_int(record.get("pageNumber", 1), 1)
    qnum     = record.get("questionNumber", "?")
    page_pil = render_page_pil(pdf_bytes, page_num, dpi=VISION_DPI)
    pw, ph   = page_pil.size

    prompt = FIND_MISSING_PROMPT.format(
        page_num      = page_num,
        qnum          = qnum,
        question_text = record.get("questionText",     "")[:300],
        image_desc    = record.get("imageDescription", "") or "Not specified",
    )
    content = [
        {"type": "input_text",  "text": prompt},
        {"type": "input_image",
         "image_url": f"data:image/jpeg;base64,{encode_pil(page_pil)}"},
    ]
    raw  = call_gpt(client, content, VISION_MODEL, max_output_tokens=500)
    data = safe_json_loads(raw, {})

    if not data.get("found"):
        return None

    try:
        x = float(data["x"]); y = float(data["y"])
        w = float(data["w"]); h = float(data["h"])
        pad = 0.02
        x1  = max(0.0, x - pad);     y1 = max(0.0, y - pad)
        x2  = min(1.0, x + w + pad); y2 = min(1.0, y + h + pad)
        if (x2 - x1) > 0.95 and (y2 - y1) > 0.90:
            return None
        crop = page_pil.crop((int(x1*pw), int(y1*ph), int(x2*pw), int(y2*ph)))
        buf  = io.BytesIO()
        crop.save(buf, format="PNG")
        name = f"p{page_num}_recovered_{qnum}.png"
        return {
            "page":           page_num,
            "name":           name,
            "source":         "recovered",
            "kind":           str(data.get("label", "visual")),
            "data":           buf.getvalue(),
            "width":          crop.width,
            "height":         crop.height,
            "questionNumber": qnum,
            "confidence":     str(data.get("confidence", "medium")).lower(),
            "notes":          str(data.get("notes", "Recovered: missing visual")),
        }
    except Exception:
        return None

def recover_all_missing(client: OpenAI, pdf_bytes: bytes,
                         records: list[dict],
                         images: list[dict],
                         image_map: dict[str, dict]) -> tuple[list[dict], dict[str, dict]]:
    missing = [r for r in records
               if r.get("hasImages") and not r.get("images")]
    if not missing:
        return images, image_map

    new_images = list(images)
    new_map    = dict(image_map)

    def try_recover(record):
        v = recover_missing_visual(client, pdf_bytes, record)
        return record["questionNumber"], v

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for qnum, visual in [f.result() for f in
                              as_completed([ex.submit(try_recover, r)
                                            for r in missing])]:
            if visual:
                new_images.append(visual)
                new_map[visual["name"]] = {
                    "questionNumber": qnum,
                    "confidence":     visual["confidence"],
                    "notes":          visual["notes"],
                    "source":         "recovered",
                }

    return new_images, new_map

# ── Question + mark scheme extraction ─────────────────────────────────────
def extract_questions_parallel(client: OpenAI,
                                pdf_bytes: bytes) -> tuple[list[dict], list[str]]:
    chunks = split_pdf(pdf_bytes, CHUNK_PAGES)
    logs: list[str] = []

    def process(i: int, chunk: bytes):
        offset  = i * CHUNK_PAGES
        pils    = pdf_chunk_to_pils(chunk)
        content = [{"type": "input_text", "text": QUESTION_PROMPT}]
        for p in pils:
            content.append({"type": "input_image",
                             "image_url": f"data:image/jpeg;base64,{encode_pil(p)}"})
        raw   = call_gpt(client, content, TEXT_MODEL)
        rows  = safe_json_loads(raw, [])
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
            ordered[i] = rows
            logs.append(msg)

    collected = []
    for i in range(len(chunks)):
        collected.extend(ordered.get(i, []))
    return collected, logs

def extract_markscheme_parallel(client: OpenAI,
                                 pdf_bytes: bytes) -> tuple[dict[str, str], list[str]]:
    chunks = split_pdf(pdf_bytes, CHUNK_PAGES)
    logs: list[str] = []

    def process(i: int, chunk: bytes):
        pils    = pdf_chunk_to_pils(chunk)
        content = [{"type": "input_text", "text": MS_PROMPT}]
        for p in pils:
            content.append({"type": "input_image",
                             "image_url": f"data:image/jpeg;base64,{encode_pil(p)}"})
        raw   = call_gpt(client, content, TEXT_MODEL)
        rows  = safe_json_loads(raw, [])
        local = {
            normalise_qnum(r.get("questionNumber", "")): str(r.get("markSchemeAnswer", "") or "")
            for r in rows
            if r.get("questionNumber") and r.get("markSchemeAnswer")
        }
        return i, local, f"MS chunk {i+1}/{len(chunks)}: {len(local)} entries"

    ordered: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, local, msg in [f.result() for f in
                               as_completed([ex.submit(process, i, c)
                                             for i, c in enumerate(chunks)])]:
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
                fields.append({"name": name, "type": "number",
                               "options": {"precision": 0}})
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
st.set_page_config(page_title="Past Paper → Airtable v2",
                   page_icon="📄", layout="wide")
st.title("📄 Past Paper → Airtable v2")
st.caption("PyMuPDF extracts everything · GPT-4V judges & maps · Recovery pass ensures no missing visuals")

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

# ── Step 1: Upload ─────────────────────────────────────────────────────────
st.subheader("1 · Upload PDFs")
col1, col2 = st.columns(2)
with col1:
    paper_name = st.text_input("Paper name", placeholder="AQA Maths P1 2024")
    exam_type  = st.text_input("Exam type",  placeholder="GCSE / A-Level / IB HL")
    paper_file = st.file_uploader("Past paper PDF *", type="pdf")
with col2:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    ms_file = st.file_uploader("Mark scheme PDF (optional)", type="pdf")

# ── Step 2: Extract ────────────────────────────────────────────────────────
st.subheader("2 · Extract with GPT")
if st.button("✨ Extract Questions", type="primary",
             disabled=not (paper_file and paper_name and exam_type and OPENAI_KEY)):

    paper_bytes = paper_file.read()
    ms_bytes    = ms_file.read() if ms_file else None
    client      = OpenAI(api_key=OPENAI_KEY)

    with st.status("Extracting…", expanded=True) as status:

        # Stage 1 + 2: extract and judge visuals
        st.write("📎 Extracting & judging visuals (PyMuPDF + GPT-4V)…")
        extract_and_judge_visuals.clear()
        images, image_map = extract_and_judge_visuals(OPENAI_KEY, paper_bytes)
        mapped = sum(1 for v in image_map.values()
                     if v.get("questionNumber") not in {"none", ""})
        st.write(f"   {len(images)} visuals kept · {mapped} mapped to questions")

        # Extract questions
        st.write("🤖 Extracting questions (parallel)…")
        questions, q_logs = extract_questions_parallel(client, paper_bytes)
        for line in q_logs:
            st.write(f"   {line}")

        # Extract mark scheme
        ms_map: dict[str, str] = {}
        if ms_bytes:
            st.write("🧠 Extracting mark scheme (parallel)…")
            ms_map, ms_logs = extract_markscheme_parallel(client, ms_bytes)
            for line in ms_logs:
                st.write(f"   {line}")

        # Build records
        records = []
        for q in questions:
            qnum = normalise_qnum(q.get("questionNumber", ""))
            records.append({
                "questionNumber":         qnum,
                "questionText":           q.get("questionText",     ""),
                "markAllocation":         clamp_int(q.get("markAllocation", 0)),
                "topic":                  q.get("topic",            ""),
                "subtopic":               q.get("subtopic",         ""),
                "markSchemeAnswer":       ms_map.get(qnum,          ""),
                "imageDescription":       q.get("imageDescription", ""),
                "hasImages":              bool(q.get("hasImages",   False)),
                "pageNumber":             clamp_int(q.get("pageNumber", 1), 1),
                "paperName":              paper_name,
                "examType":               exam_type,
                "imageMappingConfidence": "",
                "imageMappingNotes":      "",
                "images":                 [],
            })

        # Apply image mapping + fallback
        if images:
            pq_index: dict[int, list[str]] = {}
            for r in records:
                page = clamp_int(r.get("pageNumber", 0))
                qn   = normalise_qnum(r.get("questionNumber", ""))
                if page and qn:
                    pq_index.setdefault(page, [])
                    if qn not in pq_index[page]:
                        pq_index[page].append(qn)

            for img in images:
                nm = img["name"]
                if image_map.get(nm, {}).get("questionNumber") in {"", "none", None}:
                    qs = pq_index.get(img["page"], [])
                    image_map[nm] = {
                        "questionNumber": qs[-1] if qs else "none",
                        "confidence":     "low",
                        "notes":          "Fallback: last question on page",
                        "source":         "fallback",
                    }

            q_to_imgs  : dict[str, list[str]] = {}
            q_to_conf  : dict[str, list[str]] = {}
            q_to_notes : dict[str, list[str]] = {}
            for nm, meta in image_map.items():
                qn = normalise_qnum(meta.get("questionNumber", "none"))
                if qn and qn != "none":
                    q_to_imgs .setdefault(qn, []).append(nm)
                    q_to_conf .setdefault(qn, []).append(meta.get("confidence", "low"))
                    q_to_notes.setdefault(qn, []).append(
                        f"{nm}: {meta.get('notes','')} [{meta.get('source','gpt')}]")

            for r in records:
                qn    = r["questionNumber"]
                imgs  = q_to_imgs .get(qn, [])
                confs = q_to_conf .get(qn, [])
                notes = q_to_notes.get(qn, [])
                if imgs:
                    r["hasImages"]              = True
                    r["images"]                 = imgs
                    r["imageMappingConfidence"] = ("high"   if "high"   in confs else
                                                   "medium" if "medium" in confs else "low")
                    r["imageMappingNotes"]      = "\n".join(notes)

        # Stage 3: recovery pass
        missing_count = sum(1 for r in records
                            if r.get("hasImages") and not r.get("images"))
        if missing_count:
            st.write(f"🔎 Recovery pass: {missing_count} hasImages question(s) with no visual…")
            images, image_map = recover_all_missing(
                client, paper_bytes, records, images, image_map)

            for img in images:
                if img.get("source") != "recovered":
                    continue
                qn = img.get("questionNumber", "none")
                for r in records:
                    if r["questionNumber"] == qn and not r.get("images"):
                        r["hasImages"]              = True
                        r["images"]                 = [img["name"]]
                        r["imageMappingConfidence"] = img.get("confidence", "medium")
                        r["imageMappingNotes"]      = img.get("notes", "")

            recovered     = sum(1 for img in images if img.get("source") == "recovered")
            still_missing = sum(1 for r in records
                                if r.get("hasImages") and not r.get("images"))
            st.write(f"   Recovered {recovered} · {still_missing} still unresolved")

        st.session_state["records"]   = records
        st.session_state["images"]    = images
        st.session_state["image_map"] = image_map
        status.update(
            label=f"✅ Done — {len(records)} questions · {len(images)} visuals",
            state="complete"
        )

# ── Step 3: Review ─────────────────────────────────────────────────────────
if "records" in st.session_state:
    records   = st.session_state["records"]
    images    = st.session_state.get("images",    [])
    image_map = st.session_state.get("image_map", {})

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

    edited = st.data_editor(df, use_container_width=True,
                             num_rows="dynamic", height=460)

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
        with st.expander(f"🖼 Kept visuals ({len(images)})"):
            cols = st.columns(4)
            for idx, img in enumerate(images):
                with cols[idx % 4]:
                    qn  = image_map.get(img["name"], {}).get("questionNumber", "?")
                    src = img.get("source", "?")
                    st.image(img["data"],
                             caption=f"{img['name']}\nQ{qn} · {img.get('kind','?')} [{src}]",
                             use_container_width=True)

    # ── Step 4: Export / Sync ──────────────────────────────────────────────
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
            st.warning("Add your Airtable token and Base ID to sync.")
        elif not AT_TOKEN.startswith("pat"):
            st.error("❌ Token should start with `pat`.")
        elif not AT_BASE.startswith("app"):
            st.error("❌ Base ID should start with `app`.")
        else:
            token_preview = AT_TOKEN[:8] + "..." + AT_TOKEN[-4:]
            st.caption(f"Token: `{token_preview}` | Base: `{AT_BASE}` | Table: `{AT_TABLE}`")

            if st.button("🚀 Sync to Airtable", type="primary"):
                _records  = st.session_state.get("records",   [])
                _images   = st.session_state.get("images",    [])
                _imgbb    = get_secret("IMGBB_API_KEY")
                log_lines: list[str] = []
                def log(msg): log_lines.append(msg)

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
                    log("⚠️ IMGBB_API_KEY missing — visuals not attached.")

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

                    log(f"Pushing {len(payload)} records…")
                    created = create_airtable_records(AT_TOKEN, AT_BASE, AT_TABLE, payload)
                    log(f"✅ {len(created)} records synced!")
                except Exception as e:
                    log(f"❌ Sync failed: {e}")

                st.text("\n".join(log_lines))
                st.markdown(f"[Open in Airtable →](https://airtable.com/{AT_BASE})")
