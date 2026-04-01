"""
app_v3.py  –  Past Paper → Airtable  (OpenAI, Streamlit Cloud)
==============================================================
Extraction strategy:
  1. Raster images   — get_text("dict") image blocks, re-rendered at 300 DPI
  2. Tables          — find_tables(), semantically correct bboxes, 300 DPI
  3. Drawings        — individual drawing rects (NO merging), largest-first
                       dedup (contained rects skipped), 300 DPI
  4. GPT-4V          — judges each page's crops: relevant? which question?
  5. Recovery        — any hasImages question with no visual gets a
                       targeted GPT crop attempt using question + image desc

Secrets:
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
RENDER_DPI        = 150    # DPI for sending full pages to GPT
VISION_DPI        = 170    # DPI for judge/recovery page renders
EXTRACT_DPI       = 300    # DPI for actual crops (high quality)
CROP_PAD_PT       = 8      # padding in PDF points around every crop

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
   NOT relevant: blank answer lines, empty answer boxes, page borders, barcodes,
   headers, footers, "do not write" boxes, page number boxes.

2. If relevant, which question number does it belong to?
   - A visual ABOVE a question still belongs to that question if referenced by it.
   - A data table at the top of a page belongs to the first question below it that uses the data.
   - Side-by-side boxes belong to the same question.
   - Only use questionNumber = "none" if genuinely unsure.

Return ONLY a raw JSON array. No markdown. No explanation.
Each element:
{{
  "cropName": "p{page_num}_v1.png",
  "relevant": true,
  "questionNumber": "7a",
  "confidence": "high",
  "label": "pizza toppings completion table",
  "notes": "Table with SM entry beside Q7a"
}}

confidence: high, medium, or low
If not relevant set relevant=false and questionNumber="none"
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

If you cannot find any visual return: {{"found": false}}
x, y = top-left corner as fraction of page width/height (0.0-1.0)
w, h = width/height as fraction of page dimensions. Be generous.
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
def render_page_pil(pdf_bytes: bytes, page_num: int,
                    dpi: int = RENDER_DPI) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = doc[page_num - 1].get_pixmap(matrix=mat, alpha=False)
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

def pdf_chunk_to_pils(pdf_bytes: bytes,
                       dpi: int = RENDER_DPI) -> list[Image.Image]:
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
    doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = doc[page_num - 1].get_text().lower()
    doc.close()
    return not any(kw in text for kw in SKIP_PAGE_KEYWORDS)

# ── PyMuPDF extraction helpers ────────────────────────────────────────────
def is_contained_by_any(rect: fitz.Rect,
                         others: list[fitz.Rect],
                         pad: float = 4.0) -> bool:
    for o in others:
        if (rect.x0 >= o.x0 - pad and rect.y0 >= o.y0 - pad and
                rect.x1 <= o.x1 + pad and rect.y1 <= o.y1 + pad):
            return True
    return False

def crop_rect(page: fitz.Page, rect: fitz.Rect,
              pad_pt: float = CROP_PAD_PT,
              dpi: int = EXTRACT_DPI) -> bytes:
    pr     = page.rect
    padded = fitz.Rect(
        max(pr.x0, rect.x0 - pad_pt), max(pr.y0, rect.y0 - pad_pt),
        min(pr.x1, rect.x1 + pad_pt), min(pr.y1, rect.y1 + pad_pt),
    )
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=padded, alpha=False)
    return pix.tobytes("png")

def extract_all_crops(pdf_bytes: bytes,
                       debug_page: int = 0) -> tuple[list[dict], list[str]]:
    """
    Extract all visual candidates using PyMuPDF with exact coordinates.
    If debug_page > 0, return detailed logs for that page.
    """
    doc       = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_crops: list[dict] = []
    debug_log: list[str]  = []

    for page_num, page in enumerate(doc, 1):
        if not is_question_page(pdf_bytes, page_num):
            continue

        pr           = page.rect
        taken_rects: list[fitz.Rect] = []
        is_debug     = (debug_page == page_num)

        # 1. Raster images
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)["blocks"]
        for idx, block in enumerate(blocks, 1):
            if block.get("type") != 1:
                continue
            try:
                img_bytes = block.get("image")
                if not img_bytes:
                    continue
                pil = Image.open(io.BytesIO(img_bytes))
                if pil.width < 40 or pil.height < 40:
                    continue
                bbox = fitz.Rect(block.get("bbox", [0, 0, 0, 0]))
                data = crop_rect(page, bbox)
                pil2 = Image.open(io.BytesIO(data))
                taken_rects.append(bbox)
                all_crops.append({
                    "page": page_num, "name": f"p{page_num}_raster{idx}.png",
                    "source": "raster", "data": data,
                    "width": pil2.width, "height": pil2.height, "bbox": bbox,
                })
            except Exception:
                pass

        # 2. Tables
        try:
            for idx, table in enumerate(page.find_tables().tables, 1):
                rect = fitz.Rect(table.bbox)
                if rect.width < 30 or rect.height < 30:
                    continue
                data = crop_rect(page, rect)
                pil  = Image.open(io.BytesIO(data))
                taken_rects.append(rect)
                if is_debug:
                    debug_log.append(f"TABLE p{page_num}: {rect} ({rect.width:.0f}x{rect.height:.0f}pt)")
                all_crops.append({
                    "page": page_num, "name": f"p{page_num}_table{idx}.png",
                    "source": "table", "data": data,
                    "width": pil.width, "height": pil.height, "bbox": rect,
                })
        except Exception:
            pass

        # 3. Individual drawing rects — no merging
        drawing_rects: list[fitz.Rect] = []
        for drawing in page.get_drawings():
            r = drawing.get("rect")
            if not r:
                continue
            r = fitz.Rect(r)
            if r.width < 20 or r.height < 20:
                continue
            if r.width > pr.width * 0.88 or r.height > pr.height * 0.80:
                if is_debug:
                    debug_log.append(f"SKIP full-page p{page_num}: {r} ({r.width:.0f}x{r.height:.0f}pt)")
                continue
            if is_contained_by_any(r, taken_rects, pad=6):
                if is_debug:
                    debug_log.append(f"SKIP contained-by-taken p{page_num}: {r} ({r.width:.0f}x{r.height:.0f}pt)")
                continue
            drawing_rects.append(r)

        drawing_rects.sort(key=lambda r: r.width * r.height, reverse=True)
        kept: list[fitz.Rect] = []
        for r in drawing_rects:
            if not is_contained_by_any(r, kept, pad=4):
                kept.append(r)
                if is_debug:
                    debug_log.append(f"KEEP drawing p{page_num}: {r} ({r.width:.0f}x{r.height:.0f}pt)")
            else:
                if is_debug:
                    debug_log.append(f"SKIP contained-by-kept p{page_num}: {r} ({r.width:.0f}x{r.height:.0f}pt)")

        for idx, rect in enumerate(kept, 1):
            try:
                data = crop_rect(page, rect)
                pil  = Image.open(io.BytesIO(data))
                if pil.width < 20 or pil.height < 20:
                    continue
                taken_rects.append(rect)
                all_crops.append({
                    "page": page_num, "name": f"p{page_num}_draw{idx}.png",
                    "source": "drawing", "data": data,
                    "width": pil.width, "height": pil.height, "bbox": rect,
                })
            except Exception:
                pass

    doc.close()
    return all_crops, debug_log

# ── GPT judges relevance + assigns question number ────────────────────────
def judge_page_crops(client: OpenAI, pdf_bytes: bytes,
                     page_num: int, crops: list[dict]) -> list[dict]:
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
        str(r.get("cropName", "") or "").strip(): r for r in rows
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
    client    = OpenAI(api_key=openai_key)
    all_crops, _ = extract_all_crops(pdf_bytes)

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

# ── Recovery: find visuals for hasImages questions with none ──────────────
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

        # Render recovery crop at 300 DPI from fraction coords
        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[page_num - 1]
        pr   = page.rect
        rect = fitz.Rect(
            pr.x0 + x1 * pr.width,  pr.y0 + y1 * pr.height,
            pr.x0 + x2 * pr.width,  pr.y0 + y2 * pr.height,
        )
        crop_data = crop_rect(page, rect, pad_pt=4)
        doc.close()

        pil  = Image.open(io.BytesIO(crop_data))
        name = f"p{page_num}_recovered_{qnum}.png"
        return {
            "page":           page_num,
            "name":           name,
            "source":         "recovered",
            "kind":           str(data.get("label", "visual")),
            "data":           crop_data,
            "width":          pil.width,
            "height":         pil.height,
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
    missing = [r for r in records if r.get("hasImages") and not r.get("images")]
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
st.set_page_config(page_title="Past Paper → Airtable v3",
                   page_icon="📄", layout="wide")
st.title("📄 Past Paper → Airtable v3")
st.caption("PyMuPDF (no merging) + GPT-4V judge + recovery pass · 300 DPI crops")

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

with st.expander("🔧 Debug: inspect drawing extraction for a specific page"):
    debug_pg = st.number_input("Page number to debug", min_value=1, value=8, step=1)
    if st.button("Run debug extraction") and paper_file:
        paper_file.seek(0)
        _, dbg = extract_all_crops(paper_file.read(), debug_page=int(debug_pg))
        st.code("\n".join(dbg) if dbg else "No debug output for this page.")
if st.button("✨ Extract Questions", type="primary",
             disabled=not (paper_file and paper_name and exam_type and OPENAI_KEY)):

    paper_bytes = paper_file.read()
    ms_bytes    = ms_file.read() if ms_file else None
    client      = OpenAI(api_key=OPENAI_KEY)

    with st.status("Extracting…", expanded=True) as status:

        st.write("📎 Extracting visuals (PyMuPDF 300 DPI + GPT-4V judge)…")
        extract_and_judge_visuals.clear()
        images, image_map = extract_and_judge_visuals(OPENAI_KEY, paper_bytes)
        mapped = sum(1 for v in image_map.values()
                     if v.get("questionNumber") not in {"none", ""})
        st.write(f"   {len(images)} visuals kept · {mapped} mapped to questions")

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
                    img_page = img["page"]
                    qs = pq_index.get(img_page, [])
                    if not qs:
                        nearby = sorted(pq_index.keys(),
                                        key=lambda p: abs(p - img_page))
                        if nearby:
                            qs = pq_index[nearby[0]]
                    image_map[nm] = {
                        "questionNumber": qs[-1] if qs else "none",
                        "confidence":     "low",
                        "notes":          "Fallback: nearest question by page",
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

        # Recovery pass
        missing_count = sum(1 for r in records
                            if r.get("hasImages") and not r.get("images"))
        if missing_count:
            st.write(f"🔎 Recovery: {missing_count} hasImages question(s) with no visual…")
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
            still_missing = sum(1 for r   in records
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
