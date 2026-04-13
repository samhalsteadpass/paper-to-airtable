"""
app_v5.py  –  Past Paper → Airtable  (manual capture + AI assignment)
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
import pathlib
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

AT_API         = "https://api.airtable.com/v0"

# ── Auto-save helpers ─────────────────────────────────────────────────────
AUTOSAVE_PATH = pathlib.Path("autosave.json")
PDF_PATH      = pathlib.Path("current_paper.pdf")

def autosave():
    """Write session state to disk so it survives restarts."""
    try:
        save_data = {
            "paper_name":           st.session_state.get("paper_name", ""),
            "exam_type":            st.session_state.get("exam_type", ""),
            "paper_name_for_table": st.session_state.get("paper_name_for_table", ""),
            "pages":                st.session_state.get("pages", []),
            "records":              st.session_state.get("records", []),
            "boxes": {
                str(pn): [
                    {k: v for k, v in b.items() if k != "data"}
                    for b in blist
                ]
                for pn, blist in st.session_state.get("boxes", {}).items()
            },
        }
        AUTOSAVE_PATH.write_text(json.dumps(save_data, ensure_ascii=False))
    except Exception:
        pass  # never crash the app over autosave

def get_pdf() -> bytes | None:
    """Get PDF bytes — from disk if available, else session state."""
    if PDF_PATH.exists():
        return PDF_PATH.read_bytes()
    return st.session_state.get("pdf")

def autorestore():
    """On startup, reload from disk if session state is empty."""
    if "records" in st.session_state:
        return  # already loaded
    if not AUTOSAVE_PATH.exists():
        return
    try:
        save_data = json.loads(AUTOSAVE_PATH.read_text())
        if not save_data.get("records"):
            return
        st.session_state["paper_name"]           = save_data.get("paper_name", "")
        st.session_state["exam_type"]            = save_data.get("exam_type", "")
        st.session_state["paper_name_for_table"] = save_data.get("paper_name_for_table", "")
        st.session_state["pages"]                = save_data.get("pages", [])
        st.session_state["records"]              = save_data.get("records", [])
        restored = {}
        for pn_str, blist in save_data.get("boxes", {}).items():
            restored[int(pn_str)] = [
                {**b, "data": b"", "ai_qnum": b.get("ai_qnum", ""),
                 "ai_conf": b.get("ai_conf", ""), "ai_notes": b.get("ai_notes", "")}
                for b in blist
            ]
        st.session_state["boxes"] = restored
        st.session_state["_autorestore_done"] = True
    except Exception:
        pass
AT_META        = "https://api.airtable.com/v0/meta"
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
  "questionText": "Question text only (not shared preamble)",
  "markAllocation": 4,
  "topic": "Algebra",
  "subtopic": "Quadratics",
  "hasImages": false,
  "imageDescription": "Describe any diagram/graph/table. Empty string if none.",
  "pageNumber": 2
}

Rules:
- For multipart questions (e.g. Q2 with parts 2a, 2b, 2c):
  * Create a SEPARATE row for the parent (e.g. "2") with ONLY the shared
    preamble/context, markAllocation: 0, and hasImages: true if there is a diagram.
  * Then create rows for each sub-part (2a, 2b, 2c) with just their own question text.
  * Do NOT copy the preamble into each sub-part row.
- For standalone questions with no sub-parts, extract as a single row normally.
- markAllocation must be an integer (0 if missing or preamble-only).
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

You are given TWO images:
1. The FULL PAGE from the exam paper
2. A CROPPED region taken from that page (the visual to assign)

Use both images together. The full page shows the surrounding questions and layout,
which makes it easy to see which question the crop belongs to.

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
- Use the full page to identify the question number printed nearest to the crop's position.
- Use the crop itself to confirm with visual clues: labels, axis titles, table headers, figure numbers.
- Choose the single best match from the candidates list.
- If genuinely ambiguous, pick the best candidate but set confidence to low.
- questionNumber must be exactly as shown in the candidates list.
"""


COVER_PROMPT = """Look at this exam paper cover page and extract the paper name and exam type.

Return ONLY raw JSON:
{
  "paperName": "AQA Mathematics Paper 1 (Non-Calculator) 2023",
  "examType": "GCSE"
}

paperName: include the exam board, subject, paper number/name, and year if visible.
examType: one of GCSE, A-Level, AS-Level, IB HL, IB SL, or describe it briefly if none of these fit.
If a field is not clearly visible, make your best guess from context.
"""

def read_cover_page(client: OpenAI, pdf_bytes: bytes) -> tuple[str, str]:
    """Read the first page of a PDF and extract paper name and exam type."""
    page_png = render_page_cached(pdf_bytes, 1, dpi=RENDER_DPI)
    page_pil = Image.open(io.BytesIO(page_png)).convert("RGB")
    content  = [
        {"type": "input_text",  "text": COVER_PROMPT},
        {"type": "input_image",
         "image_url": f"data:image/jpeg;base64,{encode_pil(page_pil)}"},
    ]
    parsed = safe_json_loads(
        call_gpt(client, content, VISION_MODEL, max_tokens=200), {})
    return (
        str(parsed.get("paperName", "") or "").strip(),
        str(parsed.get("examType",  "") or "").strip(),
    )

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
    s = str(v or "").strip()
    s = s.replace(" ", "").replace("(", "").replace(")", "").replace(".", "")
    return s

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
        buf = io.BytesIO()
        w.save(buf)
        w.close()
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
    pb   = page_boxes(pn)
    idx  = len(pb) + 1
    data = crop_from_rel(pdf_bytes, pn, rel)
    pil  = Image.open(io.BytesIO(data))
    b = {
        "page":           pn,
        "idx":            idx,
        "name":           f"p{pn}_box{idx}.png",
        "rel":            rel,
        "data":           data,
        "width":          pil.width,
        "height":         pil.height,
        "questionNumber": qnum,
        "ai_qnum":        "",
        "ai_conf":        "",
        "ai_notes":       "",
        "notes":          notes or "manual",
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
            "questionNumber":         normalise_qnum(r.get("questionNumber", "")),
            "originalQuestionNumber": str(r.get("questionNumber", "") or "").strip(),
            "questionText":           str(r.get("questionText",     "") or ""),
            "markAllocation":         clamp_int(r.get("markAllocation", 0), 0),
            "topic":                  str(r.get("topic",            "") or ""),
            "subtopic":               str(r.get("subtopic",         "") or ""),
            "hasImages":              bool(r.get("hasImages",       False)),
            "imageDescription":       str(r.get("imageDescription", "") or ""),
            "pageNumber":             clamp_int(r.get("pageNumber", 1), 1) + offset,
        } for r in rows]
        return i, fixed, f"Chunk {i+1}/{len(chunks)}: {len(fixed)} questions"

    ordered: dict[int, list] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, rows, msg in [f.result() for f in
                              as_completed([ex.submit(process, i, c)
                                            for i, c in enumerate(chunks)])]:
            ordered[i] = rows
            logs.append(msg)

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

    ms: dict[str, str] = {}
    for i in range(len(chunks)):
        ms.update(ordered.get(i, {}))
    return ms, logs

# ── AI image assignment ───────────────────────────────────────────────────
def candidates_for_page(records: list[dict], pn: int) -> list[dict]:
    # Search same page, then widen window until we have enough candidates
    for radius in (0, 1, 2, 3):
        window = set(range(pn - radius, pn + radius + 1))
        cands  = [r for r in records if clamp_int(r.get("pageNumber")) in window]
        if len(cands) >= 3:
            # Deduplicate while preserving order
            seen: set = set()
            out = []
            for r in cands:
                qn = normalise_qnum(r.get("questionNumber", ""))
                if qn and qn not in seen:
                    out.append(r)
                    seen.add(qn)
            return out[:25]
    # Final fallback: return all records (page numbers may be unreliable)
    return records[:25]

def ai_assign(client: OpenAI, box: dict, records: list[dict],
               pdf_bytes: bytes = None) -> dict:
    cands  = records  # use all records — page filtering was causing misses
    cblock = "\n".join(
        f"- {normalise_qnum(r.get('questionNumber', ''))} | "
        f"p{clamp_int(r.get('pageNumber', 0))} | "
        f"{re.sub(chr(10), ' ', str(r.get('questionText', '') or ''))[:200]}"
        for r in cands
    )
    prompt  = AI_ASSIGN_PROMPT.format(candidates=cblock)
    # Re-crop if data is missing (e.g. after session restore)
    if not box.get("data") and pdf_bytes:
        try:
            box["data"] = crop_from_rel(pdf_bytes, box["page"], box["rel"])
        except Exception:
            pass
    if not box.get("data"):
        return {"questionNumber": "", "confidence": "low",
                "notes": "Image data missing — redraw this box"}
    crop_img = Image.open(io.BytesIO(box["data"])).convert("RGB")
    content = [{"type": "input_text", "text": prompt}]
    # Send full page first so GPT can see layout context
    if pdf_bytes:
        page_png = render_page_cached(pdf_bytes, box["page"], dpi=RENDER_DPI)
        page_pil = Image.open(io.BytesIO(page_png)).convert("RGB")
        content.append({"type": "input_image",
                         "image_url": f"data:image/jpeg;base64,{encode_pil(page_pil)}"})
    # Then the crop
    content.append({"type": "input_image",
                     "image_url": f"data:image/jpeg;base64,{encode_pil(crop_img)}"})
    parsed = safe_json_loads(
        call_gpt(client, content, VISION_MODEL, max_tokens=300), {})

    qn   = normalise_qnum(parsed.get("questionNumber", ""))
    conf = str(parsed.get("confidence", "") or "").strip().lower()
    note = str(parsed.get("notes", "") or "").strip()

    valid = {normalise_qnum(r.get("questionNumber", "")) for r in records}

    # Build an ordered list of all records for tiebreaking
    cand_qnums_ordered = [normalise_qnum(r.get("questionNumber", "")) for r in records]

    def fuzzy_match(ai_qn: str, valid_set: set) -> tuple[str, str]:
        """Return (matched_qnum, note_suffix) or ('', note_suffix) if no match."""
        if not ai_qn:
            return "", "[outside candidate set]"

        # Normalise the AI value the same way records are normalised
        # (already done via normalise_qnum above, but also strip leading zeros/Q)
        def bare(s: str) -> str:
            """Strip leading Q/q then leading zeros, keep at least one char."""
            s = s.lstrip("Qq")
            return s.lstrip("0") or s

        ai_bare = bare(ai_qn)

        # 1. Exact match
        if ai_qn in valid_set:
            return ai_qn, ""

        # 2. Match after stripping leading Q/zeros from AI value
        for v in valid_set:
            if bare(v) == ai_bare:
                return v, f"[matched {ai_qn}→{v}]"

        # 3. AI returned a parent number — find all children
        #    e.g. AI said "7", records have "7a","7b","7c"
        #    Match if stored value starts with ai_bare (after stripping its own zeros)
        def is_child(stored: str, parent_bare: str) -> bool:
            s = bare(stored)
            # child starts with the parent digits and the next char is non-digit
            if not s.startswith(parent_bare):
                return False
            rest = s[len(parent_bare):]
            return len(rest) == 0 or not rest[0].isdigit()

        children = [v for v in cand_qnums_ordered
                    if v in valid_set and is_child(v, ai_bare)]
        # Deduplicate while preserving order
        seen_c: set = set()
        children = [c for c in children if not (c in seen_c or seen_c.add(c))]
        if len(children) == 1:
            return children[0], f"[matched {ai_qn}→{children[0]}]"
        if len(children) > 1:
            first = children[0]
            return first, f"[matched {ai_qn}→{first} (first of {len(children)} sub-parts)]"

        # 4. AI returned a child, record is the parent
        #    e.g. AI said "7a", record is "7"
        parents = [v for v in valid_set if is_child(ai_bare, bare(v))]
        if len(parents) == 1:
            return parents[0], f"[matched {ai_qn}→{parents[0]}]"

        return "", "[outside candidate set]"

    matched, match_note = fuzzy_match(qn, valid)

    if not matched:
        qn   = ""
        conf = "low"
        note = (note + " " if note else "") + match_note
    else:
        if match_note:
            note = (note + " " if note else "") + match_note
        qn = matched

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
        rel    = b["rel"]
        x0, y0 = rel["x"] * w, rel["y"] * h
        x1, y1 = x0 + rel["w"] * w, y0 + rel["h"] * h
        qn     = normalise_qnum(b.get("questionNumber", ""))
        color  = "#e74c3c" if (highlight_qnum and qn == highlight_qnum) else "#e67e22"
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = str(b.get("idx", ""))
        if label:
            draw.rectangle([x0 + 1, y0 - 18, x0 + 26, y0 - 2], fill=color)
            draw.text((x0 + 5, y0 - 17), label, fill="white")
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

def get_existing_fields(token, base_id, table) -> set[str]:
    resp = requests.get(f"{AT_META}/bases/{base_id}/tables",
                        headers=at_headers(token), timeout=60)
    resp.raise_for_status()
    for t in resp.json().get("tables", []):
        if t["name"] == table:
            return {f["name"] for f in t.get("fields", [])}
    return set()

def upload_cloudinary(cloud: str, preset: str, img: dict, paper_name: str = "") -> str | None:
    if not img.get("data"):
        return None  # skip boxes with no image data
    # Include paper name in public_id so different papers never share cached images
    base = img["name"].rsplit(".", 1)[0].replace(".", "_")
    safe_paper = re.sub(r"[^a-zA-Z0-9_-]", "_", paper_name)[:40] if paper_name else "paper"
    pid  = f"{safe_paper}_{base}"
    resp = requests.post(
        f"{CLOUDINARY_API}/{cloud}/image/upload",
        data={"upload_preset": preset, "public_id": pid},
        files={"file": (img["name"], img["data"], "image/png")},
        timeout=120,
    )
    if resp.ok:
        return resp.json().get("secure_url")
    # Embed error in return so caller can log it
    try:
        err = resp.json().get("error", {}).get("message", resp.text[:200])
    except Exception:
        err = resp.text[:200]
    return f"ERROR:{err}"

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

# ═════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Past Paper → Airtable", page_icon="📄", layout="wide")

# Restore from disk on every fresh session start
autorestore()
if st.session_state.pop("_autorestore_done", False):
    st.toast("♻️ Session restored from autosave", icon="💾")

st.title("📄 Past Paper → Airtable")
st.caption("Draw boxes to capture visuals · AI suggests question assignment · Sync to Airtable")

OPENAI_KEY = get_secret("OPENAI_API_KEY")
AT_TOKEN   = get_secret("AIRTABLE_TOKEN")
AT_BASE    = get_secret("AIRTABLE_BASE_ID")
CLD_CLOUD  = get_secret("CLOUDINARY_CLOUD_NAME")
CLD_PRESET = get_secret("CLOUDINARY_UPLOAD_PRESET")

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

    default_table = st.session_state.get("paper_name_for_table", "Questions")
    AT_TABLE    = st.text_input("Table name", value=default_table,
                                help="Each paper gets its own table. Auto-set from paper name.")
    AUTO_ASSIGN = st.checkbox("Auto-assign high-confidence AI suggestions", value=True)
    st.divider()
    st.markdown("**💾 Save / Load session**")
    st.caption("Save your progress to a file and reload it later — survives restarts.")

    if st.button("💾 Save session", width='stretch'):
        save_data = {
            "paper_name":            st.session_state.get("paper_name", ""),
            "exam_type":             st.session_state.get("exam_type", ""),
            "paper_name_for_table":  st.session_state.get("paper_name_for_table", ""),
            "pages":                 st.session_state.get("pages", []),
            "records":               st.session_state.get("records", []),
            "boxes":                 {
                str(pn): [
                    {k: v for k, v in b.items() if k != "data"}
                    for b in blist
                ]
                for pn, blist in st.session_state.get("boxes", {}).items()
            },
        }
        st.session_state["_save_json"] = json.dumps(save_data, indent=2, ensure_ascii=False)
        st.success("Session saved — download below.")

    if "_save_json" in st.session_state:
        pn = st.session_state.get("paper_name", "session") or "session"
        st.download_button(
            "⬇ Download session file",
            data=st.session_state["_save_json"].encode(),
            file_name=f"{pn}_session.json",
            mime="application/json",
            width='stretch',
        )

    uploaded_session = st.file_uploader("📂 Load session file", type="json",
                                         key="session_upload")
    if uploaded_session:
        try:
            save_data = json.loads(uploaded_session.read())
            st.session_state["paper_name"]           = save_data.get("paper_name", "")
            st.session_state["exam_type"]            = save_data.get("exam_type", "")
            st.session_state["paper_name_for_table"] = save_data.get("paper_name_for_table", "")
            st.session_state["pages"]                = save_data.get("pages", [])
            st.session_state["records"]              = save_data.get("records", [])
            # Restore boxes without image data (images re-cropped on demand)
            restored_boxes = {}
            for pn_str, blist in save_data.get("boxes", {}).items():
                restored_boxes[int(pn_str)] = [
                    {**b, "data": b"", "ai_qnum": b.get("ai_qnum",""),
                     "ai_conf": b.get("ai_conf",""), "ai_notes": b.get("ai_notes","")}
                    for b in blist
                ]
            st.session_state["boxes"] = restored_boxes
            st.success(f"✅ Session loaded: {save_data.get('paper_name','')}")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load session: {e}")

    st.divider()
    st.markdown("**Required Airtable fields**")
    for name, ftype in AT_FIELDS:
        st.markdown(f"- `{name}` — {ftype}")

# ── 1. Upload ──────────────────────────────────────────────────────────────
st.subheader("1 · Upload PDFs")
c1, c2 = st.columns(2)
with c1:
    paper_file = st.file_uploader("Past paper PDF *", type="pdf")
    paper_name = st.text_input(
        "Paper name",
        value=st.session_state.get("paper_name", ""),
        placeholder="Auto-filled from cover page",
    )
    exam_type = st.text_input(
        "Exam type",
        value=st.session_state.get("exam_type", ""),
        placeholder="Auto-filled from cover page",
    )
with c2:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    ms_file = st.file_uploader("Mark scheme PDF (optional)", type="pdf")

if st.button("Load PDF", disabled=not (paper_file and OPENAI_KEY)):
    paper_file.seek(0)
    pdf    = paper_file.read()
    client = OpenAI(api_key=OPENAI_KEY)

    with st.spinner("Reading cover page…"):
        detected_name, detected_type = read_cover_page(client, pdf)

    PDF_PATH.write_bytes(pdf)
    st.session_state.pop("pdf", None)   # don't store PDF in session state
    render_page_cached.clear()          # free cached page renders from old PDF
    st.session_state["pages"]      = get_question_pages(pdf)
    st.session_state["paper_name"] = detected_name
    st.session_state["exam_type"]  = detected_type
    st.session_state.pop("records", None)
    st.session_state["boxes"] = {}
    st.session_state.pop("sync_log", None)
    st.session_state.pop("_save_json", None)
    st.session_state["sel_page_idx"] = 0
    safe_name = re.sub(r"[^a-zA-Z0-9 _-]", "", detected_name).strip() or "Questions"
    st.session_state["paper_name_for_table"] = safe_name
    # Clear autosave so old paper doesn't restore on next restart
    try:
        AUTOSAVE_PATH.unlink(missing_ok=True)
    except Exception:
        pass
    st.success(
        f"Loaded — {len(st.session_state['pages'])} question pages found.  "
        f"Detected: **{detected_name}** · **{detected_type}**"
    )
    st.rerun()

# Keep paper_name / exam_type in sync with what user may have edited
paper_name = st.session_state.get("paper_name", paper_name)
exam_type  = st.session_state.get("exam_type",  exam_type)

# ── 2. Extract ─────────────────────────────────────────────────────────────
if st.session_state.get("pages") or PDF_PATH.exists():
    st.subheader("2 · Extract questions + mark scheme")

    if st.button("✨ Extract", type="primary",
                 disabled=not (paper_name and exam_type and OPENAI_KEY)):
        pdf      = get_pdf()
        ms_bytes = ms_file.read() if ms_file else None
        client   = OpenAI(api_key=OPENAI_KEY)

        with st.status("Extracting…", expanded=True) as status:
            st.write("🤖 Questions…")
            questions, ql = extract_questions(client, pdf)
            for line in ql:
                st.write(f"   {line}")

            ms_map: dict[str, str] = {}
            if ms_bytes:
                st.write("🤖 Mark scheme…")
                ms_map, ml = extract_markscheme(client, ms_bytes)
                for line in ml:
                    st.write(f"   {line}")

            records = []
            for q in questions:
                qn = normalise_qnum(q.get("questionNumber", ""))
                records.append({
                    "questionNumber":          qn,
                    "originalQuestionNumber":  q.get("originalQuestionNumber", qn),
                    "questionText":            q.get("questionText",      ""),
                    "markAllocation":          clamp_int(q.get("markAllocation", 0)),
                    "topic":                   q.get("topic",             ""),
                    "subtopic":                q.get("subtopic",          ""),
                    "markSchemeAnswer":         ms_map.get(qn,            ""),
                    "imageDescription":         q.get("imageDescription", ""),
                    "hasImages":                bool(q.get("hasImages",   False)),
                    "pageNumber":               clamp_int(q.get("pageNumber", 1), 1),
                    "paperName":                paper_name,
                    "examType":                 exam_type,
                    "imageMappingConfidence":   "",
                    "imageMappingNotes":        "",
                    "images":                   [],
                })

            st.session_state["records"] = records
            autosave()
            status.update(label=f"✅ {len(records)} questions extracted",
                          state="complete")

# ── 3. Capture ─────────────────────────────────────────────────────────────
if st.session_state.get("pages") or PDF_PATH.exists():
    st.subheader("3 · Capture visuals")
    st.caption("① Optional: type a crop label in the left panel  "
               "② Click two corners on the page image to draw a box  "
               "③ AI instantly suggests which question it belongs to")

    pdf     = get_pdf()
    pages   = st.session_state.get("pages", [])
    records = st.session_state.get("records", [])

    if not pages:
        st.info("Load a PDF first.")
    else:
        # ── Page selection state ──────────────────────────────────────────
        if "sel_page_idx" not in st.session_state:
            st.session_state["sel_page_idx"] = 0
        idx      = max(0, min(st.session_state["sel_page_idx"], len(pages) - 1))
        sel_page = pages[idx]
        st.session_state["sel_page"] = sel_page

        left, right = st.columns([1, 2])

        with left:
            # ── Prev / Next navigation ────────────────────────────────────
            nav1, nav2, nav3 = st.columns([1, 2, 1])
            with nav1:
                if st.button("◀ Prev", disabled=idx == 0):
                    st.session_state[f"clicks_{sel_page}"] = []
                    st.session_state["sel_page_idx"] = idx - 1
                    st.session_state["_nav_ts"] = time.time()
                    st.rerun()
            with nav2:
                st.markdown(
                    f"<div style='text-align:center;padding:6px 0;font-weight:500'>"
                    f"Page {sel_page} of {pages[-1]} "
                    f"<span style='color:var(--color-text-secondary);font-weight:400'>"
                    f"({idx + 1}/{len(pages)})</span></div>",
                    unsafe_allow_html=True,
                )
            with nav3:
                if st.button("Next ▶", disabled=idx == len(pages) - 1):
                    st.session_state[f"clicks_{sel_page}"] = []
                    st.session_state["sel_page_idx"] = idx + 1
                    st.session_state["_nav_ts"] = time.time()
                    st.rerun()

            # ── Go to page ────────────────────────────────────────────────
            goto_col1, goto_col2 = st.columns([2, 1])
            with goto_col1:
                goto_val = st.number_input(
                    "Go to page",
                    min_value=pages[0],
                    max_value=pages[-1],
                    value=sel_page,
                    step=1,
                    key="goto_page_input",
                    label_visibility="collapsed",
                )
            with goto_col2:
                if st.button("Go", key="goto_page_btn"):
                    target = min(pages, key=lambda p: abs(p - int(goto_val)))
                    new_idx = pages.index(target)
                    if new_idx != idx:
                        st.session_state[f"clicks_{sel_page}"] = []
                        st.session_state["sel_page_idx"] = new_idx
                        st.session_state["_nav_ts"] = time.time()
                        st.rerun()

            # ── Assignment mode ───────────────────────────────────────────
            mode        = st.radio("Assignment mode", ["AI suggest", "Manual"],
                                   horizontal=True,
                                   help="AI suggest: GPT picks the question automatically. "
                                        "Manual: you pick from the list below.")
            manual_qnum = ""
            if mode == "Manual":
                if records:
                    qnums       = list(dict.fromkeys(
                        normalise_qnum(r.get("questionNumber", "")) for r in records))
                    manual_qnum = st.selectbox(
                        "Assign box to question",
                        qnums,
                        help="Select the question this crop belongs to before drawing the box.",
                    )
                else:
                    st.info("Run Extract first so questions are available to assign.")

            # ── Notes ─────────────────────────────────────────────────────
            st.divider()
            st.markdown("**📝 Crop label** *(optional)*")
            st.caption("Type a short label, then draw the box on the right. "
                       "The label is saved with the crop.")
            st.text_input(
                "Label for next crop",
                key="notes_input",
                placeholder="e.g. cone diagram · price table · graph Q3",
                label_visibility="collapsed",
            )
            st.divider()

            # ── Current page boxes ────────────────────────────────────────
            pb = page_boxes(sel_page)
            st.markdown(f"**{len(pb)} box(es) on this page**")
            for b in pb:
                qn    = b.get("questionNumber", "")
                ai_qn = b.get("ai_qnum", "")
                ai_cf = b.get("ai_conf", "")
                cap   = (f"→ Q{qn}" if qn
                         else f"→ AI: Q{ai_qn} ({ai_cf})" if ai_qn
                         else "unassigned")
                img_col, del_col = st.columns([5, 1])
                with img_col:
                    if b.get("data"):
                        st.image(b["data"], caption=f"Box {b['idx']} {cap}",
                                 width='stretch')
                    else:
                        st.caption(f"Box {b['idx']} {cap} *(no preview — will re-crop on sync)*")
                with del_col:
                    if st.button("🗑️", key=f"del_{sel_page}_{b['idx']}",
                                 help=f"Delete box {b['idx']}"):
                        new_pb = [x for x in pb if x["idx"] != b["idx"]]
                        set_page_boxes(sel_page, new_pb)
                        reindex(sel_page)
                        st.session_state[f"clicks_{sel_page}"] = []
                        st.rerun()

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Undo last", disabled=not pb):
                    set_page_boxes(sel_page, pb[:-1])
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
                key=f"canvas_{idx}_{sel_page}_{len(pb)}_{mode}_{manual_qnum}_{st.session_state.get('_nav_ts', 0)}",
            )

            if click:
                cx, cy = int(click["x"]), int(click["y"])
                clicks = st.session_state[ckey]
                pt     = (cx, cy)

                if not clicks or clicks[-1] != pt:
                    clicks.append(pt)
                if len(clicks) > 2:
                    clicks = clicks[-2:]
                st.session_state[ckey] = clicks

                if len(clicks) == 2:
                    (x1, y1), (x2, y2) = clicks
                    bw = abs(x2 - x1)
                    bh = abs(y2 - y1)

                    if bw > 5 and bh > 5:
                        rel = {
                            "x": min(x1, x2) / dw,
                            "y": min(y1, y2) / dh,
                            "w": bw / dw,
                            "h": bh / dh,
                        }
                        qn_for_box = normalise_qnum(manual_qnum) if mode == "Manual" else ""
                        box = add_box(
                            pdf, sel_page, rel,
                            qnum=qn_for_box,
                            notes=st.session_state.get("notes_input", "").strip() or "manual",
                        )

                        if mode == "AI suggest" and OPENAI_KEY and not records:
                            st.warning("⚠️ No questions extracted yet — run Extract first for AI assignment. Box saved without assignment.")
                        elif mode == "AI suggest" and OPENAI_KEY and records:
                            try:
                                client = OpenAI(api_key=OPENAI_KEY)
                                result = ai_assign(client, box, records, pdf_bytes=pdf)
                                pb2    = page_boxes(sel_page)
                                for b in pb2:
                                    if b["idx"] == box["idx"]:
                                        b["ai_qnum"]  = result["questionNumber"]
                                        b["ai_conf"]  = result["confidence"]
                                        b["ai_notes"] = result["notes"]
                                        if (AUTO_ASSIGN
                                                and result["questionNumber"]
                                                and result["confidence"] == "high"):
                                            b["questionNumber"] = result["questionNumber"]
                                set_page_boxes(sel_page, pb2)
                                st.toast(
                                    f"AI: Q{result['questionNumber'] or 'none'} "
                                    f"({result['confidence']})")
                            except Exception as e:
                                st.warning(f"AI assignment failed: {e}")

                        autosave()
                        st.session_state[ckey] = []
                        st.rerun()
                    else:
                        st.warning("Box too small — try again.")
                        st.session_state[ckey] = []
                else:
                    st.caption(
                        f"First corner set at {clicks[0]}. Click the second corner.")



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
        # Re-crop any boxes that lost their image data (e.g. after session restore)
        _pdf = get_pdf()
        if _pdf:
            _store = boxes()
            for _pn in _store:
                for _b in _store[_pn]:
                    if not _b.get("data"):
                        try:
                            _b["data"] = crop_from_rel(_pdf, _b["page"], _b["rel"])
                        except Exception:
                            pass
                set_page_boxes(_pn, _store[_pn])

        with st.status("Assigning…", expanded=True) as status:
            store = boxes()
            for pn in sorted(store):
                pb = store[pn]
                for b in pb:
                    if normalise_qnum(b.get("questionNumber", "")):
                        continue
                    try:
                        r = ai_assign(client, b, records, pdf_bytes=get_pdf())
                        b["ai_qnum"]  = r["questionNumber"]
                        b["ai_conf"]  = r["confidence"]
                        b["ai_notes"] = r["notes"]
                        if (AUTO_ASSIGN
                                and r["questionNumber"]
                                and r["confidence"] == "high"):
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
        "Page":           b["page"],
        "Box":            b["idx"],
        "Name":           b["name"],
        "Final Q #":      b.get("questionNumber", ""),
        "AI suggested Q": b.get("ai_qnum",        ""),
        "AI confidence":  b.get("ai_conf",         ""),
        "AI notes":       b.get("ai_notes",        ""),
        "Notes":          b.get("notes",           ""),
    } for b in ab]

    edited = st.data_editor(pd.DataFrame(rows),
                             width='stretch',
                             num_rows="fixed", height=400)

    if st.button("Save assignments"):
        store      = boxes()
        update_map = {
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
        autosave()
        st.success("Assignments saved.")
        st.rerun()

# ── Merge images into records ──────────────────────────────────────────────
if "records" in st.session_state:
    records  = st.session_state["records"]
    ab       = all_boxes()
    q_imgs:  dict[str, list[str]] = {}
    q_notes: dict[str, list[str]] = {}

    for b in ab:
        qn = normalise_qnum(b.get("questionNumber", ""))
        if not qn:
            continue
        q_imgs.setdefault(qn, []).append(b["name"])
        ai_part = (f" | AI {b['ai_qnum']} ({b['ai_conf']})"
                   if b.get("ai_qnum") else "")
        q_notes.setdefault(qn, []).append(
            f"{b['name']}: {b.get('notes', '')}{ai_part}")

    def bare_num(s: str) -> str:
        s = s.lstrip("Qq")
        return s.lstrip("0") or s

    def is_child_of(child_qn: str, parent_qn: str) -> bool:
        c = bare_num(child_qn); p = bare_num(parent_qn)
        if not c.startswith(p): return False
        rest = c[len(p):]
        return len(rest) > 0 and not rest[0].isdigit()

    for r in records:
        qn    = normalise_qnum(r.get("questionNumber", ""))
        imgs  = list(q_imgs.get(qn, []))
        notes = list(q_notes.get(qn, []))
        # Propagate images from parent to children
        # e.g. box assigned to "2" also appears on "2a", "2b", "2c"
        for parent_qn, parent_imgs in q_imgs.items():
            if parent_qn != qn and is_child_of(qn, parent_qn):
                for img in parent_imgs:
                    if img not in imgs:
                        imgs.append(img)
                for note in q_notes.get(parent_qn, []):
                    prop = f"{note} [from Q{parent_qn}]"
                    if prop not in notes:
                        notes.append(prop)
        r["images"]                 = imgs
        r["hasImages"]              = bool(imgs) or r.get("hasImages", False)
        r["imageMappingConfidence"] = (
            "manual+ai" if any("AI " in n for n in notes)
            else "manual" if imgs else "")
        r["imageMappingNotes"]      = "\n".join(notes)

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

    edited_df = st.data_editor(df, width='stretch',
                                num_rows="dynamic", height=420)

    for i, row in edited_df.iterrows():
        if i < len(records):
            records[i].update({
                "questionNumber":   normalise_qnum(row["Q #"]),
                "questionText":     row["Question Text"],
                "markAllocation":   clamp_int(row["Marks"], 0),
                "topic":            row["Topic"],
                "subtopic":         row["Subtopic"],
                "markSchemeAnswer": row["Mark Scheme"],
                "imageDescription": row["Image Desc."],
                "hasImages":        bool(row["Has Images"]),
                "pageNumber":       clamp_int(row["Page"], 1),
                "images":           [x.strip() for x in
                                     str(row["Images"] or "").split(",")
                                     if x.strip()],
            })

    if ab:
        with st.expander(f"🖼 Captured visuals ({len(ab)})"):
            cols = st.columns(4)
            for i, b in enumerate(ab):
                with cols[i % 4]:
                    qn = b.get("questionNumber", "")
                    if b.get("data"):
                        st.image(b["data"],
                                 caption=f"{b['name']} → Q{qn or '?'}",
                                 width='stretch')
                    else:
                        st.caption(f"{b['name']} → Q{qn or '?'} *(no preview)*")

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
                st.session_state["do_sync"] = True
                st.session_state.pop("sync_log", None)
                st.rerun()

            if st.session_state.get("do_sync"):
                st.session_state["do_sync"] = False
                log_lines: list[str] = []

                def log(m):
                    log_lines.append(m)

                # Re-crop all boxes from current PDF so we always have fresh image data
                _sync_pdf = get_pdf()
                if _sync_pdf:
                    _store = boxes()
                    _recrop_ok = _recrop_fail = 0
                    for _pn in _store:
                        for _b in _store[_pn]:
                            if not _b.get("data"):
                                try:
                                    _b["data"] = crop_from_rel(_sync_pdf, _b["page"], _b["rel"])
                                    _recrop_ok += 1
                                except Exception as _e:
                                    _recrop_fail += 1
                        set_page_boxes(_pn, _store[_pn])
                    if _recrop_ok or _recrop_fail:
                        log(f"Re-cropped {_recrop_ok} boxes ({_recrop_fail} failed)")
                else:
                    log("⚠️ No PDF on disk — image crops may be missing. Re-upload the PDF and sync again.")

                # Always re-read and re-merge records fresh so images are attached
                _records  = st.session_state.get("records", [])
                _ab       = all_boxes()
                _q_imgs:  dict[str, list[str]] = {}
                _q_notes: dict[str, list[str]] = {}
                for _b in _ab:
                    _qn = normalise_qnum(_b.get("questionNumber", ""))
                    if not _qn:
                        continue
                    _q_imgs.setdefault(_qn, []).append(_b["name"])
                    _ai_part = (f" | AI {_b['ai_qnum']} ({_b['ai_conf']})"
                                if _b.get("ai_qnum") else "")
                    _q_notes.setdefault(_qn, []).append(
                        f"{_b['name']}: {_b.get('notes', '')}{_ai_part}")
                def _bare(s):
                    s = s.lstrip("Qq"); return s.lstrip("0") or s
                def _is_child_of(c, p):
                    cb = _bare(c); pb = _bare(p)
                    if not cb.startswith(pb): return False
                    rest = cb[len(pb):]
                    return len(rest) > 0 and not rest[0].isdigit()

                for _r in _records:
                    _qn    = normalise_qnum(_r.get("questionNumber", ""))
                    _imgs  = list(_q_imgs.get(_qn, []))
                    _notes = list(_q_notes.get(_qn, []))
                    for _pqn, _pimgs in _q_imgs.items():
                        if _pqn != _qn and _is_child_of(_qn, _pqn):
                            for _img in _pimgs:
                                if _img not in _imgs: _imgs.append(_img)
                            for _note in _q_notes.get(_pqn, []):
                                _prop = f"{_note} [from Q{_pqn}]"
                                if _prop not in _notes: _notes.append(_prop)
                    _r["images"]                 = _imgs
                    _r["hasImages"]              = bool(_imgs) or _r.get("hasImages", False)
                    _r["imageMappingConfidence"] = (
                        "manual+ai" if any("AI " in n for n in _notes)
                        else "manual" if _imgs else "")
                    _r["imageMappingNotes"]      = "\n".join(_notes)
                records = _records
                ab      = _ab
                log(f"Syncing {len(records)} records with {len(ab)} visuals…")

                img_url_map: dict[str, str] = {}
                if ab and CLD_CLOUD and CLD_PRESET:
                    log(f"Uploading {len(ab)} visuals to Cloudinary…")

                    _paper = st.session_state.get("paper_name", "")
                    def _upload(b, _p=_paper):
                        return b["name"], upload_cloudinary(CLD_CLOUD, CLD_PRESET, b, _p)

                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                        for name, url in [f.result() for f in
                                           as_completed([ex.submit(_upload, b)
                                                         for b in ab])]:
                            if url and not str(url).startswith("ERROR:"):
                                img_url_map[name] = url
                                log(f"  ✅ {name}")
                            elif url:
                                log(f"  ❌ {name} failed: {url[6:]}")
                            else:
                                log(f"  ❌ {name} failed: no response")
                elif ab:
                    log("⚠️ Cloudinary not configured — images will not be attached.")

                try:
                    # Ensure table exists first, then fetch fields separately
                    ensure_table(AT_TOKEN, AT_BASE, AT_TABLE)
                    # Retry field fetch — newly created tables may need a moment
                    existing = set()
                    for _attempt in range(3):
                        existing = get_existing_fields(AT_TOKEN, AT_BASE, AT_TABLE)
                        if existing:
                            break
                        time.sleep(1)
                    log(f"Table fields found: {len(existing)}")

                    _pn   = st.session_state.get("paper_name", paper_name)
                    _et   = st.session_state.get("exam_type",  exam_type)

                    payload  = []
                    for r in records:
                        urls   = [img_url_map[n] for n in r.get("images", [])
                                  if n in img_url_map]
                        all_fields = {
                            "Question Number":          r.get("originalQuestionNumber", r.get("questionNumber", "")),
                            "Question Text":            r.get("questionText",            ""),
                            "Mark Allocation":          clamp_int(r.get("markAllocation", 0)),
                            "Topic":                    r.get("topic",                   ""),
                            "Subtopic":                 r.get("subtopic",                ""),
                            "Mark Scheme Answer":       r.get("markSchemeAnswer",        ""),
                            "Image Description":        r.get("imageDescription",        ""),
                            "Has Images":               bool(urls or r.get("hasImages",  False)),
                            "Images":                   [{"url": u} for u in urls],
                            "Paper Name":               r.get("paperName",    _pn),
                            "Exam Type":                r.get("examType",     _et),
                            "Page Number":              clamp_int(r.get("pageNumber", 1), 1),
                            "Image Mapping Confidence": r.get("imageMappingConfidence",  ""),
                            "Image Mapping Notes":      r.get("imageMappingNotes",       ""),
                        }
                        # Only filter if we actually got fields back
                        if existing:
                            all_fields = {k: v for k, v in all_fields.items() if k in existing}
                        payload.append({"fields": all_fields})

                    log(f"Pushing {len(payload)} records…")
                    # Push directly — ensure_table already called above
                    url = f"{AT_API}/{AT_BASE}/{requests.utils.quote(AT_TABLE, safe='')}"
                    created = []
                    for batch in chunk_list(payload, 10):
                        resp = requests.post(url, headers=at_headers(AT_TOKEN),
                                             json={"records": batch}, timeout=60)
                        if not resp.ok:
                            raise RuntimeError(f"Airtable {resp.status_code}: {resp.text[:400]}")
                        created.extend(resp.json().get("records", []))
                    log(f"✅ {len(created)} records synced!")
                except Exception as e:
                    log(f"❌ Sync failed: {e}")

                st.session_state["sync_log"] = log_lines

            if "sync_log" in st.session_state:
                st.text("\n".join(st.session_state["sync_log"]))
                st.markdown(f"[Open in Airtable →](https://airtable.com/{AT_BASE})")
