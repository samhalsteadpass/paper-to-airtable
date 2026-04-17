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
import tempfile
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
CHUNK_PAGES       = 3
MAX_WORKERS       = 2
MAX_RETRIES       = 4
BASE_BACKOFF      = 2
IMAGE_MAX_SIZE    = (1200, 1200)
JPEG_QUALITY      = 70
RENDER_DPI        = 150
EXTRACT_DPI       = 150
CANVAS_MAX_WIDTH  = 900

AT_API         = "https://api.airtable.com/v0"

# ── Auto-save helpers ─────────────────────────────────────────────────────
def autosave():
    """No-op — manual save/load handles persistence."""
    pass




def _get_pdf_path() -> str | None:
    """Return the temp file path for this session's PDF, or None if not loaded."""
    return st.session_state.get("_pdf_tmp_path")

def _set_pdf(pdf_bytes: bytes) -> str:
    """Write PDF bytes to a unique temp file. Returns the path."""
    # Delete old temp file if it exists
    old = st.session_state.get("_pdf_tmp_path")
    if old:
        try:
            import os
            os.unlink(old)
        except Exception:
            pass
    # Write to a new unique temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf_bytes)
    tmp.close()
    st.session_state["_pdf_tmp_path"] = tmp.name
    return tmp.name

def get_pdf() -> bytes | None:
    """Read PDF bytes from the session's temp file."""
    path = _get_pdf_path()
    if path:
        try:
            import os
            if os.path.exists(path):
                return open(path, "rb").read()
        except Exception:
            pass
    return None

AT_META        = "https://api.airtable.com/v0/meta"


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
- For multipart questions that have shared context/preamble printed BEFORE the
  sub-parts (e.g. a scenario, diagram, or data that all sub-parts refer to):
  * Create a SEPARATE parent row (e.g. "7") with ONLY that shared text,
    markAllocation: 0, and hasImages: true if a diagram/table/image is present.
  * Then create rows for each sub-part (7a, 7b) with just their own question text.
  * Do NOT copy the preamble text into the sub-part rows.

  EXAMPLE — this paper's Q7:
  Parent row: questionNumber "7", questionText "120 people visit a maze.
  80 are children, the rest are adults. At the start of the maze you can turn
  left or right. 45 children turn left. 75 people in total turn left.",
  markAllocation 0, hasImages true (frequency tree diagram)
  Child row: questionNumber "7a", questionText "Complete the frequency tree.",
  markAllocation 4
  Child row: questionNumber "7b", questionText "What fraction of the children
  turn left? Give your answer in its simplest form.", markAllocation 2

- If a multipart question has NO shared preamble (each sub-part is fully
  self-contained with its own complete question text), do NOT create a parent
  row — just extract each sub-part directly. Example: Q20 has 20a, 20b, 20c
  each with their own complete instruction, so extract three rows with no parent.
- For standalone single-part questions (even if they include context or a
  diagram before a single task), extract as ONE row with ALL the context
  included in questionText. Do NOT split into a parent + child.
  Example: Q26 ("A large circle and a small circle are shown...
  Work out the shaded area.") is ONE question — extract as a single row.
- A parent row is ONLY valid when: (1) there are 2+ answerable sub-parts AND
  (2) there is shared text/diagram that ALL sub-parts refer back to.
- For AQA-style dot-numbered questions (e.g. 01.1, 01.2, 03.3): NEVER create a
  parent row. Each question like "03.3" is fully standalone — any figure, image,
  or context printed before it belongs IN that question's questionText, not as a
  separate parent. Even if "Figure 2 is an image of..." appears above "03.3",
  include it in 03.3's questionText directly.
- NEVER invent a preamble that is not printed in the paper.
- questionNumber must be copied EXACTLY as printed on the paper — preserve leading zeros, dots, and spacing. e.g. "01.1" not "1.1", "02.3a" not "2.3a".
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


COVER_PROMPT = """Look at this exam paper cover page and extract the following fields.

Return ONLY raw JSON:
{
  "paperName": "AQA Mathematics Paper 1 (Non-Calculator) 2023",
  "examBoard": "AQA",
  "subject": "Mathematics",
  "level": "GCSE",
  "year": "2024",
  "paperNumber": "1"
}

paperName: full name including board, subject, paper number/name, and year.
examBoard: e.g. AQA, Edexcel, OCR, WJEC, Cambridge, IB. Leave empty if unclear.
subject: e.g. Mathematics, Physics, English Literature.
level: one of GCSE, A-Level, AS-Level, IB HL, IB SL, or describe briefly if none fit.
year: 4-digit year visible on the paper. Leave empty if not visible.
paperNumber: the paper number (e.g. 1, 2, 3) or name (e.g. "Non-Calculator"). Leave empty if not applicable.
If a field is not clearly visible, make your best guess from context.
"""

NOVA_CLASSIFY_PROMPT = """You are formatting an exam question for an e-learning platform called Nova.

Classify the question and return a JSON object with ALL required fields for the chosen type.

Question JSON:
QUESTION_DATA_PLACEHOLDER

━━━ STEP 1: Pick exactly one Nova type ━━━

- simple          → ONE single unambiguous number or word. Auto-marked exactly.
- multiple_choice → question has explicit A/B/C/D options OR is true/false.
- multiple_answer → needs 2–4 separate answer boxes, each a single unambiguous value.
- fraction        → answer must be expressed as a fraction. Use when question says "as a fraction", "in its simplest form", "what fraction", or the answer is naturally a fraction. NEVER use simple for these. Mixed numbers (e.g. 1½) must also be fraction type.
- fill_in_blank   → sentence with dropdown gaps.
- essay           → ANY question where the answer could be interpreted multiple ways, is an expression/formula (e.g. 6m+11, 3cd), is a list of values (e.g. 1,2,4,5), involves equivalent forms (e.g. 0.7 or 3/4 or 80%), requires showing working, or uses explain/describe/evaluate/justify/prove/give a reason.
- physical        → question requires physical interaction with the paper that CANNOT be done digitally: drawing, sketching, circling, annotating a diagram, completing a graph by hand, drawing arrows, labelling with lines, or any task where the student must mark on the paper itself.

DECISION RULES — apply strictly:
1. Explicit A/B/C/D options or true/false → multiple_choice
2. Requires drawing, circling, annotating, sketching, or marking on the paper → physical
3. "explain", "describe", "evaluate", "justify", "show that", "prove", "give a reason" → essay
4. Answer is or could be expressed as a fraction / mixed number → fraction
5. Answer is an algebraic expression, formula, or list of values → essay
6. Answer has multiple equivalent valid forms (e.g. ÷6 or ×1/6) → essay
7. Exactly 2–4 clearly separate numerical answers needed → multiple_answer
8. Single unambiguous number or word → simple
9. When in doubt → essay (AI marking is safer than wrong auto-marking)

━━━ STEP 2: Return ONLY this JSON — fill in ALL fields ━━━

For type = simple:
{
  "novaType": "simple",
  "friendlyName": "PAPER_NAME_PLACEHOLDER Q QNUM_PLACEHOLDER",
  "body": "<full question text, maths in [latex]...[/latex]>",
  "writtenSolution": "<full worked solution from mark scheme, or empty string>",
  "marks": <integer>,
  "difficulty": 1,
  "answerPrefix": "<text before answer box in [latex] if it contains maths, or empty string>",
  "answer": "<single unambiguous number only — e.g. '48' or '10^4' or '-8'. NEVER a fraction, list, expression, or ambiguous value>",
  "answerUnit": "<unit abbreviation in [latex]\\text{...} if needed, e.g. '[latex]\\text{cm}[/latex]' or '[latex]\\text{km}[/latex]', or empty string>"
}

For type = multiple_choice:
{
  "novaType": "multiple_choice",
  "friendlyName": "PAPER_NAME_PLACEHOLDER Q QNUM_PLACEHOLDER",
  "body": "<full question text>",
  "writtenSolution": "<correct answer explanation>",
  "marks": <integer>,
  "difficulty": 1,
  "style": "List",
  "options": [
    {"text": "<option A>", "correct": false},
    {"text": "<option B>", "correct": true},
    {"text": "<option C>", "correct": false},
    {"text": "<option D>", "correct": false}
  ]
}

For type = multiple_answer:
{
  "novaType": "multiple_answer",
  "friendlyName": "PAPER_NAME_PLACEHOLDER Q QNUM_PLACEHOLDER",
  "body": "<full question text>",
  "writtenSolution": "<worked solution>",
  "marks": <integer>,
  "difficulty": 1,
  "requireSpecificOrder": false,
  "answers": [
    {"prefix": "[latex]x=[/latex]", "answer": "<value>", "suffix": ""},
    {"prefix": "[latex]y=[/latex]", "answer": "<value>", "suffix": ""}
  ]
}

For type = fraction:
{
  "novaType": "fraction",
  "friendlyName": "PAPER_NAME_PLACEHOLDER Q QNUM_PLACEHOLDER",
  "body": "<full question text — must state answer in simplest form>",
  "writtenSolution": "<worked solution>",
  "marks": <integer>,
  "difficulty": 1,
  "answerLabel": "<text before fraction, e.g. 'Answer =' or empty string>",
  "numerator": "<top number>",
  "denominator": "<bottom number>"
}

For type = fill_in_blank:
{
  "novaType": "fill_in_blank",
  "friendlyName": "PAPER_NAME_PLACEHOLDER Q QNUM_PLACEHOLDER",
  "body": "<preamble text shown before the sentence>",
  "writtenSolution": "",
  "marks": <integer>,
  "difficulty": 1,
  "preamble": "<same as body>",
  "blankContent": "<sentence with [blank] where each dropdown goes>",
  "blanks": [
    {"options": ["opt1", "opt2", "opt3", "opt4"], "correct": "opt2", "marks": 1, "writtenSolution": "opt2"}
  ]
}

For type = essay:
{
  "novaType": "essay",
  "friendlyName": "PAPER_NAME_PLACEHOLDER Q QNUM_PLACEHOLDER",
  "body": "<full question text>",
  "writtenSolution": "<model answer shown to student>",
  "marks": <integer>,
  "difficulty": 1,
  "questionForAI": "<question text in plain English, no LaTeX>",
  "aiMarkingCriteria": "<detailed marking instructions for AI. End with: There are X possible marks, please score the answer out of X (maximum X marks)>",
  "markingCriteriaForStudent": "<mark scheme shown to student after submission>"
}

For type = physical:
{
  "novaType": "physical",
  "friendlyName": "PAPER_NAME_PLACEHOLDER Q QNUM_PLACEHOLDER",
  "body": "<full question text>",
  "writtenSolution": "<describe what the correct answer looks like, e.g. 'Circle should be drawn around the ester bond'>",
  "marks": <integer>,
  "difficulty": 1
}

━━━ Rules ━━━
- Return ONLY the JSON object — no markdown fences, no explanation.
- body must contain the FULL question text with all maths in [latex]...[/latex] tags.
- marks must equal the markAllocation from the question data.
- If question references a diagram/image, add "(See diagram)" in body.
- answer (for simple) must be a SINGLE unambiguous number. If ambiguous → use essay.
- answerPrefix and answerUnit must use [latex] tags if they contain any maths or symbols.
- Units must be abbreviated: km not kilometers, cm not centimeters, m not meters, etc.
- Fractions and mixed numbers MUST use fraction type, never simple.
- Algebraic expressions, lists of values, or answers with multiple valid forms MUST use essay.
- For multiple_choice, always provide exactly 4 options with exactly 1 marked correct: true.
- For multiple_answer, prefix must use [latex] tags e.g. "[latex]x=[/latex]".
- CRITICAL: If the question asks for a fraction OR uses the words "fraction", "simplest form", "lowest terms", or "what fraction" — you MUST use type fraction, never simple.
"""

def read_cover_page(client: OpenAI, pdf_bytes: bytes) -> dict:
    """Read the first page of a PDF and extract paper metadata."""
    page_png = render_page_cached(pdf_bytes, 1, dpi=RENDER_DPI)
    page_pil = Image.open(io.BytesIO(page_png)).convert("RGB")
    content  = [
        {"type": "input_text",  "text": COVER_PROMPT},
        {"type": "input_image",
         "image_url": f"data:image/jpeg;base64,{encode_pil(page_pil)}"},
    ]
    parsed = safe_json_loads(
        call_gpt(client, content, VISION_MODEL, max_tokens=300), {})
    return {
        "paperName":   str(parsed.get("paperName",   "") or "").strip(),
        "examBoard":   str(parsed.get("examBoard",   "") or "").strip(),
        "subject":     str(parsed.get("subject",     "") or "").strip(),
        "level":       str(parsed.get("level",       "") or "").strip(),
        "year":        str(parsed.get("year",        "") or "").strip(),
        "paperNumber": str(parsed.get("paperNumber", "") or "").strip(),
    }

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
    # Preserve AQA-style dot-separated numbers (e.g. 01.1, 02.3a) as-is
    if re.match(r'^\d{2}\.\d', s):
        return s.replace(" ", "")
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
@st.cache_data(show_spinner=False, max_entries=10)
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

@st.cache_data(show_spinner=False, max_entries=50)
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

# ── Nova Airtable schema ──────────────────────────────────────────────────
# ── Nova Airtable schema ──────────────────────────────────────────────────
# ONE table, multiple views — one view per question type.

NOVA_ALL_FIELDS: list[tuple[str, str]] = [
    # Shared
    ("Question Number",        "singleLineText"),
    ("Paper Name",             "singleLineText"),
    ("Exam Board",             "singleLineText"),
    ("Subject",                "singleLineText"),
    ("Level",                  "singleLineText"),
    ("Year",                   "singleLineText"),
    ("Paper Number",           "singleLineText"),
    ("Nova Type",              "singleLineText"),
    ("Friendly Name",          "singleLineText"),
    ("Body",                   "multilineText"),
    ("Marks",                  "number"),
    ("Difficulty",             "number"),
    ("Written Solution",       "multilineText"),
    ("Is Sub-question",        "checkbox"),
    ("Parent Question",        "singleLineText"),
    ("Images",                 "multipleAttachments"),
    # Simple
    ("Answer Prefix",          "singleLineText"),
    ("Answer",                 "singleLineText"),
    ("Answer Unit",            "singleLineText"),
    # Multiple Choice
    ("MC Style",               "singleLineText"),
    ("MC Option A",            "singleLineText"),
    ("MC Option B",            "singleLineText"),
    ("MC Option C",            "singleLineText"),
    ("MC Option D",            "singleLineText"),
    # Multiple Answer
    ("Require Specific Order", "checkbox"),
    ("MA Answer 1 Prefix",     "singleLineText"),
    ("MA Answer 1",            "singleLineText"),
    ("MA Answer 1 Suffix",     "singleLineText"),
    ("MA Answer 2 Prefix",     "singleLineText"),
    ("MA Answer 2",            "singleLineText"),
    ("MA Answer 2 Suffix",     "singleLineText"),
    ("MA Answer 3 Prefix",     "singleLineText"),
    ("MA Answer 3",            "singleLineText"),
    ("MA Answer 3 Suffix",     "singleLineText"),
    ("MA Answer 4 Prefix",     "singleLineText"),
    ("MA Answer 4",            "singleLineText"),
    ("MA Answer 4 Suffix",     "singleLineText"),
    # Fraction
    ("Answer Label",           "singleLineText"),
    ("Numerator",              "singleLineText"),
    ("Denominator",            "singleLineText"),
    # Fill in Blank
    ("Preamble",               "multilineText"),
    ("Blank Content",          "multilineText"),
    ("Blanks",                 "multilineText"),
    # Essay
    ("Question for AI",        "multilineText"),
    ("AI Marking Criteria",    "multilineText"),
    ("Marking Criteria",       "multilineText"),
    ("AI Role Prompt",         "multilineText"),
    ("ChatGPT Model",          "singleLineText"),
    ("Pass Marks",             "number"),
    ("Min Word Count",         "number"),
    ("Marking Method",         "singleLineText"),
]

# Columns shown in each type's view
_SHARED_COLS = [
    "Question Number", "Paper Name", "Nova Type", "Friendly Name",
    "Body", "Marks", "Difficulty", "Written Solution",
    "Is Sub-question", "Parent Question",
]
NOVA_VIEW_VISIBLE: dict[str, list[str]] = {
    "simple":          _SHARED_COLS + ["Answer Prefix", "Answer", "Answer Unit"],
    "multiple_choice": _SHARED_COLS + ["MC Style", "MC Option A", "MC Option B", "MC Option C", "MC Option D"],
    "multiple_answer": _SHARED_COLS + ["Require Specific Order",
                        "MA Answer 1 Prefix", "MA Answer 1", "MA Answer 1 Suffix",
                        "MA Answer 2 Prefix", "MA Answer 2", "MA Answer 2 Suffix",
                        "MA Answer 3 Prefix", "MA Answer 3", "MA Answer 3 Suffix",
                        "MA Answer 4 Prefix", "MA Answer 4", "MA Answer 4 Suffix"],
    "fraction":        _SHARED_COLS + ["Answer Label", "Numerator", "Denominator"],
    "fill_in_blank":   _SHARED_COLS + ["Preamble", "Blank Content", "Blanks"],
    "essay": [
        "Question Number", "Paper Name", "Nova Type", "Friendly Name",
        "Body", "Question for AI", "AI Marking Criteria", "Marking Criteria",
        "AI Role Prompt", "ChatGPT Model", "Pass Marks", "Min Word Count",
        "Marking Method", "Is Sub-question", "Parent Question",
    ],
    "multi_part": [
        "Question Number", "Paper Name", "Nova Type", "Friendly Name",
        "Body", "Marks",
    ],
    "physical": [
        "Question Number", "Paper Name", "Nova Type", "Friendly Name",
        "Body", "Marks",
    ],
}
NOVA_VIEW_NAMES: dict[str, str] = {
    "simple":          "Simple Questions",
    "multiple_choice": "Multiple Choice",
    "multiple_answer": "Multiple Answer",
    "fraction":        "Fraction",
    "fill_in_blank":   "Fill in the Blank",
    "essay":           "Essay (AI)",
    "multi_part":      "Multi-part (Preambles)",
    "physical":        "Physical / Cannot Complete",
}



def _build_fields_payload(fields: list[tuple[str, str]]) -> list[dict]:
    out = []
    for name, ftype in fields:
        if ftype == "number":
            out.append({"name": name, "type": "number",
                        "options": {"precision": 0}})
        elif ftype == "checkbox":
            out.append({"name": name, "type": "checkbox",
                        "options": {"icon": "check", "color": "greenBright"}})
        elif ftype == "multipleAttachments":
            out.append({"name": name, "type": "multipleAttachments"})
        else:
            out.append({"name": name, "type": ftype})
    return out


def ensure_nova_fields(token: str, base_id: str, table_id: str,
                        field_map: dict[str, str]) -> dict[str, str]:
    """
    Add any NOVA_ALL_FIELDS that are missing from the existing table.
    Returns an updated field_map.
    """
    existing_names = set(field_map.keys())
    for name, ftype in NOVA_ALL_FIELDS:
        if name in existing_names:
            continue
        # Build the field definition
        if ftype == "number":
            field_def = {"name": name, "type": "number",
                         "options": {"precision": 0}}
        elif ftype == "checkbox":
            field_def = {"name": name, "type": "checkbox",
                         "options": {"icon": "check", "color": "greenBright"}}
        elif ftype == "multipleAttachments":
            field_def = {"name": name, "type": "multipleAttachments"}
        else:
            field_def = {"name": name, "type": ftype}

        r = requests.post(
            f"{AT_META}/bases/{base_id}/tables/{table_id}/fields",
            headers=at_headers(token),
            json=field_def,
            timeout=60,
        )
        if r.ok:
            field_map[name] = r.json()["id"]
    return field_map


CLOUDINARY_API = "https://api.cloudinary.com/v1_1"


def upload_cloudinary(cloud: str, preset: str,
                      filename: str, image_bytes: bytes,
                      paper_name: str = "") -> str | None:
    """Upload image bytes to Cloudinary. Returns secure URL or None on failure."""
    base       = filename.rsplit(".", 1)[0].replace(".", "_")
    safe_paper = re.sub(r"[^a-zA-Z0-9_-]", "_", paper_name)[:40] if paper_name else "paper"
    ts         = int(time.time())
    pid        = f"{safe_paper}_{base}_{ts}"
    resp = requests.post(
        f"{CLOUDINARY_API}/{cloud}/image/upload",
        data={"upload_preset": preset, "public_id": pid},
        files={"file": (filename, image_bytes, "image/png")},
        timeout=120,
    )
    if resp.ok:
        return resp.json().get("secure_url")
    return None


def build_qnum_image_map(boxes: list[dict]) -> dict[str, list[tuple[str, bytes]]]:
    """
    Build {question_number: [(filename, image_bytes), ...]} from drawn boxes.
    Also propagates parent-assigned boxes to child question numbers.
    """
    def _bare(s: str) -> str:
        s = normalise_qnum(s).lstrip("Qq")
        return s.lstrip("0") or s

    def _is_child_of(child: str, parent: str) -> bool:
        cb, pb = _bare(child), _bare(parent)
        if not cb.startswith(pb): return False
        rest = cb[len(pb):]
        return len(rest) > 0 and not rest[0].isdigit()

    direct: dict[str, list[tuple[str, bytes]]] = {}
    for b in boxes:
        qn = normalise_qnum(b.get("questionNumber", ""))
        if not qn or not b.get("data"):
            continue
        direct.setdefault(qn, []).append((b["name"], b["data"]))

    # Propagate: if a box is assigned to parent "7", also add to "7a", "7b"
    result: dict[str, list[tuple[str, bytes]]] = {}
    for qn, imgs in direct.items():
        result.setdefault(qn, []).extend(imgs)

    # Second pass: add parent images to children
    all_qnums = list(direct.keys())
    for parent_qn, parent_imgs in direct.items():
        for child_qn in all_qnums:
            if child_qn != parent_qn and _is_child_of(child_qn, parent_qn):
                result.setdefault(child_qn, [])
                for img in parent_imgs:
                    if img not in result[child_qn]:
                        result[child_qn].append(img)

    return result


def ensure_nova_table(token: str, base_id: str,
                       table_name: str) -> str:
    """Create the Nova Questions table if it doesn't exist. Returns table_id."""
    r = requests.get(f"{AT_META}/bases/{base_id}/tables",
                     headers=at_headers(token), timeout=60)
    r.raise_for_status()
    for t in r.json().get("tables", []):
        if t["name"] == table_name:
            return t["id"]
    r2 = requests.post(
        f"{AT_META}/bases/{base_id}/tables",
        headers=at_headers(token),
        json={"name": table_name,
              "fields": _build_fields_payload(NOVA_ALL_FIELDS)},
        timeout=60,
    )
    if not r2.ok:
        raise RuntimeError(
            f"Could not create table '{table_name}': "
            f"{r2.status_code} {r2.text[:300]}")
    return r2.json()["id"]


def get_table_field_map(token: str, base_id: str,
                         table_id: str) -> dict[str, str]:
    """Return {field_name: field_id}."""
    r = requests.get(f"{AT_META}/bases/{base_id}/tables",
                     headers=at_headers(token), timeout=60)
    r.raise_for_status()
    for t in r.json().get("tables", []):
        if t["id"] == table_id:
            return {f["name"]: f["id"] for f in t.get("fields", [])}
    return {}




def nova_record_to_fields(item: dict, paper_name: str = "",
                           meta: dict | None = None) -> dict:
    """Flatten a classified nova item into a full Airtable field dict."""
    nd   = item.get("novaData") or {}
    nt   = nd.get("novaType", "")
    rec  = item.get("originalRecord") or {}
    pn   = rec.get("paperName") or paper_name
    m    = meta or {}

    # Helper: prefer record value, fall back to meta, then empty string
    def _meta(rec_key: str, meta_key: str) -> str:
        return str(rec.get(rec_key) or m.get(meta_key) or "").strip()

    # Parent preamble records have no novaData — handle separately
    if item.get("isParent"):
        return {
            "Question Number":  rec.get("originalQuestionNumber", rec.get("questionNumber", "")),
            "Paper Name":       pn,
            "Exam Board":       _meta("examBoard",   "exam_board"),
            "Subject":          _meta("subject",     "subject"),
            "Level":            _meta("level",       "level"),
            "Year":             _meta("year",        "year"),
            "Paper Number":     _meta("paperNumber", "paper_number"),
            "Nova Type":        "multi_part",
            "Friendly Name":    f"{pn} Q{rec.get('questionNumber', '')}",
            "Body":             rec.get("questionText", ""),
            "Marks":            0,
            "Difficulty":       1,
            "Written Solution": "",
            "Is Sub-question":  False,
            "Parent Question":  "",
        }

    fields: dict = {
        "Question Number":  rec.get("originalQuestionNumber", rec.get("questionNumber", "")),
        "Paper Name":       pn,
        "Exam Board":       _meta("examBoard",   "exam_board"),
        "Subject":          _meta("subject",     "subject"),
        "Level":            _meta("level",       "level"),
        "Year":             _meta("year",        "year"),
        "Paper Number":     _meta("paperNumber", "paper_number"),
        "Nova Type":        nt,
        "Friendly Name":    nd.get("friendlyName", ""),
        "Body":             nd.get("body", ""),
        "Marks":            clamp_int(nd.get("marks", 0)),
        "Difficulty":       clamp_int(nd.get("difficulty", 1), 1),
        "Written Solution": nd.get("writtenSolution", ""),
        "Is Sub-question":  bool(item.get("isSubQuestion", False)),
        "Parent Question":  item.get("parentQuestion", ""),
    }
    if nt == "simple":
        fields.update({
            "Answer Prefix": nd.get("answerPrefix", ""),
            "Answer":        nd.get("answer",       ""),
            "Answer Unit":   nd.get("answerUnit",   ""),
        })
    elif nt == "multiple_choice":
        raw_opts = nd.get("options") or []
        opts     = [o if isinstance(o, dict) else {} for o in raw_opts] if isinstance(raw_opts, list) else []
        correct  = [o for o in opts if o.get("correct")]
        wrong    = [o for o in opts if not o.get("correct")]
        ordered  = correct + wrong  # correct first → always Option A
        labels  = ["A", "B", "C", "D"]
        for i, label in enumerate(labels):
            opt = ordered[i] if i < len(ordered) else {}
            fields[f"MC Option {label}"] = str(opt.get("text", "") or "")
        fields["MC Style"] = nd.get("style", "List")
    elif nt == "multiple_answer":
        raw_answers = nd.get("answers") or []
        answers = [a if isinstance(a, dict) else {} for a in raw_answers] if isinstance(raw_answers, list) else []
        fields["Require Specific Order"] = bool(nd.get("requireSpecificOrder", False))
        for i in range(1, 5):
            ans = answers[i-1] if i-1 < len(answers) else {}
            fields[f"MA Answer {i} Prefix"] = str(ans.get("prefix", "") or "")
            fields[f"MA Answer {i}"]        = str(ans.get("answer", "") or "")
            fields[f"MA Answer {i} Suffix"] = str(ans.get("suffix", "") or "")
    elif nt == "fraction":
        fields.update({
            "Answer Label": nd.get("answerLabel", ""),
            "Numerator":    nd.get("numerator",   ""),
            "Denominator":  nd.get("denominator", ""),
        })
    elif nt == "fill_in_blank":
        fields.update({
            "Preamble":      nd.get("preamble",     ""),
            "Blank Content": nd.get("blankContent", ""),
            "Blanks":        json.dumps(nd.get("blanks", []), ensure_ascii=False),
        })
    elif nt == "essay":
        fields.update({
            "Question for AI":     nd.get("questionForAI",             ""),
            "AI Marking Criteria": nd.get("aiMarkingCriteria",         ""),
            "Marking Criteria":    nd.get("markingCriteriaForStudent",  ""),
            "AI Role Prompt":      nova_ai_role_prompt(
                                       rec.get("subject", "") or m.get("subject", ""),
                                       rec.get("level",   "") or m.get("level",   "")),
            "ChatGPT Model":       "gpt-4.1",
            "Pass Marks":          clamp_int(nd.get("marks", 0)),
            "Min Word Count":      1,
            "Marking Method":      "AI",
        })
    return fields





def push_nova_to_airtable(token: str, base_id: str, table_name: str,
                            items: list[dict],
                            paper_name: str = "",
                            boxes_list: list[dict] | None = None,
                            cld_cloud: str = "",
                            cld_preset: str = "",
                            meta: dict | None = None,
                            ) -> tuple[int, list[str]]:
    """
    Ensure the Questions table exists, push records, upload images via Cloudinary.
    Returns (records_created, log_lines, manual_view_instructions).
    """
    logs: list[str] = []

    table_id = ensure_nova_table(token, base_id, table_name)
    logs.append(f"Table '{table_name}' ready")

    field_map: dict[str, str] = {}
    for _ in range(3):
        field_map = get_table_field_map(token, base_id, table_id)
        if field_map:
            break
        time.sleep(1)
    logs.append(f"{len(field_map)} fields found")

    # Add any fields missing from an older table
    field_map = ensure_nova_fields(token, base_id, table_id, field_map)

    # Pre-register singleSelect option values so Airtable accepts them
    # Upload images to Cloudinary first so we have URLs to embed in records
    img_url_map: dict[str, str] = {}  # filename → cloudinary URL
    if boxes_list and cld_cloud and cld_preset:
        logs.append(f"Uploading images to Cloudinary…")
        for b in boxes_list:
            if not b.get("data"):
                continue
            url = upload_cloudinary(cld_cloud, cld_preset,
                                    b["name"], b["data"], paper_name)
            if url:
                img_url_map[b["name"]] = url
        logs.append(f"  {len(img_url_map)} images uploaded")
    elif boxes_list:
        logs.append("⚠ Cloudinary not configured — images will not be attached")

    # Build question → image URLs map
    img_map = build_qnum_image_map(boxes_list or [])

    payload = []
    for item in items:
        if item.get("error"):
            continue
        raw      = nova_record_to_fields(item, paper_name, meta=meta)
        filtered = {k: v for k, v in raw.items() if k in field_map}

        # Attach image URLs for this question
        qn   = normalise_qnum(
            (item.get("originalRecord") or {}).get("questionNumber",
             item.get("questionNumber", "")))
        urls = [img_url_map[fn] for fn, _ in img_map.get(qn, [])
                if fn in img_url_map]
        if urls and "Images" in field_map:
            filtered["Images"] = [{"url": u} for u in urls]

        payload.append({"fields": filtered})

    if not payload:
        logs.append("No records to push.")
        return 0, logs

    url     = f"{AT_API}/{base_id}/{requests.utils.quote(table_name, safe='')}"
    created = 0
    for batch in chunk_list(payload, 10):
        resp = requests.post(url, headers=at_headers(token),
                             json={"records": batch}, timeout=60)
        if not resp.ok:
            logs.append(f"❌ Batch failed: {resp.status_code} {resp.text[:200]}")
        else:
            created += len(resp.json().get("records", []))

    logs.append(f"✅ {created} records pushed")
    return created, logs

# ── Nova classification ───────────────────────────────────────────────────
NOVA_TYPE_LABELS = {
    "simple":          "Simple",
    "multiple_choice": "Multiple Choice",
    "multiple_answer": "Multiple Answer",
    "fraction":        "Fraction",
    "fill_in_blank":   "Fill in the Blank",
    "essay":           "Essay (AI)",
    "physical":        "Physical / Cannot Complete",
}
NOVA_TYPE_COLORS = {
    "simple":          "#27ae60",
    "multiple_choice": "#2980b9",
    "multiple_answer": "#8e44ad",
    "fraction":        "#e67e22",
    "fill_in_blank":   "#16a085",
    "essay":           "#c0392b",
    "physical":        "#95a5a6",
}
def nova_ai_role_prompt(subject: str = "", level: str = "") -> str:
    """Generate a role prompt tailored to the subject and level."""
    subj  = subject.strip() or "this subject"
    lvl   = level.strip()   or "exam"
    combo = f"{lvl} {subj}".strip()
    return (
        f"You are a fair, accurate, and constructive {combo} examiner. "
        f"Your feedback should be focused on key {subj} concepts, assessment criteria, "
        f"and the learner's application of knowledge. Ensure your comments are clear, "
        f"specific, and help learners improve their understanding of {subj} "
        f"while maintaining a professional and encouraging tone. "
        f"Accept minor spelling mistakes if the meaning is clear."
    )

# Keep a default for reference
NOVA_AI_ROLE_PROMPT = nova_ai_role_prompt()

NOVA_TWEAK_PROMPT = """You are adapting an exam question for an e-learning platform called Nova.

You will receive a classified Nova question. Your job depends on its type:

━━━ If novaType = "physical" ━━━
Convert it into a digitally-answerable question. The image stays — only the question changes.
- "Draw a circle around X" → "Which of the following best describes X?" (multiple_choice)
- "Label the diagram" / "Identify the structure labelled A" → simple or multiple_choice
- "Complete the graph" → essay asking them to describe what the graph would show
- Choose the most appropriate Nova type for the converted question.

━━━ For all other types ━━━
Slightly vary the question so it tests the same concept but with different wording or numbers:
- Swap specific numbers for similar ones (e.g. 48 → 36, 3 pounds → 4 pounds)
- Rephrase the question slightly (synonym words, different sentence structure)
- Keep the SAME novaType, same difficulty, same topic
- If the question references an image/diagram, only change the text — never change what the image shows
- The answer must change to match any number changes

Current classified question:
NOVA_DATA_PLACEHOLDER

Return ONLY the same JSON structure as the input — same fields, same novaType (unless converting from physical), all fields filled in.
No markdown fences, no explanation.
"""


def tweak_nova_question(client: OpenAI, item: dict,
                         paper_name: str = "") -> dict:
    """Tweak a classified nova item — rephrase/renumber, or convert physical to digital."""
    nd  = item.get("novaData") or {}
    rec = item.get("originalRecord") or {}

    tweak_data = {
        "novaType":     nd.get("novaType", ""),
        "friendlyName": nd.get("friendlyName", ""),
        "body":         nd.get("body", ""),
        "marks":        nd.get("marks", rec.get("markAllocation", 0)),
        "difficulty":   nd.get("difficulty", 1),
        "writtenSolution": nd.get("writtenSolution", ""),
        "markSchemeAnswer": rec.get("markSchemeAnswer", ""),
        "hasImage":     rec.get("hasImages", False),
        "imageDescription": rec.get("imageDescription", ""),
        # type-specific fields
        "answer":       nd.get("answer", ""),
        "answerPrefix": nd.get("answerPrefix", ""),
        "answerUnit":   nd.get("answerUnit", ""),
        "options":      nd.get("options", []),
        "answers":      nd.get("answers", []),
        "numerator":    nd.get("numerator", ""),
        "denominator":  nd.get("denominator", ""),
        "questionForAI": nd.get("questionForAI", ""),
        "aiMarkingCriteria": nd.get("aiMarkingCriteria", ""),
        "markingCriteriaForStudent": nd.get("markingCriteriaForStudent", ""),
        "requireSpecificOrder": nd.get("requireSpecificOrder", False),
        "style":        nd.get("style", ""),
        "preamble":     nd.get("preamble", ""),
        "blankContent": nd.get("blankContent", ""),
        "blanks":       nd.get("blanks", []),
    }

    prompt = (NOVA_TWEAK_PROMPT
              .replace("NOVA_DATA_PLACEHOLDER", json.dumps(tweak_data, indent=2)))
    content = [{"type": "input_text", "text": prompt}]

    for attempt in range(2):
        raw    = call_gpt(client, content, VISION_MODEL, max_tokens=3000)
        result = safe_json_loads(raw, {})
        if result.get("novaType") and result.get("body"):
            break

    if not result.get("novaType"):
        return nd  # return original if tweak failed

    # Apply same safety overrides as classify
    question_text = result.get("body", "").lower()
    fraction_keywords = ["as a fraction", "simplest form", "lowest terms", "what fraction"]
    if result.get("novaType") == "simple" and any(kw in question_text for kw in fraction_keywords):
        result["novaType"] = "fraction"

    if result.get("novaType") == "simple":
        ans = str(result.get("answer", "") or "")
        if any(c in ans for c in ["+", "-", "×", "÷", "or", ","]):
            result["novaType"] = "essay"
        elif "/" in ans and not any(c.isalpha() for c in ans):
            result["novaType"] = "fraction"
            parts = ans.split("/", 1)
            result["numerator"]   = parts[0].strip()
            result["denominator"] = parts[1].strip()
            result.pop("answer", None)

    return result

def classify_nova_question(client: OpenAI, record: dict,
                            paper_name: str = "") -> dict:
    """Use AI to classify one record into a Nova question type and extract fields."""
    q_data = {
        "questionNumber":   record.get("questionNumber",   ""),
        "questionText":     record.get("questionText",     ""),
        "markAllocation":   record.get("markAllocation",   0),
        "markSchemeAnswer": record.get("markSchemeAnswer", ""),
        "topic":            record.get("topic",            ""),
        "subtopic":         record.get("subtopic",         ""),
        "hasImages":        record.get("hasImages",        False),
        "imageDescription": record.get("imageDescription", ""),
    }
    prompt = (NOVA_CLASSIFY_PROMPT
              .replace("QUESTION_DATA_PLACEHOLDER", json.dumps(q_data, indent=2))
              .replace("PAPER_NAME_PLACEHOLDER", paper_name)
              .replace("QNUM_PLACEHOLDER", record.get("questionNumber", "")))
    content = [{"type": "input_text", "text": prompt}]

    # Try up to 2 times — retry if result is missing critical fields
    result = {}
    for attempt in range(2):
        raw    = call_gpt(client, content, VISION_MODEL, max_tokens=3000)
        result = safe_json_loads(raw, {})
        if result.get("novaType") and result.get("body"):
            break  # got a good result

    if not result.get("novaType"):
        result["novaType"] = "simple"

    # Safety override: force fraction type if question clearly asks for a fraction
    question_text = record.get("questionText", "").lower()
    fraction_keywords = ["as a fraction", "simplest form", "lowest terms",
                         "what fraction", "give your answer as a fraction"]
    if (result.get("novaType") == "simple"
            and any(kw in question_text for kw in fraction_keywords)):
        result["novaType"] = "fraction"
        ans = str(result.get("answer", "") or "")
        if "/" in ans:
            parts = ans.split("/", 1)
            result["numerator"]   = parts[0].strip()
            result["denominator"] = parts[1].strip()
        result.pop("answer", None)
        result.pop("answerPrefix", None)
        result.pop("answerUnit", None)

    # Safety override: force essay if answer is ambiguous/expression/list
    if result.get("novaType") == "simple":
        ans = str(result.get("answer", "") or "")
        is_expression = any(c in ans for c in ["+", "-", "×", "÷", "=", "or", ","])
        has_multiple_forms = ans.count("/") > 0 or "%" in ans
        is_list = "," in ans
        is_fraction_answer = "/" in ans and not any(c.isalpha() for c in ans)
        if is_expression or is_list:
            result["novaType"] = "essay"
            result.pop("answer", None)
            result.pop("answerPrefix", None)
            result.pop("answerUnit", None)
        elif is_fraction_answer:
            result["novaType"] = "fraction"
            parts = ans.split("/", 1)
            result["numerator"]   = parts[0].strip()
            result["denominator"] = parts[1].strip()
            result.pop("answer", None)

    # Fallback: if simple answer is empty, try to extract from writtenSolution
    if result.get("novaType") == "simple" and not result.get("answer"):
        ws = result.get("writtenSolution", "") or ""
        nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', ws)
        if nums:
            result["answer"] = nums[-1]

    return result

def group_nova_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split records into:
      standalone  – questions with no parent/child relationship
      groups      – list of {parent: record, children: [record, ...]}
    """
    def bare(s: str) -> str:
        s = normalise_qnum(s).lstrip("Qq")
        return s.lstrip("0") or s

    def is_aqa_style(qn: str) -> bool:
        """Detect AQA-style numbers like 01.1, 02.3 — dot-separated in original."""
        orig = str(qn or "").strip()
        return bool(re.match(r'^\d{2}\.\d+', orig))

    def is_child_of(child_qn: str, parent_qn: str) -> bool:
        # Never group AQA-style dot-numbered questions
        child_orig = next((r.get("originalQuestionNumber","") or r.get("questionNumber","")
                           for r in records
                           if normalise_qnum(r.get("questionNumber","")) == child_qn), "")
        parent_orig = next((r.get("originalQuestionNumber","") or r.get("questionNumber","")
                            for r in records
                            if normalise_qnum(r.get("questionNumber","")) == parent_qn), "")
        if is_aqa_style(child_orig) or is_aqa_style(parent_orig):
            return False
        cb, pb = bare(child_qn), bare(parent_qn)
        if not cb.startswith(pb):
            return False
        rest = cb[len(pb):]
        return len(rest) > 0 and not rest[0].isdigit()

    all_qnums = [normalise_qnum(r.get("questionNumber", "")) for r in records]

    parent_qnums: set[str] = set()
    child_qnums:  set[str] = set()
    for qn in all_qnums:
        for other in all_qnums:
            if other != qn and is_child_of(other, qn):
                parent_qnums.add(qn)
                child_qnums.add(other)

    groups: list[dict] = []
    for r in records:
        qn = normalise_qnum(r.get("questionNumber", ""))
        if qn in parent_qnums:
            children = [cr for cr in records
                        if normalise_qnum(cr.get("questionNumber", "")) in child_qnums
                        and is_child_of(
                            normalise_qnum(cr.get("questionNumber", "")), qn)]
            groups.append({"parent": r, "children": children})

    standalone = [r for r in records
                  if normalise_qnum(r.get("questionNumber", "")) not in parent_qnums
                  and normalise_qnum(r.get("questionNumber", "")) not in child_qnums]

    return standalone, groups

def fetch_airtable_records(token: str, base_id: str,
                            table: str) -> list[dict]:
    """Fetch all records from an Airtable table."""
    url     = f"{AT_API}/{base_id}/{requests.utils.quote(table, safe='')}"
    headers = at_headers(token)
    records = []
    offset  = None
    while True:
        params: dict = {"pageSize": 100}
        if offset:
            params["offset"] = offset
        resp = requests.get(url, headers=headers, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        for rec in data.get("records", []):
            f = rec.get("fields", {})
            records.append({
                "questionNumber":   normalise_qnum(f.get("Question Number",   "")),
                "originalQuestionNumber": str(f.get("Question Number",  "") or ""),
                "questionText":     str(f.get("Question Text",      "") or ""),
                "markAllocation":   clamp_int(f.get("Mark Allocation",  0)),
                "topic":            str(f.get("Topic",              "") or ""),
                "subtopic":         str(f.get("Subtopic",           "") or ""),
                "markSchemeAnswer": str(f.get("Mark Scheme Answer", "") or ""),
                "imageDescription": str(f.get("Image Description",  "") or ""),
                "hasImages":        bool(f.get("Has Images",        False)),
                "pageNumber":       clamp_int(f.get("Page Number",   1), 1),
                "paperName":        str(f.get("Paper Name",         "") or ""),
                "examType":         str(f.get("Exam Type",          "") or ""),
            })
        offset = data.get("offset")
        if not offset:
            break
    return records

def nova_records_to_csv(nova_classified: list[dict]) -> str:
    """Export all classified records to a flat CSV string."""

    def _safe_answers(nd: dict) -> list[dict]:
        """Return answers as a list of dicts, handling malformed data."""
        raw = nd.get("answers") or []
        if not isinstance(raw, list):
            return []
        return [a if isinstance(a, dict) else {} for a in raw]

    def _safe_options(nd: dict) -> list[dict]:
        """Return options as a list of dicts, handling malformed data."""
        raw = nd.get("options") or []
        if not isinstance(raw, list):
            return []
        return [o if isinstance(o, dict) else {} for o in raw]
    rows = []
    for item in nova_classified:
        nd  = item.get("novaData", {}) or {}
        nt  = nd.get("novaType", item.get("novaType", ""))
        row = {
            "Question Number":     item.get("questionNumber", ""),
            "Nova Type":           nt,
            "Friendly Name":       nd.get("friendlyName", ""),
            "Body":                nd.get("body", ""),
            "Marks":               nd.get("marks", ""),
            "Difficulty":          nd.get("difficulty", 1),
            "Written Solution":    nd.get("writtenSolution", ""),
            # Simple
            "Answer Prefix":       nd.get("answerPrefix", ""),
            "Answer":              nd.get("answer", ""),
            "Answer Unit":         nd.get("answerUnit", ""),
            # MC
            "MC Style":    nd.get("style", ""),
            "MC Option A": next((o.get("text","") for o in _safe_options(nd) if o.get("correct")), ""),
            "MC Option B": next((o.get("text","") for o in _safe_options(nd) if not o.get("correct")), ""),
            "MC Option C": next((o.get("text","") for i,o in enumerate(_safe_options(nd)) if not o.get("correct") and i > 0), ""),
            "MC Option D": next((o.get("text","") for i,o in enumerate(_safe_options(nd)) if not o.get("correct") and i > 1), ""),
            # Multiple answer
            "Require Specific Order": str(nd.get("requireSpecificOrder", "")),
            "MA Answer 1 Prefix": _safe_answers(nd)[0].get("prefix","") if len(_safe_answers(nd)) > 0 else "",
            "MA Answer 1":        _safe_answers(nd)[0].get("answer","") if len(_safe_answers(nd)) > 0 else "",
            "MA Answer 1 Suffix": _safe_answers(nd)[0].get("suffix","") if len(_safe_answers(nd)) > 0 else "",
            "MA Answer 2 Prefix": _safe_answers(nd)[1].get("prefix","") if len(_safe_answers(nd)) > 1 else "",
            "MA Answer 2":        _safe_answers(nd)[1].get("answer","") if len(_safe_answers(nd)) > 1 else "",
            "MA Answer 2 Suffix": _safe_answers(nd)[1].get("suffix","") if len(_safe_answers(nd)) > 1 else "",
            "MA Answer 3 Prefix": _safe_answers(nd)[2].get("prefix","") if len(_safe_answers(nd)) > 2 else "",
            "MA Answer 3":        _safe_answers(nd)[2].get("answer","") if len(_safe_answers(nd)) > 2 else "",
            "MA Answer 3 Suffix": _safe_answers(nd)[2].get("suffix","") if len(_safe_answers(nd)) > 2 else "",
            "MA Answer 4 Prefix": _safe_answers(nd)[3].get("prefix","") if len(_safe_answers(nd)) > 3 else "",
            "MA Answer 4":        _safe_answers(nd)[3].get("answer","") if len(_safe_answers(nd)) > 3 else "",
            "MA Answer 4 Suffix": _safe_answers(nd)[3].get("suffix","") if len(_safe_answers(nd)) > 3 else "",
            # Fraction
            "Answer Label":        nd.get("answerLabel", ""),
            "Numerator":           nd.get("numerator", ""),
            "Denominator":         nd.get("denominator", ""),
            # Fill in blank
            "Preamble":            nd.get("preamble", ""),
            "Blank Content":       nd.get("blankContent", ""),
            "Blanks":              json.dumps(nd.get("blanks", [])) if nd.get("blanks") else "",
            # Essay
            "Question for AI":     nd.get("questionForAI", ""),
            "AI Marking Criteria": nd.get("aiMarkingCriteria", ""),
            "Marking Criteria":    nd.get("markingCriteriaForStudent", ""),
            "AI Role Prompt":      nova_ai_role_prompt() if nt == "essay" else "",
            "ChatGPT Model":       "gpt-4.1" if nt == "essay" else "",
            "Min Word Count":      "1" if nt == "essay" else "",
            "Marking Method":      "AI" if nt == "essay" else "",
        }
        rows.append(row)
    if not rows:
        return ""
    buf = io.StringIO()
    import csv
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()

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

# ═════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Past Paper → Airtable", page_icon="📄", layout="wide")

# Restore from disk on every fresh session start


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

    AUTO_ASSIGN = st.checkbox("Auto-assign high-confidence AI suggestions", value=True)
    st.divider()
    st.markdown("**💾 Save / Load session**")
    st.caption("Save your progress to a file and reload it later — survives restarts.")

    if st.button("💾 Save session", width='stretch'):
        save_data = {
            "paper_name":            st.session_state.get("paper_name", ""),
            "exam_board":            st.session_state.get("exam_board", ""),
            "subject":               st.session_state.get("subject", ""),
            "level":                 st.session_state.get("level", ""),
            "year":                  st.session_state.get("year", ""),
            "paper_number":          st.session_state.get("paper_number", ""),
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
            st.session_state["exam_board"]           = save_data.get("exam_board", "")
            st.session_state["subject"]              = save_data.get("subject", "")
            st.session_state["level"]                = save_data.get("level", "")
            st.session_state["year"]                 = save_data.get("year", "")
            st.session_state["paper_number"]         = save_data.get("paper_number", "")
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
    st.caption("Secrets can also be set in `.streamlit/secrets.toml`.")

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
    col_board, col_subj = st.columns(2)
    with col_board:
        exam_board = st.text_input("Exam board",
            value=st.session_state.get("exam_board", ""),
            placeholder="e.g. AQA")
    with col_subj:
        subject = st.text_input("Subject",
            value=st.session_state.get("subject", ""),
            placeholder="e.g. Mathematics")
    col_lvl, col_yr, col_pn = st.columns(3)
    with col_lvl:
        level = st.text_input("Level",
            value=st.session_state.get("level", ""),
            placeholder="e.g. GCSE")
    with col_yr:
        year = st.text_input("Year",
            value=st.session_state.get("year", ""),
            placeholder="e.g. 2024")
    with col_pn:
        paper_number = st.text_input("Paper number",
            value=st.session_state.get("paper_number", ""),
            placeholder="e.g. 1")
with c2:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    ms_file = st.file_uploader("Mark scheme PDF (optional)", type="pdf")

if st.button("Load PDF", disabled=not (paper_file and OPENAI_KEY)):
    paper_file.seek(0)
    pdf    = paper_file.read()
    client = OpenAI(api_key=OPENAI_KEY)

    with st.spinner("Reading cover page…"):
        detected = read_cover_page(client, pdf)

    _set_pdf(pdf)
    render_page_cached.clear()
    st.session_state["pages"]        = get_question_pages(pdf)
    st.session_state["paper_name"]   = detected["paperName"]
    st.session_state["exam_type"]    = detected["level"]
    st.session_state["exam_board"]   = detected["examBoard"]
    st.session_state["subject"]      = detected["subject"]
    st.session_state["level"]        = detected["level"]
    st.session_state["year"]         = detected["year"]
    st.session_state["paper_number"] = detected["paperNumber"]
    st.session_state.pop("records", None)
    st.session_state["boxes"] = {}
    st.session_state.pop("_save_json", None)
    st.session_state["sel_page_idx"] = 0
    safe_name = re.sub(r"[^a-zA-Z0-9 _-]", "", detected["paperName"]).strip() or "Questions"
    st.session_state["paper_name_for_table"] = safe_name
    st.success(
        f"Loaded — {len(st.session_state['pages'])} question pages found.  "
        f"Detected: **{detected['paperName']}** · **{detected['level']}**"
    )
    st.rerun()

# Keep paper metadata in sync with what user may have edited
paper_name   = st.session_state.get("paper_name",   paper_name)
exam_board   = st.session_state.get("exam_board",   exam_board)
subject      = st.session_state.get("subject",      subject)
level        = st.session_state.get("level",        level)
year         = st.session_state.get("year",         year)
paper_number = st.session_state.get("paper_number", paper_number)

# ── 2. Extract ─────────────────────────────────────────────────────────────
if st.session_state.get("pages"):
    st.subheader("2 · Extract questions + mark scheme")

    if st.button("✨ Extract", type="primary",
                 disabled=not (paper_name and OPENAI_KEY)):
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
                    "examBoard":                exam_board,
                    "subject":                  subject,
                    "level":                    level,
                    "year":                     year,
                    "paperNumber":              paper_number,
                    "imageMappingConfidence":   "",
                    "imageMappingNotes":        "",
                    "images":                   [],
                })

            # Filter out false preamble-only rows:
            # A row with 0 marks whose question number is a prefix of other rows
            # (e.g. "03" when "03.1", "03.2" etc. also exist) is a spurious parent.
            def _is_false_preamble(r: dict, all_records: list[dict]) -> bool:
                if clamp_int(r.get("markAllocation", 0)) != 0:
                    return False
                qn = normalise_qnum(r.get("questionNumber", ""))
                if not qn:
                    return False
                # Check if any other record's number starts with this one
                # (indicating it was incorrectly split as a parent)
                for other in all_records:
                    oqn = normalise_qnum(other.get("questionNumber", ""))
                    if oqn != qn and oqn.startswith(qn):
                        return True
                return False

            before = len(records)
            records = [r for r in records if not _is_false_preamble(r, records)]
            filtered = before - len(records)
            if filtered:
                st.write(f"   Removed {filtered} false preamble row(s)")

            st.session_state["records"] = records
            autosave()
            status.update(label=f"✅ {len(records)} questions extracted",
                          state="complete")

# ── 3. Capture ─────────────────────────────────────────────────────────────
if st.session_state.get("pages"):
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
                        r = ai_assign(client, b, records, pdf_bytes=_pdf)
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

# ── Export ────────────────────────────────────────────────────────────────
if "records" in st.session_state:
    records = st.session_state["records"]
    ab      = all_boxes()

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇ Download JSON",
            data=json.dumps(records, indent=2, ensure_ascii=False).encode(),
            file_name=f"{paper_name or 'questions'}.json",
            mime="application/json",
        )
    with dl2:
        if ab:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                for b in ab:
                    if b.get("data"):
                        zf.writestr(b["name"], b["data"])
            st.download_button(
                "⬇ Download visuals (.zip)",
                data=buf.getvalue(),
                file_name=f"{paper_name or 'paper'}_visuals.zip",
                mime="application/zip",
            )

# ── 7. Nova Question Formatter ─────────────────────────────────────────────
st.divider()
st.subheader("7 · Nova Question Formatter")
st.caption(
    "Classify extracted questions into Nova question types with all fields "
    "pre-filled and ready to copy into the platform."
)

_src = st.session_state.get("records", [])
if _src:
    st.caption(f"{len(_src)} records available for classification.")

# ── Classify button ────────────────────────────────────────────────────────
classify_col, tweak_col, clear_col = st.columns([3, 3, 1])
with classify_col:
    if st.button(
        "🤖 Classify all with AI",
        type="primary",
        disabled=not (OPENAI_KEY and _src),
    ):
        st.session_state["do_nova_classify"] = True
        st.session_state.pop("nova_classified", None)
        st.rerun()

with tweak_col:
    if st.button(
        "✏️ Tweak all questions",
        disabled=not (OPENAI_KEY and "nova_classified" in st.session_state),
        help="Rephrase/renumber all questions. Converts physical questions to digital types.",
    ):
        st.session_state["do_nova_tweak"] = True
        st.rerun()

with clear_col:
    if st.button("🗑 Clear results",
                 disabled="nova_classified" not in st.session_state):
        st.session_state.pop("nova_classified", None)
        st.rerun()

if st.session_state.get("do_nova_classify"):
    st.session_state["do_nova_classify"] = False
    src_records = st.session_state.get("records", [])
    _pname      = st.session_state.get("paper_name", paper_name) or "Paper"
    client      = OpenAI(api_key=OPENAI_KEY)

    # Remove any false preamble-only rows that slipped through extraction
    def _is_false_preamble(r, all_r):
        if clamp_int(r.get("markAllocation", 0)) != 0:
            return False
        qn = normalise_qnum(r.get("questionNumber", ""))
        if not qn:
            return False
        for other in all_r:
            oqn = normalise_qnum(other.get("questionNumber", ""))
            if oqn != qn and oqn.startswith(qn):
                return True
        return False

    filtered = [r for r in src_records if not _is_false_preamble(r, src_records)]
    if len(filtered) < len(src_records):
        removed = [r.get("questionNumber","") for r in src_records if _is_false_preamble(r, src_records)]
        st.info(f"Skipping {len(src_records)-len(filtered)} false preamble row(s): {removed}")
        src_records = filtered
        st.session_state["records"] = filtered

    # Only classify questions that need answering (skip parent preamble rows)
    standalone, groups = group_nova_records(src_records)
    to_classify = list(standalone)
    for g in groups:
        to_classify.extend(g["children"])

    nova_out: list[dict] = []
    errors:   list[str]  = []

    progress_bar = st.progress(0, text="Classifying questions…")
    total = len(to_classify)

    def _classify_one(r):
        try:
            nd = classify_nova_question(client, r, _pname)
            return {"questionNumber": r["questionNumber"],
                    "originalRecord": r,
                    "novaData": nd,
                    "error": None}
        except Exception as e:
            return {"questionNumber": r["questionNumber"],
                    "originalRecord": r,
                    "novaData": {},
                    "error": str(e)}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_classify_one, r): i
                   for i, r in enumerate(to_classify)}
        done = 0
        for fut in as_completed(futures):
            nova_out.append(fut.result())
            done += 1
            progress_bar.progress(done / max(total, 1),
                                   text=f"Classified {done}/{total}…")

    progress_bar.empty()

    # Attach parent records so the UI can build multi-part groups
    parent_items = []
    for g in groups:
        parent_items.append({
            "questionNumber": g["parent"]["questionNumber"],
            "originalRecord": g["parent"],
            "novaData": None,  # parents have no nova classification
            "isParent": True,
            "childQnums": [
                normalise_qnum(c.get("questionNumber", ""))
                for c in g["children"]
            ],
            "error": None,
        })

    # Mark non-parent items
    for item in nova_out:
        item.setdefault("isParent", False)
        item.setdefault("childQnums", [])

    # Merge parents back in so display can find them
    all_nova = parent_items + nova_out
    # Sort by original record order
    qnum_order = {normalise_qnum(r.get("questionNumber", "")): i
                  for i, r in enumerate(src_records)}
    all_nova.sort(key=lambda x: qnum_order.get(
        normalise_qnum(x.get("questionNumber", "")), 9999))

    st.session_state["nova_classified"] = all_nova
    st.session_state["nova_paper_name"] = _pname
    render_page_cached.clear()  # free page render cache — no longer needed
    n_ok  = sum(1 for x in nova_out if not x["error"])
    n_err = sum(1 for x in nova_out if x["error"])
    st.success(f"✅ {n_ok} classified · {n_err} errors · {len(parent_items)} multi-part groups")
    st.rerun()

# ── Tweak all ──────────────────────────────────────────────────────────────
if st.session_state.get("do_nova_tweak"):
    st.session_state["do_nova_tweak"] = False
    all_nova = st.session_state.get("nova_classified", [])
    _pname   = st.session_state.get("nova_paper_name", "")
    client   = OpenAI(api_key=OPENAI_KEY)

    to_tweak = [x for x in all_nova if not x.get("isParent") and x.get("novaData")]
    progress_bar = st.progress(0, text="Tweaking questions…")
    total = len(to_tweak)

    def _tweak_one(item):
        try:
            nd = tweak_nova_question(client, item, _pname)
            return item["questionNumber"], nd, None
        except Exception as e:
            return item["questionNumber"], None, str(e)

    tweaked_map: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_tweak_one, item): item for item in to_tweak}
        done = 0
        for fut in as_completed(futures):
            qnum, nd, err = fut.result()
            if nd:
                tweaked_map[qnum] = nd
            done += 1
            progress_bar.progress(done / max(total, 1),
                                   text=f"Tweaked {done}/{total}…")

    progress_bar.empty()

    # Apply tweaked data back into all_nova
    for item in all_nova:
        qnum = item.get("questionNumber", "")
        if qnum in tweaked_map:
            item["novaData"]  = tweaked_map[qnum]
            item["tweaked"]   = True

    n_tweaked  = len(tweaked_map)
    n_physical = sum(1 for item in all_nova
                     if item.get("tweaked") and
                     (item.get("novaData") or {}).get("novaType") != "physical")
    st.session_state["nova_classified"] = all_nova
    st.success(f"✅ {n_tweaked} questions tweaked · physical questions converted")
    st.rerun()

# ── Display classified records ─────────────────────────────────────────────
if "nova_classified" in st.session_state:
    all_nova  = st.session_state["nova_classified"]
    pname     = st.session_state.get("nova_paper_name", "")

    # ── Filters ────────────────────────────────────────────────────────────
    all_types = sorted({
        (x.get("novaData") or {}).get("novaType", "")
        for x in all_nova
        if not x.get("isParent") and (x.get("novaData") or {}).get("novaType")
    })
    fcol1, fcol2 = st.columns([2, 3])
    with fcol1:
        type_filter = st.selectbox(
            "Filter by type",
            ["All types"] + [NOVA_TYPE_LABELS.get(t, t) for t in all_types],
            key="nova_type_filter",
        )
    with fcol2:
        search_q = st.text_input(
            "Search question number / text",
            key="nova_search",
            placeholder="e.g. 7a  or  area",
            label_visibility="collapsed",
        )

    # ── Download CSV ───────────────────────────────────────────────────────
    non_parent = [x for x in all_nova if not x.get("isParent")]
    csv_data   = nova_records_to_csv(non_parent)

    dl_col, sync_col = st.columns([1, 2])
    with dl_col:
        st.download_button(
            "⬇ Download all as CSV",
            data=csv_data.encode(),
            file_name=f"{pname or 'nova'}_questions.csv",
            mime="text/csv",
        )

    with sync_col:
        if not (AT_TOKEN and AT_BASE):
            st.warning("Add Airtable credentials in the sidebar to sync.")
        else:
            nova_tbl_name = st.text_input(
                "Nova table name",
                value="Nova Questions",
                key="nova_sync_table",
                help="All papers are appended to this one shared table. The Paper Name field tells them apart.",
            )
            st.caption(
                "All papers are appended to **one shared table**. "
                "Set up the 7 views once and they work for every paper you sync."
            )
            if st.button("🚀 Sync Nova records to Airtable", type="primary",
                         key="nova_sync_btn"):
                st.session_state["do_nova_sync"] = True
                st.session_state.pop("nova_sync_log", None)
                st.rerun()

    if st.session_state.get("do_nova_sync"):
        st.session_state["do_nova_sync"] = False
        _tbl   = st.session_state.get("nova_sync_table", "Nova Questions")
        _items = list(all_nova)  # includes parents (preambles)

        # Re-crop any boxes that lost image data after session restore
        _sync_pdf = get_pdf()
        if _sync_pdf:
            _store = boxes()
            for _pn in _store:
                for _b in _store[_pn]:
                    if not _b.get("data"):
                        try:
                            _b["data"] = crop_from_rel(_sync_pdf, _b["page"], _b["rel"])
                        except Exception:
                            pass
                set_page_boxes(_pn, _store[_pn])
        with st.status("Syncing to Airtable…", expanded=True) as _status:
            try:
                _n, _logs = push_nova_to_airtable(
                    AT_TOKEN, AT_BASE, _tbl, _items, pname,
                    boxes_list=all_boxes(),
                    cld_cloud=CLD_CLOUD,
                    cld_preset=CLD_PRESET,
                    meta={
                        "exam_board":   st.session_state.get("exam_board",   ""),
                        "subject":      st.session_state.get("subject",      ""),
                        "level":        st.session_state.get("level",        ""),
                        "year":         st.session_state.get("year",         ""),
                        "paper_number": st.session_state.get("paper_number", ""),
                    })
                for line in _logs:
                    st.write(line)
                _status.update(
                    label=f"✅ {_n} records pushed to '{_tbl}'",
                    state="complete")
                st.session_state["nova_sync_log"] = _logs
                # Free classification results from memory — no longer needed
                st.session_state.pop("nova_classified", None)
                st.session_state.pop("nova_paper_name", None)
                # Free box image bytes — already uploaded to Cloudinary
                for _pn in boxes():
                    for _b in boxes()[_pn]:
                        _b["data"] = b""
            except Exception as e:
                st.error(f"Sync failed: {e}")
                _status.update(label="❌ Sync failed", state="error")

    if "nova_sync_log" in st.session_state:
        st.markdown(f"[Open base in Airtable →](https://airtable.com/{AT_BASE})")

    st.divider()

    # ── Helper: render one field row ────────────────────────────────────────
    def _field(label: str, value: str, multiline: bool = False,
               height: int = 100, hint: str = "", key: str = ""):
        st.caption(f"**{label}**" + (f"  ·  *{hint}*" if hint else ""))
        if multiline:
            st.text_area("", value=str(value or ""), height=height,
                         key=key, label_visibility="collapsed")
        else:
            st.text_input("", value=str(value or ""), key=key,
                          label_visibility="collapsed")

    # ── Helper: render type-specific fields ─────────────────────────────────
    def _render_nova_fields(nd: dict, qn: str, uid: str = ""):
        nt = nd.get("novaType", "simple")

        # Shared fields
        col_a, col_b = st.columns(2)
        with col_a:
            _field("Friendly Name / Question",
                   nd.get("friendlyName", ""),
                   hint="Name shown in Nova — same for both fields",
                   key=f"fn_{uid or qn}")
        with col_b:
            mc1, mc2 = st.columns(2)
            with mc1:
                _field("Marks", str(nd.get("marks", "")), key=f"marks_{uid or qn}")
            with mc2:
                _field("Difficulty", str(nd.get("difficulty", 1)), key=f"diff_{uid or qn}")

        _field("Body", nd.get("body", ""), multiline=True, height=120,
               hint="Question text shown on the course. Maths in [latex]...[/latex]",
               key=f"body_{uid or qn}")

        if nt == "simple":
            col1, col2, col3 = st.columns(3)
            with col1:
                _field("Answer Prefix",
                       nd.get("answerPrefix", ""),
                       hint="Shown before the answer box",
                       key=f"apfx_{uid or qn}")
            with col2:
                _field("Answer",
                       nd.get("answer", ""),
                       hint="Exact answer (not in latex)",
                       key=f"ans_{uid or qn}")
            with col3:
                _field("Answer Unit",
                       nd.get("answerUnit", ""),
                       hint="Shown after the answer box",
                       key=f"aunit_{uid or qn}")

        elif nt == "multiple_choice":
            _field("Style", nd.get("style", "List"),
                   hint="List or Grid",
                   key=f"mcstyle_{uid or qn}")
            opts    = nd.get("options") or []
            correct = [o for o in opts if o.get("correct")]
            wrong   = [o for o in opts if not o.get("correct")]
            ordered = correct + wrong
            labels  = ["A", "B", "C", "D"]
            for i, label in enumerate(labels):
                opt = ordered[i] if i < len(ordered) else {}
                display_label = f"Option {label} (correct)" if label == "A" else f"Option {label}"
                _field(display_label,
                       str(opt.get("text", "") or ""),
                       key=f"mcopt_{uid or qn}_{i}")

        elif nt == "multiple_answer":
            _field("Require Specific Order",
                   "Yes" if nd.get("requireSpecificOrder") else "No",
                   hint="Tick if order matters",
                   key=f"maord_{uid or qn}")
            st.caption("**Answer Boxes**")
            answers = nd.get("answers") or []
            for i in range(1, 5):
                ans = answers[i-1] if i-1 < len(answers) else {}
                if not ans and i > 2:
                    continue  # skip empty optional boxes
                ac1, ac2, ac3 = st.columns(3)
                with ac1:
                    _field(f"Box {i} Prefix", str(ans.get("prefix","") or ""),
                           hint="e.g. [latex]x=[/latex]",
                           key=f"mapfx_{uid or qn}_{i}")
                with ac2:
                    _field(f"Box {i} Answer", str(ans.get("answer","") or ""),
                           key=f"maans_{uid or qn}_{i}")
                with ac3:
                    _field(f"Box {i} Suffix", str(ans.get("suffix","") or ""),
                           key=f"masfx_{uid or qn}_{i}")

        elif nt == "fraction":
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                _field("Answer Label", nd.get("answerLabel", ""),
                       hint="Shown before fraction", key=f"flabel_{uid or qn}")
            with fc2:
                _field("Numerator (top)", nd.get("numerator", ""),
                       key=f"fnum_{uid or qn}")
            with fc3:
                _field("Denominator (bottom)", nd.get("denominator", ""),
                       key=f"fden_{uid or qn}")
            st.info("💡 Remember to state in the question body that the answer must be in its simplest form.", icon="ℹ️")

        elif nt == "fill_in_blank":
            _field("Body / Preamble",
                   nd.get("preamble", nd.get("body", "")),
                   multiline=True, height=80,
                   hint="Text shown before the sentence with blanks",
                   key=f"fibpre_{uid or qn}")
            _field("Blank Question Content",
                   nd.get("blankContent", ""),
                   multiline=True, height=80,
                   hint="Use [blank] where each dropdown goes",
                   key=f"fibcont_{uid or qn}")
            st.caption("**Blanks**  ·  *For each [blank] in the content above*")
            blanks = nd.get("blanks") or []
            for bi, blank in enumerate(blanks):
                st.caption(f"Blank {bi+1}")
                bc1, bc2, bc3 = st.columns([2, 2, 1])
                with bc1:
                    opts_str = " | ".join(blank.get("options") or [])
                    _field("Options (pipe-separated)",
                           opts_str,
                           key=f"fibopt_{uid or qn}_{bi}")
                with bc2:
                    _field("Correct Answer",
                           blank.get("correct", ""),
                           key=f"fibcorr_{uid or qn}_{bi}")
                with bc3:
                    _field("Marks",
                           str(blank.get("marks", 1)),
                           key=f"fibmarks_{uid or qn}_{bi}")

        elif nt == "essay":
            _field("Question for AI",
                   nd.get("questionForAI", ""),
                   multiline=True, height=80,
                   hint="Plain English, no LaTeX — this goes to the AI",
                   key=f"essqai_{uid or qn}")
            _field("ChatGPT Model", "gpt-4.1",
                   hint="Use gpt-4.1; fall back to 'default' if unavailable",
                   key=f"essgpt_{uid or qn}")
            _field("AI Role Prompt",
                   nova_ai_role_prompt(
                       st.session_state.get("subject", ""),
                       st.session_state.get("level",   "")),
                   multiline=True, height=80,
                   key=f"essrole_{uid or qn}")
            _field("AI Marking Criteria",
                   nd.get("aiMarkingCriteria", ""),
                   multiline=True, height=150,
                   hint="Ends with 'There are X possible marks…'",
                   key=f"essaic_{uid or qn}")
            _field("Marking Criteria (shown to student)",
                   nd.get("markingCriteriaForStudent",
                          nd.get("writtenSolution", "")),
                   multiline=True, height=100,
                   key=f"essstuc_{uid or qn}")
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                _field("Pass Marks",
                       str(nd.get("marks", "")),
                       hint="Same as Marks",
                       key=f"esspass_{uid or qn}")
            with pc2:
                _field("Min Word Count", "1", key=f"esswc_{uid or qn}")
            with pc3:
                _field("Marking Method", "AI", key=f"essmm_{uid or qn}")

        # Written solution (all types except essay which shows it above differently)
        if nt not in ("essay", "physical"):
            _field("Written Solution",
                   nd.get("writtenSolution", ""),
                   multiline=True, height=100,
                   hint="Mark scheme / worked answer shown after submission",
                   key=f"ws_{uid or qn}")
        elif nt == "physical":
            st.info("⚠️ This question requires physical interaction with the paper and cannot be completed digitally.", icon="✏️")
            _field("Written Solution",
                   nd.get("writtenSolution", ""),
                   multiline=True, height=80,
                   hint="Describe what the correct answer looks like",
                   key=f"ws_{uid or qn}")

    # ── Helper: re-classify single record ──────────────────────────────────
    def _reclassify_button(item: dict, key: str):
        if st.button("🔄 Re-classify", key=f"reclassify_{key}",
                     help="Re-run AI classification for this question only"):
            _c = OpenAI(api_key=OPENAI_KEY)
            _p = st.session_state.get("nova_paper_name", "")
            try:
                nd = classify_nova_question(_c, item["originalRecord"], _p)
                item["novaData"] = nd
                item["error"]    = None
            except Exception as e:
                item["error"] = str(e)
            st.session_state["nova_classified"] = all_nova
            st.rerun()

    # ── Build display order (multi-part groups first, then standalone) ──────
    # Index by question number
    nova_by_qn   = {normalise_qnum(x["questionNumber"]): x for x in all_nova}
    parent_items_disp = [x for x in all_nova if x.get("isParent")]
    shown_qnums: set[str] = set()
    display_order: list[dict] = []  # each item: {"type": "group"|"standalone", "data": ...}

    for pi in parent_items_disp:
        grp_children = [
            nova_by_qn[cqn]
            for cqn in pi.get("childQnums", [])
            if cqn in nova_by_qn
        ]
        display_order.append({
            "dtype":    "group",
            "parent":   pi,
            "children": grp_children,
        })
        shown_qnums.add(normalise_qnum(pi["questionNumber"]))
        for c in grp_children:
            shown_qnums.add(normalise_qnum(c["questionNumber"]))

    for item in all_nova:
        qn = normalise_qnum(item["questionNumber"])
        if qn not in shown_qnums and not item.get("isParent"):
            display_order.append({"dtype": "standalone", "item": item})
            shown_qnums.add(qn)

    # ── Apply filters ────────────────────────────────────────────────────────
    def _type_of(item: dict) -> str:
        return (item.get("novaData") or {}).get("novaType", "")

    def _matches_filter(item: dict) -> bool:
        if type_filter != "All types":
            label = NOVA_TYPE_LABELS.get(_type_of(item), _type_of(item))
            if label != type_filter:
                return False
        if search_q:
            sq = search_q.lower()
            qn = item.get("questionNumber", "").lower()
            qt = (item.get("originalRecord") or {}).get("questionText", "").lower()
            if sq not in qn and sq not in qt:
                return False
        return True

    def _group_matches(grp: dict) -> bool:
        if type_filter != "All types":
            if not any(_matches_filter(c) for c in grp["children"]):
                return False
        if search_q:
            if not any(_matches_filter(c) for c in grp["children"]):
                parent_qt = (grp["parent"].get("originalRecord") or {}).get(
                    "questionText", "").lower()
                if search_q.lower() not in parent_qt:
                    return False
        return True

    # ── Render ─────────────────────────────────────────────────────────────
    rendered = 0
    for entry in display_order:

        if entry["dtype"] == "standalone":
            item = entry["item"]
            if not _matches_filter(item):
                continue
            nd = item.get("novaData") or {}
            nt = nd.get("novaType", "?")
            color = NOVA_TYPE_COLORS.get(nt, "#7f8c8d")
            label = NOVA_TYPE_LABELS.get(nt, nt)
            qn    = item.get("questionNumber", "?")
            header = (
                f"<span style='background:{color};color:#fff;padding:2px 8px;"
                f"border-radius:4px;font-size:0.75em;font-weight:600;"
                f"margin-right:8px'>{label}</span>"
                f"<strong>Q{qn}</strong>"
            )
            if item.get("error"):
                header += f"  <span style='color:#e74c3c;font-size:0.8em'>⚠ {item['error'][:60]}</span>"
            tweaked_badge = " ✏️" if item.get("tweaked") else ""
            with st.expander(f"Q{qn} · {label}{tweaked_badge}", expanded=False):
                st.markdown(header, unsafe_allow_html=True)
                st.caption(
                    f"*Original:* {(item.get('originalRecord') or {}).get('questionText', '')[:120]}")
                if item.get("error"):
                    st.error(f"Classification error: {item['error']}")
                    _reclassify_button(item, qn)
                else:
                    _render_nova_fields(nd, qn, uid=f"s{rendered}")
                    _reclassify_button(item, qn)
            rendered += 1

        else:  # group (multi-part)
            grp = entry
            if not _group_matches(grp):
                continue
            parent = grp["parent"]
            children = grp["children"]
            pqn    = parent.get("questionNumber", "?")
            ptext  = (parent.get("originalRecord") or {}).get("questionText", "")
            child_types = [
                NOVA_TYPE_LABELS.get(_type_of(c), "?")
                for c in children if not c.get("isParent")
            ]
            summary = ", ".join(dict.fromkeys(child_types)) or "sub-questions"
            with st.expander(
                f"Q{pqn} · Multi-part  ({len(children)} sub-questions: {summary})",
                expanded=False,
            ):
                st.markdown(
                    f"<span style='background:#2c3e50;color:#fff;padding:2px 8px;"
                    f"border-radius:4px;font-size:0.75em;font-weight:600'>"
                    f"Multi-part</span>  <strong>Q{pqn}</strong>",
                    unsafe_allow_html=True,
                )
                st.markdown("**Shared preamble** *(paste into the Multi-part container in Nova)*")
                st.text_area(
                    "",
                    value=ptext,
                    height=100,
                    key=f"mp_preamble_{pqn}",
                    label_visibility="collapsed",
                )
                st.caption(
                    "In Nova: create a Multi-part question, paste the preamble above, "
                    "then attach each sub-question below."
                )
                st.divider()

                for child in children:
                    nd  = child.get("novaData") or {}
                    nt  = nd.get("novaType", "?")
                    cqn = child.get("questionNumber", "?")
                    color = NOVA_TYPE_COLORS.get(nt, "#7f8c8d")
                    label = NOVA_TYPE_LABELS.get(nt, nt)
                    st.markdown(
                        f"<span style='background:{color};color:#fff;padding:2px 6px;"
                        f"border-radius:4px;font-size:0.72em;font-weight:600;"
                        f"margin-right:6px'>{label}</span>"
                        f"<strong>Sub-question Q{cqn}</strong>",
                        unsafe_allow_html=True,
                    )
                    if child.get("error"):
                        st.error(f"Classification error: {child['error']}")
                        _reclassify_button(child, cqn)
                    else:
                        _render_nova_fields(nd, cqn, uid=f"g{rendered}_{cqn}")
                        _reclassify_button(child, cqn)
                    st.divider()
            rendered += 1

    if rendered == 0:
        st.info("No questions match the current filter.")
