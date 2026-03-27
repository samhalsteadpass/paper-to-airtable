"""
app.py  –  Past Paper → Airtable  (Streamlit Cloud)
=====================================================
Deploy once, share the URL. No local setup needed.

Secrets (set in Streamlit Cloud dashboard or .streamlit/secrets.toml):
    ANTHROPIC_API_KEY  = "sk-ant-..."
    AIRTABLE_TOKEN     = "patXXXX..."
    AIRTABLE_BASE_ID   = "appXXXX..."

pip / requirements.txt:
    streamlit anthropic pymupdf pillow requests
"""

import io, json, re, base64, zipfile
from pathlib import Path

import streamlit as st
import anthropic
import requests
import fitz          # PyMuPDF
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────
MODEL      = "claude-sonnet-4-6"
MAX_TOKENS = 8000
AT_API     = "https://api.airtable.com/v0"
AT_META    = "https://api.airtable.com/v0/meta"

AT_FIELDS = [
    ("Question Number",    "singleLineText"),
    ("Question Text",      "multilineText"),
    ("Mark Allocation",    "number"),
    ("Topic",              "singleLineText"),
    ("Subtopic",           "singleLineText"),
    ("Mark Scheme Answer", "multilineText"),
    ("Image Description",  "multilineText"),
    ("Has Images",         "checkbox"),
    ("Paper Name",         "singleLineText"),
    ("Exam Type",          "singleLineText"),
]

def get_secret(key: str, fallback: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return fallback

# ── Helpers ───────────────────────────────────────────────────────────────
def pdf_to_b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode()

def clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*",     "", raw)
    raw = re.sub(r"\s*```$",     "", raw)
    return raw.strip()

def extract_images(pdf_bytes: bytes) -> list[dict]:
    """Return list of {page, name, data (bytes), width, height}."""
    doc    = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page_num, page in enumerate(doc, 1):
        for idx, img_info in enumerate(page.get_images(full=True), 1):
            xref = img_info[0]
            try:
                bi  = doc.extract_image(xref)
                img = Image.open(io.BytesIO(bi["image"]))
                if img.width < 60 or img.height < 60:
                    continue
                images.append({
                    "page":   page_num,
                    "name":   f"p{page_num}_img{idx}.{bi['ext']}",
                    "data":   bi["image"],
                    "width":  img.width,
                    "height": img.height,
                })
            except Exception:
                pass
    doc.close()
    return images

# ── Claude calls ──────────────────────────────────────────────────────────
QUESTION_PROMPT = """Extract EVERY question from this exam paper PDF.
Return ONLY a raw JSON array — no markdown fences, no preamble.

Each element:
{{
  "questionNumber":   "1a",
  "questionText":     "Full question including sub-parts and any context passage",
  "markAllocation":   4,
  "topic":            "e.g. Cell Biology",
  "subtopic":         "e.g. Mitosis",
  "hasImages":        true,
  "imageDescription": "Describe every diagram/graph/table in detail. Empty string if none.",
  "pageNumber":       2
}}

Rules:
- Split sub-questions (1a, 1b …) into separate records
- markAllocation must be an integer (0 if absent)
- Attach context text to each related child question
Paper: "{name}", Exam type: "{etype}"
"""

MS_PROMPT = """Extract ALL answers from this mark scheme PDF.
Return ONLY a raw JSON array — no markdown fences, no preamble.

Each element:
{{
  "questionNumber":   "1a",
  "markSchemeAnswer": "Full guidance: acceptable answers, key words, allow/reject/ignore, worked solutions, examiner notes"
}}
"""

def ask_claude(client: anthropic.Anthropic, pdf_b64: str, prompt: str) -> str:
    r = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": [
            {"type": "document",
             "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    return "".join(b.text for b in r.content if b.type == "text")

# ── Airtable ──────────────────────────────────────────────────────────────
def at_headers(token: str):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def ensure_table(token: str, base_id: str, table: str):
    try:
        r = requests.get(f"{AT_META}/bases/{base_id}/tables", headers=at_headers(token))
        if r.status_code == 401:
            st.info("ℹ️ Skipping auto table-creation (token lacks schema.bases:write scope). "
                    "Make sure the table and fields exist manually — see the sidebar for the field list.")
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
            else:
                fields.append({"name": name, "type": ftype})
        r2 = requests.post(f"{AT_META}/bases/{base_id}/tables",
                           headers=at_headers(token),
                           json={"name": table, "fields": fields})
        if not r2.ok:
            st.warning(f"Could not auto-create table ({r2.status_code}). Create it manually — see field list in the sidebar.")
    except Exception as e:
        st.warning(f"Table check skipped: {e}")

def push_to_airtable(token: str, base_id: str, table: str,
                     records: list[dict], progress) -> int:
    url   = f"{AT_API}/{base_id}/{requests.utils.quote(table)}"
    total = len(records)
    pushed = 0
    for i in range(0, total, 10):
        chunk = records[i:i+10]
        body  = {"records": [{"fields": {
            "Question Number":    str(r.get("questionNumber", "")),
            "Question Text":      str(r.get("questionText",   "")),
            "Mark Allocation":    int(r["markAllocation"]) if str(r.get("markAllocation","")).lstrip("-").isdigit() else 0,
            "Topic":              str(r.get("topic",       "")),
            "Subtopic":           str(r.get("subtopic",    "")),
            "Mark Scheme Answer": str(r.get("markSchemeAnswer", "")),
            "Image Description":  str(r.get("imageDescription", "")),
            "Has Images":         bool(r.get("hasImages", False)),
            "Paper Name":         str(r.get("paperName",  "")),
            "Exam Type":          str(r.get("examType",   "")),
        }} for r in chunk]}
        resp = requests.post(url, headers=at_headers(token), json=body)
        if not resp.ok:
            raise RuntimeError(f"Airtable {resp.status_code}: {resp.text[:300]}")
        pushed += len(chunk)
        progress.progress(pushed / total, text=f"Syncing… {pushed}/{total}")
    return pushed

# ── Streamlit UI ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Past Paper → Airtable", page_icon="📄", layout="wide")

st.title("📄 Past Paper → Airtable")
st.caption("Upload exam PDFs, extract questions with AI, review, then sync to Airtable.")

# Read secrets (pre-configured by admin on Streamlit Cloud)
ANTH_KEY  = get_secret("ANTHROPIC_API_KEY")
AT_TOKEN  = get_secret("AIRTABLE_TOKEN")
AT_BASE   = get_secret("AIRTABLE_BASE_ID")

# Sidebar — credentials (shown only if not in secrets)
with st.sidebar:
    st.header("⚙️ Configuration")
    if not ANTH_KEY:
        ANTH_KEY = st.text_input("Anthropic API key", type="password",
                                  placeholder="sk-ant-...")
    else:
        st.success("✓ Anthropic key loaded from secrets")

    if not AT_TOKEN:
        AT_TOKEN = st.text_input("Airtable Personal Access Token", type="password",
                                  placeholder="patXXXXXX")
    else:
        st.success("✓ Airtable token loaded from secrets")

    if not AT_BASE:
        AT_BASE = st.text_input("Airtable Base ID", placeholder="appXXXXXX")
    else:
        st.success("✓ Airtable Base ID loaded from secrets")

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
    exam_type  = st.text_input("Exam type",  placeholder="A-Level / GCSE / IB HL")
    paper_file = st.file_uploader("Past paper PDF *", type="pdf")
with col2:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    ms_file = st.file_uploader("Mark scheme PDF (optional)", type="pdf")

# ── Step 2: Extract ───────────────────────────────────────────────────────
st.subheader("2 · Extract with AI")
if st.button("✨ Extract Questions", type="primary",
             disabled=not (paper_file and paper_name and exam_type and ANTH_KEY)):
    client    = anthropic.Anthropic(api_key=ANTH_KEY)
    paper_bytes = paper_file.read()

    with st.status("Extracting…", expanded=True) as status:

        # Images
        st.write("📎 Extracting embedded images…")
        images = extract_images(paper_bytes)
        if ms_file:
            images += extract_images(ms_file.read())
            ms_file.seek(0)
        st.write(f"   Found {len(images)} images")

        # Questions
        st.write("🤖 Sending paper to Claude…")
        raw_qs = ask_claude(client, pdf_to_b64(paper_bytes),
                            QUESTION_PROMPT.format(name=paper_name, etype=exam_type))
        questions = json.loads(clean_json(raw_qs))
        st.write(f"   Extracted {len(questions)} questions")

        # Mark scheme
        ms_map: dict[str, str] = {}
        if ms_file:
            ms_file.seek(0)
            st.write("🤖 Sending mark scheme to Claude…")
            try:
                raw_ms = ask_claude(client, pdf_to_b64(ms_file.read()), MS_PROMPT)
                for item in json.loads(clean_json(raw_ms)):
                    ms_map[str(item["questionNumber"]).strip()] = item["markSchemeAnswer"]
                st.write(f"   Matched {len(ms_map)} mark scheme entries")
            except Exception as e:
                st.warning(f"⚠️ Mark scheme extraction failed: {e}\nContinuing without mark scheme answers.")
                ms_map = {}

        # Merge
        records = []
        for q in questions:
            qnum = str(q.get("questionNumber", "")).strip()
            records.append({
                "questionNumber":    qnum,
                "questionText":      q.get("questionText",     ""),
                "markAllocation":    q.get("markAllocation",   0),
                "topic":             q.get("topic",            ""),
                "subtopic":          q.get("subtopic",         ""),
                "markSchemeAnswer":  ms_map.get(qnum,          ""),
                "imageDescription":  q.get("imageDescription", ""),
                "hasImages":         bool(q.get("hasImages", False) or
                                         any(i["page"] == q.get("pageNumber") for i in images)),
                "paperName":         paper_name,
                "examType":          exam_type,
            })

        st.session_state["records"] = records
        st.session_state["images"]  = images
        status.update(label=f"✅ Done — {len(records)} questions extracted", state="complete")

# ── Step 3: Review ────────────────────────────────────────────────────────
if "records" in st.session_state:
    records = st.session_state["records"]
    images  = st.session_state.get("images", [])

    st.subheader("3 · Review & edit")
    st.caption("Click any cell to edit before syncing.")

    import pandas as pd
    df = pd.DataFrame([{
        "Q #":            r["questionNumber"],
        "Question Text":  r["questionText"],
        "Marks":          r["markAllocation"],
        "Topic":          r["topic"],
        "Subtopic":       r["subtopic"],
        "Mark Scheme":    r["markSchemeAnswer"],
        "Image Desc.":    r["imageDescription"],
        "Has Images":     r["hasImages"],
    } for r in records])

    edited = st.data_editor(df, use_container_width=True, num_rows="dynamic", height=420)

    # Sync back edits
    col_map = {
        "Q #":           "questionNumber",
        "Question Text": "questionText",
        "Marks":         "markAllocation",
        "Topic":         "topic",
        "Subtopic":      "subtopic",
        "Mark Scheme":   "markSchemeAnswer",
        "Image Desc.":   "imageDescription",
        "Has Images":    "hasImages",
    }
    for i, row in edited.iterrows():
        if i < len(records):
            for col, key in col_map.items():
                records[i][key] = row[col]

    # Image gallery
    if images:
        with st.expander(f"🖼 Extracted images ({len(images)})"):
            cols = st.columns(4)
            for idx, img in enumerate(images):
                with cols[idx % 4]:
                    st.image(img["data"], caption=img["name"], use_container_width=True)

    # Downloads
    st.subheader("4 · Export / Sync")
    dl_col, sync_col = st.columns([1, 2])

    with dl_col:
        # JSON download
        json_bytes = json.dumps(records, indent=2, ensure_ascii=False).encode()
        st.download_button("⬇ Download JSON", data=json_bytes,
                           file_name=f"{paper_name or 'questions'}.json",
                           mime="application/json")

        # Images zip download
        if images:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                for img in images:
                    zf.writestr(img["name"], img["data"])
            st.download_button("⬇ Download Images (.zip)", data=buf.getvalue(),
                               file_name=f"{paper_name or 'images'}_images.zip",
                               mime="application/zip")

    with sync_col:
        if not (AT_TOKEN and AT_BASE):
            st.warning("Add your Airtable token and Base ID in the sidebar to sync.")
        else:
            if st.button("🚀 Sync to Airtable", type="primary"):
                try:
                    ensure_table(AT_TOKEN, AT_BASE, AT_TABLE)
                    prog = st.progress(0, text="Starting…")
                    n = push_to_airtable(AT_TOKEN, AT_BASE, AT_TABLE, records, prog)
                    st.success(f"✅ {n} records synced to Airtable!")
                    st.markdown(f"[Open in Airtable →](https://airtable.com/{AT_BASE})")
                except Exception as e:
                    st.error(f"Sync failed: {e}")
