"""
app.py  –  Past Paper → Airtable  (Streamlit Cloud)
=====================================================
Secrets (set in Streamlit Cloud dashboard or .streamlit/secrets.toml):
    ANTHROPIC_API_KEY  = "sk-ant-..."
    AIRTABLE_TOKEN     = "patXXXX..."
    AIRTABLE_BASE_ID   = "appXXXX..."
    IMGBB_API_KEY      = "xxxxxxxx..."

requirements.txt:
    streamlit anthropic pymupdf pillow requests pandas
"""

import io, json, re, base64, zipfile, time
from pathlib import Path

import streamlit as st
import anthropic
import requests
import fitz
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────
MODEL       = "claude-sonnet-4-6"
MAX_TOKENS  = 8000
CHUNK_PAGES = 8
MAX_RETRIES = 4
RETRY_DELAY = 20
AT_API      = "https://api.airtable.com/v0"
AT_META     = "https://api.airtable.com/v0/meta"
IMGBB_API   = "https://api.imgbb.com/1/upload"

AT_FIELDS = [
    ("Question Number",    "singleLineText"),
    ("Question Text",      "multilineText"),
    ("Mark Allocation",    "number"),
    ("Topic",              "singleLineText"),
    ("Subtopic",           "singleLineText"),
    ("Mark Scheme Answer", "multilineText"),
    ("Image Description",  "multilineText"),
    ("Has Images",         "checkbox"),
    ("Images",             "multipleAttachments"),
    ("Paper Name",         "singleLineText"),
    ("Exam Type",          "singleLineText"),
]

# ── Secrets ───────────────────────────────────────────────────────────────
def get_secret(key: str, fallback: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return fallback

# ── PDF / image helpers ───────────────────────────────────────────────────
def pdf_to_b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode()

def clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*",     "", raw)
    raw = re.sub(r"\s*```$",     "", raw)
    return raw.strip()

def extract_images(pdf_bytes: bytes) -> list[dict]:
    doc, images = fitz.open(stream=pdf_bytes, filetype="pdf"), []
    for page_num, page in enumerate(doc, 1):
        for idx, img_info in enumerate(page.get_images(full=True), 1):
            try:
                bi  = doc.extract_image(img_info[0])
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

def split_pdf(pdf_bytes: bytes, chunk_size: int) -> list[bytes]:
    doc, chunks = fitz.open(stream=pdf_bytes, filetype="pdf"), []
    for start in range(0, len(doc), chunk_size):
        w = fitz.open()
        w.insert_pdf(doc, from_page=start, to_page=min(start + chunk_size, len(doc)) - 1)
        buf = io.BytesIO()
        w.save(buf)
        chunks.append(buf.getvalue())
        w.close()
    doc.close()
    return chunks

# ── Claude ────────────────────────────────────────────────────────────────
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
    for attempt in range(1, MAX_RETRIES + 1):
        try:
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
        except anthropic.RateLimitError:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"❌ Rate limit hit after {MAX_RETRIES} retries. "
                    "See console.anthropic.com/settings/limits or reduce CHUNK_PAGES."
                )
            wait = RETRY_DELAY * attempt
            st.toast(f"Rate limit — waiting {wait}s (retry {attempt}/{MAX_RETRIES - 1})…")
            time.sleep(wait)
        except anthropic.AuthenticationError:
            raise RuntimeError("❌ Invalid Anthropic API key.")
        except anthropic.PermissionDeniedError:
            raise RuntimeError("❌ Anthropic key has no permission.")
        except anthropic.BadRequestError as e:
            raise RuntimeError(f"❌ Bad request (PDF too large or malformed): {e}")
        except anthropic.APIStatusError as e:
            raise RuntimeError(f"❌ Anthropic API error {e.status_code}: {e.message}")

# ── Airtable ──────────────────────────────────────────────────────────────
def at_headers(token: str):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def ensure_table(token: str, base_id: str, table: str):
    try:
        r = requests.get(f"{AT_META}/bases/{base_id}/tables", headers=at_headers(token))
        if r.status_code == 401:
            st.info("ℹ️ Skipping auto table-creation (token lacks schema.bases:write).")
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
                           json={"name": table, "fields": fields})
        if not r2.ok:
            st.warning(f"Could not auto-create table ({r2.status_code}). Create it manually.")
    except Exception as e:
        st.warning(f"Table check skipped: {e}")

# ── imgbb ─────────────────────────────────────────────────────────────────
def upload_to_imgbb(api_key: str, img: dict) -> str | None:
    b64 = base64.standard_b64encode(img["data"]).decode()
    resp = requests.post(IMGBB_API, data={"key": api_key, "name": img["name"], "image": b64})
    if resp.ok:
        return resp.json()["data"]["url"]
    return None

def patch_record_images(token: str, base_id: str, table: str, record_id: str, urls: list[str]):
    resp = requests.patch(
        f"{AT_API}/{base_id}/{requests.utils.quote(table)}/{record_id}",
        headers=at_headers(token),
        json={"fields": {"Images": [{"url": u} for u in urls]}},
    )
    if not resp.ok:
        raise RuntimeError(f"{resp.status_code} — {resp.text[:200]}")

# ── Streamlit UI ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Past Paper → Airtable", page_icon="📄", layout="wide")
st.title("📄 Past Paper → Airtable")
st.caption("Upload exam PDFs, extract questions with AI, review, then sync to Airtable.")

ANTH_KEY  = get_secret("ANTHROPIC_API_KEY")
AT_TOKEN  = get_secret("AIRTABLE_TOKEN")
AT_BASE   = get_secret("AIRTABLE_BASE_ID")
IMGBB_KEY = get_secret("IMGBB_API_KEY")

with st.sidebar:
    st.header("⚙️ Configuration")
    if not ANTH_KEY:
        ANTH_KEY = st.text_input("Anthropic API key", type="password", placeholder="sk-ant-...")
    else:
        st.success("✓ Anthropic key loaded")
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
    client      = anthropic.Anthropic(api_key=ANTH_KEY)
    paper_bytes = paper_file.read()

    with st.status("Extracting…", expanded=True) as status:

        st.write("📎 Extracting embedded images…")
        images = extract_images(paper_bytes)
        if ms_file:
            images += extract_images(ms_file.read())
            ms_file.seek(0)
        st.write(f"   Found {len(images)} images")

        st.write("🤖 Sending paper to Claude…")
        try:
            chunks    = split_pdf(paper_bytes, CHUNK_PAGES)
            questions = []
            for i, chunk in enumerate(chunks, 1):
                offset = (i - 1) * CHUNK_PAGES
                st.write(f"   Chunk {i}/{len(chunks)} (pages {offset+1}–{offset+CHUNK_PAGES})…")
                raw  = ask_claude(client, pdf_to_b64(chunk),
                                  QUESTION_PROMPT.format(name=paper_name, etype=exam_type))
                rows = json.loads(clean_json(raw))
                for q in rows:
                    q["pageNumber"] = (q.get("pageNumber") or 1) + offset
                questions.extend(rows)
                if i < len(chunks):
                    time.sleep(5)
            st.write(f"   Extracted {len(questions)} questions")
        except Exception as e:
            status.update(label="Extraction failed", state="error")
            st.error(str(e))
            st.stop()

        ms_map: dict[str, str] = {}
        if ms_file:
            ms_file.seek(0)
            ms_bytes = ms_file.read()
            st.write("🤖 Sending mark scheme to Claude…")
            try:
                ms_chunks = split_pdf(ms_bytes, CHUNK_PAGES)
                for i, chunk in enumerate(ms_chunks, 1):
                    st.write(f"   Mark scheme chunk {i}/{len(ms_chunks)}…")
                    raw_ms = ask_claude(client, pdf_to_b64(chunk), MS_PROMPT)
                    for item in json.loads(clean_json(raw_ms)):
                        ms_map[str(item["questionNumber"]).strip()] = item["markSchemeAnswer"]
                    if i < len(ms_chunks):
                        time.sleep(5)
                st.write(f"   Matched {len(ms_map)} mark scheme entries")
            except Exception as e:
                st.warning(f"⚠️ Mark scheme failed: {e}")
                ms_map = {}

        records = []
        for q in questions:
            qnum      = str(q.get("questionNumber", "")).strip()
            page      = q.get("pageNumber") or 1
            chunk_idx = (page - 1) // CHUNK_PAGES   # which chunk this question came from
            records.append({
                "questionNumber":   qnum,
                "questionText":     q.get("questionText",     ""),
                "markAllocation":   q.get("markAllocation",   0),
                "topic":            q.get("topic",            ""),
                "subtopic":         q.get("subtopic",         ""),
                "markSchemeAnswer": ms_map.get(qnum,          ""),
                "imageDescription": q.get("imageDescription", ""),
                "hasImages":        bool(q.get("hasImages", False) or
                                        any(i["page"] == page for i in images)),
                "pageNumber":       page,
                "chunkIdx":         chunk_idx,
                "paperName":        paper_name,
                "examType":         exam_type,
            })

        st.session_state["records"] = records
        st.session_state["images"]  = images
        status.update(label=f"✅ Done — {len(records)} questions extracted", state="complete")

# ── Step 3: Review ────────────────────────────────────────────────────────
if "records" in st.session_state:
    import pandas as pd

    records = st.session_state["records"]
    images  = st.session_state.get("images", [])

    st.subheader("3 · Review & edit")
    st.caption("Click any cell to edit before syncing.")

    df = pd.DataFrame([{
        "Q #":           r["questionNumber"],
        "Question Text": r["questionText"],
        "Marks":         r["markAllocation"],
        "Topic":         r["topic"],
        "Subtopic":      r["subtopic"],
        "Mark Scheme":   r["markSchemeAnswer"],
        "Image Desc.":   r["imageDescription"],
        "Has Images":    r["hasImages"],
    } for r in records])

    edited = st.data_editor(df, use_container_width=True, num_rows="dynamic", height=420)

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

    if images:
        with st.expander(f"🖼 Extracted images ({len(images)})"):
            cols = st.columns(4)
            for idx, img in enumerate(images):
                with cols[idx % 4]:
                    st.image(img["data"], caption=img["name"], use_container_width=True)

    # ── Step 4: Export / Sync ─────────────────────────────────────────────
    st.subheader("4 · Export / Sync")
    dl_col, sync_col = st.columns([1, 2])

    with dl_col:
        json_bytes = json.dumps(records, indent=2, ensure_ascii=False).encode()
        st.download_button("⬇ Download JSON", data=json_bytes,
                           file_name=f"{paper_name or 'questions'}.json",
                           mime="application/json")
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
        elif not AT_TOKEN.startswith("pat"):
            st.error("❌ Token should start with `pat`. Check secrets.")
        elif not AT_BASE.startswith("app"):
            st.error("❌ Base ID should start with `app`. Check secrets.")
        else:
            token_preview = AT_TOKEN[:8] + "..." + AT_TOKEN[-4:]
            st.caption(f"Token: `{token_preview}` | Base: `{AT_BASE}` | Table: `{AT_TABLE}`")

            if st.button("🚀 Sync to Airtable", type="primary"):
                _records  = st.session_state.get("records", [])
                _images   = st.session_state.get("images",  [])
                _imgbb    = get_secret("IMGBB_API_KEY")

                log_lines = []
                def log(msg):
                    log_lines.append(msg)

                # 1. Push records
                id_map = []
                try:
                    ensure_table(AT_TOKEN, AT_BASE, AT_TABLE)
                    url   = f"{AT_API}/{AT_BASE}/{requests.utils.quote(AT_TABLE)}"
                    total = len(_records)
                    for i in range(0, total, 10):
                        chunk = _records[i:i+10]
                        body  = {"records": [{"fields": {
                            "Question Number":    str(r.get("questionNumber","")),
                            "Question Text":      str(r.get("questionText","")),
                            "Mark Allocation":    int(r["markAllocation"]) if str(r.get("markAllocation","")).lstrip("-").isdigit() else 0,
                            "Topic":              str(r.get("topic","")),
                            "Subtopic":           str(r.get("subtopic","")),
                            "Mark Scheme Answer": str(r.get("markSchemeAnswer","")),
                            "Image Description":  str(r.get("imageDescription","")),
                            "Has Images":         bool(r.get("hasImages", False)),
                            "Paper Name":         str(r.get("paperName","")),
                            "Exam Type":          str(r.get("examType","")),
                        }} for r in chunk]}
                        resp = requests.post(url, headers=at_headers(AT_TOKEN), json=body)
                        if not resp.ok:
                            raise RuntimeError(f"Airtable {resp.status_code}: {resp.text[:300]}")
                        for rec, row in zip(resp.json()["records"], chunk):
                            id_map.append({
                                "record_id":        rec["id"],
                                "hasImages":        row.get("hasImages", False),
                                "imageDescription": row.get("imageDescription", ""),
                                "chunkIdx":         row.get("chunkIdx", 0),
                            })
                    log(f"✅ {total} records pushed to Airtable")
                except Exception as e:
                    log(f"❌ Record sync failed: {e}")

                # 2. Upload images
                log(f"🖼 Images in memory: {len(_images)}")
                log(f"🔑 imgbb key: {'set ✅' if _imgbb else 'MISSING ❌'}")
                log(f"📋 id_map entries: {len(id_map)}")

                if id_map and _images and _imgbb:
                    # Group images by chunk (each chunk = CHUNK_PAGES pages)
                    chunk_img_urls: dict[int, list[str]] = {}
                    for img in _images:
                        cidx = (img["page"] - 1) // CHUNK_PAGES
                        iurl = upload_to_imgbb(_imgbb, img)
                        if iurl:
                            chunk_img_urls.setdefault(cidx, []).append(iurl)
                            log(f"  ✅ {img['name']} (chunk {cidx}) uploaded")
                        else:
                            log(f"  ❌ {img['name']} failed to upload")

                    if chunk_img_urls:
                        patched = 0
                        for m in id_map:
                            if not (m.get("hasImages") or m.get("imageDescription","").strip()):
                                continue
                            cidx     = m.get("chunkIdx", 0)
                            urls     = chunk_img_urls.get(cidx, [])
                            if not urls:
                                continue
                            try:
                                patch_record_images(AT_TOKEN, AT_BASE, AT_TABLE,
                                                    m["record_id"], urls)
                                patched += 1
                            except Exception as e:
                                log(f"  ❌ {m['record_id']}: {e}")
                        log(f"✅ Images attached to {patched} records")
                    else:
                        log("❌ All imgbb uploads failed")
                elif _images and not _imgbb:
                    log("⚠️ IMGBB_API_KEY not set — add it to Streamlit secrets")
                elif not _images:
                    log("ℹ️ No images were extracted from this PDF")

                # Show full log at once — no mid-run reruns
                st.text("\n".join(log_lines))
                st.markdown(f"[Open in Airtable →](https://airtable.com/{AT_BASE})")
