import streamlit as st
import pdfplumber
import io
import gc
from pypdf import PdfReader, PdfWriter
from transformers import pipeline
import spacy
from collections import Counter
from fpdf import FPDF

# ------------------ 1. SYSTEM INITIALIZATION ------------------
st.set_page_config(page_title="Nexus Intelligence | Pro", page_icon="⚛️", layout="wide")

if 'summary_cache' not in st.session_state: st.session_state.summary_cache = ""
if 'keywords_cache' not in st.session_state: st.session_state.keywords_cache = []
if 'question_cache' not in st.session_state: st.session_state.question_cache = []
if 'last_file' not in st.session_state: st.session_state.last_file = None

# ===================== HIGH-CONTRAST & SPACING UI =====================
st.markdown("""
<style>
    .stApp { background-color: #0F172A; color: #FFFFFF; }
    
    /* SIDEBAR SPACING: Increased gap between radio options */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] { 
        gap: 45px !important; 
        padding-top: 30px; 
    }
    [data-testid="stSidebar"] { background-color: #1E293B !important; border-right: 2px solid #334155; }
    [data-testid="stSidebar"] .stRadio label p { color: #FFFFFF !important; font-size: 1.1rem !important; }

    /* UPLOADER: Hide the default 'Browse files' text for a cleaner look */
    [data-testid="stFileUploader"] section button {
        display: none !important;
    }
    [data-testid="stFileUploader"] {
        border: 2px dashed #4F46E5;
        border-radius: 12px;
        padding: 20px;
    }

    .main-header {
        background: linear-gradient(90deg, #4F46E5, #7C3AED);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
    }

    .content-card {
        background: #1E293B;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        color: #F1F5F9;
    }

    .kw-pill {
        display: inline-block;
        background: #6366F1;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 5px;
    }

    .q-card {
        background: rgba(79, 70, 229, 0.15);
        border-left: 5px solid #818CF8;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 8px;
    }

    .stButton > button {
        width: 100%;
        background: #4F46E5 !important;
        color: white !important;
        font-weight: bold !important;
        height: 3.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ 2. PERMANENT SIDEBAR ------------------
with st.sidebar:
    st.markdown("<h2 style='color:white; text-align:center;'>⚛️ NEXUS CORE</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    module = st.radio("WORKSTREAM", ["Executive Summary", "Ask Questions", "PDF Splitter"], index=0)
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    if st.button("🗑️ RESET SYSTEM"):
        st.session_state.clear()
        st.rerun()

# ------------------ 3. CORE ENGINE ------------------
@st.cache_resource
def load_models():
    real_ai = pipeline("summarization", model="t5-small", device=-1)
    nlp = spacy.load("en_core_web_sm")
    return real_ai, nlp

real_ai, nlp = load_models()

def clean_txt(text):
    return text.encode('latin-1', 'replace').decode('latin-1')

# ------------------ 4. MAIN WORKSPACE ------------------
st.markdown('<div class="main-header"><h1>Intelligence Studio Pro</h1><p style="color: #E2E8F0;">Neural Synthesis & Document Analytics</p></div>', unsafe_allow_html=True)

file_source = st.file_uploader("Drop PDF here to activate modules", type="pdf")

if file_source:
    if st.session_state.last_file != file_source.name:
        st.session_state.summary_cache = ""
        st.session_state.keywords_cache = []
        st.session_state.question_cache = []
        st.session_state.last_file = file_source.name
        st.rerun()

    pdf_reader = PdfReader(file_source)
    total_pages = len(pdf_reader.pages)

    if module == "Executive Summary":
        if st.button("🚀 GENERATE ANALYSIS"):
            with st.status("Reading & Processing...") as status:
                try:
                    with pdfplumber.open(file_source) as pdf:
                        target_pages = [0, total_pages//2, total_pages-1]
                        raw_text = ""
                        for p in target_pages:
                            txt = pdf.pages[p].extract_text()
                            if txt: raw_text += txt + " "
                    
                    chunks = [raw_text[i:i+900] for i in range(0, min(len(raw_text), 2700), 900)]
                    summaries = []
                    for chunk in chunks:
                        res = real_ai(chunk, max_length=50, min_length=20, do_sample=False)[0]['summary_text']
                        summaries.append(res)
                    st.session_state.summary_cache = ". ".join(summaries).replace(" .", ".")
                    
                    doc_k = nlp(raw_text[:8000].lower())
                    kws = [t.text for t in doc_k if t.pos_ in ["NOUN", "PROPN"] and not t.is_stop and len(t.text) > 4]
                    st.session_state.keywords_cache = [w.upper() for w, c in Counter(kws).most_common(6)]
                    status.update(label="Analysis Ready", state="complete")
                except: st.error("Error processing document.")

        if st.session_state.summary_cache:
            st.markdown(f'<div class="content-card"><b>Neural Summary:</b><br><br>{st.session_state.summary_cache}</div>', unsafe_allow_html=True)
            
            # Keywords Section
            if st.session_state.keywords_cache:
                kw_html = "".join([f'<span class="kw-pill">{k}</span>' for k in st.session_state.keywords_cache])
                st.markdown(f'<div style="margin-top:20px; padding:10px;"><b>Key Themes Identified:</b><br>{kw_html}</div>', unsafe_allow_html=True)
            
            # PDF DOWNLOAD
            pdf_gen = FPDF()
            pdf_gen.add_page(); pdf_gen.set_font("Arial", size=12)
            pdf_gen.multi_cell(0, 10, txt=clean_txt(st.session_state.summary_cache))
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(label="📥 DOWNLOAD REPORT PDF", data=pdf_gen.output(dest='S').encode('latin-1'), file_name="Nexus_Report.pdf", mime="application/pdf")

    elif module == "Ask Questions":
        if st.button("🔍 ANALYZE SUBJECTS"):
            with st.spinner("Extracting themes..."):
                with pdfplumber.open(file_source) as pdf:
                    text = (pdf.pages[0].extract_text() or "") + " " + (pdf.pages[-1].extract_text() or "")
                doc_q = nlp(text[:12000])
                subjects = list(dict.fromkeys([chunk.text.strip() for chunk in doc_q.noun_chunks if len(chunk.text) > 6]))
                templates = [
                    "What are the primary objectives associated with {}?",
                    "How does the document evaluate the impact of {}?",
                    "What specific risks are noted regarding {}?",
                    "Can you explain the methodology used to address {}?",
                    "What are the long-term implications of {}?",
                    "How is the success of {} measured?",
                    "What evidence supports the findings on {}?",
                    "Are there any compliance factors involving {}?",
                    "How does the report suggest optimizing {}?",
                    "What is the strategic recommendation regarding {}?"
                ]
                if len(subjects) < 10: subjects += ["Operations", "Strategy", "Risk", "Outcome", "Execution"]
                st.session_state.question_cache = [templates[i].format(subjects[i]) for i in range(10)]

        for q in st.session_state.question_cache:
            st.markdown(f'<div class="q-card">{q}</div>', unsafe_allow_html=True)

    elif module == "PDF Splitter":
        st.markdown('<div class="content-card">Select page range:</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        s_p = col1.number_input("Start", 1, total_pages, 1)
        e_p = col2.number_input("End", 1, total_pages, total_pages)
        if st.button("✂️ EXPORT PDF"):
            writer = PdfWriter()
            for i in range(int(s_p)-1, int(e_p)): writer.add_page(pdf_reader.pages[i])
            st.download_button("Download", io.BytesIO(writer.write_stream()).getvalue(), "split.pdf")

gc.collect()
