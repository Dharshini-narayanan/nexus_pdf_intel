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

# ===================== UI STYLING: UNIVERSAL CONTRAST =====================
st.markdown("""
<style>
    /* Force background for the whole app to avoid Light Mode clashes */
    .stApp { background-color: #0F172A !important; color: #FFFFFF !important; }
    
    /* SIDEBAR: Compact spacing */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] { 
        gap: 20px !important; 
        padding-top: 15px; 
    }
    [data-testid="stSidebar"] { background-color: #1E293B !important; border-right: 2px solid #334155; }
    [data-testid="stSidebar"] .stRadio label p { color: #FFFFFF !important; font-size: 1rem !important; }

    /* UPLOADER: Forced visibility for both Light/Dark modes */
    [data-testid="stFileUploader"] {
        border: 2px dashed #4F46E5 !important;
        border-radius: 12px !important;
        padding: 15px !important;
        background-color: #F8FAFC !important; /* Forces a light gray area so the black button pops */
    }
    
    /* THE BUTTON: Guaranteed Solid Black */
    button[data-testid="baseButton-secondary"] {
        background-color: #000000 !important;
        border: 2px solid #000000 !important;
        border-radius: 8px !important;
        padding: 10px 30px !important;
        display: flex !important;
        justify-content: center !important;
    }

    /* THE TEXT: Forced High-Contrast White inside the black button */
    button[data-testid="baseButton-secondary"] div p {
        color: #FFFFFF !important;
        font-weight: 900 !important;
        opacity: 1 !important;
        display: block !important;
    }

    /* UI Elements */
    .main-header {
        background: linear-gradient(90deg, #4F46E5, #7C3AED);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
        color: white !important;
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
        color: white !important;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 4px;
    }

    .stButton > button {
        width: 100%;
        background: #4F46E5 !important;
        color: white !important;
        font-weight: bold !important;
        height: 3rem;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ 2. SIDEBAR ------------------
with st.sidebar:
    st.markdown("<h2 style='color:white; text-align:center;'>⚛️ NEXUS CORE</h2>", unsafe_allow_html=True)
    module = st.radio("WORKSTREAM", ["Executive Summary", "Ask Questions", "PDF Splitter"], index=0)
    st.markdown("<br><hr style='border: 1px solid #334155;'><br>", unsafe_allow_html=True)
    if st.button("🗑️ RESET"):
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
st.markdown('<div class="main-header"><h1>Intelligence Studio Pro</h1><p style="font-size:0.9rem; color: #E2E8F0;">Neural Synthesis & Document Analytics</p></div>', unsafe_allow_html=True)

file_source = st.file_uploader("Upload PDF Document", type="pdf")

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
            with st.status("Analyzing...") as status:
                try:
                    with pdfplumber.open(file_source) as pdf:
                        target_pages = [0, total_pages//2, total_pages-1]
                        raw_text = "".join([p.extract_text() or "" for p in [pdf.pages[i] for i in target_pages]])
                    
                    chunks = [raw_text[i:i+900] for i in range(0, min(len(raw_text), 2700), 900)]
                    summaries = [real_ai(c, max_length=50, min_length=20, do_sample=False)[0]['summary_text'] for c in chunks]
                    st.session_state.summary_cache = ". ".join(summaries).replace(" .", ".")
                    
                    doc_k = nlp(raw_text[:8000].lower())
                    kws = [t.text for t in doc_k if t.pos_ in ["NOUN", "PROPN"] and not t.is_stop and len(t.text) > 4]
                    st.session_state.keywords_cache = [w.upper() for w, c in Counter(kws).most_common(6)]
                    status.update(label="Complete", state="complete")
                except: st.error("Neural processing error.")

        if st.session_state.summary_cache:
            st.markdown(f'<div class="content-card"><b>Neural Summary:</b><br><br>{st.session_state.summary_cache}</div>', unsafe_allow_html=True)
            
            if st.session_state.keywords_cache:
                kw_html = "".join([f'<span class="kw-pill">{k}</span>' for k in st.session_state.keywords_cache])
                st.markdown(f'<div style="margin-top:15px; padding-left:5px; color:white;"><b>Key Themes:</b><br>{kw_html}</div>', unsafe_allow_html=True)
            
            pdf_gen = FPDF()
            pdf_gen.add_page(); pdf_gen.set_font("Arial", size=12)
            pdf_gen.multi_cell(0, 10, txt=clean_txt(st.session_state.summary_cache))
            st.download_button(label="📥 DOWNLOAD REPORT", data=pdf_gen.output(dest='S').encode('latin-1'), file_name="Nexus_Report.pdf", mime="application/pdf")

    elif module == "Ask Questions":
        if st.button("🔍 GENERATE QUESTIONS"):
            with st.spinner("Analyzing themes..."):
                with pdfplumber.open(file_source) as pdf:
                    text = (pdf.pages[0].extract_text() or "") + " " + (pdf.pages[-1].extract_text() or "")
                doc_q = nlp(text[:10000])
                subjects = list(dict.fromkeys([chunk.text.strip() for chunk in doc_q.noun_chunks if len(chunk.text) > 6]))
                templates = ["What are the primary objectives associated with {}?", "How does the document evaluate {}?", "What specific risks involve {}?", "What is the methodology for {}?", "What are the implications of {}?"]
                if len(subjects) < 10: subjects += ["Operations", "Strategy", "Execution", "Risk", "Outcome"]
                st.session_state.question_cache = [templates[i % 5].format(subjects[i]) for i in range(10)]
        for q in st.session_state.question_cache: st.markdown(f'<div class="q-card">{q}</div>', unsafe_allow_html=True)

    elif module == "PDF Splitter":
        col1, col2 = st.columns(2)
        s_p = col1.number_input("Start", 1, total_pages, 1)
        e_p = col2.number_input("End", 1, total_pages, total_pages)
        if st.button("✂️ EXPORT PDF"):
            writer = PdfWriter()
            for i in range(int(s_p)-1, int(e_p)): writer.add_page(pdf_reader.pages[i])
            st.download_button("Download", io.BytesIO(writer.write_stream()).getvalue(), "split.pdf")

gc.collect()

