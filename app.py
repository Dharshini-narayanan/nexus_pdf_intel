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

# ===================== UI STYLING: LIGHT MAIN / DARK SIDEBAR =====================
st.markdown("""
<style>
    /* MAIN BACKGROUND: Light */
    .stApp { 
        background-color: #F8FAFC !important; 
        color: #1E293B !important; 
    }
    
    /* SIDEBAR: Dark & Compact */
    [data-testid="stSidebar"] { 
        background-color: #1E293B !important; 
        border-right: 2px solid #334155; 
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] { 
        gap: 12px !important; 
        padding-top: 10px; 
    }
    [data-testid="stSidebar"] .stRadio label p { 
        color: #FFFFFF !important; 
        font-size: 0.95rem !important; 
    }
    [data-testid="stSidebar"] h3 { color: white !important; }

    /* UPLOADER: Solid background */
    [data-testid="stFileUploader"] {
        border: 2px dashed #4F46E5 !important;
        border-radius: 12px !important;
        background-color: #FFFFFF !important;
    }
    
    /* THE BROWSE BUTTON: Solid RED */
    div[data-testid="stFileUploader"] button {
        background-color: #FF0000 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 25px !important;
        font-weight: bold !important;
    }

    /* Force button text to be White */
    div[data-testid="stFileUploader"] button p {
        color: #FFFFFF !important;
        font-weight: 900 !important;
    }

    .main-header {
        background: linear-gradient(90deg, #4F46E5, #7C3AED);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
        color: white !important;
    }

    .content-card {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        color: #1E293B;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }

    .q-card {
        background: #F1F5F9;
        border-left: 5px solid #4F46E5;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 8px;
        color: #1E293B;
    }

    .kw-pill {
        display: inline-block;
        background: #4F46E5;
        color: white !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 4px;
    }

    /* Main Action Buttons */
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
    st.markdown("<h3 style='text-align:center;'>⚛️ NEXUS CORE</h3>", unsafe_allow_html=True)
    module = st.radio("WORKSTREAM", ["Executive Summary", "Ask Questions", "PDF Splitter"], index=0)
    st.markdown("<br><br>", unsafe_allow_html=True)
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
st.markdown('<div class="main-header"><h1>Intelligence Studio Pro</h1></div>', unsafe_allow_html=True)

file_source = st.file_uploader("Upload PDF to begin analysis", type="pdf")

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
        if st.button("🚀 EXECUTE NEURAL SUMMARY"):
            with st.status("Analyzing...") as status:
                try:
                    with pdfplumber.open(file_source) as pdf:
                        target_pages = [0, total_pages//2, total_pages-1]
                        raw_text = "".join([pdf.pages[i].extract_text() or "" for i in target_pages])
                    
                    chunks = [raw_text[i:i+800] for i in range(0, min(len(raw_text), 2400), 800)]
                    summaries = [real_ai(c, max_length=50, min_length=20, do_sample=False)[0]['summary_text'] for c in chunks]
                    st.session_state.summary_cache = ". ".join(summaries).replace(" .", ".")
                    
                    doc_k = nlp(raw_text[:8000].lower())
                    kws = [t.text for t in doc_k if t.pos_ in ["NOUN", "PROPN"] and not t.is_stop and len(t.text) > 4]
                    st.session_state.keywords_cache = [w.upper() for w, c in Counter(kws).most_common(6)]
                    status.update(label="Complete", state="complete")
                except: st.error("Processing Error.")

        if st.session_state.summary_cache:
            st.markdown(f'<div class="content-card"><b>Summary:</b><br><br>{st.session_state.summary_cache}</div>', unsafe_allow_html=True)
            if st.session_state.keywords_cache:
                kw_html = "".join([f'<span class="kw-pill">{k}</span>' for k in st.session_state.keywords_cache])
                st.markdown(f'<div style="margin-top:10px;"><b>Keywords:</b><br>{kw_html}</div>', unsafe_allow_html=True)
            
            pdf_gen = FPDF()
            pdf_gen.add_page(); pdf_gen.set_font("Arial", size=12)
            pdf_gen.multi_cell(0, 10, txt=clean_txt(st.session_state.summary_cache))
            st.download_button(label="📥 DOWNLOAD PDF", data=pdf_gen.output(dest='S').encode('latin-1'), file_name="Report.pdf")

    elif module == "Ask Questions":
        # RESTORED BUTTON
        if st.button("🔍 GENERATE QUESTIONS"):
            with st.spinner("Mining insights..."):
                with pdfplumber.open(file_source) as pdf:
                    text = (pdf.pages[0].extract_text() or "") + " " + (pdf.pages[-1].extract_text() or "")
                doc_q = nlp(text[:10000])
                subjects = list(dict.fromkeys([chunk.text.strip() for chunk in doc_q.noun_chunks if len(chunk.text) > 6]))
                templates = ["What are the primary objectives associated with {}?", "How does the document evaluate {}?", "What specific risks involve {}?", "What is the methodology for {}?", "What are the implications of {}?"]
                if len(subjects) < 10: subjects += ["Operations", "Strategy", "Execution", "Risk", "Outcome"]
                st.session_state.question_cache = [templates[i % 5].format(subjects[i]) for i in range(10)]
        
        for q in st.session_state.question_cache:
            st.markdown(f'<div class="q-card">{q}</div>', unsafe_allow_html=True)

    elif module == "PDF Splitter":
        col1, col2 = st.columns(2)
        s_p = col1.number_input("Start", 1, total_pages, 1)
        e_p = col2.number_input("End", 1, total_pages, total_pages)
        if st.button("✂️ EXPORT PDF"):
            writer = PdfWriter()
            for i in range(int(s_p)-1, int(e_p)): writer.add_page(pdf_reader.pages[i])
            st.download_button("Download", io.BytesIO(writer.write_stream()).getvalue(), "split.pdf")

gc.collect()
