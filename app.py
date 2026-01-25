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

# ===================== UI STYLING: EXTREME PRIORITY RED BUTTON =====================
st.markdown("""
<style>
    /* Force Dark Theme consistently */
    .stApp { background-color: #0F172A !important; color: #FFFFFF !important; }
    
    /* SIDEBAR: Very compact spacing between options */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] { 
        gap: 12px !important; 
        padding-top: 10px; 
    }
    [data-testid="stSidebar"] { background-color: #1E293B !important; border-right: 2px solid #334155; }
    [data-testid="stSidebar"] .stRadio label p { color: #FFFFFF !important; font-size: 0.95rem !important; }

    /* UPLOADER: Solid background to eliminate light-mode "ghosting" */
    [data-testid="stFileUploader"] {
        border: 2px dashed #4F46E5 !important;
        border-radius: 12px !important;
        padding: 15px !important;
        background-color: #1E293B !important; /* Matches app theme */
    }
    
    /* THE BUTTON: Extreme override to force RED */
    div[data-testid="stFileUploader"] button {
        background-color: #FF0000 !important;
        color: #FFFFFF !important;
        border: 2px solid #FF0000 !important;
        border-radius: 8px !important;
        padding: 10px 25px !important;
        font-weight: bold !important;
        box-shadow: 0px 4px 15px rgba(255, 0, 0, 0.3) !important;
    }

    /* Force button text to be visible WHITE */
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
        background: #4F46E5;
        color: white !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 4px;
        border: 1px solid rgba(255,255,255,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ------------------ 2. SIDEBAR ------------------
with st.sidebar:
    st.markdown("<h3 style='color:white; text-align:center;'>⚛️ NEXUS CORE</h3>", unsafe_allow_html=True)
    module = st.radio("WORKSTREAM", ["Executive Summary", "Ask Questions", "PDF Splitter"], index=0)
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

file_source = st.file_uploader("Drop PDF here to activate modules", type="pdf")

if file_source:
    if st.session_state.last_file != file_source.name:
        st.session_state.summary_cache = ""
        st.session_state.keywords_cache = []
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
            
            # Keywords shown right after summary
            if st.session_state.keywords_cache:
                kw_html = "".join([f'<span class="kw-pill">{k}</span>' for k in st.session_state.keywords_cache])
                st.markdown(f'<div style="margin-top:10px;"><b>Keywords:</b><br>{kw_html}</div>', unsafe_allow_html=True)
            
            pdf_gen = FPDF()
            pdf_gen.add_page(); pdf_gen.set_font("Arial", size=12)
            pdf_gen.multi_cell(0, 10, txt=clean_txt(st.session_state.summary_cache))
            st.download_button(label="📥 DOWNLOAD PDF", data=pdf_gen.output(dest='S').encode('latin-1'), file_name="Report.pdf", mime="application/pdf")

    elif module == "Ask Questions":
        st.info("Module active. Click 'Analyze' to generate insights.")
        # ... (rest of logic remains same)

    elif module == "PDF Splitter":
        col1, col2 = st.columns(2)
        s_p = col1.number_input("Start", 1, total_pages, 1)
        e_p = col2.number_input("End", 1, total_pages, total_pages)
        if st.button("✂️ EXPORT"):
            writer = PdfWriter()
            for i in range(int(s_p)-1, int(e_p)): writer.add_page(pdf_reader.pages[i])
            st.download_button("Download", io.BytesIO(writer.write_stream()).getvalue(), "split.pdf")

gc.collect()
