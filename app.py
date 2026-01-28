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

# ===================== UI STYLING: CLEAN LIGHT THEME / COMPACT SIDEBAR =====================
st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #F8FAFC !important; color: #1E293B !important; }
    
    /* SIDEBAR: Ultra-compact Dark Theme */
    [data-testid="stSidebar"] { background-color: #1E293B !important; border-right: 2px solid #334155; }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] { 
        gap: 5px !important; 
        padding-top: 0px; 
    }
    [data-testid="stSidebar"] .stRadio label p { 
        color: #FFFFFF !important; 
        font-size: 0.85rem !important; 
        margin-bottom: 0px !important;
    }
    [data-testid="stSidebar"] h3 { color: white !important; margin-bottom: 10px; }

    /* UPLOADER: Neat Alignment with RED Button */
    [data-testid="stFileUploader"] {
        border: 2px dashed #CBD5E1 !important;
        border-radius: 10px !important;
        background-color: #FFFFFF !important;
        padding: 8px !important;
    }
    
    /* THE RED BROWSE BUTTON */
    div[data-testid="stFileUploader"] button[kind="secondary"] {
        background-color: #FF0000 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 5px 15px !important;
        font-weight: bold !important;
    }
    
    div[data-testid="stFileUploader"] button[kind="secondary"] p {
        color: white !important;
    }

    .main-header {
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%);
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        color: white !important;
    }

    .content-card {
        background: #FFFFFF;
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Action Buttons (Generate/Export) */
    .stButton > button {
        width: 100%;
        background-color: #6366F1 !important;
        color: white !important;
        border-radius: 6px !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ 2. SIDEBAR ------------------
with st.sidebar:
    st.markdown("<h3 style='text-align:center;'>⚛️ NEXUS CORE</h3>", unsafe_allow_html=True)
    module = st.radio("WORKSTREAM", ["Executive Summary", "Ask Questions", "PDF Splitter"], index=0)
    st.markdown("<br>", unsafe_allow_html=True)
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

file_source = st.file_uploader("Upload PDF", type="pdf")

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
        if st.button("🚀 EXECUTE SUMMARY"):
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
                except: st.error("Neural analysis failed.")

        if st.session_state.summary_cache:
            st.markdown(f'<div class="content-card"><b>Executive Summary:</b><br><br>{st.session_state.summary_cache}</div>', unsafe_allow_html=True)
            if st.session_state.keywords_cache:
                kw_html = "".join([f'<span style="background:#E0E7FF; color:#4338CA; padding:2px 8px; border-radius:12px; margin:2px; font-size:0.7rem; font-weight:bold; display:inline-block;">{k}</span>' for k in st.session_state.keywords_cache])
                st.markdown(f'<b>Insights:</b><br>{kw_html}', unsafe_allow_html=True)
            
            pdf_gen = FPDF()
            pdf_gen.add_page(); pdf_gen.set_font("Arial", size=12)
            pdf_gen.multi_cell(0, 10, txt=clean_txt(st.session_state.summary_cache))
            st.download_button(label="📥 DOWNLOAD REPORT", data=pdf_gen.output(dest='S').encode('latin-1'), file_name="Report.pdf")

    elif module == "Ask Questions":
        if st.button("🔍 GENERATE QUESTIONS"):
            with st.spinner("Mining insights..."):
                with pdfplumber.open(file_source) as pdf:
                    text = (pdf.pages[0].extract_text() or "") + " " + (pdf.pages[-1].extract_text() or "")
                doc_q = nlp(text[:8000])
                subjects = list(dict.fromkeys([chunk.text.strip() for chunk in doc_q.noun_chunks if len(chunk.text) > 5]))
                templates = ["What are the objectives for {}?", "How is {} evaluated?", "What risks affect {}?"]
                st.session_state.question_cache = [templates[i%3].format(subjects[i]) for i in range(min(len(subjects), 10))]
        
        for q in st.session_state.question_cache:
            st.markdown(f'<div style="background:#F1F5F9; border-left:4px solid #6366F1; padding:10px; margin-bottom:8px; border-radius:4px;">{q}</div>', unsafe_allow_html=True)

    elif module == "PDF Splitter":
        st.info(f"Document has {total_pages} pages.")
        col1, col2 = st.columns(2)
        s_p = col1.number_input("Start Page", 1, total_pages, 1)
        e_p = col2.number_input("End Page", 1, total_pages, total_pages)
        if st.button("✂️ EXPORT PDF"):
            writer = PdfWriter()
            for i in range(int(s_p)-1, int(e_p)): writer.add_page(pdf_reader.pages[i])
            
            # --- FIX APPLIED HERE: .getvalue() added parentheses ---
            output_data = io.BytesIO()
            writer.write(output_data)
            st.download_button("📥 DOWNLOAD SPLIT PDF", output_data.getvalue(), "split.pdf")

gc.collect()






