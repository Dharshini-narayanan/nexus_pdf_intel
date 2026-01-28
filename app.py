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

# Persistent Session State
if 'summary_cache' not in st.session_state: st.session_state.summary_cache = ""
if 'keywords_cache' not in st.session_state: st.session_state.keywords_cache = []
if 'question_cache' not in st.session_state: st.session_state.question_cache = []
if 'last_file' not in st.session_state: st.session_state.last_file = None

# ===================== UI STYLING =====================
st.markdown("""
<style>
    .stApp { background-color: #F8FAFC !important; color: #1E293B !important; }
    [data-testid="stSidebar"] { background-color: #1E293B !important; border-right: 2px solid #334155; }
    [data-testid="stSidebar"] .stRadio label p { color: #FFFFFF !important; font-size: 0.95rem !important; }
    .main-header {
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%);
        padding: 2rem; border-radius: 16px; text-align: center;
        margin-bottom: 30px; color: white !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .content-card {
        background: #FFFFFF; padding: 2rem; border-radius: 12px;
        border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .q-card {
        background: #F1F5F9; border-left: 4px solid #6366F1;
        padding: 16px; margin-bottom: 12px; border-radius: 0 8px 8px 0;
    }
    .stButton > button {
        width: 100%; background-color: #6366F1 !important; color: white !important;
        border-radius: 8px !important; font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ 2. CORE ENGINE ------------------
# ------------------ 2. CORE ENGINE ------------------
@st.cache_resource
def load_models():
    try:
        from transformers import pipeline
        # Force the framework to PyTorch ('pt') to avoid environment confusion
        summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", framework="pt", device=-1)
        
        import en_core_web_sm
        nlp_model = en_core_web_sm.load()
        return summarizer, nlp_model
    except Exception as e:
        # This will show you the EXACT error on the screen if it fails
        st.error(f"AI Engine failed to load. Error: {e}")
        return None, None

# Global Initialization
real_ai, nlp = load_models()

# If the models failed to load, stop the app immediately so you don't get 'not defined' errors
if real_ai is None or nlp is None:
    st.warning("⚠️ The AI Engine is still warming up or installing dependencies. Please wait 2 minutes and refresh.")
    st.stop()

def clean_txt(text):
    return text.encode('latin-1', 'replace').decode('latin-1')

# ------------------ 3. SIDEBAR ------------------
with st.sidebar:
    st.markdown("<h3 style='text-align:center; color:white;'>⚛️ NEXUS CORE</h3>", unsafe_allow_html=True)
    module = st.radio("WORKSTREAM", ["Executive Summary", "Ask Questions", "PDF Splitter"], index=0)
    if st.button("🗑️ RESET SESSION"):
        st.session_state.clear()
        st.rerun()

# ------------------ 4. MAIN WORKSPACE ------------------
st.markdown('<div class="main-header"><h1>Intelligence Studio Pro</h1><p style="opacity:0.9;">Neural Synthesis & Document Analytics</p></div>', unsafe_allow_html=True)

file_source = st.file_uploader("Upload your document (PDF)", type="pdf")

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
            with st.status("Neural processing...") as status:
                try:
                    with pdfplumber.open(file_source) as pdf:
                        # Improved Sampling
                        indices = sorted(list(set([0, total_pages//2, total_pages-1])))
                        raw_text = " ".join([pdf.pages[i].extract_text() or "" for i in indices if i < total_pages])
                    
                    if not raw_text.strip():
                        st.error("No readable text found.")
                        st.stop()

                    clean_input = " ".join(raw_text.split())
                    # Mandatory for Google T5: Prefix 'summarize: '
                    chunks = ["summarize: " + clean_input[i:i+800] for i in range(0, min(len(clean_input), 2400), 800)]
                    
                    summaries = []
                    for c in chunks:
                        if len(c) > 60:
                            res = real_ai(c, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                            summaries.append(res.strip())
                    
                    st.session_state.summary_cache = ". ".join(summaries).capitalize() + "."
                    
                    # Keywords extraction
                    doc_k = nlp(clean_input[:5000].lower())
                    kws = [t.text for t in doc_k if t.pos_ in ["NOUN", "PROPN"] and not t.is_stop and len(t.text) > 4]
                    st.session_state.keywords_cache = [w.upper() for w, c in Counter(kws).most_common(6)]
                    status.update(label="Complete", state="complete")
                except Exception as e:
                    st.error(f"Analysis Error: {e}")

        if st.session_state.summary_cache:
            st.markdown(f'<div class="content-card"><b>Analysis Result:</b><br><br>{st.session_state.summary_cache}</div>', unsafe_allow_html=True)
            kw_html = "".join([f'<span style="background:#E0E7FF; color:#4338CA; padding:4px 12px; border-radius:15px; margin:4px; font-weight:bold; display:inline-block;">{k}</span>' for k in st.session_state.keywords_cache])
            st.markdown(f'<div>{kw_html}</div>', unsafe_allow_html=True)
            
            # PDF Export
            pdf_gen = FPDF(); pdf_gen.add_page(); pdf_gen.set_font("Arial", size=12)
            pdf_gen.multi_cell(0, 10, txt=clean_txt(st.session_state.summary_cache))
            st.download_button("📥 DOWNLOAD SUMMARY", pdf_gen.output(dest='S').encode('latin-1'), "Summary.pdf")

    elif module == "Ask Questions":
        if st.button("🔍 GENERATE QUESTIONS"):
            with st.spinner("Extracting insights..."):
                with pdfplumber.open(file_source) as pdf:
                    text = (pdf.pages[0].extract_text() or "")[:3000]
                
                doc_q = nlp(text)
                subjects = list(dict.fromkeys([chunk.text.strip() for chunk in doc_q.noun_chunks if len(chunk.text) > 4]))
                
                templates = ["What defines {} in this context?", "How is {} addressed?", "What are the risks of {}?"]
                st.session_state.question_cache = [templates[i%3].format(subjects[i%len(subjects)]) for i in range(6)]

        for q in st.session_state.question_cache:
            st.markdown(f'<div class="q-card">{q}</div>', unsafe_allow_html=True)

    elif module == "PDF Splitter":
        st.info(f"Document has {total_pages} pages.")
        col1, col2 = st.columns(2)
        s_p = col1.number_input("Start", 1, total_pages, 1)
        e_p = col2.number_input("End", 1, total_pages, total_pages)
        if st.button("✂️ SPLIT PDF"):
            writer = PdfWriter()
            for i in range(int(s_p)-1, int(e_p)): writer.add_page(pdf_reader.pages[i])
            out = io.BytesIO()
            writer.write(out)
            st.download_button("📥 DOWNLOAD SPLIT", out.getvalue(), "split.pdf")

gc.collect()







