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

# Persistent State
if 'summary_cache' not in st.session_state: st.session_state.summary_cache = ""
if 'question_cache' not in st.session_state: st.session_state.question_cache = []
if 'last_file' not in st.session_state: st.session_state.last_file = None

# ===================== ADVANCED UI STYLING =====================
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #F8FAFC;
    }

    /* Permanent Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] { 
        gap: 25px !important; 
        padding-top: 20px; 
    }
    
    /* Main Header Card */
    .main-header {
        background: linear-gradient(90deg, #4F46E5, #9333EA);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }

    /* Glassmorphism Cards */
    .content-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
        color: #E2E8F0;
        line-height: 1.7;
    }

    .q-card {
        background: rgba(99, 102, 241, 0.1);
        border-left: 4px solid #818CF8;
        padding: 15px;
        margin-bottom: 12px;
        border-radius: 8px;
        color: #F1F5F9;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px !important;
        background: linear-gradient(90deg, #2563EB, #3B82F6) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.6rem !important;
        transition: 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(37, 99, 235, 0.4);
    }

    /* Mobile adjustments */
    @media (max-width: 640px) {
        .main-header { padding: 1rem; }
        .main-header h1 { font-size: 1.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# ------------------ 2. PERMANENT SIDEBAR ------------------
with st.sidebar:
    st.markdown("<h2 style='color:white; text-align:center;'>⚛️ NEXUS CORE</h2>", unsafe_allow_html=True)
    st.markdown("---")
    module = st.radio("SELECT WORKSTREAM", ["Executive Summary", "Ask Questions", "PDF Splitter"], index=0)
    st.markdown("---")
    st.info("Upload your document in the main studio to begin analysis.")

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
st.markdown('<div class="main-header"><h1>Intelligence Studio Pro</h1><p>Advanced PDF Analytics & Neural Synthesis</p></div>', unsafe_allow_html=True)

file_source = st.file_uploader("Upload PDF Document", type="pdf", label_visibility="collapsed")

if not file_source:
    st.markdown('<div class="content-card" style="text-align:center;"><h3>Welcome.</h3><p>Please upload a PDF file to activate the neural modules.</p></div>', unsafe_allow_html=True)
else:
    # Reset logic
    if st.session_state.last_file != file_source.name:
        st.session_state.summary_cache = ""
        st.session_state.question_cache = []
        st.session_state.last_file = file_source.name
        st.rerun()

    pdf_reader = PdfReader(file_source)
    total_pages = len(pdf_reader.pages)

    if module == "Executive Summary":
        if st.button("🚀 EXECUTE NEURAL SUMMARY"):
            with st.status("Synthesizing Content...") as status:
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
                    status.update(label="Analysis Complete", state="complete")
                except:
                    st.error("Memory Limit: Please try a smaller PDF.")

        if st.session_state.summary_cache:
            st.markdown(f'<div class="content-card"><b>Neural Summary:</b><br><br>{st.session_state.summary_cache}</div>', unsafe_allow_html=True)
            
            # PDF DOWNLOAD BUTTON (Always appears after summary)
            pdf_gen = FPDF()
            pdf_gen.add_page()
            pdf_gen.set_font("Arial", size=12)
            pdf_gen.multi_cell(0, 10, txt=clean_txt(st.session_state.summary_cache))
            
            st.download_button(
                label="📥 DOWNLOAD SUMMARY PDF",
                data=pdf_gen.output(dest='S').encode('latin-1'),
                file_name="Nexus_Summary.pdf",
                mime="application/pdf"
            )

    elif module == "Ask Questions":
        if st.button("🔍 GENERATE ANALYTICAL QUESTIONS"):
            with st.spinner("Decoding document structure..."):
                with pdfplumber.open(file_source) as pdf:
                    text = (pdf.pages[0].extract_text() or "") + " " + (pdf.pages[-1].extract_text() or "")
                doc_q = nlp(text[:12000])
                subjects = list(dict.fromkeys([chunk.text.strip() for chunk in doc_q.noun_chunks if len(chunk.text) > 6]))
                
                templates = [
                    "What are the primary objectives associated with {}?",
                    "How does the document evaluate the impact of {}?",
                    "What specific risks are noted regarding {}?",
                    "Can you explain the methodology used to address {}?",
                    "What are the long-term implications of {} for stakeholders?",
                    "How is the success of {} measured in this context?",
                    "What evidence supports the findings on {}?",
                    "Are there any compliance factors involving {}?",
                    "How does the report suggest optimizing {}?",
                    "What is the final strategic recommendation regarding {}?"
                ]
                if len(subjects) < 10: subjects += ["Operations", "Strategy", "Execution", "Risk", "Outcome"]
                st.session_state.question_cache = [templates[i].format(subjects[i]) for i in range(10)]

        for q in st.session_state.question_cache:
            st.markdown(f'<div class="q-card">{q}</div>', unsafe_allow_html=True)

    elif module == "PDF Splitter":
        st.markdown('<div class="content-card">Extract specific page ranges for focus study.</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        s_p = col1.number_input("Start Page", 1, total_pages, 1)
        e_p = col2.number_input("End Page", 1, total_pages, total_pages)
        if st.button("✂️ DOWNLOAD SPLIT PDF"):
            writer = PdfWriter()
            for i in range(int(s_p)-1, int(e_p)): writer.add_page(pdf_reader.pages[i])
            out = io.BytesIO()
            writer.write(out)
            st.download_button("Download Now", out.getvalue(), "split_segment.pdf")

gc.collect()
