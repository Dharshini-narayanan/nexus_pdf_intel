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

# ===================== UI STYLING =====================
st.markdown("""
<style>
    :root { --text-col: inherit; }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: 30px !important; padding-top: 30px; }
    .content-card { padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(128,128,128,0.2); 
                    margin-bottom: 15px; background: rgba(128,128,128,0.05); color: var(--text-col); }
    .q-card { border-left: 5px solid #6366F1; padding: 15px; margin-bottom: 12px; 
              background: rgba(99, 102, 241, 0.08); border-radius: 8px; color: var(--text-col);
              font-size: 1rem; font-weight: 500; }
    .scope-pill { display: inline-block; background: #4F46E5; color: white !important; 
                  padding: 5px 12px; border-radius: 20px; font-size: 0.75rem; margin: 4px; font-weight:700; }
    .stButton > button { width: 100%; border-radius: 10px !important; background: #2563EB !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ------------------ 2. CORE ENGINE ------------------
@st.cache_resource
def load_models():
    real_ai = pipeline("summarization", model="t5-small", device=-1)
    nlp = spacy.load("en_core_web_sm")
    return real_ai, nlp

real_ai, nlp = load_models()

# ------------------ 3. MAIN INTERFACE ------------------
st.title("⚛️ Intelligence Studio")
file_source = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")

if file_source:
    if st.session_state.last_file != file_source.name:
        for k in ['summary_cache','keywords_cache','question_cache']: st.session_state[k] = [] if 'cache' in k else ""
        st.session_state.last_file = file_source.name
        st.rerun()

    pdf_reader = PdfReader(file_source)
    total_pages = len(pdf_reader.pages)

    with st.sidebar:
        st.markdown("<h2 style='color:white;'>NEXUS CORE</h2>", unsafe_allow_html=True)
        module = st.radio("WORKSTREAM", ["Executive Summary", "Ask Questions", "PDF Splitter"])

    if module == "Executive Summary":
        if st.button("GENERATE SUMMARY"):
            with st.status("Analyzing...") as status:
                try:
                    with pdfplumber.open(file_source) as pdf:
                        target_pages = [0, total_pages//2, total_pages-1]
                        raw_text = ""
                        for p in target_pages:
                            page_text = pdf.pages[p].extract_text()
                            if page_text: raw_text += page_text + " "
                    
                    chunks = [raw_text[i:i+900] for i in range(0, min(len(raw_text), 2700), 900)]
                    unique_sentences = []
                    for chunk in chunks:
                        res = real_ai(chunk, max_length=50, min_length=20, do_sample=False)[0]['summary_text']
                        unique_sentences.append(res)
                    st.session_state.summary_cache = ". ".join(unique_sentences).replace(" .", ".")
                    status.update(label="Complete", state="complete")
                except: st.error("Processing error.")

        if st.session_state.summary_cache:
            st.markdown(f'<div class="content-card">{st.session_state.summary_cache}</div>', unsafe_allow_html=True)

    elif module == "Ask Questions":
        if st.button("GENERATE ANALYSIS QUESTIONS"):
            with st.spinner("Identifying key themes..."):
                with pdfplumber.open(file_source) as pdf:
                    text = (pdf.pages[0].extract_text() or "") + " " + (pdf.pages[-1].extract_text() or "")
                
                doc_q = nlp(text[:12000])
                # Filter for proper nouns and complex noun phrases
                subjects = list(dict.fromkeys([chunk.text.strip() for chunk in doc_q.noun_chunks if len(chunk.text) > 6]))
                
                # PROFESSIONAL TEMPLATES
                templates = [
                    "What are the primary objectives associated with {}?",
                    "How does the document evaluate the impact of {}?",
                    "What specific risks or challenges are noted regarding {}?",
                    "Can you explain the methodology used to address {}?",
                    "What are the long-term implications of {} for the stakeholders?",
                    "How is the success of {} measured within this context?",
                    "What evidence supports the current findings on {}?",
                    "Are there any regulatory or compliance factors involving {}?",
                    "How does the report suggest optimizing {}?",
                    "What is the final strategic recommendation regarding {}?"
                ]
                
                if len(subjects) < 10:
                    subjects += ["Operational Framework", "Strategic Planning", "Project Outcomes", "Compliance Standards", "Resource Management"]

                # Pair templates with subjects dynamically
                st.session_state.question_cache = [templates[i].format(subjects[i]) for i in range(10)]

        for q in st.session_state.question_cache:
            st.markdown(f'<div class="q-card">{q}</div>', unsafe_allow_html=True)

    elif module == "PDF Splitter":
        s_p = st.number_input("Start", 1, total_pages, 1)
        e_p = st.number_input("End", 1, total_pages, total_pages)
        if st.button("SPLIT PDF"):
            writer = PdfWriter()
            for i in range(int(s_p)-1, int(e_p)): writer.add_page(pdf_reader.pages[i])
            out = io.BytesIO()
            writer.write(out)
            st.download_button("Download", out.getvalue(), "split.pdf")

