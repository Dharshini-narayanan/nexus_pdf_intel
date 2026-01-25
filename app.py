rimport streamlit as st
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
if 'last_uploaded_file' not in st.session_state: st.session_state.last_uploaded_file = None

# ===================== MOBILE & DARK MODE CSS =====================
st.markdown("""
<style>
    /* Global Reset for Theme Consistency */
    [data-testid="stAppViewContainer"] { padding: 1rem; }
    
    /* Auto-adjusting Text & Cards for Dark/Light Mode */
    .content-card {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 15px;
        background-color: rgba(128, 128, 128, 0.05);
        line-height: 1.6;
        word-wrap: break-word;
    }

    /* Questions: Mobile Friendly Padding */
    .q-card {
        border-left: 5px solid #6366F1;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 8px;
        background-color: rgba(99, 102, 241, 0.08);
        font-size: 0.95rem;
    }

    /* Keyword Pills: Responsive Wrapping */
    .kw-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 15px;
    }
    .scope-pill {
        background: #4F46E5;
        color: white !important;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    /* Mobile Button Optimization */
    .stButton > button {
        width: 100% !important;
        height: 3rem;
        border-radius: 10px !important;
        font-weight: 600 !important;
        background-color: #2563EB !important;
        color: white !important;
    }

    /* Hide Sidebar on very small screens for better focus */
    @media (max-width: 640px) {
        [data-testid="stSidebar"] { width: 100vw !important; }
        .main-title { font-size: 1.5rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# ------------------ 2. CORE AI ENGINE ------------------
@st.cache_resource
def load_ai():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", device=-1)
    nlp = spacy.load("en_core_web_sm")
    return summarizer, nlp

summarizer, nlp = load_ai()

def clean_txt(text):
    return text.encode('latin-1', 'replace').decode('latin-1')

# ------------------ 3. SIDEBAR ------------------
with st.sidebar:
    st.markdown("<h2 style='color:white;'>NEXUS CORE</h2>", unsafe_allow_html=True)
    module = st.radio("WORKSTREAM", ["Executive Summary", "Ask Questions", "PDF Splitter"], index=0)

# ------------------ 4. MAIN INTERFACE ------------------
st.markdown("<h1 class='main-title'>⚛️ Intelligence Studio</h1>", unsafe_allow_html=True)
file_source = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")

if file_source:
    if st.session_state.last_uploaded_file != file_source.name:
        st.session_state.summary_cache = ""
        st.session_state.keywords_cache = []
        st.session_state.question_cache = []
        st.session_state.last_uploaded_file = file_source.name
        st.rerun()

    pdf_reader = PdfReader(file_source)
    total_pages = len(pdf_reader.pages)

    if module == "Executive Summary":
        if st.button("START MOBILE AUDIT"):
            with st.status("Processing...") as status:
                try:
                    with pdfplumber.open(file_source) as pdf:
                        # Strategic sampling for 100+ pages to save RAM
                        indices = [0, total_pages//2, total_pages-1]
                        full_text = ""
                        summaries = []
                        for i, idx in enumerate(indices):
                            text = pdf.pages[idx].extract_text()
                            if text:
                                full_text += text + " "
                                s = summarizer(text[:1500], max_length=100, min_length=40, truncation=True)[0]['summary_text']
                                summaries.append(f"**Key Point {i+1}:** {s}")
                                gc.collect()
                    st.session_state.summary_cache = "\n\n".join(summaries)
                    doc = nlp(full_text[:8000].lower())
                    words = [t.text for t in doc if t.pos_ in ["NOUN", "PROPN"] and not t.is_stop and len(t.text) > 4]
                    st.session_state.keywords_cache = [w.upper() for w, count in Counter(words).most_common(8)]
                    status.update(label="Complete", state="complete")
                except: st.error("Out of Memory. Refresh and try again.")

        if st.session_state.summary_cache:
            # Responsive Keywords
            kw_html = "".join([f'<span class="scope-pill">{kw}</span>' for kw in st.session_state.keywords_cache])
            st.markdown(f'<div class="kw-container">{kw_html}</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="content-card">{st.session_state.summary_cache}</div>', unsafe_allow_html=True)
            
            # PDF Download (Binary stream for cross-device compatibility)
            try:
                pdf_gen = FPDF()
                pdf_gen.add_page()
                pdf_gen.set_font("Arial", size=12)
                pdf_gen.multi_cell(0, 10, txt=clean_txt(st.session_state.summary_cache))
                st.download_button("📥 Download Report (PDF)", data=pdf_gen.output(dest='S').encode('latin-1'), file_name="Report.pdf")
            except: pass

    elif module == "Ask Questions":
        if st.button("SYNTHESIZE 10 QUESTIONS"):
            with pdfplumber.open(file_source) as pdf:
                text = (pdf.pages[0].extract_text() or "") + " " + (pdf.pages[-1].extract_text() or "")
            doc = nlp(text[:6000])
            ents = list(set([e.text for e in doc.ents if len(e.text) > 3]))
            if len(ents) < 10: ents += ["Operations", "Budget", "Strategy", "Impact", "Overview", "Safety", "Policy", "Team", "Goals", "Success"]
            st.session_state.question_cache = [f"{i+1}. Insight regarding {ents[i]}?" for i in range(10)]

        for q in st.session_state.question_cache: st.markdown(f'<div class="q-card">{q}</div>', unsafe_allow_html=True)

    elif module == "PDF Splitter":
        s_p = st.number_input("Start", 1, total_pages, 1)
        e_p = st.number_input("End", 1, total_pages, total_pages)
        if st.button("DOWNLOAD FRAGMENT"):
            writer = PdfWriter()
            for i in range(int(s_p)-1, int(e_p)): writer.add_page(pdf_reader.pages[i])
            out = io.BytesIO()
            writer.write(out)
            st.download_button("Save Split", out.getvalue(), "split.pdf")
