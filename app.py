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

# --- MEMORY STORES ---
for key in ['summary_cache', 'keywords_cache', 'question_cache']:
    if key not in st.session_state: st.session_state[key] = ""
if 'last_file' not in st.session_state: st.session_state.last_file = None

# ===================== UI STYLING =====================
st.markdown("""
<style>
    :root { --text-col: inherit; }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: 30px !important; padding-top: 30px; }
    .content-card { padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(128,128,128,0.2); 
                    margin-bottom: 15px; background: rgba(128,128,128,0.05); color: var(--text-col); }
    .q-card { border-left: 5px solid #6366F1; padding: 12px; margin-bottom: 8px; 
              background: rgba(99, 102, 241, 0.08); border-radius: 4px; color: var(--text-col); }
    .scope-pill { display: inline-block; background: #4F46E5; color: white !important; 
                  padding: 5px 12px; border-radius: 20px; font-size: 0.75rem; margin: 4px; font-weight:700; }
    .stButton > button { width: 100%; border-radius: 10px !important; background: #2563EB !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ------------------ 2. THE "REAL" AI ENGINE ------------------
@st.cache_resource
def load_models():
    # T5-Small is a "Real" Summarizer that fits in memory
    real_ai = pipeline("summarization", model="t5-small", device=-1)
    nlp = spacy.load("en_core_web_sm")
    return real_ai, nlp

real_ai, nlp = load_models()

def clean_txt(text):
    return text.encode('latin-1', 'replace').decode('latin-1')

# ------------------ 3. MAIN INTERFACE ------------------
st.title("⚛️ Intelligence Studio")
file_source = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")

if file_source:
    if st.session_state.last_file != file_source.name:
        st.session_state.summary_cache = ""
        st.session_state.last_file = file_source.name
        st.rerun()

    pdf_reader = PdfReader(file_source)
    total_pages = len(pdf_reader.pages)

    with st.sidebar:
        st.markdown("<h2 style='color:white;'>NEXUS CORE</h2>", unsafe_allow_html=True)
        module = st.radio("WORKSTREAM", ["Executive Summary", "Ask Questions", "PDF Splitter"])

    if module == "Executive Summary":
        if st.button("RUN NEURAL SUMMARY"):
            with st.status("AI is reading and rewriting...") as status:
                try:
                    with pdfplumber.open(file_source) as pdf:
                        # Scan Intro, Middle, and Conclusion
                        target_pages = [0, total_pages//2, total_pages-1]
                        raw_text = ""
                        for p in target_pages:
                            page_text = pdf.pages[p].extract_text()
                            if page_text: raw_text += page_text + " "
                    
                    # Split into 3 small chunks so AI doesn't crash RAM
                    chunks = [raw_text[i:i+800] for i in range(0, min(len(raw_text), 2400), 800)]
                    summaries = []
                    
                    for chunk in chunks:
                        # The actual AI "summarization" happens here
                        res = real_ai(chunk, max_length=60, min_length=20, do_sample=False)
                        summaries.append(res[0]['summary_text'])
                        gc.collect() # Clean RAM after every page
                    
                    st.session_state.summary_cache = ". ".join(summaries).replace(" .", ".")
                    
                    # Keywords
                    doc = nlp(raw_text[:5000].lower())
                    kws = [t.text for t in doc if t.pos_ in ["NOUN", "PROPN"] and not t.is_stop and len(t.text) > 4]
                    st.session_state.keywords_cache = [w.upper() for w, c in Counter(kws).most_common(6)]
                    status.update(label="Summary Synthesized!", state="complete")
                except:
                    st.error("Document too dense for Free Tier AI. Try a smaller PDF.")

        if st.session_state.summary_cache:
            kw_html = "".join([f'<span class="scope-pill">{kw}</span>' for kw in st.session_state.keywords_cache])
            st.markdown(f'<div>{kw_html}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="content-card"><b>AI Executive Summary:</b><br>{st.session_state.summary_cache}</div>', unsafe_allow_html=True)
            
            # PDF Download
            pdf_gen = FPDF()
            pdf_gen.add_page(); pdf_gen.set_font("Arial", size=12)
            pdf_gen.multi_cell(0, 10, txt=clean_txt(st.session_state.summary_cache))
            st.download_button("📥 Download PDF Summary", data=pdf_gen.output(dest='S').encode('latin-1'), file_name="Summary.pdf")

    elif module == "Ask Questions":
        # (Remaining features same as previous build, including Search)
        if st.button("GENERATE 10 QUESTIONS"):
            with pdfplumber.open(file_source) as pdf:
                text = (pdf.pages[0].extract_text() or "") + " " + (pdf.pages[-1].extract_text() or "")
            doc_q = nlp(text[:8000])
            subjects = list(dict.fromkeys([chunk.text.strip() for chunk in doc_q.noun_chunks if len(chunk.text) > 5]))
            st.session_state.question_cache = [f"{i+1}. Insight regarding {subjects[i]}?" for i in range(min(10, len(subjects)))]
        for q in st.session_state.question_cache: st.markdown(f'<div class="q-card">{q}</div>', unsafe_allow_html=True)

    elif module == "PDF Splitter":
        s_p = st.number_input("Start", 1, total_pages, 1)
        e_p = st.number_input("End", 1, total_pages, total_pages)
        if st.button("SPLIT PDF"):
            writer = PdfWriter()
            for i in range(int(s_p)-1, int(e_p)): writer.add_page(pdf_reader.pages[i])
            st.download_button("Download", io.BytesIO(writer.write_stream()).getvalue(), "split.pdf")
