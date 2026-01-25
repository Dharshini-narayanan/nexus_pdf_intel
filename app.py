import streamlit as st
import pdfplumber
import io
import gc
import re
from pypdf import PdfReader, PdfWriter
import spacy
from collections import Counter
from fpdf import FPDF

# ------------------ 1. SYSTEM INITIALIZATION ------------------
st.set_page_config(page_title="Nexus Intelligence | Pro", page_icon="⚛️", layout="wide")

if 'summary_cache' not in st.session_state: st.session_state.summary_cache = ""
if 'keywords_cache' not in st.session_state: st.session_state.keywords_cache = []
if 'question_cache' not in st.session_state: st.session_state.question_cache = []
if 'last_file' not in st.session_state: st.session_state.last_file = None

# ===================== MOBILE & THEME UI =====================
st.markdown("""
<style>
    :root { --text-col: inherit; }
    .content-card { padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(128,128,128,0.2); 
                    margin-bottom: 15px; background: rgba(128,128,128,0.05); color: var(--text-col); }
    .q-card { border-left: 5px solid #6366F1; padding: 12px; margin-bottom: 8px; 
              background: rgba(99, 102, 241, 0.08); border-radius: 4px; color: var(--text-col); }
    .scope-pill { display: inline-block; background: #4F46E5; color: white !important; 
                  padding: 5px 12px; border-radius: 20px; font-size: 0.75rem; margin: 4px; font-weight:700; }
    .stButton > button { width: 100%; border-radius: 10px !important; background: #2563EB !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ------------------ 2. LIGHTWEIGHT ENGINE ------------------
@st.cache_resource
def load_nlp():
    # Load ONLY the small language model (No heavy Transformers)
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

def generate_light_summary(text, limit=5):
    doc = nlp(text[:15000]) # Scan a large chunk safely
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text) > 30]
    # Simple Lexical Ranking (Most important sentences based on keyword density)
    words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    word_freq = Counter(words)
    
    scored_sentences = []
    for sent in sentences:
        score = sum(word_freq[word.lower()] for word in sent.split() if word.lower() in word_freq)
        scored_sentences.append((score, sent))
    
    top_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)[:limit]
    return " ".join([s[1] for s in top_sentences])

def clean_txt(text):
    return text.encode('latin-1', 'replace').decode('latin-1')

# ------------------ 3. MAIN INTERFACE ------------------
st.title("⚛️ Intelligence Studio")
file_source = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")

if file_source:
    if st.session_state.last_file != file_source.name:
        for k in ['summary_cache','keywords_cache','question_cache']: st.session_state[k] = ""
        st.session_state.last_file = file_source.name
        st.rerun()

    pdf_reader = PdfReader(file_source)
    total_pages = len(pdf_reader.pages)

    with st.sidebar:
        st.markdown("<h2 style='color:white;'>NEXUS CORE</h2>", unsafe_allow_html=True)
        module = st.radio("WORKSTREAM", ["Executive Summary", "Ask Questions", "PDF Splitter"])

    if module == "Executive Summary":
        if st.button("RUN STABLE AUDIT"):
            with st.spinner("Analyzing high-density sections..."):
                try:
                    with pdfplumber.open(file_source) as pdf:
                        # Sample 3 key pages to avoid OOM
                        pages = [0, total_pages//2, total_pages-1]
                        text_data = ""
                        for p in pages:
                            content = pdf.pages[p].extract_text()
                            if content: text_data += content + " "
                    
                    st.session_state.summary_cache = generate_light_summary(text_data)
                    # Keyword extraction
                    doc_k = nlp(text_data[:10000].lower())
                    kws = [t.text for t in doc_k if t.pos_ in ["NOUN", "PROPN"] and not t.is_stop and len(t.text) > 4]
                    st.session_state.keywords_cache = [w.upper() for w, c in Counter(kws).most_common(8)]
                    gc.collect()
                except: st.error("Document too complex. Try a smaller version.")

        if st.session_state.summary_cache:
            kw_html = "".join([f'<span class="scope-pill">{kw}</span>' for kw in st.session_state.keywords_cache])
            st.markdown(f'<div style="margin-bottom:10px;">{kw_html}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="content-card"><b>Key Findings:</b><br>{st.session_state.summary_cache}</div>', unsafe_allow_html=True)
            
            pdf_gen = FPDF()
            pdf_gen.add_page()
            pdf_gen.set_font("Arial", size=12)
            pdf_gen.multi_cell(0, 10, txt=clean_txt(st.session_state.summary_cache))
            st.download_button("📥 Download PDF", data=pdf_gen.output(dest='S').encode('latin-1'), file_name="Summary.pdf")

    elif module == "Ask Questions":
        if st.button("GENERATE 10 QUESTIONS"):
            with pdfplumber.open(file_source) as pdf:
                text = (pdf.pages[0].extract_text() or "") + " " + (pdf.pages[-1].extract_text() or "")
            doc_q = nlp(text[:8000])
            ents = list(set([e.text for e in doc_q.ents if len(e.text) > 3]))
            if len(ents) < 10: ents += ["Scope", "Timeline", "Stakeholders", "Costs", "Risks", "Methodology", "Data", "Security", "Results", "Future"]
            st.session_state.question_cache = [f"{i+1}. Insight regarding {ents[i]}?" for i in range(10)]

        for q in st.session_state.question_cache: st.markdown(f'<div class="q-card">{q}</div>', unsafe_allow_html=True)

    elif module == "PDF Splitter":
        s_p = st.number_input("Start", 1, total_pages, 1)
        e_p = st.number_input("End", 1, total_pages, total_pages)
        if st.button("SPLIT PDF"):
            writer = PdfWriter()
            for i in range(int(s_p)-1, int(e_p)): writer.add_page(pdf_reader.pages[i])
            out = io.BytesIO()
            writer.write(out)
            st.download_button("Download", out.getvalue(), "split.pdf")
