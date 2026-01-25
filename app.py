import streamlit as st
import pdfplumber
import io
import gc
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

# ===================== UI STYLING =====================
st.markdown("""
<style>
    :root { --text-col: inherit; }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: 30px !important; padding-top: 30px; }
    .content-card { padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(128,128,128,0.2); 
                    margin-bottom: 15px; background: rgba(128,128,128,0.05); color: var(--text-col); }
    .q-card { border-left: 5px solid #6366F1; padding: 12px; margin-bottom: 8px; 
              background: rgba(99, 102, 241, 0.08); border-radius: 4px; color: var(--text-col); }
    .search-result { font-size: 0.85rem; padding: 8px; border-bottom: 1px solid rgba(128,128,128,0.1); }
    .scope-pill { display: inline-block; background: #4F46E5; color: white !important; 
                  padding: 5px 12px; border-radius: 20px; font-size: 0.75rem; margin: 4px; font-weight:700; }
    .stButton > button { width: 100%; border-radius: 10px !important; background: #2563EB !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ------------------ 2. CORE ENGINE ------------------
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

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
            with st.spinner("Analyzing..."):
                try:
                    with pdfplumber.open(file_source) as pdf:
                        pages = [0, total_pages//2, total_pages-1]
                        text_data = ""
                        for p in pages:
                            content = pdf.pages[p].extract_text()
                            if content: text_data += content + " "
                    
                    # Frequency-based summary
                    doc = nlp(text_data[:15000])
                    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text) > 40]
                    words = [t.text.lower() for t in doc if t.is_alpha and not t.is_stop]
                    word_freq = Counter(words)
                    scored = sorted([(sum(word_freq[w.lower()] for w in s.split()), s) for s in sentences], reverse=True)
                    st.session_state.summary_cache = " ".join([s[1] for s in scored[:5]])
                    
                    kws = [t.text for t in doc if t.pos_ in ["NOUN", "PROPN"] and not t.is_stop and len(t.text) > 4]
                    st.session_state.keywords_cache = [w.upper() for w, c in Counter(kws).most_common(8)]
                    gc.collect()
                except: st.error("Processing error.")

        if st.session_state.summary_cache:
            kw_html = "".join([f'<span class="scope-pill">{kw}</span>' for kw in st.session_state.keywords_cache])
            st.markdown(f'<div>{kw_html}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="content-card"><b>Analysis:</b><br>{st.session_state.summary_cache}</div>', unsafe_allow_html=True)
            
            pdf_gen = FPDF()
            pdf_gen.add_page(); pdf_gen.set_font("Arial", size=12)
            pdf_gen.multi_cell(0, 10, txt=clean_txt(st.session_state.summary_cache))
            st.download_button("📥 Download PDF", data=pdf_gen.output(dest='S').encode('latin-1'), file_name="Summary.pdf")

    elif module == "Ask Questions":
        st.info("💡 Tip: Use the search box below to find answers to these questions.")
        if st.button("GENERATE 10 QUESTIONS"):
            with pdfplumber.open(file_source) as pdf:
                text = (pdf.pages[0].extract_text() or "") + " " + (pdf.pages[total_pages//2].extract_text() or "") + " " + (pdf.pages[-1].extract_text() or "")
            doc_q = nlp(text[:12000])
            subjects = list(dict.fromkeys([chunk.text.strip() for chunk in doc_q.noun_chunks if len(chunk.text) > 5]))
            if len(subjects) < 10: subjects += ["Objectives", "Timeline", "Stakeholders", "Risks", "Methodology"]
            st.session_state.question_cache = [f"{i+1}. What are the details regarding {subjects[i]}?" for i in range(10)]

        for q in st.session_state.question_cache: st.markdown(f'<div class="q-card">{q}</div>', unsafe_allow_html=True)
        
        # --- NEW SEARCH FEATURE ---
        st.markdown("---")
        query = st.text_input("🔍 Search Document for Answers:", placeholder="Enter a keyword from the questions above...")
        if query:
            with pdfplumber.open(file_source) as pdf:
                results = []
                for i in range(min(total_pages, 20)): # Scan first 20 pages for speed
                    page_text = pdf.pages[i].extract_text()
                    if page_text and query.lower() in page_text.lower():
                        # Find sentences containing the query
                        found = [s.strip() for s in page_text.split('.') if query.lower() in s.lower()]
                        for f in found: results.append(f"**[Page {i+1}]**: {f}...")
                
                if results:
                    st.success(f"Found {len(results[:5])} matches:")
                    for r in results[:5]: st.markdown(f'<div class="search-result">{r}</div>', unsafe_allow_html=True)
                else:
                    st.warning("No direct matches found in the first 20 pages.")

    elif module == "PDF Splitter":
        s_p = st.number_input("Start", 1, total_pages, 1)
        e_p = st.number_input("End", 1, total_pages, total_pages)
        if st.button("SPLIT PDF"):
            writer = PdfWriter()
            for i in range(int(s_p)-1, int(e_p)): writer.add_page(pdf_reader.pages[i])
            out = io.BytesIO()
            writer.write(out)
            st.download_button("Download", out.getvalue(), "split.pdf")

