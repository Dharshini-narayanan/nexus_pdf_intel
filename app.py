import streamlit as st
import pdfplumber
import io
from pypdf import PdfReader, PdfWriter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer
import spacy
from collections import Counter
from fpdf import FPDF 

# ------------------ 1. SYSTEM INITIALIZATION & RESET LOGIC ------------------
st.set_page_config(page_title="Nexus Intelligence | Pro", page_icon="⚛️", layout="wide")

# Persistent state initialization
if 'summary_cache' not in st.session_state: st.session_state.summary_cache = ""
if 'keywords_cache' not in st.session_state: st.session_state.keywords_cache = []
if 'question_cache' not in st.session_state: st.session_state.question_cache = []
if 'last_uploaded_file' not in st.session_state: st.session_state.last_uploaded_file = None

# ===================== UI STYLING (SAME AS BEFORE) =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp { background-color: #F8FAFC; font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background-color: #0F172A !important; border-right: 1px solid #1E293B; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: 25px !important; display: flex; flex-direction: column; padding-top: 20px; }
[data-testid="stSidebar"] .stRadio label p { color: #FFFFFF !important; font-weight: 500; font-size: 1rem !important; }
.scope-pill { display: inline-block; background: #EEF2FF; color: #4F46E5; padding: 6px 15px; border-radius: 6px; font-size: 0.85rem; font-weight: 700; margin: 5px; border: 1px solid #C7D2FE; text-transform: uppercase; }
.main-header { background: #FFFFFF; padding: 2rem; border-radius: 12px; border-left: 6px solid #2563EB; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 2rem; }
.content-card { background: #FFFFFF; padding: 2rem; border-radius: 12px; border: 1px solid #E2E8F0; line-height: 1.8; margin-bottom: 20px;}
.q-card { background: #F8FAFC; border-left: 4px solid #6366F1; padding: 15px; margin-bottom: 10px; border-radius: 4px; color: #1E293B; font-weight: 500;}
.stButton > button { background-color: #2563EB !important; color: #FFFFFF !important; border-radius: 8px !important; font-weight: 600 !important; height: 48px !important; }
.info-box { background-color: #F1F5F9; color: #475569; padding: 15px; border-radius: 8px; border: 1px solid #CBD5E1; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ------------------ 2. CORE UTILITIES ------------------
@st.cache_resource
def load_ai_models():
    try:
        model_id = "sshleifer/distilbart-cnn-12-6"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        summarizer = pipeline("summarization", model=model_id, device=-1)
        nlp = spacy.load("en_core_web_sm")
        return summarizer, tokenizer, nlp
    except Exception: return None, None, None

summarizer, tokenizer, nlp = load_ai_models()

def extract_scope_keywords(text):
    if not nlp: return []
    try:
        doc = nlp(text[:25000].lower())
        words = [t.text for t in doc if t.pos_ in ["NOUN", "PROPN"] and not t.is_stop and len(t.text) > 4]
        return [w.upper() for w, count in Counter(words).most_common(12)]
    except Exception: return []

def clean_txt(text):
    rep = {'\u2013':'-', '\u2014':'-', '\u2018':"'", '\u2019':"'", '\u201c':'"', '\u201d':'"', '\u2022':'*', '\u2026':'...'}
    for u, s in rep.items(): text = text.replace(u, s)
    return text.encode('latin-1', 'replace').decode('latin-1')

# ------------------ 3. SIDEBAR ------------------
with st.sidebar:
    st.markdown("<h2 style='color:white;'>NEXUS CORE</h2>", unsafe_allow_html=True)
    module = st.radio("WORKSTREAM", ["Executive Summary", "Ask Questions", "PDF Splitter"], index=0)
    st.markdown("---")
    if st.session_state.keywords_cache:
        st.markdown("<p style='color:#94A3B8; font-size:0.8rem;'>SCOPE CONTEXT</p>", unsafe_allow_html=True)
        for kw in st.session_state.keywords_cache[:5]:
            st.markdown(f"<div style='color:#6366F1; font-size:0.85rem; font-weight:600;'># {kw}</div>", unsafe_allow_html=True)

# ------------------ 4. MAIN WORKSPACE & FILE HANDLER ------------------
st.markdown('<div class="main-header"><h1>Intelligence Studio</h1></div>', unsafe_allow_html=True)
file_source = st.file_uploader("Upload PDF Document", type="pdf", label_visibility="collapsed")

# --- UPDATE 1: AUTO-RESET ON NEW FILE ---
if file_source:
    if st.session_state.last_uploaded_file != file_source.name:
        st.session_state.summary_cache = ""
        st.session_state.keywords_cache = []
        st.session_state.question_cache = []
        st.session_state.last_uploaded_file = file_source.name
        st.rerun()

    try:
        pdf_reader = PdfReader(file_source)
        total_pages = len(pdf_reader.pages)
    except Exception:
        st.markdown('<div class="info-box">Error reading PDF structure.</div>', unsafe_allow_html=True)
        st.stop()

    # --- MODULES ---
    if module == "Executive Summary":
        if st.button("RUN DEEP AUDIT", type="primary"):
            with st.status("Analyzing Large Document...") as status:
                # --- UPDATE 2: ERROR HANDLING ON TEXT EXTRACTION ---
                try:
                    with pdfplumber.open(file_source) as pdf:
                        full_text = " ".join([p.extract_text() or "" for p in pdf.pages])
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2800, chunk_overlap=300)
                    chunks = text_splitter.split_text(full_text)
                    
                    summaries = []
                    limit = min(len(chunks), 10)
                    for i in range(limit):
                        try:
                            s = summarizer(chunks[i], max_length=150, min_length=50, truncation=True)[0]['summary_text']
                            summaries.append(f"**Section {i+1}:** {s}")
                        except Exception: continue # Skip chunk if error
                    
                    st.session_state.summary_cache = "\n\n".join(summaries)
                    st.session_state.keywords_cache = extract_scope_keywords(full_text)
                    status.update(label="Analysis Finalized", state="complete")
                except Exception:
                    st.markdown('<div class="info-box">Processing failed. Please ensure PDF is text-readable.</div>', unsafe_allow_html=True)

        if st.session_state.summary_cache:
            st.markdown("### Top Scope Entities")
            kw_html = "".join([f'<span class="scope-pill">{kw}</span>' for kw in st.session_state.keywords_cache])
            st.markdown(f'<div style="margin-bottom:20px;">{kw_html}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="content-card">{st.session_state.summary_cache}</div>', unsafe_allow_html=True)
            
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", '', 10)
                pdf.multi_cell(0, 7, clean_txt(st.session_state.summary_cache))
                st.download_button("📥 DOWNLOAD PDF REPORT", data=bytes(pdf.output()), file_name="Nexus_Report.pdf", use_container_width=True)
            except Exception: pass

    elif module == "Ask Questions":
        st.markdown('<div class="content-card"><h3>Smart Inquiry Engine</h3>Generate 10 investigative questions based on the document\'s full context.</div>', unsafe_allow_html=True)
        
        if st.button("SYNTHESIZE 10 QUESTIONS", type="primary"):
            with st.spinner("Decoding document entities across all pages..."):
                try:
                    with pdfplumber.open(file_source) as pdf:
                        # Scan 5 points: Start, 25%, 50%, 75%, End
                        intervals = [0, total_pages//4, total_pages//2, (3*total_pages)//4, total_pages-1]
                        sample_text = ""
                        for pg_num in intervals:
                            if pg_num < total_pages:
                                page = pdf.pages[pg_num].extract_text()
                                if page: sample_text += page + " "
                    
                    doc_nlp = nlp(sample_text[:20000])
                    entities = [ent.text for ent in doc_nlp.ents if len(ent.text) > 3]
                    concepts = [chunk.text for chunk in doc_nlp.noun_chunks if len(chunk.text) > 5]
                    
                    # Create a unique pool of subjects (Removed bolding syntax)
                    pool = list(dict.fromkeys(entities + concepts)) 
                    
                    if len(pool) >= 10:
                        st.session_state.question_cache = [
                            f"1. How is the concept of {pool[0]} introduced and justified in the document?",
                            f"2. What are the primary data points or conclusions associated with {pool[1]}?",
                            f"3. Does the text suggest any specific risks or challenges regarding {pool[2]}?",
                            f"4. How does {pool[3]} impact the overall scope and objectives of the report?",
                            f"5. What evidence is provided to support the claims made about {pool[4]}?",
                            f"6. Are there any notable correlations between {pool[5]} and {pool[6]} discussed?",
                            f"7. How does the author address the long-term implications of {pool[7]}?",
                            f"8. What methodologies are employed to analyze {pool[8]}?",
                            f"9. Does the document highlight any specific stakeholders related to {pool[9]}?",
                            f"10. Based on the summary, what is the final recommendation regarding {pool[2]}?"
                        ]
                    else:
                        # Professional fallback if document is short on entities
                        topics = ["Objectives", "Methodology", "Data Analysis", "Results", "Stakeholders", "Risks", "Timeline", "Budget", "Conclusion", "Recommendations"]
                        st.session_state.question_cache = [f"{i+1}. What does the document specify regarding {t}?" for i, t in enumerate(topics)]
                
                except Exception:
                    st.session_state.question_cache = ["Analysis error. Please try generating questions again."]

        # DISPLAY 10 QUESTIONS (Clean text, no asterisks)
        if st.session_state.question_cache:
            for q in st.session_state.question_cache:
                st.markdown(f'<div class="q-card">{q}</div>', unsafe_allow_html=True)
    elif module == "PDF Splitter":
        col1, col2 = st.columns(2)
        s_p, e_p = col1.number_input("Start", 1, total_pages, 1), col2.number_input("End", 1, total_pages, total_pages)
        if st.button("SPLIT & DOWNLOAD", type="primary"):
            try:
                writer = PdfWriter()
                for i in range(int(s_p)-1, int(e_p)): writer.add_page(pdf_reader.pages[i])
                out = io.BytesIO(); writer.write(out); st.download_button("Save Split PDF", out.getvalue(), "split.pdf")
            except Exception: st.markdown('<div class="info-box">Split failed. Check page range.</div>', unsafe_allow_html=True)
else:
    st.markdown("<div style='text-align:center; padding:100px; color:#94A3B8; border: 2px dashed #E2E8F0; border-radius:12px;'>Awaiting PDF Document Upload</div>", unsafe_allow_html=True)