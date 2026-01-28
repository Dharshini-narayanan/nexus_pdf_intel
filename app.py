import streamlit as st
import pdfplumber
import io
import gc
from pypdf import PdfReader, PdfWriter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import spacy
from collections import Counter
from fpdf import FPDF

# 1. INITIALIZATION
st.set_page_config(page_title="Nexus Intelligence | Pro", page_icon="⚛️", layout="wide")

if 'summary_cache' not in st.session_state: st.session_state.summary_cache = ""
if 'keywords_cache' not in st.session_state: st.session_state.keywords_cache = []
if 'question_cache' not in st.session_state: st.session_state.question_cache = []

# 2. CORE ENGINE (Manual Loading for Stability)
@st.cache_resource
def load_engine():
    try:
        model_name = "t5-small"
        # Manual loading bypasses the "Unknown task" error
        tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        import en_core_web_sm
        nlp = en_core_web_sm.load()
        return tokenizer, model, nlp
    except Exception as e:
        st.error(f"Engine Startup Failed: {e}")
        return None, None, None

tokenizer, model, nlp = load_engine()

if not tokenizer or not model:
    st.warning("⚠️ AI Engine is initializing. Please wait 60 seconds and refresh.")
    st.stop()

# 3. HELPER FUNCTIONS
def generate_summary(text_input):
    input_text = "summarize: " + text_input # T5 mandatory prefix
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def clean_txt(text):
    return text.encode('latin-1', 'replace').decode('latin-1')

# 4. UI STYLING
st.markdown("""<style>
    .stApp { background-color: #F8FAFC !important; }
    .main-header { background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%); padding: 2rem; border-radius: 16px; text-align: center; color: white; margin-bottom: 20px;}
    .content-card { background: #FFFFFF; padding: 1.5rem; border-radius: 12px; border: 1px solid #E2E8F0; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .q-card { background: #F1F5F9; border-left: 4px solid #6366F1; padding: 12px; margin-bottom: 10px; border-radius: 0 8px 8px 0; }
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h3 style='color:white;'>⚛️ NEXUS CORE</h3>", unsafe_allow_html=True)
    module = st.radio("WORKSTREAM", ["Executive Summary", "Ask Questions", "PDF Splitter"])
    if st.button("🗑️ RESET"):
        st.session_state.clear()
        st.rerun()

st.markdown('<div class="main-header"><h1>Intelligence Studio Pro</h1></div>', unsafe_allow_html=True)
file_source = st.file_uploader("Upload Document (PDF)", type="pdf")

if file_source:
    pdf_reader = PdfReader(file_source)
    total_pages = len(pdf_reader.pages)

    if module == "Executive Summary":
        if st.button("🚀 EXECUTE SUMMARY"):
            with st.status("Neural Synthesis...") as status:
                with pdfplumber.open(file_source) as pdf:
                    indices = [0, total_pages//2, total_pages-1]
                    raw_text = " ".join([pdf.pages[i].extract_text() or "" for i in indices if i < total_pages])
                
                if raw_text.strip():
                    st.session_state.summary_cache = generate_summary(raw_text[:2000])
                    doc_k = nlp(raw_text[:3000].lower())
                    kws = [t.text for t in doc_k if t.pos_ in ["NOUN", "PROPN"] and not t.is_stop and len(t.text) > 4]
                    st.session_state.keywords_cache = [w.upper() for w, c in Counter(kws).most_common(5)]
                    status.update(label="Complete", state="complete")

        if st.session_state.summary_cache:
            st.markdown(f'<div class="content-card"><b>Summary:</b><br><br>{st.session_state.summary_cache}</div>', unsafe_allow_html=True)
            kw_html = "".join([f'<span style="background:#E0E7FF; color:#4338CA; padding:4px 12px; border-radius:15px; margin:4px; font-weight:bold; display:inline-block;">{k}</span>' for k in st.session_state.keywords_cache])
            st.markdown(f'<div>{kw_html}</div>', unsafe_allow_html=True)
            
            pdf_gen = FPDF(); pdf_gen.add_page(); pdf_gen.set_font("Arial", size=12)
            pdf_gen.multi_cell(0, 10, txt=clean_txt(st.session_state.summary_cache))
            st.download_button("📥 DOWNLOAD REPORT", pdf_gen.output(dest='S').encode('latin-1'), "Summary.pdf")

    elif module == "Ask Questions":
        if st.button("🔍 ANALYZE QUESTIONS"):
            with pdfplumber.open(file_source) as pdf:
                text = (pdf.pages[0].extract_text() or "")[:3000]
            doc_q = nlp(text)
            subjects = list(dict.fromkeys([chunk.text.strip() for chunk in doc_q.noun_chunks if len(chunk.text) > 5]))
            templates = ["What are the implications of {}?", "How does the text define {}?", "What is the role of {}?"]
            st.session_state.question_cache = [templates[i%3].format(subjects[i%len(subjects)]) for i in range(min(6, len(subjects)))]
        
        for q in st.session_state.question_cache:
            st.markdown(f'<div class="q-card">{q}</div>', unsafe_allow_html=True)

    elif module == "PDF Splitter":
        col1, col2 = st.columns(2)
        s_p = col1.number_input("Start Page", 1, total_pages, 1)
        e_p = col2.number_input("End Page", 1, total_pages, total_pages)
        if st.button("✂️ EXPORT PAGES"):
            writer = PdfWriter()
            for i in range(int(s_p)-1, int(e_p)): writer.add_page(pdf_reader.pages[i])
            out = io.BytesIO()
            writer.write(out)
            st.download_button("📥 DOWNLOAD PDF", out.getvalue(), "split.pdf")

gc.collect()
