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

# 2. CORE ENGINE (Manual Loading)
@st.cache_resource
def load_engine():
    try:
        model_name = "t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        import en_core_web_sm
        nlp = en_core_web_sm.load()
        return tokenizer, model, nlp
    except Exception as e:
        st.error(f"Engine Startup Failed: {e}")
        return None, None, None

tokenizer, model, nlp = load_engine()

# Safeguard to prevent "NameError"
if not tokenizer or not model:
    st.warning("⚠️ AI Engine is still installing dependencies. Please wait 60 seconds and refresh.")
    st.stop()

# 3. SUMMARIZATION LOGIC
def generate_summary(text_input):
    # Mandatory T5 prefix
    input_text = "summarize: " + text_input
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Direct model generation
    summary_ids = model.generate(
        inputs, 
        max_length=150, 
        min_length=40, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 4. UI COMPONENTS (Keep your existing layout)
st.markdown("""<style>
    .stApp { background-color: #F8FAFC !important; }
    .main-header { background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%); padding: 2rem; border-radius: 16px; text-align: center; color: white; }
    .content-card { background: #FFFFFF; padding: 2rem; border-radius: 12px; border: 1px solid #E2E8F0; margin-bottom: 20px; }
</style>""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>Intelligence Studio Pro</h1></div>', unsafe_allow_html=True)

file_source = st.file_uploader("Upload PDF", type="pdf")

if file_source:
    pdf_reader = PdfReader(file_source)
    total_pages = len(pdf_reader.pages)
    
    if st.button("🚀 EXECUTE SUMMARY"):
        with st.status("Neural processing...") as status:
            with pdfplumber.open(file_source) as pdf:
                # Sample pages for memory efficiency
                indices = [0, total_pages//2, total_pages-1]
                raw_text = " ".join([pdf.pages[i].extract_text() or "" for i in indices if i < total_pages])
            
            if raw_text.strip():
                # Process in one clean batch
                summary = generate_summary(raw_text[:2000])
                st.session_state.summary_cache = summary
                
                # Keywords
                doc_k = nlp(raw_text[:3000].lower())
                kws = [t.text for t in doc_k if t.pos_ in ["NOUN", "PROPN"] and not t.is_stop and len(t.text) > 4]
                st.session_state.keywords_cache = [w.upper() for w, c in Counter(kws).most_common(5)]
                status.update(label="Complete", state="complete")

    if 'summary_cache' in st.session_state and st.session_state.summary_cache:
        st.markdown(f'<div class="content-card"><b>Summary:</b><br>{st.session_state.summary_cache}</div>', unsafe_allow_html=True)








