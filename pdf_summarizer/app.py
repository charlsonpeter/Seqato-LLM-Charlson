import streamlit as st
import PyPDF2
from transformers import pipeline

# ✅ Set page config BEFORE any other Streamlit commands
st.set_page_config(page_title="PDF Summarizer", layout="centered")

# Load summarizer model
@st.cache_resource
def load_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_model()

st.title("📄 Document Summarizer")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text.strip()

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)

    st.subheader("📖 Extracted Text")
    with st.expander("Click to view full text"):
        st.text_area("Text from PDF", text, height=300)

    if st.button("📝 Summarize"):
        with st.spinner("Summarizing..."):
            # Chunking if text is long
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            summary = ""
            for chunk in chunks:
                out = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
                summary += out[0]['summary_text'] + " "
            st.subheader("📌 Summary")
            st.write(summary)
