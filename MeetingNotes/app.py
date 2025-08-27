import streamlit as st
import whisper
from transformers import pipeline

# Load Whisper (local speech-to-text)
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

# Load summarizer
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

whisper_model = load_whisper()
summarizer = load_summarizer()

st.title("🎙️ Meeting Notes & Action Item Extractor")

uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/mp3")

    with st.spinner("Transcribing with Whisper..."):
        # Save temp file
        with open("temp_audio.mp3", "wb") as f:
            f.write(uploaded_file.read())
        result = whisper_model.transcribe("temp_audio.mp3")
        transcript = result["text"]

    st.subheader("📝 Transcript")
    st.write(transcript)

    with st.spinner("Summarizing..."):
        summary = summarizer(transcript, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

    st.subheader("📋 Meeting Notes")
    st.write(summary)

    # Very simple action item extraction
    st.subheader("✅ Action Items")
    action_items = [sent for sent in transcript.split(".") if any(x in sent.lower() for x in ["will", "need to", "should", "action", "task"])]
    if action_items:
        for i, item in enumerate(action_items, 1):
            st.write(f"{i}. {item.strip()}")
    else:
        st.write("No explicit action items detected.")
