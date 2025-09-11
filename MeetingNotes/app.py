import streamlit as st
import whisper
try:
    from transformers import pipeline as hf_pipeline  # local summarizer
except Exception:
    hf_pipeline = None

st.set_page_config(page_title="Meeting Notes & Action Item Extractor", page_icon="🎙️")

# Load Whisper (local speech-to-text)
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

# Load summarizer
@st.cache_resource
def load_summarizer():
    return hf_pipeline("summarization", model="facebook/bart-large-cnn")

# --- Helpers for long-text summarization ---
def _chunk_text_by_tokens(text: str, tokenizer, max_tokens: int):
    # Fallback if tokenizer isn't available for some reason
    if tokenizer is None or max_tokens <= 0:
        return [text]
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return [text]
    chunks = []
    for i in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
    return chunks


def summarize_long_text(text: str, summarizer_pipeline, max_summary_len: int = 150, min_summary_len: int = 50) -> str:
    if summarizer_pipeline is None:
        raise RuntimeError("Summarizer pipeline is not available. Please install transformers[torch].")

    tokenizer = getattr(summarizer_pipeline, "tokenizer", None)
    # Keep some headroom for special tokens
    model_max = getattr(tokenizer, "model_max_length", 1024)
    # Some tokenizers report very large max (int(1e30)); clamp to a sensible default for BART
    if model_max is None or model_max > 2048:
        model_max = 1024
    chunk_budget = max(128, min(900, model_max - 50))

    chunks = _chunk_text_by_tokens(text, tokenizer, chunk_budget)

    # If it fits in one chunk, do a single pass
    if len(chunks) == 1:
        return summarizer_pipeline(
            chunks[0], max_length=max_summary_len, min_length=min_summary_len, do_sample=False, truncation=True
        )[0]["summary_text"]

    # Map-reduce summarization: summarize each chunk, then summarize the concatenated summaries
    progress = st.progress(0, text="Summarizing chunks...")
    partial_summaries = []
    total = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        part = summarizer_pipeline(
            chunk, max_length=max_summary_len, min_length=min_summary_len, do_sample=False, truncation=True
        )[0]["summary_text"]
        partial_summaries.append(part)
        progress.progress(idx / total, text=f"Summarized chunk {idx}/{total}")

    progress.empty()
    combined = "\n".join(partial_summaries)

    # Final reduce step (allow a bit longer output)
    final = summarizer_pipeline(
        combined, max_length=max_summary_len, min_length=min_summary_len, do_sample=False, truncation=True
    )[0]["summary_text"]
    return final


# --- Helper to load audio from Streamlit upload without writing to disk ---
import numpy as np

def load_audio_array_from_upload(uploaded_file) -> np.ndarray:
    try:
        from pydub import AudioSegment
    except Exception as exc:
        raise RuntimeError("pydub is required to decode uploaded audio in-memory. Install with: pip install pydub") from exc

    # Ensure stream at start
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    # Let pydub detect format from header; fallback to filename extension
    file_bytes = uploaded_file.read()
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    try:
        audio_seg = AudioSegment.from_file(io.BytesIO(file_bytes))
    except Exception:
        # Retry with explicit format from filename extension if available
        name = getattr(uploaded_file, "name", None)
        fmt = name.split(".")[-1].lower() if name and "." in name else None
        if fmt:
            audio_seg = AudioSegment.from_file(io.BytesIO(file_bytes), format=fmt)
        else:
            # Re-raise original decoding error
            raise

    # Convert to 16kHz mono float32 numpy array in range [-1, 1]
    audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio_seg.get_array_of_samples())
    # Normalize based on sample width
    if audio_seg.sample_width == 2:
        audio = samples.astype(np.float32) / 32768.0
    elif audio_seg.sample_width == 4:
        audio = samples.astype(np.float32) / 2147483648.0
    else:
        # Generic normalization
        max_abs = max(1, np.max(np.abs(samples)))
        audio = samples.astype(np.float32) / float(max_abs)
    return audio.astype(np.float32)

import io

whisper_model = load_whisper()
summarizer = load_summarizer() if hf_pipeline is not None else None

st.title("🎙️ Meeting Notes & Action Item Extractor")

uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/mp3")

    with st.spinner("Transcribing with Whisper..."):
        try:
            audio_array = load_audio_array_from_upload(uploaded_file)
            result = whisper_model.transcribe(audio_array)
            transcript = result["text"]
        except Exception as e:
            st.exception(e)
            transcript = ""

    if transcript:
        st.subheader("📝 Transcript")
        st.write(transcript)

        if summarizer is None:
            st.error("Summarizer not available. Please install the transformers library with a compatible torch backend.")
        else:
            with st.spinner("Summarizing..."):
                try:
                    summary = summarize_long_text(transcript, summarizer, max_summary_len=150, min_summary_len=50)
                except Exception as e:
                    st.exception(e)
                    summary = ""

            if summary:
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
