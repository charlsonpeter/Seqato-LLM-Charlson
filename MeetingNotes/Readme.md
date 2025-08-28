# Meeting Notes & Action Item Extractor

## Objective
Build a Streamlit application that accepts an audio file upload, transcribes speech to text using Whisper, summarizes the transcript using a Hugging Face transformer model, and extracts action items from the transcript.

## Topics Learned
1. **Streamlit Basics**  
   - Used `st.file_uploader()` for uploading audio files.  
   - Displayed audio with `st.audio()`.  
   - Added progress indicators using `st.spinner()`.  
   - Wrote results to the app using `st.write()` and `st.subheader()`.

2. **Whisper (Speech-to-Text)**  
   - Installed and used OpenAI Whisper locally.  
   - Loaded a Whisper model with `whisper.load_model("base")`.  
   - Transcribed audio files using `whisper_model.transcribe("temp_audio.mp3")`.

3. **Hugging Face Transformers (Summarization)**  
   - Used the `pipeline()` API for summarization.  
   - Selected the DistilBART CNN model (`sshleifer/distilbart-cnn-12-6`).  
   - Generated concise meeting summaries with `max_length` and `min_length` settings.

4. **Action Item Extraction (Rule-Based)**  
   - Extracted potential tasks by keyword matching (`will`, `need to`, `should`, `action`, `task`).  
   - Split transcript into sentences and filtered for these keywords.

5. **Streamlit Caching**  
   - Used `@st.cache_resource` to cache model loading, improving performance when re-running the app.

## Issues Encountered and Solutions
1. **Large Model Loading Time**  
   - Problem: Whisper and Hugging Face models take time to load every run.  
   - Solution: Used `@st.cache_resource` to load models only once.

2. **Memory/Performance Limits**  
   - Problem: Large audio files and transcripts can slow summarization.  
   - Solution: Limited summarization length with `max_length=150, min_length=50`.

3. **Temporary File Handling**  
   - Problem: Uploaded files need to be written to disk before Whisper can process them.  
   - Solution: Saved the uploaded file as `temp_audio.mp3` before transcription.

4. **Extracting Action Items**  
   - Problem: Not all action items are explicitly worded.  
   - Solution: Implemented a keyword-based filter; future improvement could use NLP-based extraction.

## Future Improvements
- Replace keyword filtering with an NLP task extraction model (e.g., using `spacy` or fine-tuned transformers).  
- Add speaker diarization to identify who said what.  
- Integrate OpenAI API for improved summaries and structured action items.  
- Store transcripts, summaries, and action items in a database for long-term tracking.

## Outcome
Successfully created a Streamlit app that uploads audio, transcribes it with Whisper, summarizes the transcript, and extracts action items. This provides structured meeting notes and a task list from raw audio input.
