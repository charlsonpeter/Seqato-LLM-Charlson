# 📘 Learning Log: Local Multi-Modal Assistant (Ollama + Streamlit)

## Objective
Build a Streamlit application that allows users to ask questions about images and text using a local Ollama model (`llava`). The app should work fully offline and handle both text-only and image+text queries.

## Topics Learned
1. **Streamlit Basics**  
   - Created text input areas using `st.text_area()`.  
   - Handled file uploads with `st.file_uploader()`.  
   - Displayed uploaded images using `st.image()`.  
   - Used buttons (`st.button()`) to trigger actions and `st.spinner()` to show processing status.  
   - Provided user feedback using `st.warning()` and `st.error()`.

2. **Local Ollama API Integration**  
   - Interfaced with Ollama’s local API endpoint (`http://localhost:11434/api/generate`).  
   - Sent JSON requests containing the model, prompt, and optional images.  
   - Handled the JSON response to extract the model’s answer.

3. **Image Handling**  
   - Saved uploaded images to a temporary file using `tempfile.NamedTemporaryFile`.  
   - Encoded images in Base64 before sending to the Ollama API.  
   - Cleaned up temporary files after processing to avoid disk clutter.

4. **Error Handling**  
   - Managed request timeouts using different timeout settings for text-only vs. image queries.  
   - Caught connection errors, general request exceptions, and unexpected errors.  
   - Displayed meaningful error messages to the user.

5. **Offline Multi-Modal Processing**  
   - Used Ollama’s `llava` model locally, avoiding reliance on external servers.  
   - Designed the app to handle both pure text and text+image inputs.

---

## Issues Encountered and Solutions
1. **Timeouts for Image Queries**  
   - Problem: Image processing takes longer than text-only requests.  
   - Solution: Set a longer timeout (`timeout=120`) when sending requests with images.

2. **Temporary File Management**  
   - Problem: Uploaded images need to be written to disk before encoding.  
   - Solution: Used `tempfile.NamedTemporaryFile` and deleted the file after processing.

3. **Connection Errors**  
   - Problem: Ollama server not running results in failed requests.  
   - Solution: Added clear error messages prompting the user to run `ollama serve`.

4. **Unexpected API Responses**  
   - Problem: API might return unexpected JSON or no `response` key.  
   - Solution: Checked for `response` in the returned JSON and provided a fallback message.

---

## Future Improvements
- Add **batch image uploads** for multiple-image questions.  
- Improve **actionable feedback** by highlighting detected objects or concepts in the image.  
- Add **speech-to-text integration** to allow asking questions via audio.  
- Store a **history of questions and answers** for offline reference.  
- Enhance UI with **interactive image annotations** to show model reasoning.

---

## Outcome
Successfully created a Streamlit app that:
- Accepts text questions and optional images.  
- Sends requests to a local Ollama model (`llava`) for processing.  
- Handles offline multi-modal queries with robust error handling.  
- Returns answers clearly, with the uploaded image displayed alongside the response.
