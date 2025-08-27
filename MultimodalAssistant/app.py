# app.py
import streamlit as st
import requests
import tempfile
import os
import base64
import json

st.title("🖼️ Local Multi-Modal Assistant (Ollama + Streamlit)")
st.write("Ask questions about images + text, fully offline.")

question = st.text_area("Enter your question:", "What is in this image?")
uploaded_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])

def ask_ollama(question, image_path=None):
    """Ask Ollama using the API endpoint"""
    url = "http://localhost:11434/api/generate"
    
    # Prepare the request data
    data = {
        "model": "llava",
        "prompt": question,
        "stream": False
    }
    
    # If image is provided, encode it and add to the request
    if image_path:
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                data["images"] = [image_data]
        except Exception as e:
            return f"Error processing image: {str(e)}"
    
    try:
        # Use longer timeout for image processing
        timeout = 120 if image_path else 60
        
        response = requests.post(url, json=data, timeout=timeout)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response received")
        else:
            st.error(f"API Error: {response.status_code}")
            return f"Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "Request timed out. The model might be busy processing another request. Please try again in a few moments."
    except requests.exceptions.ConnectionError:
        return "Connection error. Please make sure Ollama is running (ollama serve)."
    except requests.exceptions.RequestException as e:
        return f"Request error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

if st.button("Ask"):
    if not question and not uploaded_file:
        st.warning("Please enter a question or upload an image.")
    else:
        image_path = None
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_file.read())
                image_path = tmp.name
            st.image(image_path, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Thinking locally..."):
            answer = ask_ollama(question, image_path)

        st.subheader("💡 Answer:")
        st.write(answer)

        if image_path:
            os.remove(image_path)
