import streamlit as st
import requests

FASTAPI_URL = "http://localhost:8000/chat"

st.title("🧠 LLM Chat App")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
prompt = st.chat_input("Ask something...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send to backend
    with st.spinner("Thinking..."):
        response = requests.post(FASTAPI_URL, json={"messages": st.session_state.messages})
        reply = response.json()["message"]["content"]

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)

