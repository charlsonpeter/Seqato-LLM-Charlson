import streamlit as st
import requests

st.set_page_config(page_title="Local LLM Chat", layout="centered")
st.title("💬 Chat with Local LLM")

# Chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.chat_input("Say something...")

if user_input:
    st.session_state.chat.append(("user", user_input))

    with st.spinner("Thinking..."):
        response = requests.post(
            "http://localhost:8000/chat",
            json={"prompt": user_input}
        )

        reply = response.json().get("response", "")
        st.session_state.chat.append(("llm", reply))

# Display chat history
for sender, message in st.session_state.chat:
    if sender == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)
