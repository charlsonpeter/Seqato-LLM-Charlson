import os, uuid
import streamlit as st

try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.vectorstores import Chroma
    from langchain_community.llms import Ollama
except:
    from langchain.document_loaders import PyPDFLoader, TextLoader
    from langchain.vectorstores import Chroma
    from langchain.llms import Ollama

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="Chatbot Q&A", layout="centered")
st.title("💬 Document Chatbot")

# --- Custom CSS for right/left alignment ---
st.markdown("""
<style>
.user-msg {
    background-color: #2b313e;
    color: white;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px;
    text-align: right;
    float: right;
    clear: both;
    max-width: 70%;
}
.bot-msg {
    background-color: #4f5250;
    color: white;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px;
    text-align: left;
    float: left;
    clear: both;
    max-width: 70%;
}
</style>
""", unsafe_allow_html=True)

# Default model
model_name = "mistral"

uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

QA_TEMPLATE = """Answer the question using only the given context.
If the answer is not in the context, say "I don't know from the document."

Context:
{context}

Question: {question}
Answer:"""
qa_prompt = PromptTemplate.from_template(QA_TEMPLATE)

def load_document(path):
    if path.lower().endswith(".pdf"):
        return PyPDFLoader(path).load()
    return TextLoader(path).load()

def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_dir = os.path.join(".chroma", str(uuid.uuid4())[:8])
    return Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Create QA chain after upload
if uploaded and st.session_state.qa_chain is None:
    save_path = os.path.join(".", uploaded.name)
    with open(save_path, "wb") as f:
        f.write(uploaded.getbuffer())

    docs = load_document(save_path)
    vdb = create_vectorstore(docs)
    retriever = vdb.as_retriever()

    llm = Ollama(model=model_name, temperature=0.2)
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt},
    )

# Chat flow
if st.session_state.qa_chain:
    user_input = st.chat_input("Ask something...")

    if user_input:
        # Add user message immediately
        st.session_state.messages.append(("user", user_input))

        # Rerun so user sees their msg instantly
        st.rerun()

    # Render chat
    for i, (role, msg) in enumerate(st.session_state.messages):
        if role == "user":
            st.markdown(f"<div class='user-msg'>{msg}</div>", unsafe_allow_html=True)

            # If this is the latest user message and no bot reply yet
            if i == len(st.session_state.messages) - 1:
                with st.spinner("Thinking..."):
                    answer = st.session_state.qa_chain.run(msg)
                st.session_state.messages.append(("bot", answer))
                st.rerun()

        else:
            st.markdown(f"<div class='bot-msg'>{msg}</div>", unsafe_allow_html=True)
