# 📘 Learning Log: Custom Chatbot Q&A (RAG Application)

## 🗓️ Project Goal
Develop an AI-powered chatbot using **LangChain** and **ChromaDB** for document-based Q&A, with a **Streamlit UI**.  
The chatbot should:
- Allow users to upload documents (`PDF`, `TXT`).
- Create embeddings and store them in ChromaDB.
- Retrieve relevant context using **RetrievalQA**.
- Use **Ollama** (local LLM, e.g., `mistral`) for responses.
- Provide a clean chatbot-style UI (user messages on the right, bot responses on the left).

---

## ✅ Topics Learned

### 1. **LangChain Document Loaders**
- **`PyPDFLoader`** for reading PDFs page by page.
- **`TextLoader`** for plain text documents.
- Learned about **community vs. core imports** (`langchain_community` vs. `langchain`).

---

### 2. **Text Splitting**
- Used **`RecursiveCharacterTextSplitter`**.
- Split documents into chunks (`chunk_size=1000`, `overlap=200`).
- Ensured better retrieval accuracy by avoiding context loss.

---

### 3. **Embeddings**
- Integrated **HuggingFaceEmbeddings** (`sentence-transformers/all-MiniLM-L6-v2`).
- Learned about vector representations and how they enable semantic search in ChromaDB.

---

### 4. **Vector Database (ChromaDB)**
- Used **Chroma** to store and query embeddings.
- Configured with a persistent directory (`.chroma/uuid`).
- Learned about `retriever` interface for RAG pipelines.

---

### 5. **LLM Integration with Ollama**
- Connected **Ollama** as the local model provider.
- Default model: **Mistral**.
- Used parameters like `temperature=0.2` for more factual answers.

---

### 6. **RetrievalQA Chain**
- Combined retriever + Ollama model via:
  ```python
  RetrievalQA.from_chain_type(
      llm=llm,
      retriever=retriever,
      chain_type="stuff",
      chain_type_kwargs={"prompt": qa_prompt},
  )
  ```
- Learned prompt templating to enforce:
  - Context-only answers.
  - `"I don't know from the document"` fallback.

---

### 7. **Streamlit Chat UI**
- Built **custom chat layout** using CSS:
  - User messages (`right aligned`).
  - Bot messages (`left aligned`).
- Used **`st.session_state`** to persist conversation.
- Integrated **`st.chat_input`** for smoother UX.

---

## ⚠️ Issues Encountered & Solutions

### 1. **Import Errors**
- Problem: Different LangChain versions caused missing imports (`langchain_community` vs. `langchain`).
- **Solution:** Added `try-except` fallback for imports.

---

### 2. **Persistence of Chat State**
- Problem: Messages disappeared on page refresh.
- **Solution:** Used `st.session_state.messages` to persist chat history across reruns.

---

### 3. **Bot Reply Synchronization**
- Problem: User message and bot reply rendering in wrong order.
- **Solution:** 
  - Added reruns (`st.rerun()`) after user input and bot reply.
  - Ensured bot response is computed only after the latest user input.

---

### 4. **Vectorstore Duplication**
- Problem: Each upload created a new ChromaDB directory, consuming space.
- **Solution:** Used unique temporary directories with `uuid` but noted optimization needed (clean-up unused dirs).

---

### 5. **Model Responses Not Accurate**
- Problem: Model sometimes hallucinated outside the document.
- **Solution:** Used a strict prompt template (`Answer only from context...`).

---

## 🚀 Key Takeaways
- **LangChain** simplifies the RAG pipeline but version mismatches require careful handling.
- **ChromaDB** is lightweight and well-suited for local vector search.
- **Ollama** allows fully local LLM inference (no API keys required).
- **Streamlit** with custom CSS gives a neat chatbot-like interface.
- Prompt engineering is essential to keep responses grounded.

---

## 📌 Next Steps
- Add support for **multi-document uploads**.
- Implement **conversation memory** (contextual continuity across Q&A).
- Optimize **vectorstore persistence** (clean-up old `.chroma` dirs).
- Explore **different Ollama models** (e.g., `llama2`, `codellama`).
- Add **export chat history** feature.
