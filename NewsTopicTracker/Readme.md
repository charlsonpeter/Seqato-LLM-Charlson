# Global News Topic Tracker

This document records the learning journey while developing the **Global News Topic Tracker** Streamlit app.

---

## 🗓️ Timeline of Learning

### 1. **Google News RSS Feeds**
- **Learned**:  
  - Google News provides **RSS feeds** for different countries and topics.
  - Custom search queries can also be built using RSS URLs.
- **Issue**: Initially unclear how to switch between **topics** and **search queries**.  
- **Solution**:  
  - Implemented `build_rss_urls()` function to generate correct URLs based on user selection (topic vs query).

---

### 2. **Feed Parsing**
- **Learned**:  
  - Used `feedparser` library to read and parse RSS feeds into structured data.
  - Extracted fields: `title`, `link`, `summary`, `source`, `published_dt`.
- **Issue**: Some feeds lacked `published_parsed`.  
- **Solution**:  
  - Fallback to `updated_parsed` if `published_parsed` is missing.
  - Used safe datetime conversion to avoid errors.

---

### 3. **Article Grouping (Clustering)**
- **Learned**:  
  - Simple **title normalization** (lowercasing, removing punctuation/stopwords) works for grouping near-duplicate stories.
- **Issue**: Over-grouping or under-grouping when headlines differ slightly.  
- **Solution**:  
  - Implemented `topic_key_from_title()` → selects top keywords from title.  
  - Good enough for headlines, though semantic embeddings could improve accuracy.

---

### 4. **Scoring & Ranking**
- **Learned**:  
  - Need to balance **recency** and **cluster size**.  
- **Issue**: Older but large clusters were dominating.  
- **Solution**:  
  - Implemented `recency_score()` with exponential decay (half-life of 24h).
  - Final cluster score = `(cluster size * 0.7) + (recency * 10.0 * 0.3)`.

---

### 5. **Summarization**
- **Learned**:  
  - Integrated both **local Hugging Face summarizer** (`distilbart-cnn-12-6`) and **OpenAI GPT** models.  
- **Issues**:  
  1. Local summarizer too slow for long text.  
  2. OpenAI requires API key.  
- **Solutions**:  
  - Added text cleaning (`clean_for_summary`) to shorten inputs.  
  - Implemented fallback extractive summarizer (first few sentences).  
  - Sidebar control for selecting summarizer and entering OpenAI key.

---

### 6. **Streamlit UI**
- **Learned**:  
  - Used `st.sidebar` for settings.  
  - Used `st.container` and `st.expander` for clean layout.  
  - `st.download_button` for exporting summaries.  
- **Issue**: Summarization was slow and blocked UI.  
- **Solution**:  
  - Added `with st.spinner("Summarizing…")` for better UX.  
  - Cached local summarizer with `@st.cache_resource`.

---

### 7. **Exporting Summaries**
- **Learned**:  
  - Markdown export is simple and portable.  
- **Issue**: Wanted to preserve titles + summaries in one file.  
- **Solution**:  
  - Compiled summaries into a single Markdown blob and provided download.

---

## ✅ Key Takeaways
- Google News RSS + Feedparser = reliable way to fetch latest headlines.
- Simple text normalization works for headline grouping (though embeddings would be better).
- Summarization tradeoff:
  - Hugging Face → free but slower and less accurate.
  - OpenAI GPT → faster, more accurate, but requires API key.
- Streamlit provides a quick and interactive UI for news analysis.

---

## 📌 Future Improvements
- Use **semantic embeddings** for clustering (e.g., OpenAI embeddings or Sentence Transformers).
- Add **charts** to visualize topic frequency over time.
- Improve **multi-language summarization** support.
- Optimize **batch summarization** for larger datasets.

---
