from __future__ import annotations
import os
import re
import html
from typing import List, Dict, Tuple
from datetime import datetime
from urllib.parse import quote_plus

import streamlit as st
import feedparser
import pandas as pd

# Optional libs (loaded lazily)
try:
    from transformers import pipeline as hf_pipeline  # local summarizer
except Exception:
    hf_pipeline = None

try:
    from openai import OpenAI  # OpenAI client (>=1.0)
except Exception:
    OpenAI = None  # type: ignore

# -----------------------------
# UI & CONSTANTS
# -----------------------------
st.set_page_config(page_title="Global News Topic Tracker", page_icon="🗞️", layout="wide")
st.title("🗞️ Global News Topic Tracker")
st.caption("Fetch from Google News (RSS) → group similar stories → summarize clearly.")

ISO_COUNTRIES: Dict[str, Tuple[str, str]] = {
    "India": ("IN", "en"),
    "United States": ("US", "en"),
    "United Kingdom": ("GB", "en"),
    "Australia": ("AU", "en"),
    "Canada": ("CA", "en"),
    "Germany": ("DE", "de"),
    "France": ("FR", "fr"),
    "Spain": ("ES", "es"),
    "Italy": ("IT", "it"),
    "Brazil": ("BR", "pt"),
    "Japan": ("JP", "ja"),
    "South Korea": ("KR", "ko"),
}

TOPIC_MAP: Dict[str, str | None] = {
    "Top stories": None,
    "World": "WORLD",
    "Nation": "NATION",
    "Business": "BUSINESS",
    "Technology": "TECHNOLOGY",
    "Entertainment": "ENTERTAINMENT",
    "Sports": "SPORTS",
    "Science": "SCIENCE",
    "Health": "HEALTH",
}

STOPWORDS = set("""
a the an and of to in on for with by from about into over after before under above
is are was were be being been it its as at that this these those their his her they them you your our
""".split())

# -----------------------------
# Helper functions
# -----------------------------

def build_rss_urls(query: str | None, country_code: str, language: str, topic_label: str | None) -> List[str]:
    """Return one or more Google News RSS URLs for the chosen edition/topic or search query."""
    base = f"https://news.google.com/rss?hl={language}-{country_code}&gl={country_code}&ceid={country_code}:{language}"
    if query:
        q = quote_plus(query)
        return [f"https://news.google.com/rss/search?q={q}&hl={language}-{country_code}&gl={country_code}&ceid={country_code}:{language}"]
    if topic_label and TOPIC_MAP.get(topic_label):
        return [base + f"&topic={TOPIC_MAP[topic_label]}"]
    return [base]


def fetch_rss(url: str):
    return feedparser.parse(url)


def parse_entries(feeds: List) -> pd.DataFrame:
    rows = []
    for feed in feeds:
        for e in feed.entries:
            title = e.get("title", "").strip()
            link = e.get("link", "")
            summary = html.unescape(re.sub(r"<[^>]+>", " ", e.get("summary", "").strip()))
            published_parsed = e.get("published_parsed") or e.get("updated_parsed")
            published_dt = None
            if published_parsed and hasattr(published_parsed, "tm_year"):
                published_dt = datetime(*published_parsed[:6])
            source = None
            # feedparser may nest source as dict; handle safely
            src = e.get("source")
            if isinstance(src, dict):
                source = src.get("title")
            elif isinstance(src, str):
                source = src
            rows.append({
                "title": title,
                "link": link,
                "summary": summary,
                "source": source or "",
                "published_dt": published_dt,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["link"]).reset_index(drop=True)
    return df


def normalize_title(title: str) -> List[str]:
    # Lowercase, remove punctuation, split, drop stopwords
    title = re.sub(r"[^\w\s]", " ", title.lower())
    words = [w for w in title.split() if w and w not in STOPWORDS]
    return words


def topic_key_from_title(title: str, top_k: int = 5) -> str:
    words = normalize_title(title)
    return " ".join(words[:top_k]) or title.lower()


def group_articles(df: pd.DataFrame) -> Dict[str, List[int]]:
    """Group near-duplicate stories by a simple normalized key from the title.
    This is easy to read and good enough for headlines.
    """
    groups: Dict[str, List[int]] = {}
    for i, row in df.iterrows():
        key = topic_key_from_title(row["title"])
        groups.setdefault(key, []).append(i)
    return groups


def recency_score(published_dt: datetime | None, now: datetime) -> float:
    # Halve the score every 24 hours; unknown date gets a small constant
    if not published_dt:
        return 0.3
    hours = (now - published_dt).total_seconds() / 3600.0
    return 1.0 / (2 ** (hours / 24.0))


def clean_for_summary(text: str) -> str:
    text = re.sub(r"http\S+", "", text)               # remove URLs
    text = re.sub(r"\bLINK:.*", "", text)             # drop LINK lines
    text = re.sub(r"\s+", " ", text).strip()          # collapse spaces
    return text


def format_as_bullets(summary: str, max_bullets: int = 5) -> str:
    parts = re.split(r"(?<=[.!?])\s+", summary)
    bullets = [f"• {p.strip()}" for p in parts if p.strip()]
    return "\n".join(bullets[:max_bullets])

# Add: paragraph formatter for summaries

def format_as_paragraph(summary: str) -> str:
    return re.sub(r"\s+", " ", summary).strip()

# -----------------------------
# Summarizers (local + OpenAI)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_local_summarizer():
    if hf_pipeline is None:
        return None
    try:
        # small, fast summarizer
        return hf_pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception:
        return None


def summarize_with_local(text: str) -> str:
    summarizer = load_local_summarizer()
    if summarizer is None:
        return "(Local summarizer not available. Install `transformers`.)"
    text = clean_for_summary(text)[:4000]
    if not text:
        return "(No content to summarize)"
    words = len(text.split())
    # Aim for a longer paragraph; lengths are in tokens, we estimate conservatively
    max_len = min(300, max(120, words // 2))
    min_len = min(max(80, words // 4), max_len - 30) if max_len > 120 else 80
    try:
        out = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        raw = out[0].get("summary_text", "").strip()
        return format_as_paragraph(raw)
    except Exception:
        # extractive fallback (first 3-5 sentences as a paragraph)
        parts = re.split(r"(?<=[.!?])\s+", text)
        raw = " ".join([p.strip() for p in parts[:5] if p.strip()])
        return format_as_paragraph(raw)


def summarize_with_openai(text: str, api_key: str | None, model: str = "gpt-4o-mini") -> str:
    if OpenAI is None:
        return "(OpenAI library not installed. Install `openai`.)"
    if not api_key and not os.getenv("OPENAI_API_KEY"):
        return "(Set OPENAI_API_KEY or paste your key in the sidebar.)"
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    text = clean_for_summary(text)[:8000]
    try:
        prompt = f"""
Write a clear, factual single-paragraph summary (5–7 sentences) of the following news cluster.
Cover who/what, what happened, when/where (if known), and why it matters.
Avoid speculation. No bullet points, no headings, no preamble or outro.

TEXT:
{text}
"""
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(OpenAI error: {e})"

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Settings")
    country_name = st.selectbox("Edition / Country", list(ISO_COUNTRIES.keys()), index=0)
    country_code, language = ISO_COUNTRIES[country_name]

    topic_choice = st.selectbox("Topic", list(TOPIC_MAP.keys()), index=0)
    query = st.text_input("Keyword search (overrides Topic)", placeholder="e.g., elections, climate, semiconductor")

    max_articles = st.slider("Max articles", 20, 200, 80, 10)

    st.divider()
    summarizer_choice = st.radio("Summarizer", ["Local (free)", "OpenAI"], index=0)
    openai_key = None
    openai_model = "gpt-4o-mini"
    if summarizer_choice == "OpenAI":
        openai_key = st.text_input("OpenAI API Key (optional if set as env)", type="password")
        openai_model = st.text_input("OpenAI model", value="gpt-4o-mini")

# -----------------------------
# Fetch + prepare data
# -----------------------------
urls = build_rss_urls(query or None, country_code, language, topic_choice)
feeds = [fetch_rss(u) for u in urls]
df = parse_entries(feeds)

if df.empty:
    st.warning("No news found. Try a different topic or keyword.")
    st.stop()

# Sort newest first and cap to user limit
if "published_dt" in df.columns:
    df = df.sort_values("published_dt", ascending=False)
df = df.head(max_articles).reset_index(drop=True)

# Group into simple topic buckets
buckets = group_articles(df)

# Score clusters by size + recency (simple heuristic)
now = datetime.utcnow()
cluster_table = []
for key, idxs in buckets.items():
    rec = 0.0
    for i in idxs:
        rec = max(rec, recency_score(df.loc[i, "published_dt"], now))
    cluster_table.append({
        "key": key,
        "count": len(idxs),
        "recency": rec,
        "score": len(idxs) * 0.7 + rec * 10.0 * 0.3,
        "idxs": idxs,
    })
cluster_table.sort(key=lambda r: r["score"], reverse=True)

# -----------------------------
# Display clusters
# -----------------------------
st.subheader("Trending Topic Clusters")

all_summaries_md: List[str] = []

for row in cluster_table:
    idxs = row["idxs"]
    sub = df.loc[idxs]

    # Build a compact cluster text for summarization
    parts = []
    for _, r in sub.head(10).iterrows():
        t = f"- {r['title']}"
        if pd.notnull(r.get("published_dt")):
            t += f" (🕒 {r['published_dt'].strftime('%Y-%m-%d %H:%M')})"
        if r["summary"]:
            t += f"\n  {r['summary'][:280]}"
        t += f"\n  LINK: {r['link']}"
        parts.append(t)
    cluster_text = "\n".join(parts)

    # Summarize
    with st.container(border=True):
        st.markdown(f"### {sub.iloc[0]['title']}")
        with st.spinner("Summarizing…"):
            if summarizer_choice == "OpenAI":
                summary_md = summarize_with_openai(cluster_text, api_key=openai_key, model=openai_model)
            else:
                summary_md = summarize_with_local(cluster_text)
        st.markdown(summary_md)
        all_summaries_md.append(f"#### {sub.iloc[0]['title']}\n" + summary_md)

        # Expand: list all stories
        with st.expander("All stories in this topic"):
            for _, r in sub.iterrows():
                time_str = r["published_dt"].strftime("%Y-%m-%d %H:%M") if pd.notnull(r["published_dt"]) else ""
                st.markdown(f"- [{r['title']}]({r['link']})")
                if r["summary"]:
                    st.caption(r["summary"][:220] + ("…" if len(r["summary"]) > 220 else ""))
                if time_str:
                    st.caption(f"🕒 {time_str}")

# -----------------------------
# Raw data + Export
# -----------------------------
with st.expander("View raw articles table"):
    st.dataframe(df)

md_blob = "\n\n".join(all_summaries_md) if all_summaries_md else "(No summaries)"
st.download_button("Download summaries (Markdown)", md_blob, file_name="news_summaries.md")
