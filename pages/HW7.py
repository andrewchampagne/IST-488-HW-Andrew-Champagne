"""
app.py  –  News Reporting Bot for Law Firm Client Monitoring
------------------------------------------------------------
Pre-requisite: run `python build_rag_db.py` once to build rag_db.pkl.
Then:  streamlit run app.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
import tiktoken


DB_PATH = "rag_db.pkl"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
TOP_K = 8          # articles retrieved per query
BUFFER = 6         # last N messages kept in history


@st.cache_resource
def load_db():
    if not os.path.exists(DB_PATH):
        st.error(
            f"**RAG database not found.**  "
            f"Please run `python build_rag_db.py` first, then restart the app."
        )
        st.stop()
    with open(DB_PATH, "rb") as f:
        return pickle.load(f)

db = load_db()
df: pd.DataFrame = db["df"]
embeddings: np.ndarray = db["embeddings"]
companies: list = db["companies"]

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def embed_query(text: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec


def retrieve(query: str, company_filter: list[str] | None = None, k: int = TOP_K) -> pd.DataFrame:
    """Return top-k rows most similar to the query, with optional company filter."""
    q_vec = embed_query(query)
    scores = embeddings @ q_vec          # cosine sim (vectors are normalised)

    candidate_idx = np.argsort(scores)[::-1]

    if company_filter:
        mask = df["company_name"].isin(company_filter).values
        candidate_idx = [i for i in candidate_idx if mask[i]]

    top_idx = candidate_idx[:k]
    result = df.iloc[top_idx].copy()
    result["_score"] = scores[top_idx]
    return result


def format_articles_for_context(rows: pd.DataFrame) -> str:
    parts = []
    for rank, (_, row) in enumerate(rows.iterrows(), 1):
        doc_preview = str(row["Document"])[:600]
        parts.append(
            f"[Article {rank}]\n"
            f"Company: {row['company_name']}\n"
            f"Date: {row['Date'].strftime('%Y-%m-%d')}\n"
            f"Content: {doc_preview}\n"
            f"URL: {row['URL']}"
        )
    return "\n\n".join(parts)



def count_tokens(messages: list[dict], model: str = CHAT_MODEL) -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    n = sum(4 + sum(len(enc.encode(v)) for v in m.values()) for m in messages)
    return n + 2


def apply_buffer(messages: list[dict]) -> list[dict]:
    return messages[-BUFFER:] if len(messages) > BUFFER else messages


def build_system_prompt(retrieved: pd.DataFrame | None = None) -> str:
    base = (
        "You are LexMonitor, an expert news analyst bot for a global law firm. "
        "Your job is to help attorneys and client relationship managers stay informed "
        "about news affecting their clients.\n\n"
        "Guidelines:\n"
        "- ONLY discuss news from the articles provided to you. Do not invent facts.\n"
        "- When asked for 'interesting' or 'top' news, rank articles by legal/business "
        "  significance: regulatory actions, litigation, M&A, leadership changes, and "
        "  financial distress rank highest.\n"
        "- For each article you mention, include: company name, date, a 1-2 sentence "
        "  summary, why it matters legally or strategically, and the URL.\n"
        "- Use numbered lists when returning multiple articles.\n"
        "- If the user asks about a company not in the database, say so clearly.\n"
        "- Keep a professional, precise tone suitable for a law firm audience."
    )
    if retrieved is not None and not retrieved.empty:
        context = format_articles_for_context(retrieved)
        return f"{base}\n\n--- RELEVANT ARTICLES ---\n{context}\n--- END ARTICLES ---"
    return base

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = None


with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Scales_of_justice_2.svg/240px-Scales_of_justice_2.svg.png",
        width=60,
    )
    st.title("⚖️ LexMonitor")
    st.caption("Client News Intelligence")

    st.divider()
    st.subheader("Filter by Client")
    selected_companies = st.multiselect(
        "Show news only for:",
        options=companies,
        placeholder="All clients (no filter)",
    )

    st.divider()
    st.subheader("Quick Queries")
    col1, col2 = st.columns(2)
    if col1.button("🔥 Top News", use_container_width=True):
        st.session_state._quick_query = "Find the most interesting and legally significant news"
    if col2.button("⚠️ Legal Risk", use_container_width=True):
        st.session_state._quick_query = "Find news involving regulatory actions, lawsuits, or legal risk"
    if col1.button("💼 M&A / Deals", use_container_width=True):
        st.session_state._quick_query = "Find news about mergers, acquisitions, or major deals"
    if col2.button("📉 Financials", use_container_width=True):
        st.session_state._quick_query = "Find news about financial results, earnings, or distress"

    st.divider()
    st.subheader("Stats")
    st.metric("Total Articles", len(df))
    st.metric("Clients Covered", df["company_name"].nunique())
    date_min = df["Date"].min().strftime("%b %d")
    date_max = df["Date"].max().strftime("%b %d, %Y")
    st.caption(f"Coverage: {date_min} – {date_max}")

    if st.session_state.messages:
        st.divider()
        st.subheader("Token Usage")
        sys_tokens = count_tokens([{"role": "system", "content": build_system_prompt()}])
        buf = apply_buffer(st.session_state.messages)
        msg_tokens = count_tokens(buf)
        st.write(f"System: ~{sys_tokens:,} tokens")
        st.write(f"History: ~{msg_tokens:,} tokens")
        st.write(f"**Total: ~{sys_tokens + msg_tokens:,}**")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_retrieved = None
        st.rerun()


st.title("Client News Bot")
st.caption(
    "Ask about news affecting your clients. "
    "Try: *'Find the most interesting news'*, *'Find news about Apple'*, "
    "or *'Any regulatory issues this week?'*"
)

if st.session_state.last_retrieved is not None and not st.session_state.last_retrieved.empty:
    with st.expander(f"📎 {len(st.session_state.last_retrieved)} source articles used for last response"):
        for _, row in st.session_state.last_retrieved.iterrows():
            st.markdown(
                f"**{row['company_name']}** · {row['Date'].strftime('%Y-%m-%d')}  \n"
                f"{str(row['Document'])[:120]}...  \n"
                f"[Read article]({row['URL']})"
            )
            st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if "_quick_query" in st.session_state:
    prompt = st.session_state.pop("_quick_query")
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    do_generate = True
else:
    do_generate = False
    prompt = None

if user_input := st.chat_input("Ask about client news…"):
    prompt = user_input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    do_generate = True

if do_generate and prompt:
    # Retrieve relevant articles
    filter_cos = selected_companies if selected_companies else None
    retrieved = retrieve(prompt, company_filter=filter_cos, k=TOP_K)
    st.session_state.last_retrieved = retrieved

    system_prompt = build_system_prompt(retrieved)
    buffered = apply_buffer(st.session_state.messages)
    api_messages = [{"role": "system", "content": system_prompt}] + buffered

    with st.chat_message("assistant"):
        container = st.empty()
        full_response = ""
        try:
            stream = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=api_messages,
                stream=True,
                temperature=0.3,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    container.markdown(full_response + "▌")
            container.markdown(full_response)
        except Exception as e:
            st.error(f"Error: {e}")
            full_response = f"Sorry, I encountered an error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()
