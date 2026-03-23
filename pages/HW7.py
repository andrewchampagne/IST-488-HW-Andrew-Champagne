import re
from pathlib import Path

import chromadb
import pandas as pd
import streamlit as st
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI


st.title("Client News Reporting Bot")

PROJECT_ROOT    = Path(__file__).resolve().parent
CSV_PATH        = PROJECT_ROOT / "news.csv"
CHROMA_DB_DIR   = PROJECT_ROOT / "vector_db" / "hw7_chroma"
CHROMA_DB_FILE  = CHROMA_DB_DIR / "chroma.sqlite3"
COLLECTION_NAME = "HW7NewsCollection"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL      = "gpt-4o-mini"
TOP_K           = 8    
MAX_MEMORY      = 6   

@st.cache_data
def load_news_df() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.sort_values("Date", ascending=False).reset_index(drop=True)
    return df

df = load_news_df()
companies = sorted(df["company_name"].dropna().unique().tolist())

def build_documents_from_csv(dataframe: pd.DataFrame) -> tuple[list[str], list[dict], list[str]]:
    """Convert each CSV row into a ChromaDB document with metadata."""
    documents, metadatas, ids = [], [], []
    for idx, row in dataframe.iterrows():
        text = (
            f"Company: {row['company_name']}\n"
            f"Date: {row['Date'].strftime('%Y-%m-%d') if pd.notna(row['Date']) else 'Unknown'}\n"
            f"Article: {str(row['Document'])[:1000]}"
        )
        documents.append(text)
        metadatas.append({
            "company":  str(row["company_name"]),
            "date":     row["Date"].strftime("%Y-%m-%d") if pd.notna(row["Date"]) else "",
            "url":      str(row["URL"]),
            "doc_text": str(row["Document"])[:300],
        })
        ids.append(f"article::{idx}")
    return documents, metadatas, ids


def get_or_create_vector_db():
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name=EMBEDDING_MODEL,
    )

    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    db_already_exists = CHROMA_DB_FILE.exists()

    client     = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    if not db_already_exists:
        documents, metadatas, ids = build_documents_from_csv(df)
        if documents:
            # Upsert in batches of 100 to avoid API timeouts
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                collection.add(
                    documents=documents[i : i + batch_size],
                    metadatas=metadatas[i : i + batch_size],
                    ids=ids[i : i + batch_size],
                )
        st.info(f"Vector DB built and saved to `{CHROMA_DB_DIR}`.")

    return collection


if "HW7_VectorDB" not in st.session_state:
    with st.spinner("Preparing vector database from news articles (first run only)…"):
        st.session_state.HW7_VectorDB = get_or_create_vector_db()

collection = st.session_state.HW7_VectorDB

def retrieve_articles(query: str, company_filter: list[str] | None = None, k: int = TOP_K) -> list[dict]:
    where = {"company": {"$in": company_filter}} if company_filter else None
    result = collection.query(
        query_texts=[query],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    docs      = result.get("documents",  [[]])[0]
    metas     = result.get("metadatas",  [[]])[0]
    distances = result.get("distances",  [[]])[0]
    return [
        {
            "text":     doc,
            "company":  meta.get("company",  ""),
            "date":     meta.get("date",     ""),
            "url":      meta.get("url",      ""),
            "doc_text": meta.get("doc_text", ""),
            "distance": dist,
        }
        for doc, meta, dist in zip(docs, metas, distances)
    ]


def build_rag_context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(
            f"[Article {i}]\n"
            f"Company: {c['company']}\n"
            f"Date: {c['date']}\n"
            f"Content: {c['text']}\n"
            f"URL: {c['url']}"
        )
    return "\n\n".join(blocks)


def build_memory_text(conversation: list[dict]) -> str:
    if not conversation:
        return "No prior interactions."
    lines = []
    for i, turn in enumerate(conversation, 1):
        lines.append(f"Turn {i} – User: {turn['user']}\nTurn {i} – Assistant: {turn['assistant']}")
    return "\n\n".join(lines)


def trim_buffer(conversation: list[dict]) -> list[dict]:
    return conversation[-MAX_MEMORY:] if len(conversation) > MAX_MEMORY else conversation

with st.sidebar:

    st.subheader("Filter by Client")
    selected_companies = st.multiselect(
        "Restrict search to:",
        options=companies,
        placeholder="All clients (no filter)",
    )

    st.divider()
    st.subheader("Quick Queries")
    quick_queries = {
        "Most Interesting":  "Find the most interesting and legally significant news",
        "Legal & Regulatory": "Find news involving regulatory actions, lawsuits, or legal risk",
        "M&A / Deals":       "Find news about mergers, acquisitions, or major business deals",
        "Financials":         "Find news about financial results, earnings, or distress",
    }
    for label, query_text in quick_queries.items():
        if st.button(label, use_container_width=True):
            st.session_state._quick_query = query_text

    st.divider()
    st.subheader("Database Stats")
    st.metric("Articles Indexed", collection.count())
    st.metric("Clients Covered",  df["company_name"].nunique())
    date_min = df["Date"].min().strftime("%b %d")
    date_max = df["Date"].max().strftime("%b %d, %Y")
    st.caption(f"Coverage: {date_min} – {date_max}")

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.hw7_conversation = []
        st.session_state.pop("_last_sources", None)
        st.rerun()

if "hw7_conversation" not in st.session_state:
    st.session_state.hw7_conversation = []

openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.divider()
if st.session_state.get("_last_sources"):
    with st.expander(f"📎 {len(st.session_state['_last_sources'])} source articles used for last response"):
        for chunk in st.session_state["_last_sources"]:
            st.markdown(
                f"**{chunk['company']}** · {chunk['date']}  \n"
                f"{chunk['doc_text']}…  \n"
                f"[Read article]({chunk['url']})"
            )
            st.divider()

for turn in st.session_state.hw7_conversation:
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        st.markdown(turn["assistant"])

prompt = None
if "_quick_query" in st.session_state:
    prompt = st.session_state.pop("_quick_query")
    with st.chat_message("user"):
        st.markdown(prompt)

if user_input := st.chat_input("Ask about client news… e.g. 'Find news about Apple'"):
    prompt = user_input
    with st.chat_message("user"):
        st.markdown(prompt)

if prompt:
    filter_cos = selected_companies if selected_companies else None
    retrieved  = retrieve_articles(prompt, company_filter=filter_cos, k=TOP_K)
    st.session_state["_last_sources"] = retrieved

    rag_context  = build_rag_context(retrieved)
    memory_text  = build_memory_text(st.session_state.hw7_conversation)

    system_prompt = (
        "You are an expert news analyst for a global law firm. "
        "Your job is to help attorneys and client relationship managers stay informed "
        "about news affecting their clients.\n\n"
        "Guidelines:\n"
        "- ONLY discuss news from the articles provided. Do not invent facts.\n"
        "- When asked for 'interesting' or 'top' news, rank by legal/business significance: "
        "regulatory actions, litigation, M&A, leadership changes, and financial distress rank highest.\n"
        "- For each article, include: company name, date, 1-2 sentence summary, "
        "why it matters legally or strategically, and the URL.\n"
        "- Use numbered lists when returning multiple articles.\n"
        "- If a company is not in the database, say so clearly.\n"
        "- Keep a professional, precise tone suitable for a law firm audience."
    )

    user_prompt = (
        f"Current question:\n{prompt}\n\n"
        f"Conversation memory (last {MAX_MEMORY} turns):\n{memory_text}\n\n"
        f"Retrieved articles:\n{rag_context if rag_context else 'No relevant articles found.'}"
    )

    with st.chat_message("assistant"):
        container     = st.empty()
        full_response = ""
        try:
            stream = openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                stream=True,
                temperature=0.3,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    container.markdown(full_response + "▌")
            container.markdown(full_response)

            sources_used = list({c["company"] for c in retrieved})[:5]
            if sources_used:
                st.caption(f"Sources: {', '.join(sources_used)}")

        except Exception as exc:
            st.error(f"Error generating response: {exc}")
            full_response = f"Sorry, I encountered an error: {exc}"

    st.session_state.hw7_conversation.append({"user": prompt, "assistant": full_response})
    st.session_state.hw7_conversation = trim_buffer(st.session_state.hw7_conversation)
    st.rerun()