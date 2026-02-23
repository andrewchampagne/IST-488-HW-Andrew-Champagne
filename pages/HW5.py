import re
from html import unescape
from pathlib import Path

import chromadb
import streamlit as st
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

st.title("HW5: Short-Term Memory Chatbot")
st.write(
    "This page uses a retrieval function (`relative_club_info`) to fetch relevant "
    "club information from ChromaDB, then sends those retrieval results to the LLM."
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HTML_SOURCE_DIR = PROJECT_ROOT / "su_orgs"
CHROMA_DB_DIR = PROJECT_ROOT / "vector_db" / "hw4_chroma"
CHROMA_DB_FILE = CHROMA_DB_DIR / "chroma.sqlite3"
COLLECTION_NAME = "HW4HtmlCollection"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-mini"
MAX_MEMORY_INTERACTIONS = 5


def extract_text_from_html(html_content: str) -> str:
    without_scripts = re.sub(
        r"<script[\s\S]*?</script>|<style[\s\S]*?</style>",
        " ",
        html_content,
        flags=re.IGNORECASE,
    )
    text_only = re.sub(r"<[^>]+>", " ", without_scripts)
    return " ".join(unescape(text_only).split())


def split_into_two_chunks(text: str) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    midpoint = len(cleaned) // 2
    first_half = cleaned[:midpoint].strip()
    second_half = cleaned[midpoint:].strip()
    return [chunk for chunk in (first_half, second_half) if chunk]


def build_html_chunks() -> tuple[list[str], list[dict], list[str]]:
    html_files = sorted(HTML_SOURCE_DIR.glob("*.html"))
    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for html_path in html_files:
        raw_html = html_path.read_text(encoding="utf-8", errors="ignore")
        doc_text = extract_text_from_html(raw_html)
        chunks = split_into_two_chunks(doc_text)

        for chunk_idx, chunk_text in enumerate(chunks):
            documents.append(chunk_text)
            metadatas.append(
                {
                    "source": html_path.name,
                    "chunk_index": chunk_idx,
                    "doc_type": "html",
                }
            )
            ids.append(f"{html_path.stem}::chunk{chunk_idx}")

    return documents, metadatas, ids


def get_or_create_vector_db():
    embedding_function = OpenAIEmbeddingFunction(
        api_key=st.secrets["OPENAI_API_KEY"],
        model_name=EMBEDDING_MODEL,
    )

    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    db_already_exists = CHROMA_DB_FILE.exists()

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    if not db_already_exists:
        documents, metadatas, ids = build_html_chunks()
        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
        st.info("Created the persistent ChromaDB for HW4/HW5 data.")

    return collection


if "HW5_VectorDB" not in st.session_state:
    with st.spinner("Preparing vector database..."):
        st.session_state.HW5_VectorDB = get_or_create_vector_db()

collection = st.session_state.HW5_VectorDB
st.success("Vector DB ready")
st.write(f"Collection: `{collection.name}`")
st.write(f"Chunk count: `{collection.count()}`")


def relative_club_info(query: str, k: int = 5) -> dict:
    result = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    context_blocks = []
    sources = []
    for idx, (doc, meta, distance) in enumerate(zip(docs, metas, distances), start=1):
        source = meta.get("source", "Unknown")
        chunk_index = meta.get("chunk_index", "Unknown")
        if source not in sources:
            sources.append(source)
        context_blocks.append(
            f"[Result {idx}] Source: {source} | chunk: {chunk_index} | distance: {distance}\n"
            f"{doc}"
        )

    return {"context": "\n\n".join(context_blocks), "sources": sources}


def build_memory_text(history: list[dict]) -> str:
    if not history:
        return "No prior interactions."

    lines = []
    for idx, item in enumerate(history, start=1):
        lines.append(
            f"Interaction {idx} User: {item['user']}\n"
            f"Interaction {idx} Assistant: {item['assistant']}"
        )
    return "\n\n".join(lines)


def trim_history() -> None:
    if len(st.session_state.hw5_history) > MAX_MEMORY_INTERACTIONS:
        st.session_state.hw5_history = st.session_state.hw5_history[
            -MAX_MEMORY_INTERACTIONS:
        ]


st.divider()
st.subheader("Chat")
st.write("Ask about Syracuse organizations.")

if "hw5_history" not in st.session_state:
    st.session_state.hw5_history = []

for interaction in st.session_state.hw5_history:
    with st.chat_message("user"):
        st.markdown(interaction["user"])
    with st.chat_message("assistant"):
        st.markdown(interaction["assistant"])

openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if user_question := st.chat_input("Ask a question about organizations"):
    with st.chat_message("user"):
        st.markdown(user_question)

    retrieval = relative_club_info(user_question, k=5)
    memory_text = build_memory_text(st.session_state.hw5_history)

    # Per HW5 instructions, we pass retrieval results directly into prompts 
    system_prompt = (
        "You are a campus organization assistant. Use the retrieved club information "
        "below as your main evidence. If evidence is incomplete, say so clearly and "
        "then provide a cautious best-effort answer.\n\n"
        f"Retrieved club information:\n{retrieval['context'] if retrieval['context'] else 'No retrieval results.'}"
    )
    user_prompt = (
        f"Conversation memory (last {MAX_MEMORY_INTERACTIONS} interactions):\n{memory_text}\n\n"
        f"Current user question:\n{user_question}"
    )

    with st.chat_message("assistant"):
        try:
            completion = openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            assistant_text = completion.choices[0].message.content or ""
            st.markdown(assistant_text)

            if retrieval["sources"]:
                st.caption(f"Sources used: {', '.join(retrieval['sources'][:5])}")

            st.session_state.hw5_history.append(
                {"user": user_question, "assistant": assistant_text}
            )
            trim_history()
        except Exception as exc:
            st.error(f"Error generating response: {exc}")