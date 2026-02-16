import re
from html import unescape
from pathlib import Path

import chromadb
import streamlit as st
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

st.title("HW4: HTML RAG Vector DB + Chatbot")
st.write(
    "Builds a persistent Chroma vector database from the provided HTML files and "
    "uses it to augment responses in a Streamlit chatbot."
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
    normalized = " ".join(unescape(text_only).split())
    return normalized


def split_into_two_chunks(text: str) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    # Chunking method used:
    # We split each document into exactly two sequential halves (first half + second half).
    # This guarantees "two mini-documents per source document" as required, keeps
    # each chunk semantically contiguous, and is simple/reproducible for this homework.
    midpoint = len(cleaned) // 2

    first_half = cleaned[:midpoint].strip()
    second_half = cleaned[midpoint:].strip()
    chunks = [chunk for chunk in (first_half, second_half) if chunk]
    return chunks


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
            chunk_id = f"{html_path.stem}::chunk{chunk_idx}"
            documents.append(chunk_text)
            metadatas.append(
                {
                    "source": html_path.name,
                    "chunk_index": chunk_idx,
                    "doc_type": "html",
                }
            )
            ids.append(chunk_id)

    return documents, metadatas, ids


def get_or_create_hw4_vector_db():
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    embedding_function = OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name=EMBEDDING_MODEL,
    )

    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    db_already_exists = CHROMA_DB_FILE.exists()

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    # Only build/populate when the persistent DB file does not already exist.
    # This lets the app be rerun many times without rebuilding embeddings.
    if not db_already_exists:
        documents, metadatas, ids = build_html_chunks()
        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
        st.info(f"Created new vector DB at `{CHROMA_DB_DIR}`.")

    return collection


if "HW4_VectorDB" not in st.session_state:
    with st.spinner("Preparing persistent vector DB from HTML files..."):
        st.session_state.HW4_VectorDB = get_or_create_hw4_vector_db()

collection = st.session_state.HW4_VectorDB
st.success("Vector DB is ready.")
st.write(f"Collection: `{collection.name}`")
st.write(f"Document chunks in collection: `{collection.count()}`")


def retrieve_relevant_chunks(query_text: str, k: int = 5) -> list[dict]:
    result = collection.query(
        query_texts=[query_text],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    rows = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        rows.append(
            {
                "chunk_text": document,
                "source": metadata.get("source", "Unknown"),
                "chunk_index": metadata.get("chunk_index", "Unknown"),
                "distance": distance,
            }
        )
    return rows


def build_rag_context(chunks: list[dict]) -> tuple[str, list[str]]:
    blocks = []
    unique_sources = []
    for idx, chunk in enumerate(chunks, start=1):
        source = chunk["source"]
        if source not in unique_sources:
            unique_sources.append(source)
        blocks.append(
            f"[Context {idx}] Source: {source} (chunk {chunk['chunk_index']})\n"
            f"{chunk['chunk_text']}"
        )
    return "\n\n".join(blocks), unique_sources


st.divider()
st.subheader("HW4 RAG Chatbot")
st.write("Ask a question. The app retrieves relevant HTML chunks from ChromaDB.")

openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "hw4_conversation" not in st.session_state:
    st.session_state.hw4_conversation = []


def trim_conversation_buffer() -> None:
    if len(st.session_state.hw4_conversation) > MAX_MEMORY_INTERACTIONS:
        st.session_state.hw4_conversation = st.session_state.hw4_conversation[
            -MAX_MEMORY_INTERACTIONS:
        ]


for interaction in st.session_state.hw4_conversation:
    with st.chat_message("user"):
        st.markdown(interaction["user"])
    with st.chat_message("assistant"):
        st.markdown(interaction["assistant"])

if user_question := st.chat_input("Ask about the SU organization HTML pages"):
    with st.chat_message("user"):
        st.markdown(user_question)

    retrieved_chunks = retrieve_relevant_chunks(user_question, k=5)
    rag_context, sources_used = build_rag_context(retrieved_chunks)

    memory_text = ""
    if st.session_state.hw4_conversation:
        memory_lines = []
        for idx, interaction in enumerate(st.session_state.hw4_conversation, start=1):
            memory_lines.append(
                f"Interaction {idx} - User: {interaction['user']}\n"
                f"Interaction {idx} - Assistant: {interaction['assistant']}"
            )
        memory_text = "\n\n".join(memory_lines)

    system_prompt = (
        "You are a helpful campus organizations assistant. Use the retrieved context "
        "when it is relevant to answer accurately. If the context is not enough, state "
        "what is missing and then provide your best helpful answer."
    )

    user_prompt = (
        f"Current question:\n{user_question}\n\n"
        f"Conversation memory (last {MAX_MEMORY_INTERACTIONS} interactions):\n"
        f"{memory_text if memory_text else 'No prior interactions.'}\n\n"
        f"Retrieved vector DB context:\n{rag_context if rag_context else 'No retrieved chunks.'}"
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
            if sources_used:
                st.caption(f"Retrieved from: {', '.join(sources_used[:5])}")

            st.session_state.hw4_conversation.append(
                {"user": user_question, "assistant": assistant_text}
            )
            trim_conversation_buffer()
        except Exception as exc:
            st.error(f"Error generating response: {exc}")