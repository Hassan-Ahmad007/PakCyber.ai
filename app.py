import io
import os
import re
from dataclasses import dataclass
from typing import List

import faiss
import numpy as np
import requests
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

GOOGLE_DRIVE_FILE_ID = "16nmBDF_69B_5IVok23N9iJkEtjB5UjgI"
GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
MIN_RELEVANCE_SCORE = 0.30


# ============================================================
# Data structures
# ============================================================

@dataclass
class Chunk:
    text: str
    page: int
    chunk_id: int


# ============================================================
# Streamlit page
# ============================================================

st.set_page_config(
    page_title="Pakistani Cyber Crime RAG Assistant",
    page_icon="🔐",
    layout="wide",
)

st.title("🔐 Pakistani Cyber Crime RAG Assistant")
st.caption(
    "Ask questions about the Pakistani cyber-crime legislation provided as "
    "the application's source document."
)

st.info(
    "This assistant answers from the configured source document. "
    "It is an educational/research tool and is not a substitute for advice "
    "from a qualified Pakistani lawyer."
)


# ============================================================
# Document loading
# ============================================================

@st.cache_data(show_spinner=False)
def download_source_pdf() -> bytes:
    """Download the public Google Drive PDF."""
    urls = [
        (
            "https://drive.usercontent.google.com/download"
            f"?id={GOOGLE_DRIVE_FILE_ID}&export=download&confirm=t"
        ),
        (
            "https://drive.google.com/uc"
            f"?export=download&id={GOOGLE_DRIVE_FILE_ID}"
        ),
    ]

    last_error = None

    for url in urls:
        try:
            response = requests.get(
                url,
                timeout=60,
                allow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            if response.content.startswith(b"%PDF") or "pdf" in content_type:
                return response.content

            # Google Drive can sometimes return an HTML confirmation page.
            if b"<html" in response.content[:1000].lower():
                last_error = RuntimeError(
                    "Google Drive returned an HTML page instead of the PDF."
                )
            else:
                last_error = RuntimeError(
                    "The Google Drive response was not recognized as a PDF."
                )

        except requests.RequestException as exc:
            last_error = exc

    raise RuntimeError(
        "Could not download the source document from Google Drive. "
        "Make sure the file is shared as 'Anyone with the link - Viewer'."
    ) from last_error


@st.cache_data(show_spinner=False)
def extract_pages(pdf_bytes: bytes) -> List[tuple[int, str]]:
    """Extract text page-by-page so retrieved chunks retain page references."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        if text:
            pages.append((page_number, text))

    if not pages:
        raise RuntimeError(
            "No selectable text was found in the PDF. "
            "The document may be scanned/image-only and would need OCR."
        )

    return pages


# ============================================================
# Chunking
# ============================================================

def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping word-based chunks."""
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = max(end - overlap, start + 1)

    return chunks


@st.cache_data(show_spinner=False)
def build_chunks(pages: List[tuple[int, str]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    chunk_id = 0

    for page_number, page_text in pages:
        for text in split_text(page_text):
            chunks.append(
                Chunk(
                    text=text,
                    page=page_number,
                    chunk_id=chunk_id,
                )
            )
            chunk_id += 1

    return chunks


# ============================================================
# Embedding model + FAISS
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Building FAISS knowledge base...")
def build_faiss_index(chunks: List[Chunk]):
    model = load_embedding_model()

    texts = [chunk.text for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def retrieve_chunks(
    question: str,
    index,
    chunks: List[Chunk],
    model,
    top_k: int = TOP_K,
) -> List[tuple[Chunk, float]]:
    """Retrieve the most semantically relevant document chunks."""
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, index_id in zip(scores[0], indices[0]):
        if index_id < 0:
            continue

        results.append((chunks[int(index_id)], float(score)))

    return results


# ============================================================
# Groq
# ============================================================

def get_groq_api_key() -> str:
    """Read GROQ_API_KEY from Streamlit secrets first, then environment."""
    try:
        secret_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        secret_key = None

    api_key = secret_key or os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Add it to Streamlit Secrets "
            "when deploying, or set it as an environment variable locally."
        )

    return api_key


def build_prompt(question: str, retrieved: List[tuple[Chunk, float]]) -> str:
    context_parts = []

    for chunk, score in retrieved:
        context_parts.append(
            f"[Page {chunk.page} | Relevance {score:.3f}]\n{chunk.text}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return f"""
You are a Pakistani cyber-crime legal information assistant.

Your task is to answer the user's question using ONLY the supplied source
document excerpts.

STRICT RULES:
1. Do not invent laws, sections, penalties, dates, definitions, procedures,
   authorities, or legal interpretations.
2. If the supplied excerpts do not contain enough information to answer,
   clearly say that the answer was not found in the provided document.
3. Do not use your general knowledge to fill missing legal information.
4. Preserve the meaning of the source document.
5. When discussing a legal provision, mention the relevant section number
   when it is available in the retrieved text.
6. Include source page references in the answer using [Page X].
7. Keep the answer clear and understandable.
8. This is an educational information system, not legal advice.

SOURCE DOCUMENT EXCERPTS:
{context}

USER QUESTION:
{question}

ANSWER:
""".strip()


def generate_answer(question: str, retrieved: List[tuple[Chunk, float]]) -> str:
    client = Groq(api_key=get_groq_api_key())

    prompt = build_prompt(question, retrieved)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=GROQ_MODEL,
        temperature=0.1,
    )

    return chat_completion.choices[0].message.content.strip()


# ============================================================
# Initialize knowledge base
# ============================================================

try:
    with st.spinner("Loading Pakistani cyber-crime document..."):
        pdf_bytes = download_source_pdf()
        pages = extract_pages(pdf_bytes)
        chunks = build_chunks(pages)
        embedding_model = load_embedding_model()
        faiss_index = build_faiss_index(chunks)

except Exception as exc:
    st.error(f"Knowledge base initialization failed: {exc}")
    st.stop()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("Knowledge Base")

    st.write(f"**Source pages:** {len(pages)}")
    st.write(f"**Text chunks:** {len(chunks)}")
    st.write(f"**Embedding model:** `{EMBEDDING_MODEL}`")
    st.write(f"**LLM:** `{GROQ_MODEL}`")

    st.divider()

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "The FAISS index is created automatically from the configured "
        "Google Drive document."
    )


# ============================================================
# Chat history
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# User interaction
# ============================================================

question = st.chat_input(
    "Ask a question about Pakistani cyber-crime law..."
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the document..."):
                retrieved = retrieve_chunks(
                    question=question,
                    index=faiss_index,
                    chunks=chunks,
                    model=embedding_model,
                    top_k=TOP_K,
                )

            # If the best semantic match is weak, do not send unrelated
            # material to the LLM.
            if not retrieved or retrieved[0][1] < MIN_RELEVANCE_SCORE:
                answer = (
                    "I could not find enough relevant information in the "
                    "provided cyber-crime document to answer this question."
                )
            else:
                answer = generate_answer(question, retrieved)

            st.markdown(answer)

            with st.expander("Retrieved source passages"):
                for chunk, score in retrieved:
                    st.markdown(
                        f"**Page {chunk.page} — similarity {score:.3f}**"
                    )
                    st.write(chunk.text)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception as exc:
            error_message = f"Error while generating the answer: {exc}"
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message}
            )
