# 🔐 Pakistani Cyber Crime RAG Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions about Pakistani cyber-crime legislation using a provided Pakistani legal document as its knowledge source.

The application uses:

- **Streamlit** for the web interface
- **FAISS** for vector similarity search
- **Sentence Transformers** for semantic embeddings
- **Groq** for large-language-model generation
- **Llama 3.3 70B Versatile** as the LLM
- **Google Drive** as the source location for the PDF document

> **Important:** This project is an educational/legal-information demonstration. It is not a substitute for advice from a qualified Pakistani lawyer.

---

## 1. Project Overview

Normal LLM applications can answer questions using information learned during model training. That creates a problem for legal applications because the model may provide information that is outdated, incomplete, or unrelated to the specific legal document being studied.

This project uses **Retrieval-Augmented Generation (RAG)**.

Instead of directly asking the LLM a question, the application first searches the Pakistani cyber-crime document for relevant passages. Those passages are then provided to the LLM as context.

The basic flow is:

```text
                    Pakistani Cyber-Crime PDF
                              │
                              ▼
                       Extract PDF Text
                              │
                              ▼
                         Text Chunking
                              │
                              ▼
                     Sentence Transformer
                         Embeddings
                              │
                              ▼
                          FAISS Index
                              │
                     User asks question
                              │
                              ▼
                    Embed the user question
                              │
                              ▼
                    FAISS similarity search
                              │
                              ▼
                     Relevant text chunks
                              │
                              ▼
                  Groq Llama 3.3 70B
                              │
                              ▼
                       Grounded answer
                              │
                              ▼
                         Streamlit UI
```

---

# 2. Source Document

The application uses the Pakistani cyber-crime document provided through Google Drive.

The configured Google Drive file ID is:

```text
16nmBDF_69B_5IVok23N9iJkEtjB5UjgI
```

The document is downloaded by the application at runtime.

The source document contains the **Prevention of Electronic Crimes Act, 2016 (PECA)** material, including provisions concerning electronic crimes, offences, punishments, investigation, and related matters.

The application does not require the PDF to be committed to the GitHub repository.

---

# 3. Why RAG?

RAG combines two components:

### Retrieval

FAISS searches the document's vector representations and finds passages that are semantically similar to the user's question.

### Generation

The retrieved passages are sent to Groq's Llama 3.3 70B Versatile model. The model generates an understandable response based on those passages.

Therefore:

```text
Question
   ↓
Semantic Search
   ↓
Relevant legal passages
   ↓
LLM
   ↓
Answer grounded in source
```

This reduces the risk of the LLM answering a legal question entirely from its general training knowledge.

---

# 4. Technologies

## Streamlit

Streamlit provides the interactive browser-based application.

## FAISS

FAISS is used as the vector database/search engine.

The application uses:

```python
faiss.IndexFlatIP
```

The document embeddings and query embeddings are normalized, so inner-product similarity corresponds to cosine similarity.

## Sentence Transformers

The application uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

for generating embeddings.

This model is relatively lightweight and suitable for a Streamlit Cloud demonstration.

## Groq

Groq provides the LLM inference API.

The application uses:

```text
llama-3.3-70b-versatile
```

---

# 5. Project Structure

The repository only requires three files:

```text
cyber-crime-rag/
│
├── app.py
├── requirements.txt
└── README.md
```

### `app.py`

Contains the complete Streamlit application, including:

- Google Drive PDF downloading
- PDF text extraction
- Text cleaning
- Chunking
- Embedding generation
- FAISS index creation
- Similarity retrieval
- Groq API integration
- Prompt construction
- Chat interface
- Retrieved-source display

### `requirements.txt`

Contains the Python dependencies required by the application.

### `README.md`

Contains project documentation and deployment instructions.

---

# 6. RAG Pipeline

## Step 1 — Download document

The application downloads the PDF from Google Drive.

The Google Drive file must be publicly accessible.

Use:

```text
Anyone with the link → Viewer
```

The application does not use a Google account login.

---

## Step 2 — Extract text

`pypdf` extracts text from the PDF page by page.

Keeping the page number is important because the application can later show the user where retrieved information came from.

Example metadata:

```text
Page 3
Page 4
Page 12
```

---

## Step 3 — Chunking

Large pages are divided into smaller overlapping pieces.

Current configuration:

```python
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
```

The overlap helps prevent important information from being lost when a legal provision crosses a chunk boundary.

Each chunk stores:

```text
chunk text
page number
chunk ID
```

---

# 7. Embeddings

Each text chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

For example:

```text
Legal text
    ↓
Embedding model
    ↓
[0.023, -0.184, 0.512, ...]
```

The vectors are normalized before being stored in FAISS.

---

# 8. FAISS Retrieval

When a user asks:

```text
What is unauthorized access?
```

the question is converted into an embedding.

FAISS compares the question vector with the document chunk vectors.

The application retrieves the top five most relevant chunks:

```python
TOP_K = 5
```

The similarity score is also retained.

---

# 9. Grounded Generation

The retrieved passages are inserted into a prompt for Groq.

The prompt instructs the model to:

1. Use only the provided source excerpts.
2. Avoid inventing legal information.
3. Mention section numbers when available.
4. Include page references.
5. Say when the answer cannot be found in the source.
6. Avoid presenting the output as professional legal advice.

The Groq call uses the requested syntax:

```python
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="llama-3.3-70b-versatile",
)

answer = chat_completion.choices[0].message.content
```

The deployed application additionally supports Streamlit Secrets for the API key.

---

# 10. Source References

The generated response is instructed to include page references such as:

```text
[Page 4]
```

The application also provides a:

```text
Retrieved source passages
```

section below the answer.

This allows the user to inspect the actual passages retrieved by FAISS.

---

# 11. Hallucination Control

Because this is a legal-information application, hallucination control is important.

The system prompt tells the LLM:

```text
Do not invent laws, sections, penalties, dates,
definitions, procedures, authorities, or legal interpretations.
```

It also tells the model to say when the retrieved document does not contain enough information.

The application additionally checks the highest FAISS similarity score.

If the best result is below:

```python
MIN_RELEVANCE_SCORE = 0.30
```

the application does not send the unrelated retrieved material to the LLM.

Instead, it returns:

```text
I could not find enough relevant information in the provided cyber-crime document to answer this question.
```

The threshold can be adjusted later after testing with real questions.

---

# 12. Installation

## Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/cyber-crime-rag.git
cd cyber-crime-rag
```

## Step 2 — Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

# 13. Configure Groq API Key

The application requires:

```text
GROQ_API_KEY
```

## Local development

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

Windows CMD:

```cmd
set GROQ_API_KEY=your_groq_api_key
```

macOS/Linux:

```bash
export GROQ_API_KEY="your_groq_api_key"
```

Do **not** put the API key directly inside `app.py`.

Do **not** commit the API key to GitHub.

---

# 14. Run Locally

Run:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, normally:

```text
http://localhost:8501
```

Open that URL in your browser.

---

# 15. Streamlit Cloud Deployment

## Step 1 — Push the project to GitHub

Your repository should contain:

```text
app.py
requirements.txt
README.md
```

Example:

```bash
git add app.py requirements.txt README.md
git commit -m "Initial cyber crime RAG application"
git push
```

---

## Step 2 — Create Streamlit Cloud application

Create a new application and select the GitHub repository.

Set:

```text
Main file path:
app.py
```

---

## Step 3 — Add Groq Secret

In Streamlit Cloud, open the application's **Secrets** configuration.

Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Do not put the actual key into GitHub.

---

# 16. Google Drive Configuration

The current source document is identified in `app.py` by:

```python
GOOGLE_DRIVE_FILE_ID = "16nmBDF_69B_5IVok23N9iJkEtjB5UjgI"
```

If the document changes, replace this ID with the new Google Drive file ID.

For example, if the Google Drive URL is:

```text
https://drive.google.com/file/d/FILE_ID/view
```

then:

```text
FILE_ID
```

is the value used by the application.

The new document must be publicly accessible to anyone with the link.

---

# 17. Example Questions

The application can be used for questions such as:

```text
What is unauthorized access under the Act?
```

```text
What is the punishment for unauthorized copying or transmission of data?
```

```text
What does the Act say about interference with an information system?
```

```text
What is spoofing?
```

```text
What does the Act say about critical infrastructure?
```

```text
What is the purpose of the Prevention of Electronic Crimes Act?
```

The answer should be based on retrieved passages from the source document.

---

# 18. Example RAG Process

Suppose the user asks:

```text
What is the punishment for unauthorized access?
```

The application performs:

```text
User question
      ↓
Embedding
      ↓
FAISS search
      ↓
Relevant passage
      ↓
Section 3
      ↓
Groq Llama 3.3 70B
      ↓
Answer + page reference
```

The application therefore does not simply ask the LLM:

```text
"What is the punishment?"
```

Instead, it provides relevant source material first.

---

# 19. Important Limitation

The current application is intentionally based on the supplied document.

This means it should **not** be treated as a complete, always-current Pakistani legal database.

If Pakistani cyber-crime legislation is amended, the source document should be replaced with an authoritative updated document and the application should be redeployed/restarted.

For real legal use, the source should be maintained from authoritative Pakistani legal sources.

---

# 20. Security

Never commit:

```text
GROQ_API_KEY
```

to GitHub.

Avoid code such as:

```python
client = Groq(api_key="gsk_...")
```

Instead use environment variables or Streamlit Secrets.

For Git repositories, a `.gitignore` can also be added later if local `.env` files are used.

---

# 21. Performance

The application uses Streamlit caching for:

- Downloaded PDF
- Extracted pages
- Chunks
- Embedding model
- FAISS index

This prevents the application from rebuilding the complete knowledge base unnecessarily on every Streamlit interaction.

The first startup can take longer because the embedding model needs to be loaded.

Subsequent interactions use the cached resources.

---

# 22. Current Configuration

| Component | Configuration |
|---|---|
| UI | Streamlit |
| Vector database | FAISS |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM | `llama-3.3-70b-versatile` |
| LLM provider | Groq |
| PDF extraction | pypdf |
| Source | Pakistani cyber-crime PDF |
| Retrieval | Cosine similarity via normalized FAISS vectors |
| Retrieved chunks | 5 |
| Chunk size | 900 words |
| Chunk overlap | 150 words |
| Minimum similarity | 0.30 |
| Deployment | Streamlit Cloud |

---

# 23. Future Improvements

Possible future improvements include:

- Hybrid keyword + semantic retrieval
- Reranking retrieved passages
- Better legal-domain embeddings
- Conversation-aware retrieval
- Multiple Pakistani legal documents
- Document version tracking
- Section-aware chunking
- Better handling of tables
- Urdu-language questions and answers
- Voice input
- User authentication
- Admin document management
- Persistent FAISS index storage
- Automatic document update detection
- More advanced legal citation formatting

---

# 24. Disclaimer

This application is designed for educational, research, and demonstration purposes.

It does not establish an attorney-client relationship and does not provide professional legal advice.

Users should consult a qualified legal professional for advice about an actual cyber-crime case, investigation, complaint, prosecution, or legal proceeding.

---

## License

You may add an appropriate open-source license to the repository depending on how you intend to distribute the project.
