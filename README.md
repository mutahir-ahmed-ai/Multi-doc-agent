# 📚 Multi-Document Intelligence Agent

A multi-turn chat interface over uploaded PDF documents — powered by FAISS vector search and Groq's Llama 3.3 70B. Every answer shows exactly which document and page it came from.

---

## What It Does

Upload up to 5 PDFs → ask questions in natural language → get answers with source citations.

- Ask questions that span **multiple documents** — "compare how these two reports describe X"
- Ask **follow-up questions** — the agent remembers your conversation
- Every answer ends with the **source filename, page number, and a direct quote** from the relevant chunk

---

## Demo

**Uploaded documents:** `annual_report_2024.pdf`, `market_analysis.pdf`

**User:** What revenue figures are mentioned across the documents?

**Agent:** The annual report states total revenue of $4.2B for FY2024, representing 12% year-over-year growth. The market analysis contextualises this against the sector average of 8% growth.

> 📎 **Source:** `annual_report_2024.pdf` (Page 7) — *"Total revenue for the fiscal year ended..."*
> 📎 **Source:** `market_analysis.pdf` (Page 3) — *"Sector average growth rate for comparable firms..."*

**User:** Which document goes into more detail on this?

**Agent:** The annual report provides more granular breakdown — it includes revenue by segment...

---

## Architecture

```
User Question
      ↓
HuggingFace all-MiniLM-L6-v2 embeds the question → 384-dim vector
      ↓
FAISS similarity_search(k=4) finds the 4 most relevant chunks
      ↓
Retrieved chunks (with source metadata) → prompt context
      ↓
Groq Llama 3.3 70B streams the answer
      ↓
Source attribution block appended (filename + page + quote)
      ↓
Streamlit st.chat_message renders it all
```

### The RAG Pipeline — How Source Attribution Works

The key insight is **metadata propagation**:

1. When a PDF is processed, every text chunk is wrapped in a LangChain `Document` object with `metadata = {"source": "filename.pdf", "page": 3}`
2. FAISS stores both the vector AND the metadata together
3. When retrieval happens, the returned documents carry their metadata
4. `app.py` reads `doc.metadata["source"]` to build the citation

The source filename is never lost — it travels from upload through chunking through indexing through retrieval all the way to the UI.

### Two-Phase Processing

**Upload time (slow, done once):**
- PDF bytes → page text via PyMuPDF
- Page text → 1000-char overlapping chunks via `RecursiveCharacterTextSplitter`
- Chunks → 384-dim embedding vectors via `all-MiniLM-L6-v2`
- Vectors → FAISS index stored in `st.session_state`

**Query time (fast, per question):**
- Question → embedding vector (~10ms)
- FAISS nearest-neighbour search (~5ms)
- Build prompt with retrieved context + chat history
- Groq streams response (~1-3 seconds)

---

## Tech Stack

| Component        | Tool                                           |
|------------------|------------------------------------------------|
| Vector Store     | FAISS (Facebook AI Similarity Search)          |
| Embeddings       | HuggingFace all-MiniLM-L6-v2                  |
| LLM              | Llama 3.3 70B via Groq API (free tier)         |
| PDF Processing   | PyMuPDF (fitz)                                 |
| Framework        | LangChain (Document, TextSplitter)             |
| UI               | Streamlit                                      |
| Deployment       | Streamlit Cloud                                |

---

## Project Structure

```
multi-doc-intelligence/
├── rag/
│   ├── __init__.py              # Makes rag/ a Python package
│   ├── document_processor.py   # PDF → LangChain Document chunks with metadata
│   └── retriever.py            # FAISS build + Groq streaming query
├── app.py                      # Streamlit UI, session state, chat interface
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Deployment (Streamlit Cloud)

### 1. Push to GitHub

```bash
# Create a new repo on github.com named multi-doc-intelligence
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/mutahir-ahmed-ai/multi-doc-intelligence.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Select repository: `mutahir-ahmed-ai/multi-doc-intelligence`
4. Main file path: `app.py`
5. Click **Advanced settings** → **Secrets**

### 3. Add Secret

```toml
GROQ_API_KEY = "your_groq_key_here"
```

Get your free key at [console.groq.com](https://console.groq.com)

### 4. Deploy

Click **Deploy** — first build takes ~3 minutes (installing sentence-transformers downloads the model).

---

## Run Locally

```bash
git clone https://github.com/mutahir-ahmed-ai/multi-doc-intelligence
cd multi-doc-intelligence
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_key_here"
```

```bash
streamlit run app.py
```

---

## Key Technical Decisions

**Why FAISS instead of a hosted vector DB (Pinecone, Weaviate)?**
FAISS runs entirely in-process with no external service, no API key, and no latency overhead. For up to 5 PDFs the index fits comfortably in Streamlit Cloud's 1 GB RAM. A hosted DB would be overkill and add a dependency.

**Why all-MiniLM-L6-v2 instead of OpenAI Ada?**
It's free, runs on CPU, and performs comparably to Ada for retrieval tasks on English documents. OpenAI Ada would cost money and add a second API key to manage.

**Why `@st.cache_resource` on the embedding model?**
Without caching, the ~80 MB model would reload on every user interaction (Streamlit reruns the full script on every event). Caching loads it once and keeps it in memory for the session — critical for usability.

**Why k=4 chunks for retrieval?**
Four chunks (~4000 characters) gives the LLM enough context to synthesise a good answer without exceeding Groq's tokens-per-minute limit on the free tier. Increasing to k=8 risks rate limit errors on complex documents.

**Why `temperature=0.1` for the LLM?**
Document Q&A requires factual, grounded answers. Low temperature means the model sticks closely to the retrieved text rather than generating creative (and potentially hallucinated) content.

---

## Related Projects

- [AI Research Agent](https://github.com/mutahir-ahmed-ai/Research-agent) — autonomous web research + PDF report generation
- [HR Assistant Chatbot](https://github.com/mutahir-ahmed-ai/hr-assistant-chatbot) — RAG on PDF documents
- [YouTube Q&A Bot](https://github.com/mutahir-ahmed-ai/youtube-QA-bot) — dynamic RAG on video transcripts

---

## Author

**Mutahir Ahmed** — AI Developer | NLP & RAG Systems
[LinkedIn](https://www.linkedin.com/in/mutahir-ahmed-8229341b5/) · [GitHub](https://github.com/mutahir-ahmed-ai)
