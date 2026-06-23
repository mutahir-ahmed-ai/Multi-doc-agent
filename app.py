import streamlit as st
from rag.document_processor import process_uploaded_pdfs
from rag.retriever import get_embeddings, build_vector_store, query_documents

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# Same pattern as your Research Agent — "centered" keeps content readable.
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Document Intelligence",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Multi-Document Intelligence Agent")
st.markdown(
    "Upload multiple PDFs and chat with all of them at once. "
    "Every answer shows **which document** it came from."
)

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
#
# Streamlit re-runs the entire script from top to bottom on every user action
# (a button click, a keystroke). st.session_state is a dictionary that survives
# those reruns — it's how you build stateful apps in Streamlit.
#
# We initialize each key only if it doesn't already exist (the "not in" check).
# Without this guard, every rerun would wipe out the stored data.
# ──────────────────────────────────────────────────────────────────────────────

if "vector_store" not in st.session_state:
    # Holds the FAISS index after documents are processed.
    # Built once, reused for every query — expensive to rebuild.
    st.session_state.vector_store = None

if "loaded_docs" not in st.session_state:
    # List of PDF filenames currently in the index.
    # Used to display "Loaded Documents" in the sidebar.
    st.session_state.loaded_docs = []

if "messages" not in st.session_state:
    # Chat history for the UI — list of dicts: {"role": "user"/"assistant", "content": "..."}
    # Streamlit's st.chat_message() reads from this list to render bubbles.
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    # Chat history in LangChain format — list of (human_msg, ai_msg) tuples.
    # Passed to the retriever so the LLM understands follow-up questions.
    # Example: [("What is RAG?", "RAG stands for..."), ("How does it work?", "It works by...")]
    st.session_state.chat_history = []

if "embeddings" not in st.session_state:
    # Cached embedding model — loaded once via @st.cache_resource, stored here
    # so we don't need to re-load it every time the app reruns.
    st.session_state.embeddings = None

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Document Management
#
# The sidebar is separate from the main content area. It's always visible.
# We put the file uploader here so the chat area stays clean.
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Document Manager")

    uploaded_files = st.file_uploader(
        "Upload PDFs (max 5)",
        type="pdf",
        accept_multiple_files=True,   # Allows selecting multiple files at once
        help="Each PDF is tagged with its filename so every answer can cite its source."
    )

    # Enforce the 5-PDF limit to stay within Streamlit Cloud's 1 GB RAM ceiling.
    # sentence-transformers + FAISS + 5 PDFs comfortably fits; more gets risky.
    if uploaded_files and len(uploaded_files) > 5:
        st.warning("⚠️ Max 5 PDFs. Only the first 5 will be processed.")
        uploaded_files = uploaded_files[:5]

    # The "Process Documents" button triggers the full pipeline:
    # read PDFs → chunk text → build FAISS index
    if uploaded_files:
        if st.button("🔄 Process Documents", type="primary", use_container_width=True):

            # Step 1: Load the HuggingFace embedding model.
            # get_embeddings() is decorated with @st.cache_resource in retriever.py,
            # so the ~80 MB model downloads once and is reused for all subsequent calls.
            with st.spinner("Loading embedding model (first run downloads ~80 MB)..."):
                embeddings = get_embeddings()
                st.session_state.embeddings = embeddings

            # Step 2: Read each PDF and split into chunks with filename metadata.
            with st.spinner(f"Reading {len(uploaded_files)} PDF(s)..."):
                try:
                    chunks = process_uploaded_pdfs(uploaded_files)
                except Exception as e:
                    st.error(f"Error reading PDFs: {e}")
                    st.stop()

            # Step 3: Build the FAISS vector index from the chunks.
            # This converts every chunk to a 384-dimensional embedding vector
            # and stores them in a FAISS index for fast similarity search.
            with st.spinner("Building search index..."):
                try:
                    st.session_state.vector_store = build_vector_store(
                        chunks, st.session_state.embeddings
                    )
                except Exception as e:
                    st.error(f"Error building index: {e}")
                    st.stop()

            # Save the filenames and reset chat for the new document set.
            st.session_state.loaded_docs = [f.name for f in uploaded_files]
            st.session_state.messages = []
            st.session_state.chat_history = []

            st.success(
                f"✅ {len(chunks)} chunks indexed from "
                f"{len(uploaded_files)} document(s). Start chatting!"
            )

    # ── Loaded Documents Display ────────────────────────────────────────────
    # Show which documents are currently in the FAISS index.
    # This is important UX — the user needs to know what the agent can see.
    if st.session_state.loaded_docs:
        st.divider()
        st.markdown("**📋 Loaded Documents:**")
        for i, name in enumerate(st.session_state.loaded_docs, 1):
            st.markdown(f"{i}. 📄 `{name}`")

        st.divider()

        # The clear button wipes everything: index, filenames, and chat history.
        # st.rerun() forces an immediate full refresh so the UI updates instantly.
        if st.button("🗑️ Clear All & Reset", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.loaded_docs = []
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.embeddings = None
            st.rerun()
    else:
        st.info("No documents loaded. Upload PDFs above to begin.")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN CHAT AREA
#
# Two states:
#   1. No documents loaded → show instructions
#   2. Documents loaded → show full chat interface
# ──────────────────────────────────────────────────────────────────────────────

if not st.session_state.loaded_docs:
    # Onboarding instructions shown before any documents are uploaded.
    st.markdown("""
    ### 👈 How to use this app

    1. **Upload up to 5 PDFs** in the sidebar
    2. Click **Process Documents** — the agent reads and indexes everything
    3. **Ask questions** in the chat below
    4. Every answer ends with the **source document and a direct quote**

    ---
    **Example questions you can ask:**
    - *"Summarise the key findings from all documents"*
    - *"What does [document A] say about X?"*
    - *"Compare how the two reports describe Y"*
    - *"Find anything related to Z across all uploads"*
    """)

else:
    # ── Render Chat History ─────────────────────────────────────────────────
    # Loop through st.session_state.messages and render each bubble.
    # st.chat_message("user") creates a human bubble on the right.
    # st.chat_message("assistant") creates an AI bubble on the left.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ── Chat Input ──────────────────────────────────────────────────────────
    # st.chat_input() renders a sticky input bar at the bottom of the page.
    # It returns the submitted text (or None if nothing was typed yet).
    # The walrus operator (:=) assigns and checks in one line — a Streamlit idiom.
    if user_query := st.chat_input("Ask anything about your documents..."):

        # 1. Show the user's message in the chat immediately.
        with st.chat_message("user"):
            st.markdown(user_query)

        # 2. Save it to session_state so it persists on the next rerun.
        st.session_state.messages.append({"role": "user", "content": user_query})

        # 3. Generate and stream the assistant's response.
        with st.chat_message("assistant"):

            # message_placeholder is an empty container we'll write into
            # during streaming — updating it creates the word-by-word effect.
            message_placeholder = st.empty()

            try:
                # query_documents returns:
                #   answer_stream — a generator that yields text chunks from Groq
                #   source_docs   — LangChain Document objects (chunk + metadata)
                answer_stream, source_docs = query_documents(
                    query=user_query,
                    vector_store=st.session_state.vector_store,
                    chat_history=st.session_state.chat_history
                )

                # ── Stream the answer word by word ──────────────────────────
                # Each iteration adds a new token from Groq to the accumulated text.
                # We append "▌" (a block cursor) to signal that more is coming.
                # When the loop ends, the cursor is removed.
                streamed_answer = ""
                for token in answer_stream:
                    streamed_answer += token
                    message_placeholder.markdown(streamed_answer + "▌")

                # ── Build Source Attribution Block ───────────────────────────
                # source_docs is a list of Document objects.
                # doc.metadata["source"] = the original PDF filename
                # doc.metadata["page"]   = page number within that PDF
                # doc.page_content       = the raw text chunk
                #
                # We deduplicate by filename — if two chunks came from the same
                # document, we only cite that document once.
                source_block = "\n\n---\n"
                seen = set()

                for doc in source_docs:
                    filename = doc.metadata.get("source", "Unknown document")
                    page = doc.metadata.get("page", "?")

                    if filename not in seen:
                        seen.add(filename)
                        # Truncate the chunk to 120 chars for the inline quote.
                        # Replace newlines with spaces so it reads inline cleanly.
                        snippet = (
                            doc.page_content[:120]
                            .replace("\n", " ")
                            .strip()
                        )
                        source_block += (
                            f"\n📎 **Source:** `{filename}` (Page {page}) "
                            f"— *\"{snippet}...\"*"
                        )

                # Final response = full answer + source citations
                final_response = streamed_answer + source_block

                # Replace the streaming placeholder with the final complete text.
                message_placeholder.markdown(final_response)

                # 4. Save to session state.
                st.session_state.messages.append(
                    {"role": "assistant", "content": final_response}
                )

                # Update LangChain chat history with just the answer text
                # (no source block) — the LLM doesn't need to see its own citations.
                st.session_state.chat_history.append(
                    (user_query, streamed_answer)
                )

            except Exception as e:
                message_placeholder.markdown(f"❌ Error: {str(e)}")
                st.info(
                    "If you're seeing a rate limit error, wait a few seconds and try again. "
                    "Groq's free tier has generous limits but brief cooldown periods."
                )
