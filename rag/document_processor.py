import fitz  # PyMuPDF — already in your stack from Project 4
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ──────────────────────────────────────────────────────────────────────────────
# WHY THIS FILE EXISTS
#
# Before text can go into FAISS, it needs to be:
#   1. Extracted from the PDF binary
#   2. Broken into manageable chunks (the LLM has a context window limit)
#   3. Tagged with its source filename — THIS is what makes attribution possible
#
# The metadata dict attached to every Document object travels with the chunk
# through all processing steps and ends up in FAISS. When we retrieve a chunk,
# we also get its metadata back — that's how we know "this came from report.pdf, page 3".
# ──────────────────────────────────────────────────────────────────────────────


def process_uploaded_pdfs(uploaded_files: list) -> list[Document]:
    """
    Reads Streamlit-uploaded PDF files and returns a list of LangChain
    Document objects, each chunk tagged with its source filename and page number.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects (BytesIO wrappers)

    Returns:
        List of LangChain Document objects ready for embedding
    """

    # ── Text Splitter Configuration ──────────────────────────────────────────
    #
    # Why do we need to split at all?
    # A 50-page PDF might be 200,000 characters. The LLM's context window can't
    # hold all of that, and neither can one embedding vector represent it meaningfully.
    # We split into overlapping chunks so each piece is small enough to embed well.
    #
    # chunk_size=1000: each chunk is roughly 1000 characters (~200 words).
    #   This is a sweet spot — large enough to contain a complete idea,
    #   small enough for the embedding to capture it precisely.
    #
    # chunk_overlap=200: the last 200 chars of chunk N also appear at the start
    #   of chunk N+1. This prevents answers from being cut in half at a boundary.
    #   Example: if a sentence spans the end of one chunk, it's captured in both.
    #
    # separators: the splitter tries these in order — it prefers to split on
    #   paragraph breaks first (\n\n), then lines (\n), then sentences (.), etc.
    #   This keeps sentences intact wherever possible.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_documents = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name

        # Streamlit gives us a BytesIO-like object (the raw PDF binary).
        # PyMuPDF's fitz.open() can read directly from bytes — no need to
        # save to disk first. This works perfectly on Streamlit Cloud where
        # we don't have write access to arbitrary paths.
        pdf_bytes = uploaded_file.read()
        try:
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            # If a file is corrupted or not a real PDF, skip it gracefully.
            print(f"Warning: Could not open {filename}: {e}")
            continue

        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]

            # get_text() extracts all readable text from the page.
            # Note: scanned PDFs (image-only) will return an empty string here
            # because there's no embedded text layer. We skip those pages.
            text = page.get_text()

            if not text.strip():
                continue  # Skip blank or image-only pages

            # ── The Key Step: Attach Metadata ──────────────────────────────
            #
            # We wrap the text in a LangChain Document object with a metadata dict.
            # This metadata dict is what makes source attribution possible later.
            #
            # metadata["source"] = filename
            #   → When we retrieve this chunk, we know which file it came from.
            #     FAISS stores this alongside the vector — retrieval returns both.
            #
            # metadata["page"] = page_num + 1
            #   → 1-indexed page number for human-readable citations.
            #
            # metadata["total_pages"] = total page count
            #   → Useful context for the user ("Page 3 of 47").
            page_document = Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": page_num + 1,
                    "total_pages": len(pdf_doc)
                }
            )
            all_documents.append(page_document)

        pdf_doc.close()

    if not all_documents:
        raise ValueError(
            "No readable text found in the uploaded PDFs. "
            "This tool requires PDFs with a text layer — scanned image PDFs are not supported."
        )

    # ── Split into chunks ─────────────────────────────────────────────────────
    #
    # split_documents() splits each Document's page_content into multiple smaller
    # chunks, and COPIES the original metadata to every resulting chunk.
    #
    # So if page 5 of "report.pdf" gets split into 4 chunks, all 4 chunks have:
    #   metadata = {"source": "report.pdf", "page": 5, "total_pages": 20}
    #
    # This is the inheritance of metadata that makes attribution work end-to-end.
    chunks = text_splitter.split_documents(all_documents)

    return chunks
