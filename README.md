# ⚡ Electrical RAG

A Retrieval-Augmented Generation (RAG) system that answers electrical engineering questions from your own PDF documents, with page-number citations.

Upload a textbook or notes, ask a question, and get an answer grounded in the document, along with the exact passages it came from.

## Features

- Upload one or more PDFs through a simple web interface
- Semantic search over the document (not just keyword matching)
- Answers generated only from the retrieved passages, with citations like [1], [2]
- Expandable **Sources** section showing the file, page number, and matching text
- Says so when the answer is not in the uploaded documents

## How it works

1. **Extract:** text is read from each PDF page with PyMuPDF.
2. **Chunk:** pages are split into overlapping chunks of about 350 words.
3. **Embed:** each chunk is converted to a vector with the `BAAI/bge-small-en-v1.5` sentence-transformers model.
4. **Retrieve:** the question is embedded, and the 5 most similar chunks are found with cosine similarity (NumPy).
5. **Generate:** the chunks and the question are sent to an LLM on Groq, which is instructed to answer only from the context and cite sources.
6. **Display:** the answer and sources are shown in a Gradio web page.

## Tech stack

- Python
- PyMuPDF (PDF text extraction)
- sentence-transformers (embeddings)
- NumPy (vector search)
- Groq API (LLM)
- Gradio (frontend)
- Google Colab (development and demo)

## Files

| File | Purpose |
|------|---------|
| `Electrical_RAG.ipynb` | Colab notebook with the full working project |
| `app.py` | Standalone version of the app (backend + Gradio frontend) |
| `requirements.txt` | Python dependencies |

## Run it in Google Colab

1. Open the notebook in Google Colab (optional: Runtime, then Change runtime type, then T4 GPU).
2. Run the cells in order: install, API key and model setup, backend, web page.
3. Paste your own free Groq API key when asked (get one at console.groq.com).
4. Open the Gradio link, upload a PDF, click **Index documents**, and ask questions.

## Run it locally

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here      # Windows: set GROQ_API_KEY=your_key_here
python app.py
```

Then open the local URL printed in the terminal.

## Notes and limitations

- Works with text-based PDFs. Scanned PDFs need OCR first.
- Diagrams and circuit figures are images and are not understood; only the surrounding text is used.
- Mathematical formulas may appear as plain text or raw LaTeX.
- The index lives in memory, so PDFs must be re-indexed after a restart.
- API keys are never stored in the code. Provide your own key.
- No copyrighted textbooks are included in this repository. Upload your own documents.
