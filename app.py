"""Electrical RAG - Hugging Face Space version (Gradio + Groq)."""
import os
import glob

import gradio as gr
import numpy as np
import pymupdf
from groq import Groq
from sentence_transformers import SentenceTransformer

# ---------- Setup ----------
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

api_key = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=api_key) if api_key else None

PREFERRED = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b",
             "llama-3.1-8b-instant", "qwen/qwen3-32b"]
MODEL = None
if groq_client:
    try:
        available = [m.id for m in groq_client.models.list().data]
        MODEL = next((m for m in PREFERRED if m in available), None)
        if MODEL is None:
            MODEL = next(m for m in available
                         if not any(x in m for x in ("whisper", "guard", "tts", "orpheus")))
    except Exception as e:
        print("Could not list Groq models:", e)

SYSTEM = (
    "You are an electrical engineering study assistant. Answer using ONLY the numbered "
    "context passages provided. Cite sources like [1] or [2] after the claims they support. "
    "Keep formulas, values and units exactly as written in the context; never invent numbers. "
    "If the context doesn't contain the answer, say you couldn't find it in the uploaded documents. "
    "Explain step by step for a student. For anything safety-critical (wiring, high voltage, "
    "code compliance), remind the user to follow the local electrical code and consult a "
    "licensed electrician."
)

chunks = []
vecs = np.zeros((0, 384), dtype="float32")


# ---------- Backend ----------
def index_paths(paths):
    """Read PDFs, split into chunks, embed, and replace the current index."""
    global chunks, vecs
    new_chunks = []
    for path in paths:
        name = os.path.basename(path)
        for page_no, page in enumerate(pymupdf.open(path), 1):
            words = page.get_text().split()
            if len(words) < 30:
                continue
            for start in range(0, len(words), 300):
                new_chunks.append({"source": name, "page": page_no,
                                   "text": " ".join(words[start:start + 350])})
                if start + 350 >= len(words):
                    break
    if not new_chunks:
        return "No readable text found. Scanned PDFs need OCR first."
    vecs = embedder.encode([c["text"] for c in new_chunks], batch_size=64,
                           normalize_embeddings=True)
    chunks = new_chunks
    return f"Done! {len(chunks)} chunks indexed. You can ask questions now."


def add_pdfs(files):
    if not files:
        return "Please upload at least one PDF first."
    paths = [f if isinstance(f, str) else f.name for f in files]
    return index_paths(paths)


def answer(question):
    if groq_client is None or MODEL is None:
        return "The server has no GROQ_API_KEY set. Add it in Space settings, then restart.", ""
    if not chunks:
        return "Upload and index a PDF first.", ""
    if not question.strip():
        return "Type a question first.", ""
    q = embedder.encode(["Represent this sentence for searching relevant passages: " + question],
                        normalize_embeddings=True)[0]
    top = np.argsort(-(vecs @ q))[:5]
    hits = [chunks[i] for i in top]
    context = "\n\n".join(f"[{n}] ({h['source']}, page {h['page']})\n{h['text']}"
                          for n, h in enumerate(hits, 1))
    try:
        reply = groq_client.chat.completions.create(
            model=MODEL, max_tokens=1500,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}])
    except Exception as e:
        return f"The AI service returned an error: {e}", ""
    sources = "\n\n".join(f"**[{n}] {h['source']}, page {h['page']}**: {h['text'][:250]}..."
                          for n, h in enumerate(hits, 1))
    return reply.choices[0].message.content, sources


# Optional: auto-index any PDFs placed in a "docs" folder (only use books you may share)
startup_pdfs = sorted(glob.glob("docs/*.pdf"))
startup_status = index_paths(startup_pdfs) if startup_pdfs else ""

# ---------- Frontend ----------
with gr.Blocks(title="Electrical RAG") as demo:
    gr.Markdown("# ⚡ Electrical RAG\nUpload your PDFs, click **Index documents**, then ask questions.")
    files = gr.File(label="Upload PDFs", file_count="multiple", file_types=[".pdf"])
    index_btn = gr.Button("Index documents")
    status = gr.Textbox(label="Status", interactive=False, value=startup_status)
    question = gr.Textbox(label="Your question", placeholder="e.g. State Kirchhoff's voltage law")
    ask_btn = gr.Button("Ask", variant="primary")
    answer_box = gr.Markdown()
    with gr.Accordion("Sources", open=False):
        sources_box = gr.Markdown()

    index_btn.click(add_pdfs, files, status)
    ask_btn.click(answer, question, [answer_box, sources_box])
    question.submit(answer, question, [answer_box, sources_box])

demo.launch()
