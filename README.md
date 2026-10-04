# ELECTRICAL-RAG
Electrical RAG is a Retrieval-Augmented Generation system that answers electrical engineering questions from uploaded PDF documents. It extracts text with PyMuPDF, splits it into overlapping chunks, converts them into embeddings with the BAAI/bge-small-en-v1.5 model, and retrieves the most relevant passages using cosine similarity.
