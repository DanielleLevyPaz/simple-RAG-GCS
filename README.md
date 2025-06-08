# 🧙‍♂️ Harry Potter RAG Application

This is a Retrieval-Augmented Generation (RAG) app built in Python that allows you to ask questions about the **Harry Potter book (PDF)** stored in Google Cloud Storage. It uses Google's **Gemini Pro LLM** for answering questions based on vector-retrieved chunks from the text.

---

## 📦 Features

- ✅ Downloads a PDF from Google Cloud Storage (GCS)
- ✅ Extracts and chunks the PDF text
- ✅ Embeds the chunks using Gemini Embedding API
- ✅ Stores embeddings in an in-memory FAISS vector store
- ✅ Uses Gemini Pro LLM to answer user questions based on relevant chunks
- ✅ Simple interactive CLI interface