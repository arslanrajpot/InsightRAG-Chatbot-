# InsightRAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets users upload documents and ask natural-language questions about their content, with grounded, accurate answers instead of generic AI guesses.

## Overview

Users upload one or more documents (PDF), which are parsed and split into chunks, embedded with HuggingFace sentence embeddings, and indexed in a FAISS vector store. When a user asks a question, the system retrieves the most relevant chunks and uses an LLM (via Groq) to generate an answer grounded in the uploaded content, along with a confidence score so users can judge how reliable each answer is.

## Key Features

Document upload and parsing directly from PDF files. Semantic search over document content using FAISS and HuggingFace embeddings. LLM-generated answers grounded in the retrieved document chunks to minimize hallucination. Confidence scoring on responses so users know how trustworthy an answer is. A modular FastAPI backend built for easy extension to new document types or retrieval strategies.

## Tech Stack

FastAPI for the backend API. LangChain and LangGraph for the RAG and conversation orchestration. Groq-hosted LLMs for answer generation. HuggingFace Sentence Transformers for embeddings. FAISS as the vector store. PyPDF for PDF text extraction.

## Project Structure

```
backend/
└── app/
    main.py       FastAPI app entrypoint
    config.py     app configuration
    models/       request/response and data models
    routers/      API route definitions
    services/     RAG pipeline and document processing logic
    utils/        PDF parsing utilities
```

## Getting Started

Install dependencies and set your API key, then run the server.

```bash
cd backend
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env
uvicorn app.main:app --reload
```

Once running, upload a document through the API (or connected frontend) and start asking questions about its content.

## Roadmap

Planned improvements include support for additional file types (Word, plain text), multi-document conversations, and persistent chat history per document.

## Author

Built by **Arslan Arshad**, Full-Stack & AI Engineer.
Portfolio: https://arslan-arshad.netlify.app/ · Email: arslanarshad1018@gmail.com
