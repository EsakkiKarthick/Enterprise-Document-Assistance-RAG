# Enterprise Document Assistance

A simple Retrieval-Augmented Generation (RAG) project for enterprise document Q&A.
It ingests PDF documents, chunks text into semantic units, stores them in a Chroma vector database, and answers queries using a local LLM service.

## Repository Structure

- `chunking.py` - text chunking utilities for splitting documents into statement-aware chunks.
- `RAG_Pipeline.py` - vector store manager using Chroma and sentence-transformer embeddings.
- `RAG_Agent.py` - retrieval pipeline that loads nearest chunks, builds context, and queries an Ollama-based LLM.
- `requirements.txt` - Python dependencies for the project.
- `Input/` - directory intended for PDF documents and other source files to ingest.

## Key Features

- PDF ingestion with `PyMuPDF` (`fitz`) for extracting text.
- Chunking with sentence/statement-aware logic.
- Persistent vector store using `chromadb`.
- Semantic search using sentence-transformer embeddings.
- LLM generation through a local Ollama HTTP endpoint.

## Requirements

- Python 3.11+ (recommended)
- A Python virtual environment
- Local Ollama service available at `http://localhost:11434`

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Ensure Ollama is running locally.

## Usage

### Prepare documents

Place PDF files in the `Input/` folder or any location accessible to your scripts.

### Ingest documents into the vector store

Open `RAG_Pipeline.py` and update the example `filePath` / `fileName` values in the `__main__` block if needed.

Run:

```powershell
python RAG_Pipeline.py
```

This creates a persistent Chroma collection under your home directory in `Vector_Store` and loads chunked document text.

### Query the RAG agent

Run:

```powershell
python RAG_Agent.py
```

This script uses the configured collection name and sends a query through `retrieve_node`, `context_data`, and `generate_data`.

## Configuration

- `VECTOR_STORE_PATH` in `RAG_Pipeline.py` is set to `~/Vector_Store` by default.
- `RAG_Agent.py` expects an Ollama endpoint at `http://localhost:11434/api/generate` and the model `llama3`.
- Sentence embedding model is `all-MiniLM-L6-v2` by default.

## Notes

- The pipeline currently relies on a local Ollama service for generation.
- The `RAG_Agent.py` output prints retrieved chunks and final answers, including source metadata.
- Chunking uses a token-count approximation via word count and sentence segmentation.

## Troubleshooting

- If vector store collections are empty, confirm documents are loaded successfully in `RAG_Pipeline.py`.
- If the LLM request fails, verify Ollama is running and reachable.
- If you see encoding or PDF extraction issues, ensure your PDFs are readable and supported.

## License

This repository does not include a license file. Add one if you want to clarify reuse terms.
