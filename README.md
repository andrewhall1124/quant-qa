# Quantitative Finance RAG QA System

A Retrieval-Augmented Generation (RAG) system for answering questions about quantitative finance research papers using Hugging Face tools and Apple Silicon MPS acceleration.

## Features

- **PDF Ingestion**: Extract and chunk text from academic papers using PyMuPDF
- **Semantic Search**: BAAI/bge-base-en-v1.5 embeddings with FAISS vector search
- **Answer Generation**: HuggingFaceH4/zephyr-7b-beta with 4-bit quantization
- **MPS Acceleration**: Optimized for Apple Silicon Macs
- **Source Citations**: Automatic citation of paper sources with page numbers

## Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) - Fast Python package installer and resolver

Install uv if you haven't already:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd quant-qa
```

2. Install dependencies using uv:
```bash
uv sync
```

This will create a virtual environment and install all required dependencies specified in [pyproject.toml](pyproject.toml).

3. Create the data directory and add your PDF files:
```bash
mkdir -p data
# Place your quantitative finance PDFs in the data/ directory
```

## Usage

The application is run as a Python module using the `app` package. All commands use the `uv run` prefix to execute within the virtual environment.

### Available Commands

#### 1. Build Index
Build the vector index from all PDFs in the `data/` directory:
```bash
uv run python app build
```

This command:
- Scans the `data/` directory for PDF files
- Extracts and chunks text (~400 tokens per chunk with 50 token overlap)
- Generates embeddings using BGE-base-en-v1.5
- Creates a FAISS vector index
- Saves the index to the `index/` directory

#### 2. Ask a Question
Query the system with a single question:
```bash
uv run python app ask "Why does the momentum trading strategy make money?"
```

This command:
- Loads the pre-built index and models
- Retrieves the top 5 most relevant document chunks
- Generates an answer using the Zephyr-7B-beta model
- Returns the answer with source citations (paper names and page numbers)

## How It Works

The RAG system operates in two phases:

### Phase 1: Indexing (Offline)
1. **PDF Ingestion** ([app/ingest.py](app/ingest.py))
   - Extracts text from PDF files using PyMuPDF
   - Splits text into semantic chunks (~400 tokens each)
   - Maintains metadata (filename, page numbers)

2. **Embedding & Indexing** ([app/embed.py](app/embed.py))
   - Generates dense vector embeddings using BAAI/bge-base-en-v1.5 (768 dimensions)
   - Builds FAISS index with inner product similarity
   - Persists index to disk for reuse

### Phase 2: Question Answering (Online)
1. **Retrieval** ([app/retrieve.py](app/retrieve.py))
   - Embeds the user's question using the same model
   - Performs similarity search in FAISS index
   - Returns top-k most relevant chunks (default: 5)

2. **Generation** ([app/generate.py](app/generate.py))
   - Formats retrieved chunks as context
   - Prompts Zephyr-7B-beta (4-bit quantized) to generate answer
   - Uses MPS acceleration on Apple Silicon or falls back to CPU
   - Adds source citations from retrieved chunks

## Architecture

```
quant-qa/
├── app/                  # Main application package
│   ├── __main__.py      # CLI interface (Click-based)
│   ├── ingest.py        # PDF parsing and chunking
│   ├── embed.py         # Embedding and vector store
│   ├── retrieve.py      # Query and retrieval
│   └── generate.py      # LLM answer generation
├── data/                # Raw PDF files (you create this)
├── index/               # Persisted FAISS vector index
├── pyproject.toml       # Project dependencies and metadata
└── README.md
```

## Technical Details

- **Chunking**: ~400 tokens with 50 token overlap for context preservation
- **Embeddings**: BAAI/bge-base-en-v1.5 (768 dimensions, optimized for retrieval)
- **Vector Search**: FAISS with inner product similarity (fast approximate nearest neighbors)
- **Generation**: HuggingFaceH4/zephyr-7b-beta with 4-bit quantization (reduces memory to ~4GB)
- **Device Support**: Automatic detection - MPS (Apple Silicon) > CUDA (NVIDIA) > CPU

## Configuration

The application uses default configuration values in [app/__main__.py](app/__main__.py):
- `INDEX_PATH = "index"` - Where the FAISS index is stored
- `TOP_K = 5` - Number of chunks to retrieve per query
- `DATA_DIR = "data"` - Where PDF files are read from
