# Quantitative Finance RAG QA System

A Retrieval-Augmented Generation (RAG) system for answering questions about quantitative finance research papers using Hugging Face tools and Apple Silicon MPS acceleration.

## Features

- **PDF Ingestion**: Extract and chunk text from academic papers using PyMuPDF
- **Semantic Search**: BAAI/bge-base-en-v1.5 embeddings with FAISS vector search
- **Answer Generation**: HuggingFaceH4/zephyr-7b-beta with 4-bit quantization
- **MPS Acceleration**: Optimized for Apple Silicon Macs
- **Source Citations**: Automatic citation of paper sources with page numbers

## Installation

1. Install dependencies:
```bash
uv sync
```

2. Place PDF files in the `data/papers/` directory

## Usage

### Build Index
```bash
cd src
python pipeline.py build ../data/papers/
```

### Ask Single Question
```bash
python pipeline.py ask "What are the main risk factors in portfolio optimization?"
```

### Interactive Mode
```bash
python pipeline.py interactive
```

### Command Options
- `--index-path`: Custom index storage location (default: `index`)
- `--top-k`: Number of chunks to retrieve (default: 5)
- `--simple-generator`: Use fallback generator if main model fails

## Architecture

```
quant-qa/
├── data/papers/          # Raw PDFs
├── src/
│   ├── ingest.py        # PDF parsing and chunking
│   ├── embed.py         # Embedding and vector store
│   ├── retrieve.py      # Query and retrieval
│   ├── generate.py      # LLM answer generation
│   └── pipeline.py      # End-to-end interface
├── index/               # Persisted vector index
└── requirements.txt
```

## Technical Details

- **Chunking**: ~400 tokens with 50 token overlap
- **Embeddings**: BGE-base-en-v1.5 (768 dimensions)
- **Vector Search**: FAISS with inner product similarity
- **Generation**: Zephyr-7B-beta with 4-bit quantization
- **Device Support**: MPS (Apple Silicon), CUDA, CPU fallback

## Example

```bash
# Build index from papers
python pipeline.py build ../data/papers/

# Ask a question
python pipeline.py ask "How does volatility clustering affect option pricing?"
```

The system will retrieve relevant paper segments and generate answers with source citations.