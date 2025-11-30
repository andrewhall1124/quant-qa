import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from ingest import DocumentChunk
import torch

class EmbeddingStore:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", index_path: str = "index"):
        self.model_name = model_name
        self.index_path = index_path
        self.device = self._get_device()

        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

        self.index = None
        self.chunks = []
        self.chunk_metadata = []

    def _get_device(self):
        """Get the best available device, preferring MPS for Apple Silicon."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def embed_chunks(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Generate embeddings for document chunks."""
        print(f"Generating embeddings for {len(chunks)} chunks...")

        texts = [chunk.text for chunk in chunks]

        # Generate embeddings in batches to manage memory
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings

    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """Build FAISS index from document chunks."""
        print("Building FAISS index...")

        embeddings = self.embed_chunks(chunks)
        dimension = embeddings.shape[1]

        # Use Inner Product index (which works well with normalized embeddings)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))

        self.chunks = chunks
        self.chunk_metadata = [chunk.metadata for chunk in chunks]

        print(f"Index built with {self.index.ntotal} vectors")

    def save_index(self) -> None:
        """Save index and metadata to disk."""
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)

        # Save FAISS index
        index_file = os.path.join(self.index_path, "faiss_index.bin")
        faiss.write_index(self.index, index_file)

        # Save chunks and metadata
        chunks_file = os.path.join(self.index_path, "chunks.pkl")
        with open(chunks_file, 'wb') as f:
            pickle.dump((self.chunks, self.chunk_metadata), f)

        # Save model name for consistency
        config_file = os.path.join(self.index_path, "config.pkl")
        with open(config_file, 'wb') as f:
            pickle.dump({
                'model_name': self.model_name,
                'dimension': self.index.d if self.index else None
            }, f)

        print(f"Index saved to {self.index_path}")

    def load_index(self) -> bool:
        """Load index and metadata from disk."""
        index_file = os.path.join(self.index_path, "faiss_index.bin")
        chunks_file = os.path.join(self.index_path, "chunks.pkl")
        config_file = os.path.join(self.index_path, "config.pkl")

        if not all(os.path.exists(f) for f in [index_file, chunks_file, config_file]):
            print("Index files not found")
            return False

        try:
            # Load configuration
            with open(config_file, 'rb') as f:
                config = pickle.load(f)

            # Verify model consistency
            if config['model_name'] != self.model_name:
                print(f"Warning: Loaded index was built with {config['model_name']}, "
                      f"but current model is {self.model_name}")

            # Load FAISS index
            self.index = faiss.read_index(index_file)

            # Load chunks and metadata
            with open(chunks_file, 'rb') as f:
                self.chunks, self.chunk_metadata = pickle.load(f)

            print(f"Loaded index with {self.index.ntotal} vectors")
            return True

        except Exception as e:
            print(f"Error loading index: {e}")
            return False

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query string."""
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding[0]

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks given a query."""
        if self.index is None:
            raise ValueError("Index not built or loaded")

        query_embedding = self.embed_query(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        # Search the index
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                result = {
                    'chunk': self.chunks[idx],
                    'score': float(scores[0][i]),
                    'metadata': self.chunk_metadata[idx]
                }
                results.append(result)

        return results

if __name__ == "__main__":
    import sys
    from ingest import PDFIngester

    if len(sys.argv) < 2:
        print("Usage: python embed.py <command> [args]")
        print("Commands:")
        print("  build <pdf_directory> - Build index from PDFs")
        print("  search <query> - Search existing index")
        sys.exit(1)

    command = sys.argv[1]
    store = EmbeddingStore()

    if command == "build":
        if len(sys.argv) != 3:
            print("Usage: python embed.py build <pdf_directory>")
            sys.exit(1)

        pdf_directory = sys.argv[2]
        if not os.path.exists(pdf_directory):
            print(f"Directory {pdf_directory} does not exist")
            sys.exit(1)

        # Ingest PDFs
        ingester = PDFIngester()
        chunks = ingester.ingest_directory(pdf_directory)

        if not chunks:
            print("No chunks created")
            sys.exit(1)

        # Build and save index
        store.build_index(chunks)
        store.save_index()

    elif command == "search":
        if len(sys.argv) != 3:
            print("Usage: python embed.py search <query>")
            sys.exit(1)

        query = sys.argv[2]

        # Load index
        if not store.load_index():
            print("Could not load index. Build it first with 'build' command.")
            sys.exit(1)

        # Search
        results = store.search(query)

        print(f"\nTop {len(results)} results for: {query}")
        print("=" * 50)

        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.4f}")
            print(f"   Source: {result['metadata']['filename']} (page {result['metadata']['page_number']})")
            print(f"   Text: {result['chunk'].text[:200]}...")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)