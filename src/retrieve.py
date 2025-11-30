from typing import List, Dict, Any
from embed import EmbeddingStore


class Retriever:
    def __init__(
        self, index_path: str = "index", model_name: str = "BAAI/bge-base-en-v1.5"
    ):
        self.embedding_store = EmbeddingStore(
            model_name=model_name, index_path=index_path
        )
        self.is_loaded = False

    def load(self) -> bool:
        """Load the embedding store and index."""
        if self.embedding_store.load_index():
            self.is_loaded = True
            return True
        else:
            print("Failed to load index")
            return False

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant chunks for a query."""
        if not self.is_loaded:
            raise RuntimeError("Retriever not loaded. Call load() first.")

        results = self.embedding_store.search(query, k=top_k)
        return results

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved results into a context string for generation."""
        if not results:
            return "No relevant context found."

        context_parts = []
        for i, result in enumerate(results, 1):
            metadata = result["metadata"]
            source = f"{metadata['filename']} (page {metadata['page_number']})"
            text = result["chunk"].text

            context_part = f"[Source {i}: {source}]\n{text}"
            context_parts.append(context_part)

        return "\n\n".join(context_parts)

    def get_sources(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract source citations from results."""
        sources = []
        for result in results:
            metadata = result["metadata"]
            source = f"{metadata['filename']} (page {metadata['page_number']})"
            if source not in sources:
                sources.append(source)
        return sources
