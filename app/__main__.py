import click
import sys
import os
from typing import Optional

from ingest import PDFIngester
from embed import EmbeddingStore
from retrieve import Retriever
from generate import Generator

# Configuration constants
INDEX_PATH = "index"
TOP_K = 5
DATA_DIR = "data"


class App:
    def __init__(self, index_path: str = "index"):
        self.index_path = index_path
        self.retriever = None
        self.generator = None

    def build_index(self) -> bool:
        """Build the vector index from PDFs in the specified directory."""
        if not os.path.exists(DATA_DIR):
            print(f"Error: Directory {DATA_DIR} does not exist")
            return False

        print("Step 1/3: Ingesting PDFs...")
        ingester = PDFIngester()
        chunks = ingester.ingest_directory(DATA_DIR)

        if not chunks:
            print("No chunks created from PDFs")
            return False

        print("Step 2/3: Building embeddings and index...")
        store = EmbeddingStore(index_path=self.index_path)
        store.build_index(chunks)

        print("Step 3/3: Saving index...")
        store.save_index()

        print(f"✓ Index built successfully with {len(chunks)} chunks")
        return True

    def load_models(self) -> bool:
        """Load the retrieval and generation models."""
        print("Loading models...")

        # Load retriever
        self.retriever = Retriever(index_path=self.index_path)
        if not self.retriever.load():
            print("Error: Could not load retrieval index")
            return False

        # Load generator
        try:
            self.generator = Generator()
            self.generator.load_model()
            print("✓ Models loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading generator: {e}")
            return False

    def answer_question(self, question: str, top_k: int = 5) -> Optional[str]:
        """Answer a question using the RAG pipeline."""
        if not self.retriever or not self.generator:
            print("Models not loaded. Call load_models() first.")
            return None

        print(f"Retrieving relevant context for: {question}")

        # Retrieve relevant chunks
        results = self.retriever.retrieve(question, top_k=top_k)

        if not results:
            return "I couldn't find any relevant information in the indexed papers for your question."

        # Format context
        context = self.retriever.format_context(results)

        print("Generating answer...")

        # Generate answer
        answer = self.generator.generate_answer(context, question)

        # Add source citations
        sources = self.retriever.get_sources(results)
        if sources:
            citation_text = "\n\nSources:\n" + "\n".join(
                f"• {source}" for source in sources
            )
            answer += citation_text

        return answer


@click.group()
def cli():
    """Quantitative Finance RAG QA System"""
    pass


@cli.command()
def build():
    """Build index from PDFs in the specified directory."""
    app = App(index_path=INDEX_PATH)
    success = app.build_index()
    if not success:
        sys.exit(1)


@cli.command()
@click.argument("question")
def ask(question):
    """Ask a question using the RAG system."""
    app = App(index_path=INDEX_PATH)

    if not app.load_models():
        sys.exit(1)

    answer = app.answer_question(question, top_k=TOP_K)
    if answer:
        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print("=" * 60)
        print(answer)
    else:
        sys.exit(1)


def main():
    cli()


if __name__ == "__main__":
    main()
