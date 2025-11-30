#!/usr/bin/env python3
import argparse
import sys
import os
from typing import Optional

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingest import PDFIngester
from embed import EmbeddingStore
from retrieve import Retriever
from generate import Generator, SimpleGenerator

class QuantQAPipeline:
    def __init__(self, index_path: str = "index"):
        self.index_path = index_path
        self.retriever = None
        self.generator = None

    def build_index(self, pdf_directory: str) -> bool:
        """Build the vector index from PDFs in the specified directory."""
        if not os.path.exists(pdf_directory):
            print(f"Error: Directory {pdf_directory} does not exist")
            return False

        print("Step 1/3: Ingesting PDFs...")
        ingester = PDFIngester()
        chunks = ingester.ingest_directory(pdf_directory)

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

    def load_models(self, use_simple_generator: bool = False) -> bool:
        """Load the retrieval and generation models."""
        print("Loading models...")

        # Load retriever
        self.retriever = Retriever(index_path=self.index_path)
        if not self.retriever.load():
            print("Error: Could not load retrieval index")
            return False

        # Load generator
        try:
            if use_simple_generator:
                self.generator = SimpleGenerator()
            else:
                self.generator = Generator()
            self.generator.load_model()
            print("✓ Models loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading generator: {e}")
            print("You can try running with --simple-generator flag")
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
            citation_text = "\n\nSources:\n" + "\n".join(f"• {source}" for source in sources)
            answer += citation_text

        return answer

def main():
    parser = argparse.ArgumentParser(description="Quantitative Finance RAG QA System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build index from PDFs")
    build_parser.add_argument("pdf_directory", help="Directory containing PDF files")
    build_parser.add_argument("--index-path", default="index", help="Path to store index (default: index)")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("--index-path", default="index", help="Path to index (default: index)")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    ask_parser.add_argument("--simple-generator", action="store_true", help="Use simple fallback generator")

    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive QA mode")
    interactive_parser.add_argument("--index-path", default="index", help="Path to index (default: index)")
    interactive_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    interactive_parser.add_argument("--simple-generator", action="store_true", help="Use simple fallback generator")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    pipeline = QuantQAPipeline(index_path=args.index_path)

    if args.command == "build":
        success = pipeline.build_index(args.pdf_directory)
        if not success:
            sys.exit(1)

    elif args.command == "ask":
        if not pipeline.load_models(use_simple_generator=args.simple_generator):
            sys.exit(1)

        answer = pipeline.answer_question(args.question, top_k=args.top_k)
        if answer:
            print("\n" + "="*60)
            print(f"Question: {args.question}")
            print("="*60)
            print(answer)
        else:
            sys.exit(1)

    elif args.command == "interactive":
        if not pipeline.load_models(use_simple_generator=args.simple_generator):
            sys.exit(1)

        print("\n" + "="*60)
        print("Quantitative Finance QA System - Interactive Mode")
        print("="*60)
        print("Type 'quit' or 'exit' to stop")
        print()

        while True:
            try:
                question = input("Question: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if not question:
                    continue

                answer = pipeline.answer_question(question, top_k=args.top_k)
                if answer:
                    print("\n" + "-"*50)
                    print(answer)
                    print("-"*50 + "\n")
                else:
                    print("Sorry, I couldn't generate an answer.\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break

if __name__ == "__main__":
    main()