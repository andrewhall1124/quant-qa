import os
import pymupdf
from typing import List, Dict, Any
import re

class DocumentChunk:
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata

class PDFIngester:
    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF with page-level metadata."""
        doc = pymupdf.open(pdf_path)
        pages = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()

            if text.strip():  # Only include non-empty pages
                pages.append({
                    'text': text.strip(),
                    'page_number': page_num + 1,
                    'filename': os.path.basename(pdf_path)
                })

        doc.close()
        return pages

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation using word count * 1.3."""
        words = len(text.split())
        return int(words * 1.3)

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk text into segments of approximately chunk_size tokens with overlap."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        current_tokens = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self.estimate_tokens(sentence)

            # If adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk with current content
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_id'] = len(chunks)
                chunks.append(DocumentChunk(current_chunk.strip(), chunk_metadata))

                # Start new chunk with overlap
                overlap_text = ""
                overlap_tokens = 0

                # Go backwards to find sentences for overlap
                j = i - 1
                while j >= 0 and overlap_tokens < self.overlap:
                    prev_sentence = sentences[j]
                    prev_tokens = self.estimate_tokens(prev_sentence)

                    if overlap_tokens + prev_tokens <= self.overlap:
                        overlap_text = prev_sentence + " " + overlap_text
                        overlap_tokens += prev_tokens
                        j -= 1
                    else:
                        break

                current_chunk = overlap_text
                current_tokens = overlap_tokens

            # Add current sentence to chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_tokens += sentence_tokens
            i += 1

        # Add final chunk if there's remaining content
        if current_chunk.strip():
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_id'] = len(chunks)
            chunks.append(DocumentChunk(current_chunk.strip(), chunk_metadata))

        return chunks

    def ingest_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """Process a single PDF and return chunked documents."""
        pages = self.extract_text_from_pdf(pdf_path)
        all_chunks = []

        for page_data in pages:
            chunks = self.chunk_text(
                page_data['text'],
                {
                    'filename': page_data['filename'],
                    'page_number': page_data['page_number']
                }
            )
            all_chunks.extend(chunks)

        return all_chunks

    def ingest_directory(self, directory_path: str) -> List[DocumentChunk]:
        """Process all PDFs in a directory."""
        all_chunks = []

        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(directory_path, filename)
                print(f"Processing {filename}...")

                try:
                    chunks = self.ingest_pdf(pdf_path)
                    all_chunks.extend(chunks)
                    print(f"  Created {len(chunks)} chunks")
                except Exception as e:
                    print(f"  Error processing {filename}: {e}")

        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python ingest.py <pdf_directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        sys.exit(1)

    ingester = PDFIngester()
    chunks = ingester.ingest_directory(directory)

    # Print sample chunk for verification
    if chunks:
        print("\nSample chunk:")
        print(f"Text: {chunks[0].text[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")