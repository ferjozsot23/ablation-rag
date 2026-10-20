"""
Document chunker for RAG.

Splits long documents into manageable chunks while preserving context.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text fragment with metadata."""
    chunk_id: str              # unique identifier
    document_id: str           # source document ID
    company_id: str            # company ticker
    form_type: str             # 10-K or 10-Q
    period_end: str            # fiscal period

    section_item: str          # Item number (1, 1A, 7, etc.)
    section_title: str         # Item title

    text: str                  # chunk content
    chunk_index: int           # chunk index within the section (0, 1, 2...)
    total_chunks_in_section: int  # total chunks in this section

    # Statistics
    word_count: int
    char_count: int

    # Additional metadata
    is_priority_section: bool  # whether it belongs to a priority Item

    def to_dict(self) -> Dict:
        """Converts to a dictionary."""
        return asdict(self)

    def get_context_header(self) -> str:
        """Builds the context header of the chunk."""
        return (
            f"Company: {self.company_id} | "
            f"Document: {self.form_type} {self.period_end} | "
            f"Section: Item {self.section_item} - {self.section_title}"
        )


class DocumentChunker:
    """Chunker for financial documents."""

    def __init__(
        self,
        target_chunk_size: int = 800,      # target tokens
        min_chunk_size: int = 600,         # minimum tokens
        max_chunk_size: int = 1000,        # maximum tokens
        overlap_size: int = 100,           # overlap tokens
        chars_per_token: float = 4.0       # chars/token approximation
    ):
        """
        Initializes the chunker.

        Args:
            target_chunk_size: target size in tokens
            min_chunk_size: minimum size in tokens
            max_chunk_size: maximum size in tokens
            overlap_size: overlap between chunks in tokens
            chars_per_token: approximate characters per token
        """
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.chars_per_token = chars_per_token

        # Convert to characters to simplify the splitting
        self.target_chars = int(target_chunk_size * chars_per_token)
        self.min_chars = int(min_chunk_size * chars_per_token)
        self.max_chars = int(max_chunk_size * chars_per_token)
        self.overlap_chars = int(overlap_size * chars_per_token)

    def chunk_document(self, doc_data: Dict) -> List[Chunk]:
        """
        Chunks a whole document.

        Args:
            doc_data: normalized document (JSON)

        Returns:
            List of Chunks
        """
        all_chunks = []

        for section in doc_data['sections']:
            section_chunks = self._chunk_section(
                section_text=section['text'],
                document_id=doc_data['document_id'],
                company_id=doc_data['company_id'],
                form_type=doc_data['form_type'],
                period_end=doc_data['period_end'],
                section_item=section['item_number'],
                section_title=section['item_title'],
                is_priority=section.get('priority', False)
            )

            all_chunks.extend(section_chunks)

        return all_chunks

    def _chunk_section(
        self,
        section_text: str,
        document_id: str,
        company_id: str,
        form_type: str,
        period_end: str,
        section_item: str,
        section_title: str,
        is_priority: bool
    ) -> List[Chunk]:
        """
        Splits a section into chunks.

        Strategy:
        1. If the section is small (< max_chunk_size), return it as 1 chunk
        2. If it is large, split into paragraphs and group them into chunks
        3. Respect paragraph boundaries whenever possible
        """
        # Case 1: small section
        if len(section_text) <= self.max_chars:
            return [self._create_chunk(
                text=section_text,
                document_id=document_id,
                company_id=company_id,
                form_type=form_type,
                period_end=period_end,
                section_item=section_item,
                section_title=section_title,
                chunk_index=0,
                total_chunks=1,
                is_priority=is_priority
            )]

        # Case 2: large section - split
        paragraphs = self._split_into_paragraphs(section_text)
        chunks = []
        current_chunk_text = ""

        for para in paragraphs:
            # Adding this paragraph would exceed max_chars: close the chunk
            if len(current_chunk_text) + len(para) > self.max_chars and current_chunk_text:
                chunks.append(current_chunk_text)

                # Start the next chunk with overlap
                overlap_text = self._extract_overlap(current_chunk_text)
                current_chunk_text = overlap_text + "\n\n" + para
            else:
                # Append the paragraph to the current chunk
                if current_chunk_text:
                    current_chunk_text += "\n\n" + para
                else:
                    current_chunk_text = para

        # Add the last chunk
        if current_chunk_text:
            chunks.append(current_chunk_text)

        # Convert the texts to Chunk objects
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            chunk_obj = self._create_chunk(
                text=chunk_text,
                document_id=document_id,
                company_id=company_id,
                form_type=form_type,
                period_end=period_end,
                section_item=section_item,
                section_title=section_title,
                chunk_index=i,
                total_chunks=len(chunks),
                is_priority=is_priority
            )
            chunk_objects.append(chunk_obj)

        return chunk_objects

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Splits text into paragraphs.

        Separators:
        - Blank line (\\n\\n)
        """
        # Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Split on blank lines
        paragraphs = text.split('\n\n')

        # Drop empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        return paragraphs

    def _extract_overlap(self, text: str) -> str:
        """
        Extracts the last N characters for the overlap.

        Tries to cut at a sentence boundary.
        """
        if len(text) <= self.overlap_chars:
            return text

        # Take the last overlap_chars characters
        overlap = text[-self.overlap_chars:]

        # Start after the first sentence end
        first_period = overlap.find('. ')
        if first_period > 0:
            overlap = overlap[first_period + 2:]  # +2 to skip ". "

        return overlap.strip()

    def _create_chunk(
        self,
        text: str,
        document_id: str,
        company_id: str,
        form_type: str,
        period_end: str,
        section_item: str,
        section_title: str,
        chunk_index: int,
        total_chunks: int,
        is_priority: bool
    ) -> Chunk:
        """Creates a Chunk object."""
        chunk_id = f"{document_id}_item{section_item}_chunk{chunk_index}"

        # Prepend a context header to the text
        context_header = (
            f"[Company: {company_id} | "
            f"Document: {form_type} {period_end} | "
            f"Section: Item {section_item} - {section_title}]\n\n"
        )

        full_text = context_header + text

        return Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            company_id=company_id,
            form_type=form_type,
            period_end=period_end,
            section_item=section_item,
            section_title=section_title,
            text=full_text,
            chunk_index=chunk_index,
            total_chunks_in_section=total_chunks,
            word_count=len(text.split()),
            char_count=len(text),
            is_priority_section=is_priority
        )


def main():
    """Chunker demo (run from the project root: `python -m src.retrieval.chunker`)."""
    import sys

    logging.basicConfig(level=logging.INFO)

    # Load a sample document
    processed_dir = Path("data/processed")
    sample_files = list(processed_dir.glob("AAPL_10-K*.json"))

    if not sample_files:
        print("No Apple documents found")
        sys.exit(1)

    sample_file = sample_files[0]
    doc_data = json.loads(sample_file.read_text())

    print("="*60)
    print("Demo: Document Chunker")
    print("="*60)
    print(f"\nDocument: {doc_data['document_id']}")
    print(f"Sections: {len(doc_data['sections'])}")
    print(f"Total words: {doc_data['total_words']:,}\n")

    chunker = DocumentChunker(
        target_chunk_size=800,
        overlap_size=100
    )

    chunks = chunker.chunk_document(doc_data)

    print(f"Result: {len(chunks)} chunks generated\n")

    # Statistics
    words_per_chunk = [c.word_count for c in chunks]
    print(f"Chunk statistics:")
    print(f"  Total: {len(chunks)}")
    print(f"  Average words: {sum(words_per_chunk)/len(words_per_chunk):.0f}")
    print(f"  Min words: {min(words_per_chunk)}")
    print(f"  Max words: {max(words_per_chunk)}")

    # Priority chunks
    priority_chunks = [c for c in chunks if c.is_priority_section]
    print(f"  Priority chunks: {len(priority_chunks)}")

    # Preview
    print(f"\nChunk preview:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n  Chunk #{i+1}:")
        print(f"    ID: {chunk.chunk_id}")
        print(f"    Section: Item {chunk.section_item} - {chunk.section_title[:40]}")
        print(f"    Words: {chunk.word_count}")
        print(f"    Text preview:")
        preview = chunk.text[:200].replace('\n', ' ')
        print(f"      {preview}...")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
