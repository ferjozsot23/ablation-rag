"""
Schema for normalized documents.

Defines the standard DocumentItem structure.
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path


@dataclass
class DocumentSection:
    """A section (Item) inside a document."""
    item_number: str          # "1", "1A", "7", "8"
    item_title: str           # "Business", "Risk Factors", "MD&A"
    text: str                 # clean text of the section
    word_count: int           # number of words
    char_count: int           # number of characters
    priority: bool = False    # priority Items (1, 1A, 7, 8)

    def to_dict(self) -> Dict:
        """Converts to a dictionary."""
        return asdict(self)


@dataclass
class DocumentItem:
    """
    Normalized financial document.

    Standard structure for 10-K, 10-Q and other SEC filings.
    """
    # Identifiers
    document_id: str          # unique document hash
    company_id: str           # ticker (AAPL, NVDA, MSFT)
    cik: str                  # SEC CIK

    # Filing metadata
    form_type: str            # "10-K" or "10-Q"
    filing_date: Optional[str] = None  # filing date YYYY-MM-DD
    period_end: str = None    # fiscal period YYYY-MM-DD
    accession_number: str = None  # SEC accession number

    # Content
    sections: List[DocumentSection] = field(default_factory=list)

    # Statistics
    total_words: int = 0
    total_sections: int = 0

    # Provenance
    source_file: str = None   # path of the original file
    source_url: Optional[str] = None  # SEC URL
    sha256_hash: str = None   # hash of the original content

    # Timestamps
    parsed_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    @classmethod
    def from_parsed_html(
        cls,
        parsed_data: Dict,
        source_file: Path
    ) -> 'DocumentItem':
        """
        Creates a DocumentItem from the HTML parser output.

        Args:
            parsed_data: dict returned by SECHTMLParser
            source_file: path of the original HTML file

        Returns:
            Populated DocumentItem
        """
        # Read the original file to compute its hash
        original_content = source_file.read_bytes()
        sha256 = hashlib.sha256(original_content).hexdigest()

        # Build the sections
        sections = [
            DocumentSection(**section_data)
            for section_data in parsed_data['sections']
        ]

        # Unique document_id
        # Format: {ticker}_{form_type}_{period_end}_{short_hash}
        ticker = parsed_data.get('ticker', 'UNKNOWN')
        form_type = parsed_data.get('form_type', 'UNKNOWN')
        period_end = parsed_data.get('period_end', 'UNKNOWN')
        hash_short = sha256[:8]

        document_id = f"{ticker}_{form_type}_{period_end}_{hash_short}"

        # Build the SEC URL
        cik = parsed_data.get('cik', '').lstrip('0')  # without padding
        acc = parsed_data.get('accession_number', '').replace('-', '')
        source_url = None
        if cik and acc:
            source_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik}/{acc}/{source_file.name}"
            )

        return cls(
            document_id=document_id,
            company_id=ticker,
            cik=parsed_data.get('cik'),
            form_type=form_type,
            period_end=period_end,
            accession_number=parsed_data.get('accession_number'),
            sections=sections,
            total_words=parsed_data.get('total_words', 0),
            total_sections=len(sections),
            source_file=str(source_file),
            source_url=source_url,
            sha256_hash=sha256,
            parsed_at=parsed_data.get('parsed_at')
        )

    def to_dict(self) -> Dict:
        """Converts to a dictionary (JSON-serializable)."""
        data = asdict(self)
        data['sections'] = [s.to_dict() for s in self.sections]
        return data

    def to_json(self, indent: int = 2) -> str:
        """Serializes to JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_to_file(self, output_path: Path):
        """Saves the document to a JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            self.to_json(),
            encoding='utf-8'
        )

    def get_priority_sections(self) -> List[DocumentSection]:
        """Returns only the priority sections."""
        return [s for s in self.sections if s.priority]

    def get_section(self, item_number: str) -> Optional[DocumentSection]:
        """Gets a section by Item number."""
        for section in self.sections:
            if section.item_number == item_number:
                return section
        return None

    def __str__(self) -> str:
        """String representation."""
        return (
            f"DocumentItem({self.document_id}: "
            f"{self.form_type} {self.company_id} "
            f"{self.period_end}, {self.total_sections} sections, "
            f"{self.total_words:,} words)"
        )


def main():
    """Schema demo (`python -m src.ingestion.document_schema`)."""
    sections = [
        DocumentSection(
            item_number="1",
            item_title="Business",
            text="Apple Inc. designs, manufactures and markets...",
            word_count=2500,
            char_count=15000,
            priority=True
        ),
        DocumentSection(
            item_number="1A",
            item_title="Risk Factors",
            text="The following summarizes factors...",
            word_count=8500,
            char_count=52000,
            priority=True
        )
    ]

    doc = DocumentItem(
        document_id="AAPL_10-K_2025-09-27_abc12345",
        company_id="AAPL",
        cik="0000320193",
        form_type="10-K",
        period_end="2025-09-27",
        filing_date="2025-10-31",
        accession_number="0000320193-25-000079",
        sections=sections,
        total_words=11000,
        total_sections=2,
        source_file="data/raw/10-K_0000320193_...",
        sha256_hash="abc123..."
    )

    print("="*60)
    print("Demo: DocumentItem Schema")
    print("="*60)
    print(f"\n{doc}\n")

    print("JSON output (preview):")
    print("-"*60)
    json_str = doc.to_json()
    print(json_str[:500])
    print("...")

    print(f"\nPriority sections: {len(doc.get_priority_sections())}")
    for section in doc.get_priority_sections():
        print(f"  - Item {section.item_number}: {section.item_title} ({section.word_count:,} words)")


if __name__ == "__main__":
    main()
