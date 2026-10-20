"""
HTML parser for SEC documents (10-K, 10-Q).

Extracts clean text and preserves the Item structure. Tables are kept as
flattened text; `extract_tables_from_html` can extract them in a structured
form but it is not part of the pipeline.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from bs4 import BeautifulSoup, NavigableString
from datetime import datetime

logger = logging.getLogger(__name__)


class SECHTMLParser:
    """Parser for SEC HTML documents."""

    # Regex that identifies Items
    ITEM_PATTERN = re.compile(
        r'Item\s+(\d+[A-Z]?)\.?\s+([A-Z][^\n]{5,120})',
        re.MULTILINE | re.IGNORECASE
    )

    # Priority Items
    PRIORITY_ITEMS = {
        '1': 'Business',
        '1A': 'Risk Factors',
        '1B': 'Unresolved Staff Comments',
        '7': 'MD&A',
        '7A': 'Market Risk',
        '8': 'Financial Statements',
    }

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def parse(self, html_path: Path) -> Dict:
        """
        Parses an SEC HTML document.

        Args:
            html_path: path to the HTML file

        Returns:
            Dict with the parsed document
        """
        self.logger.info(f"Parsing: {html_path.name}")

        html = html_path.read_text(encoding='utf-8')

        soup = BeautifulSoup(html, 'html.parser')

        # Step 1: remove noise
        self._clean_html(soup)

        # Step 2: extract metadata from the file name
        metadata = self._extract_metadata_from_filename(html_path.name)

        # Step 3: extract the full text
        full_text = soup.get_text()

        # Step 4: split by Items
        sections = self._extract_items(full_text)

        result = {
            **metadata,
            'source_file': str(html_path),
            'total_sections': len(sections),
            'sections': sections,
            'total_words': sum(s['word_count'] for s in sections),
            'parsed_at': datetime.now().isoformat()
        }

        self.logger.info(
            f"  Extracted {len(sections)} Items, "
            f"{result['total_words']:,} words"
        )

        return result

    def _clean_html(self, soup: BeautifulSoup):
        """
        Removes unwanted elements from the HTML.

        Modifies soup in place.
        """
        # 1. Remove scripts, styles, metadata
        for tag in soup(['script', 'style', 'meta', 'link']):
            tag.decompose()

        # 2. Remove navigation/TOC (divs with many links)
        for div in soup.find_all('div'):
            links = div.find_all('a')
            # More than 10 links: probably navigation
            if len(links) > 10:
                # Check that the links are internal (#item1, #item2)
                internal_links = [
                    a for a in links
                    if a.get('href', '').startswith('#')
                ]
                if len(internal_links) > 8:
                    div.decompose()

        # 3. Remove HTML comments
        for comment in soup.find_all(
            string=lambda text: isinstance(text, NavigableString) and
            text.strip().startswith('<!--')
        ):
            comment.extract()

    def _extract_metadata_from_filename(self, filename: str) -> Dict:
        """
        Extracts metadata from the file name.

        Format: {form_type}_{cik}_{accession}_{primary_doc}
        Example: 10-K_0000320193_0000320193-25-000079_aapl-20250927.htm
        """
        parts = filename.replace('.htm', '').split('_')

        metadata = {
            'form_type': None,
            'cik': None,
            'accession_number': None,
            'ticker': None,
            'period_end': None
        }

        if len(parts) >= 4:
            metadata['form_type'] = parts[0]
            metadata['cik'] = parts[1]
            metadata['accession_number'] = parts[2]

            # Ticker and date from the primary document
            # Format: aapl-20250927
            primary = parts[3]
            match = re.match(r'([a-z]+)-(\d{8})', primary, re.I)
            if match:
                metadata['ticker'] = match.group(1).upper()
                date_str = match.group(2)
                # Convert YYYYMMDD to YYYY-MM-DD
                metadata['period_end'] = (
                    f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                )

        return metadata

    def _extract_items(self, text: str) -> List[Dict]:
        """
        Extracts the sections (Items) from the text.

        Args:
            text: full text of the document

        Returns:
            List of dicts, one per Item
        """
        sections = []

        # Find every occurrence of "Item X."
        matches = list(self.ITEM_PATTERN.finditer(text))

        if not matches:
            self.logger.warning("  No Items found in the document")
            # Return the whole text as a single section
            return [{
                'item_number': 'FULL',
                'item_title': 'Full Document',
                'text': self._normalize_text(text),
                'word_count': len(text.split()),
                'char_count': len(text)
            }]

        for i, match in enumerate(matches):
            item_number = match.group(1).upper()
            item_title = match.group(2).strip()

            # Start of this Item
            start = match.start()

            # End of this Item (start of the next one, or end of the document)
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(text)

            item_text = text[start:end]

            item_text_clean = self._normalize_text(item_text)

            # Drop very short Items (probably false positives)
            if len(item_text_clean.split()) < 20:
                continue

            sections.append({
                'item_number': item_number,
                'item_title': item_title,
                'text': item_text_clean,
                'word_count': len(item_text_clean.split()),
                'char_count': len(item_text_clean),
                'priority': item_number in self.PRIORITY_ITEMS
            })

        # Sort by Item number (1, 1A, 1B, 2, ...)
        sections.sort(key=lambda x: self._item_sort_key(x['item_number']))

        return sections

    def _normalize_text(self, text: str) -> str:
        """
        Normalizes whitespace and line breaks.

        Args:
            text: text to normalize

        Returns:
            Normalized text
        """
        # 1. Remove XBRL metadata (URLs and namespaces)
        text = re.sub(r'http://[^\s]+', '', text)

        # 2. Remove document references
        text = re.sub(r'[a-z]+-\d{8}(false|true)', '', text, flags=re.I)

        # 3. Collapse repeated spaces
        text = re.sub(r'[ \t]+', ' ', text)

        # 4. Collapse repeated line breaks (maximum 2)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # 5. Strip spaces at the start/end of lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        # 6. Strip empty lines at the start and end
        text = text.strip()

        return text

    def _item_sort_key(self, item_number: str) -> Tuple:
        """
        Builds a sort key for Items.

        Converts "1", "1A", "1B", "2" into sortable tuples.
        """
        match = re.match(r'(\d+)([A-Z]?)', item_number)
        if match:
            num = int(match.group(1))
            letter = match.group(2) or ''
            return (num, letter)
        return (9999, item_number)

    def extract_tables_from_html(
        self,
        soup: BeautifulSoup,
        max_tables: int = 10
    ) -> List[Dict]:
        """
        Extracts tables from the HTML into a structured format.

        Args:
            soup: BeautifulSoup object
            max_tables: maximum number of tables to extract

        Returns:
            List of tables as dicts
        """
        tables_data = []

        tables = soup.find_all('table')[:max_tables]

        for i, table in enumerate(tables):
            rows = table.find_all('tr')

            if not rows:
                continue

            # Headers and data
            table_rows = []
            for row in rows:
                cells = row.find_all(['th', 'td'])
                row_data = [
                    cell.get_text(strip=True)
                    for cell in cells
                ]
                # Skip empty rows
                if any(cell for cell in row_data):
                    table_rows.append(row_data)

            if table_rows:
                tables_data.append({
                    'table_index': i,
                    'num_rows': len(table_rows),
                    'num_cols': len(table_rows[0]) if table_rows else 0,
                    'data': table_rows
                })

        return tables_data


def main():
    """Parser demo (`python -m src.ingestion.html_parser`)."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Look for a sample 10-K
    raw_dir = Path("data/raw")
    sample_files = list(raw_dir.glob("10-K_0000320193*.htm"))

    if not sample_files:
        print("No Apple 10-K files found")
        sys.exit(1)

    # Use the most recent one
    sample_file = sorted(sample_files)[-1]

    print(f"\n{'='*60}")
    print(f"Demo: SEC HTML parser")
    print(f"{'='*60}\n")
    print(f"File: {sample_file.name}\n")

    parser = SECHTMLParser()
    result = parser.parse(sample_file)

    print(f"\n{'='*60}")
    print(f"Parsing result")
    print(f"{'='*60}")
    print(f"\nMetadata:")
    print(f"  Form Type: {result['form_type']}")
    print(f"  Ticker: {result['ticker']}")
    print(f"  CIK: {result['cik']}")
    print(f"  Period End: {result['period_end']}")
    print(f"  Total Words: {result['total_words']:,}")

    print(f"\nSections extracted: {len(result['sections'])}")
    print(f"\n{'Item':<6} {'Title':<40} {'Words':>8} {'Priority'}")
    print(f"{'-'*70}")
    for section in result['sections'][:15]:
        priority_mark = '*' if section.get('priority') else ''
        print(
            f"{section['item_number']:<6} "
            f"{section['item_title'][:40]:<40} "
            f"{section['word_count']:>8,} "
            f"{priority_mark}"
        )

    if len(result['sections']) > 15:
        print(f"... and {len(result['sections']) - 15} more")

    print(f"\n{'='*60}")
    print(f"Preview: Item 1 (Business)")
    print(f"{'='*60}")
    item1 = next((s for s in result['sections'] if s['item_number'] == '1'), None)
    if item1:
        preview = item1['text'][:500]
        print(preview)
        print("...")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
