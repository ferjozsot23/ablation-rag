"""
Processes and normalizes all downloaded HTML documents.

Pipeline:
1. Reads HTML files from data/raw/
2. Parses them with SECHTMLParser
3. Converts them to DocumentItem
4. Saves normalized JSON files in data/processed/
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from src.ingestion.html_parser import SECHTMLParser
from src.ingestion.document_schema import DocumentItem

# Configure logging
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes SEC HTML documents into the normalized format."""

    def __init__(
        self,
        input_dir: Path = Path("data/raw"),
        output_dir: Path = Path("data/processed"),
        manifest_dir: Path = Path("data/manifests")
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.manifest_dir = manifest_dir

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

        # Parser
        self.parser = SECHTMLParser()

        # Stats
        self.stats = {
            "processed": 0,
            "failed": 0,
            "total_words": 0,
            "by_form_type": {},
            "by_company": {},
            "errors": []
        }

    def process_all(self):
        """Processes every HTML file in input_dir."""
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Document processing - started")
        logger.info(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'#'*60}\n")

        # Find every HTML file (10-K and 10-Q)
        html_files = list(self.input_dir.glob("10-*.htm"))

        logger.info(f"Found {len(html_files)} HTML files")
        logger.info(f"Input:  {self.input_dir}")
        logger.info(f"Output: {self.output_dir}\n")

        start_time = datetime.now()

        for i, html_file in enumerate(html_files, 1):
            logger.info(f"\n[{i}/{len(html_files)}] {html_file.name}")

            try:
                # Parse the HTML
                parsed_data = self.parser.parse(html_file)

                # Convert to DocumentItem
                doc = DocumentItem.from_parsed_html(parsed_data, html_file)

                # Save the normalized JSON
                output_file = self.output_dir / f"{doc.document_id}.json"
                doc.save_to_file(output_file)

                # Update stats
                self.stats["processed"] += 1
                self.stats["total_words"] += doc.total_words

                # Per-form-type stats
                form_type = doc.form_type
                if form_type not in self.stats["by_form_type"]:
                    self.stats["by_form_type"][form_type] = {
                        "count": 0,
                        "words": 0
                    }
                self.stats["by_form_type"][form_type]["count"] += 1
                self.stats["by_form_type"][form_type]["words"] += doc.total_words

                # Per-company stats
                company = doc.company_id
                if company not in self.stats["by_company"]:
                    self.stats["by_company"][company] = {
                        "count": 0,
                        "words": 0
                    }
                self.stats["by_company"][company]["count"] += 1
                self.stats["by_company"][company]["words"] += doc.total_words

                logger.info(f"  Saved: {output_file.name}")

            except Exception as e:
                error_msg = f"Error in {html_file.name}: {e}"
                logger.error(f"  {error_msg}")
                self.stats["failed"] += 1
                self.stats["errors"].append(error_msg)

        # Save per-company manifests
        self._generate_manifests()

        # Final report
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        self._print_summary(len(html_files), duration)

    def _generate_manifests(self):
        """Generates one JSONL manifest per company."""
        logger.info(f"\n{'='*60}")
        logger.info("Generating per-company manifests...")
        logger.info(f"{'='*60}\n")

        # Group documents by company
        company_docs = {}

        for json_file in self.output_dir.glob("*.json"):
            doc_data = json.loads(json_file.read_text())
            company = doc_data["company_id"]

            if company not in company_docs:
                company_docs[company] = []

            company_docs[company].append({
                "document_id": doc_data["document_id"],
                "form_type": doc_data["form_type"],
                "period_end": doc_data["period_end"],
                "filing_date": doc_data.get("filing_date"),
                "total_words": doc_data["total_words"],
                "total_sections": doc_data["total_sections"],
                "source_file": doc_data["source_file"],
                "sha256_hash": doc_data["sha256_hash"]
            })

        # One manifest per company
        for company, docs in company_docs.items():
            # Sort by date (handle None)
            docs.sort(key=lambda x: x["period_end"] or "")

            manifest_file = self.manifest_dir / f"{company}_manifest.jsonl"

            with open(manifest_file, 'w', encoding='utf-8') as f:
                for doc in docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')

            logger.info(
                f"  {company}: {len(docs)} documents → "
                f"{manifest_file.name}"
            )

    def _print_summary(self, total_files: int, duration: float):
        """Prints the final summary."""
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Processing completed")
        logger.info(f"{'#'*60}\n")

        logger.info(f"Statistics:")
        logger.info(f"  Files processed: {self.stats['processed']}/{total_files}")
        logger.info(f"  Files failed: {self.stats['failed']}")
        logger.info(f"  Total words: {self.stats['total_words']:,}")
        logger.info(f"  Total time: {duration:.1f}s ({duration/60:.1f} min)")
        logger.info(
            f"  Speed: {self.stats['processed']/(duration/60):.1f} docs/min"
        )

        logger.info(f"\nBy document type:")
        for form_type, data in sorted(self.stats["by_form_type"].items()):
            logger.info(
                f"  {form_type}: {data['count']} docs, "
                f"{data['words']:,} words"
            )

        logger.info(f"\nBy company:")
        for company, data in sorted(self.stats["by_company"].items()):
            logger.info(
                f"  {company}: {data['count']} docs, "
                f"{data['words']:,} words"
            )

        if self.stats["errors"]:
            logger.warning(f"\n{len(self.stats['errors'])} errors:")
            for error in self.stats["errors"][:5]:
                logger.warning(f"  - {error}")
            if len(self.stats["errors"]) > 5:
                logger.warning(
                    f"  ... and {len(self.stats['errors']) - 5} more"
                )

        logger.info(f"\nNormalized documents: {self.output_dir}/")
        logger.info(f"Manifests: {self.manifest_dir}/")
        logger.info(f"Logs: logs/processing.log\n")


def main():
    """Runs the processing (`python -m src.ingestion.process_documents`)."""
    processor = DocumentProcessor()
    processor.process_all()


if __name__ == "__main__":
    main()
