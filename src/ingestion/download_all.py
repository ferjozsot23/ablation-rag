"""
SEC document download pipeline for the 10 companies.

Downloads:
- Submissions (metadata)
- 10-K (annual reports)
- 10-Q (quarterly reports)
- Company Facts (XBRL)

For the period 2021-01-01 to 2026-09-30.
"""

import csv
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from src.ingestion.sec_client import SECClient

# Configure logging
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DownloadPipeline:
    """SEC document download pipeline."""

    def __init__(
        self,
        universe_csv: Path = Path("data/universe.csv"),
        output_dir: Path = Path("data/raw"),
        start_date: str = "2021-01-01",
        end_date: str = "2026-09-30"
    ):
        self.universe_csv = universe_csv
        self.output_dir = output_dir
        self.start_date = start_date
        self.end_date = end_date

        # SEC client
        self.client = SECClient(output_dir=output_dir)

        # Stats
        self.stats = {
            "companies_processed": 0,
            "companies_failed": 0,
            "filings_10k": 0,
            "filings_10q": 0,
            "company_facts": 0,
            "total_downloads": 0,
            "errors": []
        }

    def load_universe(self) -> List[Dict[str, str]]:
        """Loads the company universe from the CSV."""
        companies = []

        with open(self.universe_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                companies.append({
                    "ticker": row["ticker"],
                    "cik": row["cik"],
                    "name": row["company_name"]
                })

        logger.info(f"Universe loaded: {len(companies)} companies")
        return companies

    def download_company(self, company: Dict[str, str]) -> Dict[str, any]:
        """
        Downloads every document of a company.

        Returns:
            Dict with the download stats
        """
        ticker = company["ticker"]
        cik = company["cik"]
        name = company["name"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {name} ({ticker}) - CIK {cik}")
        logger.info(f"{'='*60}")

        company_stats = {
            "ticker": ticker,
            "cik": cik,
            "name": name,
            "filings_10k": 0,
            "filings_10q": 0,
            "company_facts": False,
            "errors": []
        }

        try:
            # 1. Download submissions
            logger.info(f"[1/3] Downloading submissions...")
            submissions = self.client.download_submissions(cik)

            if not submissions:
                error_msg = f"Could not download submissions for {ticker}"
                logger.error(error_msg)
                company_stats["errors"].append(error_msg)
                self.stats["companies_failed"] += 1
                return company_stats

            # 2. Filter and download 10-K and 10-Q
            logger.info(f"[2/3] Filtering 10-K and 10-Q ({self.start_date} - {self.end_date})...")

            # 10-K (annual)
            filings_10k = self.client.get_filings_by_type(
                submissions,
                form_types=["10-K"],
                start_date=self.start_date,
                end_date=self.end_date
            )
            logger.info(f"  Found: {len(filings_10k)} 10-Ks")

            # 10-Q (quarterly)
            filings_10q = self.client.get_filings_by_type(
                submissions,
                form_types=["10-Q"],
                start_date=self.start_date,
                end_date=self.end_date
            )
            logger.info(f"  Found: {len(filings_10q)} 10-Qs")

            # Download the 10-Ks
            for i, filing in enumerate(filings_10k, 1):
                logger.info(f"  Downloading 10-K {i}/{len(filings_10k)}: {filing['filing_date']}")
                path = self.client.download_filing(
                    cik=cik,
                    accession_number=filing["accession_number"],
                    primary_document=filing["primary_document"],
                    filing_type="10-K"
                )
                if path:
                    company_stats["filings_10k"] += 1
                    self.stats["filings_10k"] += 1
                    self.stats["total_downloads"] += 1
                else:
                    error_msg = f"10-K download failed {filing['filing_date']}"
                    company_stats["errors"].append(error_msg)

            # Download the 10-Qs
            for i, filing in enumerate(filings_10q, 1):
                logger.info(f"  Downloading 10-Q {i}/{len(filings_10q)}: {filing['filing_date']}")
                path = self.client.download_filing(
                    cik=cik,
                    accession_number=filing["accession_number"],
                    primary_document=filing["primary_document"],
                    filing_type="10-Q"
                )
                if path:
                    company_stats["filings_10q"] += 1
                    self.stats["filings_10q"] += 1
                    self.stats["total_downloads"] += 1
                else:
                    error_msg = f"10-Q download failed {filing['filing_date']}"
                    company_stats["errors"].append(error_msg)

            # 3. Download Company Facts
            logger.info(f"[3/3] Downloading Company Facts...")
            facts = self.client.download_company_facts(cik)
            if facts:
                company_stats["company_facts"] = True
                self.stats["company_facts"] += 1
                self.stats["total_downloads"] += 1
            else:
                error_msg = "Company Facts not available"
                logger.warning(f"  {error_msg}")
                company_stats["errors"].append(error_msg)

            # Company summary
            logger.info(f"\n{name} completed:")
            logger.info(f"  - 10-Ks: {company_stats['filings_10k']}")
            logger.info(f"  - 10-Qs: {company_stats['filings_10q']}")
            logger.info(f"  - Company Facts: {'yes' if company_stats['company_facts'] else 'no'}")

            self.stats["companies_processed"] += 1

        except Exception as e:
            error_msg = f"Error processing {ticker}: {e}"
            logger.error(error_msg)
            company_stats["errors"].append(error_msg)
            self.stats["companies_failed"] += 1
            self.stats["errors"].append(error_msg)

        return company_stats

    def run(self):
        """Runs the full pipeline."""
        logger.info(f"\n{'#'*60}")
        logger.info(f"# SEC download pipeline - started")
        logger.info(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"# Period: {self.start_date} - {self.end_date}")
        logger.info(f"{'#'*60}\n")

        start_time = datetime.now()

        # Load the universe
        companies = self.load_universe()

        # Process every company
        all_company_stats = []
        for i, company in enumerate(companies, 1):
            logger.info(f"\n[{i}/{len(companies)}] {company['name']}")
            company_stats = self.download_company(company)
            all_company_stats.append(company_stats)

        # Save the inventory
        self._save_inventory(all_company_stats)

        # Final report
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"\n{'#'*60}")
        logger.info(f"# Pipeline completed")
        logger.info(f"{'#'*60}")
        logger.info(f"\nStatistics:")
        logger.info(f"  Companies processed: {self.stats['companies_processed']}/{len(companies)}")
        logger.info(f"  Companies failed: {self.stats['companies_failed']}")
        logger.info(f"  10-Ks downloaded: {self.stats['filings_10k']}")
        logger.info(f"  10-Qs downloaded: {self.stats['filings_10q']}")
        logger.info(f"  Company Facts: {self.stats['company_facts']}")
        logger.info(f"  Total downloads: {self.stats['total_downloads']}")
        logger.info(f"  Total time: {duration:.1f}s ({duration/60:.1f} min)")
        logger.info(f"  Speed: {self.stats['total_downloads']/(duration/60):.1f} docs/min")

        if self.stats["errors"]:
            logger.warning(f"\n{len(self.stats['errors'])} errors:")
            for error in self.stats["errors"][:10]:
                logger.warning(f"  - {error}")
            if len(self.stats["errors"]) > 10:
                logger.warning(f"  ... and {len(self.stats['errors']) - 10} more")

        logger.info(f"\nInventory saved in: data/raw/inventory.json")
        logger.info(f"Logs saved in: logs/download.log\n")

    def _save_inventory(self, company_stats: List[Dict]):
        """Saves the inventory of downloaded documents."""
        inventory = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "period": f"{self.start_date} to {self.end_date}",
                "total_companies": len(company_stats)
            },
            "summary": {
                "companies_processed": self.stats["companies_processed"],
                "companies_failed": self.stats["companies_failed"],
                "filings_10k": self.stats["filings_10k"],
                "filings_10q": self.stats["filings_10q"],
                "company_facts": self.stats["company_facts"],
                "total_downloads": self.stats["total_downloads"]
            },
            "companies": company_stats
        }

        output_path = self.output_dir / "inventory.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(inventory, f, indent=2, ensure_ascii=False)

        logger.info(f"\nInventory saved: {output_path}")


def main():
    """Runs the download pipeline (`python -m src.ingestion.download_all`)."""
    pipeline = DownloadPipeline(
        universe_csv=Path("data/universe.csv"),
        output_dir=Path("data/raw"),
        start_date="2021-01-01",
        end_date="2026-09-30"
    )

    pipeline.run()


if __name__ == "__main__":
    main()
