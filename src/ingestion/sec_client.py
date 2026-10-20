"""
SEC EDGAR API Client with Rate Limiting

Downloads SEC documents following the fair-access policy:
- Max 10 requests per second
- Mandatory User-Agent with a contact email
- Error handling and retries

References:
- SEC API docs: https://www.sec.gov/data-research/sec-api-documentation
- Rate limit policy: https://www.sec.gov/os/accessing-edgar-data
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SECClient:
    """Client for downloading documents from SEC EDGAR."""

    BASE_URL = "https://data.sec.gov"
    ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

    # The SEC requires an identifying User-Agent (set SEC_USER_AGENT in .env)
    USER_AGENT = os.getenv("SEC_USER_AGENT", "NLP-RAG-Project/1.0 (your-email@example.com)")

    # Rate limiting: max 10 requests per second
    MIN_REQUEST_INTERVAL = 0.1  # 100 ms between requests = 10 req/s

    def __init__(self, output_dir: Path = Path("data/raw")):
        """
        Initializes the SEC client.

        Args:
            output_dir: directory where downloaded documents are saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Timestamp of the last request (rate limiting)
        self._last_request_time = 0.0

        # Session with retries
        self.session = self._create_session()

        logger.info(f"SECClient initialized. Output dir: {self.output_dir}")

    def _create_session(self) -> requests.Session:
        """Creates an HTTP session with automatic retries."""
        session = requests.Session()

        # Retries for transient errors
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,  # 1s, 2s, 4s
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Mandatory headers
        session.headers.update({
            "User-Agent": self.USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        })

        return session

    def _rate_limit(self):
        """Applies rate limiting (max 10 req/s)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            sleep_time = self.MIN_REQUEST_INTERVAL - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def download_submissions(self, cik: str) -> Optional[Dict[str, Any]]:
        """
        Downloads the submissions.json file of a company.

        It contains the metadata of every filing submitted.

        Args:
            cik: company CIK (10 digits, zero-padded)

        Returns:
            Dict with the submissions data, or None on failure

        Example:
            >>> client = SECClient()
            >>> data = client.download_submissions("0000320193")  # Apple
            >>> print(data["name"])
            "Apple Inc."
        """
        # Normalize the CIK to 10 digits
        cik_padded = cik.zfill(10)

        url = f"{self.BASE_URL}/submissions/CIK{cik_padded}.json"
        output_path = self.output_dir / f"submissions_{cik_padded}.json"

        logger.info(f"Downloading submissions for CIK {cik_padded}...")

        try:
            self._rate_limit()
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Save to disk
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Submissions downloaded: {data['name']} "
                f"({len(data['filings']['recent']['form'])} filings)"
            )

            return data

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"CIK {cik_padded} not found")
            else:
                logger.error(f"HTTP error {e.response.status_code}: {e}")
            return None

        except Exception as e:
            logger.error(f"Error downloading submissions: {e}")
            return None

    def download_filing(
        self,
        cik: str,
        accession_number: str,
        primary_document: str,
        filing_type: str = "10-K"
    ) -> Optional[Path]:
        """
        Downloads a specific filing (10-K, 10-Q, etc.) as HTML.

        Args:
            cik: company CIK
            accession_number: filing accession number (e.g. "0000320193-23-000106")
            primary_document: HTML file name (e.g. "aapl-20230930.htm")
            filing_type: document type, used for logging

        Returns:
            Path of the downloaded file, or None on failure

        Example:
            >>> client = SECClient()
            >>> path = client.download_filing(
            ...     cik="0000320193",
            ...     accession_number="0000320193-23-000106",
            ...     primary_document="aapl-20230930.htm",
            ...     filing_type="10-K"
            ... )
        """
        # CIK without padding for the Archives URL
        cik_clean = str(int(cik))

        # Accession number without dashes for the path
        acc_no_dashes = accession_number.replace("-", "")

        url = f"{self.ARCHIVES_URL}/{cik_clean}/{acc_no_dashes}/{primary_document}"

        filename = f"{filing_type}_{cik.zfill(10)}_{accession_number}_{primary_document}"
        output_path = self.output_dir / filename

        logger.info(f"Downloading {filing_type} for CIK {cik}...")

        try:
            self._rate_limit()
            response = self.session.get(url, timeout=60)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                f.write(response.content)

            size_kb = len(response.content) / 1024
            logger.info(f"{filing_type} downloaded: {output_path.name} ({size_kb:.1f} KB)")

            return output_path

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code}: {url}")
            return None

        except Exception as e:
            logger.error(f"Error downloading filing: {e}")
            return None

    def download_company_facts(self, cik: str) -> Optional[Dict[str, Any]]:
        """
        Downloads Company Facts (structured XBRL data).

        It contains every financial metric reported, as JSON.

        Args:
            cik: company CIK

        Returns:
            Dict with the company facts data, or None on failure

        Example:
            >>> client = SECClient()
            >>> facts = client.download_company_facts("0000320193")
            >>> revenue = facts["facts"]["us-gaap"]["Revenues"]
        """
        cik_padded = cik.zfill(10)

        url = f"{self.BASE_URL}/api/xbrl/companyfacts/CIK{cik_padded}.json"
        output_path = self.output_dir / f"company_facts_{cik_padded}.json"

        logger.info(f"Downloading Company Facts for CIK {cik_padded}...")

        try:
            self._rate_limit()
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            num_facts = len(data.get("facts", {}).get("us-gaap", {}))
            logger.info(f"Company Facts downloaded: {num_facts} GAAP metrics")

            return data

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Company Facts not available for CIK {cik_padded}")
            else:
                logger.error(f"HTTP error {e.response.status_code}: {e}")
            return None

        except Exception as e:
            logger.error(f"Error downloading Company Facts: {e}")
            return None

    def get_filings_by_type(
        self,
        submissions_data: Dict[str, Any],
        form_types: list[str],
        start_date: str = "2021-01-01",
        end_date: str = "2026-09-30"
    ) -> list[Dict[str, str]]:
        """
        Filters filings by type and date range.

        Args:
            submissions_data: data from submissions.json
            form_types: list of types (e.g. ["10-K", "10-Q"])
            start_date: start date YYYY-MM-DD
            end_date: end date YYYY-MM-DD

        Returns:
            List of dicts with the info of each filing

        Example:
            >>> subs = client.download_submissions("0000320193")
            >>> filings = client.get_filings_by_type(subs, ["10-K"], "2021-01-01", "2023-12-31")
            >>> print(len(filings))
            3
        """
        filings = []
        recent = submissions_data["filings"]["recent"]

        for i in range(len(recent["form"])):
            form_type = recent["form"][i]
            filing_date = recent["filingDate"][i]

            # Filter by type and date
            if form_type in form_types and start_date <= filing_date <= end_date:
                filings.append({
                    "form_type": form_type,
                    "filing_date": filing_date,
                    "accession_number": recent["accessionNumber"][i],
                    "primary_document": recent["primaryDocument"][i],
                    "report_date": recent.get("reportDate", [None] * len(recent["form"]))[i]
                })

        # Sort by date (most recent first)
        filings.sort(key=lambda x: x["filing_date"], reverse=True)

        return filings


def main():
    """SEC client demo (`python -m src.ingestion.sec_client`)."""
    import sys

    client = SECClient(output_dir=Path("data/raw"))

    # Example: Apple (CIK 0000320193)
    cik = "0000320193"

    print(f"\n{'='*60}")
    print(f"Demo: downloading Apple data (CIK {cik})")
    print(f"{'='*60}\n")

    # 1. Download submissions
    print("1. Downloading submissions.json...")
    submissions = client.download_submissions(cik)
    if not submissions:
        print("Error downloading submissions")
        sys.exit(1)

    print(f"   Company: {submissions['name']}")
    print(f"   Ticker: {submissions['tickers'][0]}")
    print(f"   Total filings: {len(submissions['filings']['recent']['form'])}\n")

    # 2. Recent 10-Ks
    print("2. Filtering 10-Ks (2021-2026)...")
    filings_10k = client.get_filings_by_type(
        submissions,
        form_types=["10-K"],
        start_date="2021-01-01",
        end_date="2026-09-30"
    )
    print(f"   Found: {len(filings_10k)} 10-Ks\n")

    for filing in filings_10k[:3]:
        print(f"   - {filing['form_type']} | {filing['filing_date']} | {filing['primary_document']}")

    # 3. Download the most recent 10-K
    if filings_10k:
        print(f"\n3. Downloading the most recent 10-K...")
        latest = filings_10k[0]
        path = client.download_filing(
            cik=cik,
            accession_number=latest["accession_number"],
            primary_document=latest["primary_document"],
            filing_type="10-K"
        )
        if path:
            print(f"   Saved to: {path}")

    # 4. Download Company Facts
    print(f"\n4. Downloading Company Facts...")
    facts = client.download_company_facts(cik)
    if facts:
        gaap_facts = list(facts.get("facts", {}).get("us-gaap", {}).keys())[:5]
        print(f"   Example GAAP metrics: {', '.join(gaap_facts)}")

    print(f"\n{'='*60}")
    print("Demo completed")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
