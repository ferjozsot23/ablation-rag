"""
Tool: get_financial_facts

Extracts structured financial data from SEC Company Facts (XBRL).
"""

import json
from pathlib import Path
from typing import List, Dict, Optional


class GetFinancialFacts:
    """
    Tool for querying structured financial data.

    Reads Company Facts JSON files (XBRL) and extracts specific metrics.
    """

    # Common aliases mapped to exact XBRL names
    METRIC_ALIASES = {
        'Revenues': 'RevenueFromContractWithCustomerExcludingAssessedTax',
        'Revenue': 'RevenueFromContractWithCustomerExcludingAssessedTax',
        'TotalRevenue': 'RevenueFromContractWithCustomerExcludingAssessedTax',
        'Sales': 'RevenueFromContractWithCustomerExcludingAssessedTax',
        'NetIncome': 'NetIncomeLoss',
        'Profit': 'NetIncomeLoss',
    }

    def __init__(self, company_facts_dir: Path):
        """
        Initializes the tool.

        Args:
            company_facts_dir: directory with the CompanyFacts JSON files
        """
        self.company_facts_dir = company_facts_dir
        self.facts_cache = {}  # cache per company
        self._load_company_facts()

    def _load_company_facts(self):
        """Loads every available Company Facts file."""
        if not self.company_facts_dir.exists():
            return

        # Manual CIK → ticker mapping (for files without tickers in the JSON)
        CIK_TO_TICKER = {
            '320193': 'AAPL',
            '0000320193': 'AAPL',
        }

        for facts_file in self.company_facts_dir.glob("CIK*.json"):
            data = json.loads(facts_file.read_text())

            ticker = None
            if 'tickers' in data and data['tickers']:
                ticker = data['tickers'][0]
            else:
                # Fall back to the CIK mapping
                cik = str(data.get('cik', ''))
                ticker = CIK_TO_TICKER.get(cik)

            if ticker:
                self.facts_cache[ticker] = data

    def get_fact(
        self,
        company: str,
        metric: str,
        period: Optional[str] = None,
        form: Optional[str] = None
    ) -> List[Dict]:
        """
        Gets a specific financial fact.

        Args:
            company: company ticker (e.g. "AAPL")
            metric: XBRL metric (e.g. "Revenues", "Assets", "NetIncomeLoss")
            period: specific period (e.g. "2023", "2023-Q1") [optional]
            form: form type (e.g. "10-K", "10-Q") [optional]

        Returns:
            List of facts matching the filters
        """
        if company not in self.facts_cache:
            return []

        company_data = self.facts_cache[company]

        # Look the metric up in us-gaap
        if 'facts' not in company_data:
            return []

        if 'us-gaap' not in company_data['facts']:
            return []

        gaap_facts = company_data['facts']['us-gaap']

        # Map the alias to the real XBRL name, if any
        actual_metric = self.METRIC_ALIASES.get(metric, metric)

        # Exact match first, then case-insensitive
        metric_data = None
        if actual_metric in gaap_facts:
            metric_data = gaap_facts[actual_metric]
        else:
            for key in gaap_facts:
                if key.lower() == actual_metric.lower():
                    metric_data = gaap_facts[key]
                    break

        if not metric_data or 'units' not in metric_data:
            return []

        # Collect the values (typically in USD)
        results = []
        for unit, values in metric_data['units'].items():
            for entry in values:
                # Filter by period if given
                if period:
                    if 'end' in entry:
                        end_date = entry['end']
                        if period not in end_date:
                            continue

                # Filter by form if given
                if form:
                    if 'form' in entry and entry['form'] != form:
                        continue

                results.append({
                    'company': company,
                    'metric': metric,
                    'value': entry.get('val'),
                    'unit': unit,
                    'start': entry.get('start'),
                    'end': entry.get('end'),
                    'fiscal_year': entry.get('fy'),
                    'fiscal_period': entry.get('fp'),
                    'form': entry.get('form'),
                    'filed': entry.get('filed')
                })

        return results

    def format_for_agent(self, facts: List[Dict]) -> str:
        """
        Formats facts for the agent.

        Args:
            facts: list of facts

        Returns:
            Formatted string
        """
        if not facts:
            return "No financial data found."

        output = []
        output.append(f"Found {len(facts)} financial records:\n")

        for i, fact in enumerate(facts, 1):
            output.append(f"\n[{i}] {fact['company']} - {fact['metric']}")
            output.append(f"    Value: {fact['value']:,} {fact['unit']}")
            output.append(f"    Period: {fact.get('start', 'N/A')} → {fact.get('end', 'N/A')}")
            output.append(f"    Fiscal Year: {fact.get('fiscal_year', 'N/A')}, Period: {fact.get('fiscal_period', 'N/A')}")
            output.append(f"    Form: {fact.get('form', 'N/A')}")

        return "\n".join(output)


# Singleton
_facts_tool = None


def get_facts_tool() -> GetFinancialFacts:
    """Returns the singleton instance."""
    global _facts_tool

    if _facts_tool is None:
        project_root = Path(__file__).parent.parent.parent
        facts_dir = project_root / "data/company_facts"
        _facts_tool = GetFinancialFacts(facts_dir)

    return _facts_tool


def get_financial_facts(
    company: str,
    metric: str,
    period: Optional[str] = None,
    form: Optional[str] = None
) -> List[Dict]:
    """
    Convenience function to get facts.

    Args:
        company: ticker
        metric: XBRL metric
        period: period [optional]
        form: form type [optional]

    Returns:
        List of facts
    """
    tool = get_facts_tool()
    return tool.get_fact(company, metric, period, form)


def main():
    """Tool demo (run with `python -m src.tools.get_financial_facts`)."""
    print("\n" + "="*60)
    print("TOOL: get_financial_facts")
    print("="*60)

    print("\n1. Initializing tool...")
    tool = get_facts_tool()
    print(f"   {len(tool.facts_cache)} companies with Company Facts")
    print(f"   Companies: {list(tool.facts_cache.keys())}")

    print("\n2. Searching: Apple revenue 2023...")
    facts = tool.get_fact(
        company="AAPL",
        metric="Revenues",
        period="2023",
        form="10-K"
    )

    if facts:
        print(f"\n3. Results ({len(facts)} records):")
        formatted = tool.format_for_agent(facts[:3])
        print(formatted)
    else:
        print("\n3. No data found.")
        print("   (Expected if no Company Facts JSON files were downloaded)")

    print("\n" + "="*60)
    print("Demo completed")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
