"""
Agent Tools: LangChain-compatible tools for System 5

All tools use the @tool decorator so they can be bound to a LangGraph agent.
"""

from pathlib import Path
from langchain.tools import tool

# Import underlying implementations
from src.tools.search_documents import search_documents as _search_docs
from src.tools.calculate import Calculate
from src.tools.get_financial_facts import GetFinancialFacts


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

@tool
def search(query: str) -> str:
    """Search SEC documents using hybrid retrieval + knowledge graph.

    Use this for:
    - Finding financial data in 10-K/10-Q documents
    - Getting context about company performance
    - Exploring detailed financial information

    Args:
        query: Natural language search query about financial data

    Returns:
        Relevant document excerpts with metadata
    """
    return _search_docs(query, top_k=15)


@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions safely.

    Use this for:
    - Growth rates: ((new - old) / old) * 100
    - Ratios and percentages
    - Basic arithmetic: +, -, *, /, **

    Args:
        expression: Math expression like "((383.3 - 394.3) / 394.3) * 100"

    Returns:
        Calculated result or error message
    """
    calc = Calculate()
    result = calc.calculate(expression)

    if result['error']:
        return f"Error: {result['error']}"

    return f"Result: {result['result']}"


@tool
def get_financial_facts(company: str, metric: str = "Revenues") -> str:
    """Get structured XBRL financial data from SEC Company Facts.

    Use this for:
    - Precise financial metrics (Revenues, NetIncome, Assets, etc.)
    - Historical data across multiple periods
    - Structured data that doesn't require document search

    Available companies: AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, etc.
    Common metrics: Revenues, NetIncome, Assets, Liabilities, CashAndCashEquivalents

    Args:
        company: Stock ticker (e.g., "AAPL")
        metric: XBRL metric name (e.g., "Revenues")

    Returns:
        JSON with recent financial data or error message
    """
    import json

    # agent_tools.py is in src/tools/, so the project root is 2 levels up
    project_root = Path(__file__).parent.parent.parent

    facts_tool = GetFinancialFacts(project_root / "data/company_facts")
    facts = facts_tool.get_fact(company, metric)

    if not facts:
        return f"No facts found for {company} - {metric}"

    # Return the last 3 periods
    recent = facts[-3:]
    return json.dumps(recent, indent=2)


# ============================================================================
# TOOLS LIST (single source of truth)
# ============================================================================

TOOLS = [
    search,
    calculator,
    get_financial_facts
]
