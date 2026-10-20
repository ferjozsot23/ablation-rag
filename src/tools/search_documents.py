"""
Tool: search_documents

Searches relevant documents with hybrid retrieval + knowledge graph.
Used by System 5: combines BM25 + FAISS + RRF + graph expansion.
"""

from pathlib import Path
from typing import List, Dict, Optional

from src.retrieval.bm25_index import BM25Index
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.hybrid_search import HybridSearch
from src.knowledge_graph.graph_search import graph_search_pipeline
from src.knowledge_graph.graph_builder import load_graph


class SearchDocuments:
    """
    Document search tool.

    Uses hybrid retrieval to find relevant chunks.
    """

    def __init__(
        self,
        bm25_path: Path,
        faiss_path: Path,
        chunks_path: Path,
        graph_path: Optional[Path] = None
    ):
        """
        Initializes the search tool.

        Args:
            bm25_path: path to the BM25 index (without extension)
            faiss_path: path to the FAISS index (without extension)
            chunks_path: path to chunks.jsonl
            graph_path: path to the knowledge graph .pkl (optional)
        """
        self.bm25 = BM25Index.load(bm25_path)
        self.faiss = FAISSIndex.load(faiss_path)

        self.hybrid = HybridSearch(
            bm25_index=self.bm25,
            faiss_index=self.faiss,
            embedder=None,  # loaded lazily
            rrf_k=60
        )

        self.hybrid.load_chunks_text(chunks_path)

        # Knowledge graph (optional)
        self.graph = None
        if graph_path and graph_path.exists():
            self.graph = load_graph(graph_path)

    def search(
        self,
        query: str,
        top_k: int = 8,
        include_text: bool = True
    ) -> Dict:
        """
        Searches relevant documents with Hybrid + Graph.

        Args:
            query: query string
            top_k: number of results (default: 8)
            include_text: whether to include the full text (default: True)

        Returns:
            Dict with chunks and graph_context (when available)
        """
        # 1. Hybrid search (BM25 + Vector + RRF)
        chunks = self.hybrid.search(
            query=query,
            top_k=top_k,
            include_text=include_text
        )

        # 2. Graph search (when available)
        graph_context = None
        if self.graph:
            graph_results = graph_search_pipeline(
                query,
                self.graph,
                top_k_entities=5,
                expansion_depth=2,
                max_expanded_nodes=10
            )

            graph_lines = []
            if graph_results['found_entities']:
                graph_lines.append("Entities: " + ", ".join([
                    f"{e[0]} ({e[2]})"
                    for e in graph_results['found_entities'][:5]
                ]))

            graph_context = "\n".join(graph_lines) if graph_lines else None

        return {
            'chunks': chunks,
            'graph_context': graph_context
        }

    def format_results_for_agent(
        self,
        results: List[Dict],
        graph_context: Optional[str] = None,
        include_full_text: bool = False
    ) -> str:
        """
        Formats the results for the LLM agent.

        Args:
            results: search results
            graph_context: knowledge graph context (optional)
            include_full_text: whether to include the full text or only a preview

        Returns:
            String formatted for the LLM
        """
        if not results:
            return "No relevant documents found."

        output = []
        output.append(f"Found {len(results)} relevant documents:\n")

        for i, result in enumerate(results, 1):
            meta = result['metadata']

            output.append(f"\n[{i}] {meta['company_id']} {meta['form_type']} ({meta['period_end']})")

            if 'text' in result:
                text = result['text']
                # 800-character preview for agents
                preview = text[:800] + "..." if len(text) > 800 else text
                output.append(f"    {preview}\n")

        if graph_context:
            output.append(f"\nKnowledge Graph Context:\n{graph_context}")

        return "\n".join(output)


# Global singleton to avoid reloading the indexes
_search_tool = None


def get_search_tool() -> SearchDocuments:
    """Returns the singleton SearchDocuments instance."""
    global _search_tool

    if _search_tool is None:
        # Paths relative to the project
        project_root = Path(__file__).parent.parent.parent

        bm25_path = project_root / "data/indexes/bm25_index"
        faiss_path = project_root / "data/indexes/faiss_index"
        chunks_path = project_root / "data/chunks/chunks.jsonl"
        graph_path = project_root / "data/knowledge_graph/knowledge_graph.pkl"

        _search_tool = SearchDocuments(
            bm25_path=bm25_path,
            faiss_path=faiss_path,
            chunks_path=chunks_path,
            graph_path=graph_path
        )

    return _search_tool


def search_documents(
    query: str,
    top_k: int = 8
) -> str:
    """
    Convenience function for search (used by agents).

    Args:
        query: query string
        top_k: number of results

    Returns:
        Formatted string with the results
    """
    tool = get_search_tool()
    result = tool.search(query, top_k=top_k, include_text=True)
    return tool.format_results_for_agent(result['chunks'], result['graph_context'])


def main():
    """Tool demo (run with `python -m src.tools.search_documents`)."""
    print("\n" + "="*60)
    print("TOOL: search_documents")
    print("="*60)

    print("\n1. Initializing tool...")
    tool = get_search_tool()
    print("   Indexes loaded")

    query = "What was Apple's revenue in 2023?"

    print(f"\n2. Search: '{query}'")
    results = tool.search(query, top_k=5, include_text=False)

    print(f"\n3. Results ({len(results['chunks'])} chunks):")
    formatted = tool.format_results_for_agent(results['chunks'], results['graph_context'])
    print(formatted)

    print("\n" + "="*60)
    print("Demo completed")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
