"""
System 4: Hybrid + Graph RAG

Architecture:
1. Hybrid retrieval: BM25 + Vector + RRF → top-K chunks
2. Graph search: query → relevant entities → expand context → subgraph
3. Context fusion: chunks + graph knowledge
4. LLM generation: answer from the enriched context

Expected behavior:
- Better recall than Systems 1, 2 and 3 (graph expansion)
- More contextualized answers (relationships between entities)
- Useful for multi-hop queries (who sold what to whom?)
"""

import time
import json
from pathlib import Path
from typing import Dict, List

from src.llm.llm_client import LLMClient, glm_endpoint
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.bm25_index import BM25Index
from src.retrieval.faiss_index import FAISSIndex
from src.knowledge_graph.graph_search import graph_search_pipeline
from src.knowledge_graph.graph_builder import load_graph


class System4_HybridGraph:
    """
    System 4: Hybrid Search + Graph Expansion + LLM.

    Pipeline:
    - Query → Hybrid Search (BM25+Vector+RRF) → chunks
    - Query → Graph Search → entities → expand → subgraph
    - Fusion: chunks + graph context → prompt
    - LLM → answer
    """

    def __init__(
        self,
        chunks_file: Path,
        graph_file: Path,
        temperature: float = 0.0,
        top_k: int = 8,
        graph_top_k: int = 5,
        graph_expansion_depth: int = 2,
        graph_max_nodes: int = 30
    ):
        """
        Initializes System 4.

        Args:
            chunks_file: path to chunks.jsonl
            graph_file: path to knowledge_graph.pkl
            temperature: LLM temperature
            top_k: top-K chunks from the hybrid search
            graph_top_k: top-K entities from the graph search
            graph_expansion_depth: expansion depth in the graph
            graph_max_nodes: maximum nodes in the expanded subgraph
        """
        print("\n[System 4: Hybrid + Graph RAG]")
        print("-" * 60)

        # IMPORTANT: load BGE-M3 FIRST (before BM25/FAISS)
        print("  Pre-loading BGE-M3 embedder...")
        from FlagEmbedding import BGEM3FlagModel
        embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
        print("    BGE-M3 loaded")

        print("  Loading BM25 and FAISS indexes...")
        project_root = chunks_file.parent.parent
        bm25_path = project_root / "indexes/bm25_index"
        faiss_path = project_root / "indexes/faiss_index"

        bm25 = BM25Index.load(bm25_path)
        faiss = FAISSIndex.load(faiss_path)

        print("  Loading chunks...")
        self.chunks = self._load_chunks(chunks_file)

        print("  Initializing Hybrid Search (BM25 + Vector + RRF)...")
        self.hybrid_search = HybridSearch(
            bm25_index=bm25,
            faiss_index=faiss,
            embedder=embedder,
            rrf_k=60
        )
        self.hybrid_search.load_chunks_text(chunks_file)

        print("  Loading Knowledge Graph...")
        self.graph = load_graph(graph_file)
        print(f"    Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

        # LLM client: always the self-hosted endpoint configured through GLM_* variables
        print("  Initializing LLM client (GLM endpoint)...")
        self.llm = LLMClient(
            **glm_endpoint(),
            temperature=temperature
        )

        self.top_k = top_k
        self.graph_top_k = graph_top_k
        self.graph_expansion_depth = graph_expansion_depth
        self.graph_max_nodes = graph_max_nodes

        print(f"  Config: top_k={top_k}, graph_entities={graph_top_k}, expansion_depth={graph_expansion_depth}, max_nodes={graph_max_nodes}")

        print("-" * 60 + "\n")

    def _load_chunks(self, chunks_file: Path) -> Dict[str, Dict]:
        """Loads chunks keyed by ID."""
        chunks = {}
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                chunks[chunk['chunk_id']] = chunk
        return chunks

    def answer(
        self,
        query: str,
        include_trace: bool = False
    ) -> Dict:
        """
        Answers a question with Hybrid + Graph RAG.

        Args:
            query: user question
            include_trace: whether to include the detailed trace

        Returns:
            Dict with the answer, metadata and, optionally, the trace
        """
        start_time = time.time()

        # 1. Hybrid retrieval
        retrieved_chunks = self.hybrid_search.search(query, top_k=self.top_k)

        # 2. Graph search
        graph_results = graph_search_pipeline(
            query,
            self.graph,
            top_k_entities=self.graph_top_k,
            expansion_depth=self.graph_expansion_depth,
            max_expanded_nodes=self.graph_max_nodes
        )

        # 3. Context fusion
        context = self._build_context(retrieved_chunks, graph_results)

        # 4. LLM generation
        prompt = self._build_prompt(query, context)

        llm_result = self.llm.generate(
            prompt=prompt,
            system_message="You are a financial analyst. Answer questions based on SEC filings data."
        )

        if llm_result.get('error'):
            return {
                'error': llm_result['error'],
                'query': query
            }

        end_time = time.time()
        result = {
            'query': query,
            'answer': llm_result['answer'],
            'retrieved_chunks': len(retrieved_chunks),
            'graph_entities_found': len(graph_results['found_entities']),
            'graph_entities_expanded': len(graph_results['expanded_entities']),
            'graph_subgraph_nodes': graph_results['subgraph_nodes'],
            'graph_subgraph_edges': graph_results['subgraph_edges'],
            'total_tokens': llm_result.get('total_tokens', 0),
            'prompt_tokens': llm_result.get('prompt_tokens', 0),
            'completion_tokens': llm_result.get('completion_tokens', 0),
            'time_seconds': end_time - start_time
        }

        if include_trace:
            result['trace'] = {
                'hybrid_results': retrieved_chunks,
                'graph_results': {
                    'found_entities': graph_results['found_entities'],
                    'expanded_entities': graph_results['expanded_entities'][:10],  # first 10
                    'subgraph_nodes': graph_results['subgraph_nodes'],
                    'subgraph_edges': graph_results['subgraph_edges']
                },
                'context': context,
                'prompt': prompt
            }

        return result

    def _build_context(
        self,
        chunks: List[Dict],
        graph_results: Dict
    ) -> Dict:
        """
        Fuses chunk context and graph context.

        Args:
            chunks: chunks retrieved by the hybrid search
            graph_results: graph search results

        Returns:
            Dict with the fused context
        """
        # Chunk context
        chunk_texts = []
        for i, result in enumerate(chunks, 1):
            chunk_id = result['chunk_id']
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                continue

            meta = result.get('metadata', {})
            company = meta.get('company_id', 'UNKNOWN')
            form = meta.get('form_type', 'UNKNOWN')

            chunk_texts.append(
                f"[Chunk {i}] {company} {form}\n{chunk['text'][:1000]}"  # first 1000 chars of each chunk
            )

        # Graph context
        graph_context = []

        # Entities found
        if graph_results['found_entities']:
            graph_context.append("ENTITIES FOUND:")
            for entity, similarity, entity_type in graph_results['found_entities'][:5]:
                graph_context.append(f"  - {entity} ({entity_type})")

        # Subgraph relations (only the first 5, to keep the prompt short)
        if graph_results.get('subgraph'):
            subgraph = graph_results['subgraph']
            graph_context.append(f"\nRELATED ENTITIES ({subgraph.number_of_nodes()} total):")

            edge_count = 0
            for u, v, data in subgraph.edges(data=True):
                if edge_count >= 5:
                    break
                relacion = data.get('relacion', 'RELATED_TO')
                graph_context.append(f"  {u} --[{relacion}]--> {v}")
                edge_count += 1

        return {
            'chunks': chunk_texts,
            'graph': graph_context
        }

    def _build_prompt(self, query: str, context: Dict) -> str:
        """Builds the LLM prompt.

        Chunks come first (main source), the graph context follows (complement).
        """

        chunks_text = "\n\n".join(context['chunks'])

        # Graph context (structured complement, only when it has real content)
        graph_section = ""
        if context['graph'] and len(context['graph']) > 2:
            graph_text = "\n".join(context['graph'])
            graph_section = f"""

ADDITIONAL CONTEXT (Entity Relationships):
{graph_text}"""

        prompt = f"""Answer the following question based on the provided SEC filing excerpts.

DOCUMENT EXCERPTS:
{chunks_text}{graph_section}

QUESTION: {query}

INSTRUCTIONS:
1. Answer based on the document excerpts above
2. Be precise and cite specific numbers/facts when available
3. If the answer is partially in the documents, provide what you can find
4. Only say "I don't have enough information" if the documents contain NO relevant information

ANSWER:"""

        return prompt


def main():
    """Tests System 4 with the same question used for Systems 1-3."""
    print("\n" + "="*80)
    print("SYSTEM 4: Hybrid + Graph RAG - TEST")
    print("="*80)

    project_root = Path(__file__).parent.parent.parent
    chunks_file = project_root / "data/chunks/chunks.jsonl"
    graph_file = project_root / "data/knowledge_graph/knowledge_graph.pkl"

    query = "What was Apple's total revenue in fiscal year 2023?"

    print(f"\nQuestion: {query}")
    print("="*80)

    system4 = System4_HybridGraph(
        chunks_file=chunks_file,
        graph_file=graph_file,
        temperature=0.0,
        top_k=8,
        graph_top_k=5,
        graph_expansion_depth=2,
        graph_max_nodes=10  # smaller subgraph to reduce noise
    )

    result = system4.answer(query, include_trace=False)

    if result.get('error'):
        print(f"\nERROR: {result['error']}")
    else:
        print(f"\nAnswer: {result['answer'][:300]}...")
        print(f"\nMetrics:")
        print(f"  Retrieved chunks: {result['retrieved_chunks']}")
        print(f"  Graph entities found: {result['graph_entities_found']}")
        print(f"  Graph entities expanded: {result['graph_entities_expanded']}")
        print(f"  Graph subgraph: {result['graph_subgraph_nodes']} nodes, {result['graph_subgraph_edges']} edges")
        print(f"  Tokens: {result['total_tokens']} ({result['prompt_tokens']}+{result['completion_tokens']})")
        print(f"  Time: {result['time_seconds']:.2f}s")

        # Cost estimate (gpt-4o-mini pricing)
        input_cost = (result['prompt_tokens'] / 1_000_000) * 0.15
        output_cost = (result['completion_tokens'] / 1_000_000) * 0.60
        total_cost = input_cost + output_cost
        print(f"  Cost: ${total_cost:.6f}")

    print("\n" + "="*80)
    print(" System 4 test completed")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
