"""
System 3: Hybrid RAG (BM25 + Vector + RRF)

Pipeline: Query → HybridSearch.search(top_k=8) → format_context → LLM → Answer

Reciprocal Rank Fusion (RRF) of BM25 and FAISS results.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from src.retrieval.device import default_device
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.bm25_index import BM25Index
from src.retrieval.faiss_index import FAISSIndex
from src.llm.llm_client import LLMClient


class System3_Hybrid:
    """
    System 3: Hybrid RAG (BM25 + FAISS + RRF).

    Retrieves documents with hybrid search (RRF fusion) and generates the
    answer with the LLM.
    """

    def __init__(
        self,
        chunks_file: Path,
        temperature: float = 0.0,
        top_k: int = 8,
        rrf_k: int = 60
    ):
        """
        Initializes System 3.

        Args:
            chunks_file: path to chunks.jsonl
            temperature: LLM temperature (default: 0.0)
            top_k: number of chunks to retrieve (default: 8)
            rrf_k: RRF k parameter (default: 60)
        """
        # IMPORTANT: load BGE-M3 FIRST (before BM25/FAISS).
        # BGE-M3 uses multiprocessing (loky), which conflicts if loaded afterwards.
        device = default_device()
        print(f"[System3] Pre-loading BGE-M3 embedder on {device}...")
        from FlagEmbedding import BGEM3FlagModel
        embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=device.startswith("cuda"), device=device)
        print("[System3] BGE-M3 loaded")

        print("[System3] Loading chunks...")
        self.chunks = self._load_chunks(chunks_file)

        # Load BM25 and FAISS indexes (AFTER BGE-M3)
        print("[System3] Loading BM25 and FAISS indexes...")
        project_root = Path(__file__).parent.parent.parent
        bm25_path = project_root / "data/indexes/bm25_index"
        faiss_path = project_root / "data/indexes/faiss_index"

        bm25 = BM25Index.load(bm25_path)
        faiss = FAISSIndex.load(faiss_path)

        # Build the hybrid search with the already-loaded embedder
        print("[System3] Initializing hybrid search...")
        self.hybrid = HybridSearch(
            bm25_index=bm25,
            faiss_index=faiss,
            embedder=embedder,
            rrf_k=rrf_k
        )

        self.hybrid.load_chunks_text(chunks_file)

        self.top_k = top_k

        print("[System3] Initializing LLM client...")
        self.llm = LLMClient(temperature=temperature)

        print(f"[System3] Ready - Hybrid search with {len(self.chunks)} chunks")

    def _load_chunks(self, chunks_file: Path) -> Dict[str, Dict]:
        """Loads chunks keyed by ID."""
        chunks = {}
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                chunks[chunk['chunk_id']] = chunk
        return chunks

    def _format_context(self, results: List[Dict]) -> str:
        """
        Formats the retrieved chunks as LLM context.

        Args:
            results: list of dicts with chunk_id, metadata, rrf_score, etc.

        Returns:
            Formatted string
        """
        if not results:
            return "No relevant documents found."

        context_parts = []
        context_parts.append(f"Retrieved {len(results)} relevant documents:\n")

        for i, result in enumerate(results, 1):
            chunk_id = result['chunk_id']
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                continue

            metadata = result['metadata']

            # Header
            context_parts.append(f"\n[Document {i}]")
            context_parts.append(f"Source: {metadata['company_id']} {metadata['form_type']} ({metadata['period_end']})")
            context_parts.append(f"Section: Item {metadata.get('section_item', 'N/A')}")
            context_parts.append(f"Relevance: RRF={result['rrf_score']:.4f}\n")

            # Text
            context_parts.append(chunk['text'])
            context_parts.append("\n" + "-"*80)

        return "\n".join(context_parts)

    def answer(
        self,
        question: str,
        include_trace: bool = False
    ) -> Dict[str, Any]:
        """
        Answers a question with hybrid search + LLM.

        Args:
            question: user question
            include_trace: whether to include the execution trace (default: False)

        Returns:
            Dict with the answer, token counts, timing and, optionally, the trace
        """
        trace = []

        # Step 1: Hybrid search (BM25 + FAISS + RRF)
        trace.append(f"[Step 1] Hybrid search: '{question}'")
        results = self.hybrid.search(
            query=question,
            top_k=self.top_k,
            include_text=False  # chunks are already loaded
        )
        trace.append(f"[Step 1] Retrieved {len(results)} chunks via RRF")

        # Step 2: Format context
        context = self._format_context(results)
        trace.append(f"[Step 2] Formatted context ({len(context)} chars)")

        # Step 3: Create prompt
        system_message = """You are a financial analyst assistant. Answer questions based ONLY on the provided documents.

Rules:
- Cite sources with [Document N] references
- If information is not in the documents, say so
- For numerical answers, include units and time periods
- Be concise and factual"""

        prompt = f"""Based on the following documents, answer the question.

{context}

Question: {question}

Answer:"""

        trace.append(f"[Step 3] Created prompt ({len(prompt)} chars)")

        # Step 4: LLM generation
        trace.append("[Step 4] Calling LLM...")
        result = self.llm.generate(
            prompt=prompt,
            system_message=system_message
        )

        if result.get('error'):
            trace.append(f"[Step 4] ERROR: {result['error']}")
            return {
                'question': question,
                'answer': None,
                'error': result['error'],
                'system': 'System3_Hybrid',
                'retrieved_chunks': len(results),
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'time_seconds': result['time_seconds'],
                'trace': trace if include_trace else None
            }

        trace.append(f"[Step 4] Generated answer ({len(result['answer'])} chars)")
        trace.append(f"[Step 4] Tokens: {result['total_tokens']} ({result['prompt_tokens']}+{result['completion_tokens']})")

        # Extract chunk texts for RAGAS
        chunk_texts = [self.chunks.get(r['chunk_id'], {}).get('text', '') for r in results]

        return {
            'question': question,
            'answer': result['answer'],
            'system': 'System3_Hybrid',
            'retrieved_chunks': len(results),
            'chunk_ids': [r['chunk_id'] for r in results],
            'chunk_texts': chunk_texts,  # used by RAGAS
            'rrf_scores': [r['rrf_score'] for r in results],
            'prompt_tokens': result['prompt_tokens'],
            'completion_tokens': result['completion_tokens'],
            'total_tokens': result['total_tokens'],
            'reasoning_tokens': result.get('reasoning_tokens'),
            'llm_calls': result.get('llm_calls', 1),
            'time_seconds': result['time_seconds'],
            'model': result['model'],
            'trace': trace if include_trace else None
        }


def main():
    """System 3 demo (run with `python -m src.systems.system3_hybrid`)."""
    print("\n" + "="*60)
    print("SYSTEM 3: Hybrid RAG (BM25 + Vector + RRF)")
    print("="*60)

    project_root = Path(__file__).parent.parent.parent
    chunks_path = project_root / "data/chunks/chunks.jsonl"

    print("\n1. Initializing System 3...")
    system = System3_Hybrid(
        chunks_file=chunks_path,
        temperature=0.0,
        top_k=8
    )

    question = "What was Apple's total revenue in fiscal year 2023?"

    print(f"\n2. Question: '{question}'")
    print("-" * 60)

    result = system.answer(question, include_trace=True)

    if result.get('error'):
        print(f"\nERROR: {result['error']}")
    else:
        print(f"\n3. Answer:")
        print(result['answer'])
        print(f"\n4. Metadata:")
        print(f"   Retrieved: {result['retrieved_chunks']} chunks")
        print(f"   Tokens: {result['total_tokens']} ({result['prompt_tokens']}+{result['completion_tokens']})")
        print(f"   Time: {result['time_seconds']:.2f}s")
        print(f"   Model: {result['model']}")

        if result.get('trace'):
            print(f"\n5. Trace:")
            for step in result['trace']:
                print(f"   {step}")

    print("\n" + "="*60)
    print("Demo completed")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
