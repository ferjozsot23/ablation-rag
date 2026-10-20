"""
System 1: BM25 Only RAG

Pipeline: Query → BM25.search(top_k=8) → format_context → LLM → Answer

Lexical baseline: keyword search only.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from src.retrieval.bm25_index import BM25Index
from src.llm.llm_client import LLMClient


class System1_BM25Only:
    """
    System 1: BM25-only RAG.

    Retrieves documents with BM25 (lexical search) only and generates the
    answer with the LLM.
    """

    def __init__(
        self,
        bm25_index_path: Path,
        chunks_file: Path,
        temperature: float = 0.0,
        top_k: int = 8
    ):
        """
        Initializes System 1.

        Args:
            bm25_index_path: path to the BM25 index (without extension)
            chunks_file: path to chunks.jsonl
            temperature: LLM temperature (default: 0.0)
            top_k: number of chunks to retrieve (default: 8)
        """
        print("[System1] Loading BM25 index...")
        self.bm25 = BM25Index.load(bm25_index_path)

        print("[System1] Loading chunks...")
        self.chunks = self._load_chunks(chunks_file)

        self.top_k = top_k

        print("[System1] Initializing LLM client...")
        self.llm = LLMClient(temperature=temperature)

        print(f"[System1] Ready - BM25 index with {len(self.chunks)} chunks")

    def _load_chunks(self, chunks_file: Path) -> Dict[str, Dict]:
        """Loads chunks keyed by ID."""
        chunks = {}
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                chunks[chunk['chunk_id']] = chunk
        return chunks

    def _format_context(self, results: List[tuple]) -> str:
        """
        Formats the retrieved chunks as LLM context.

        Args:
            results: list of (chunk_id, score, metadata)

        Returns:
            Formatted string
        """
        if not results:
            return "No relevant documents found."

        context_parts = []
        context_parts.append(f"Retrieved {len(results)} relevant documents:\n")

        for i, (chunk_id, score, metadata) in enumerate(results, 1):
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                continue

            # Header
            context_parts.append(f"\n[Document {i}]")
            context_parts.append(f"Source: {metadata['company_id']} {metadata['form_type']} ({metadata['period_end']})")
            context_parts.append(f"Section: Item {metadata.get('section_item', 'N/A')}")
            context_parts.append(f"Relevance: BM25={score:.4f}\n")

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
        Answers a question with BM25 + LLM.

        Args:
            question: user question
            include_trace: whether to include the execution trace (default: False)

        Returns:
            Dict with the answer, token counts, timing and, optionally, the trace
        """
        trace = []

        # Step 1: BM25 retrieval
        trace.append(f"[Step 1] BM25 search: '{question}'")
        results = self.bm25.search(question, top_k=self.top_k)
        trace.append(f"[Step 1] Retrieved {len(results)} chunks")

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
            # Extract chunk texts for RAGAS
            chunk_texts = [self.chunks.get(cid, {}).get('text', '') for cid, _, _ in results]
            return {
                'question': question,
                'answer': None,
                'error': result['error'],
                'system': 'System1_BM25Only',
                'retrieved_chunks': len(results),
                'chunk_texts': chunk_texts,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'time_seconds': result['time_seconds'],
                'trace': trace if include_trace else None
            }

        trace.append(f"[Step 4] Generated answer ({len(result['answer'])} chars)")
        trace.append(f"[Step 4] Tokens: {result['total_tokens']} ({result['prompt_tokens']}+{result['completion_tokens']})")

        # Extract chunk texts for RAGAS
        chunk_texts = [self.chunks.get(cid, {}).get('text', '') for cid, _, _ in results]

        return {
            'question': question,
            'answer': result['answer'],
            'system': 'System1_BM25Only',
            'retrieved_chunks': len(results),
            'chunk_ids': [cid for cid, _, _ in results],
            'chunk_texts': chunk_texts,  # used by RAGAS
            'bm25_scores': [score for _, score, _ in results],
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
    """System 1 demo (run with `python -m src.systems.system1_bm25_only`)."""
    print("\n" + "="*60)
    print("SYSTEM 1: BM25 Only RAG")
    print("="*60)

    project_root = Path(__file__).parent.parent.parent
    bm25_path = project_root / "data/indexes/bm25_index"
    chunks_path = project_root / "data/chunks/chunks.jsonl"

    print("\n1. Initializing System 1...")
    system = System1_BM25Only(
        bm25_index_path=bm25_path,
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
