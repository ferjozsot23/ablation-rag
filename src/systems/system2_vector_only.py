"""
System 2: Vector Only RAG

Pipeline: Query → BGE-M3.encode → FAISS.search(top_k=8) → format_context → LLM → Answer

Semantic baseline: vector search only (cosine similarity).
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.retrieval.device import default_device
from src.retrieval.faiss_index import FAISSIndex
from src.llm.llm_client import LLMClient

# BGE-M3 embedder
from FlagEmbedding import BGEM3FlagModel


class System2_VectorOnly:
    """
    System 2: Vector-only RAG.

    Retrieves documents with FAISS (semantic search) only and generates the
    answer with the LLM.
    """

    def __init__(
        self,
        faiss_index_path: Path,
        chunks_file: Path,
        embedder_model: str = "BAAI/bge-m3",
        temperature: float = 0.0,
        top_k: int = 8,
        device: Optional[str] = None
    ):
        """
        Initializes System 2.

        Args:
            faiss_index_path: path to the FAISS index (without extension)
            chunks_file: path to chunks.jsonl
            embedder_model: BGE-M3 model (default: "BAAI/bge-m3")
            temperature: LLM temperature (default: 0.0)
            top_k: number of chunks to retrieve (default: 8)
            device: torch device for the embedder (default: env EMBEDDER_DEVICE,
                then "cuda" if available, otherwise "cpu")
        """
        print("[System2] Loading FAISS index...")
        self.faiss = FAISSIndex.load(faiss_index_path)

        print("[System2] Loading chunks...")
        self.chunks = self._load_chunks(chunks_file)

        device = device or default_device()

        print(f"[System2] Loading embedder: {embedder_model} on {device}...")
        self.embedder = BGEM3FlagModel(
            embedder_model,
            use_fp16=device.startswith("cuda"),
            device=device
        )

        self.top_k = top_k

        print("[System2] Initializing LLM client...")
        self.llm = LLMClient(temperature=temperature)

        print(f"[System2] Ready - FAISS index with {len(self.chunks)} chunks")

    def _load_chunks(self, chunks_file: Path) -> Dict[str, Dict]:
        """Loads chunks keyed by ID."""
        chunks = {}
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                chunks[chunk['chunk_id']] = chunk
        return chunks

    def _encode_query(self, query: str) -> np.ndarray:
        """
        Embeds the query with BGE-M3.

        Args:
            query: query string

        Returns:
            Normalized embedding (1024 dims)
        """
        embedding = self.embedder.encode(
            [query],
            batch_size=1,
            max_length=8192
        )['dense_vecs'][0]

        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype('float32')

    def _format_context(self, results: List[tuple]) -> str:
        """
        Formats the retrieved chunks as LLM context.

        Args:
            results: list of (chunk_id, similarity, metadata)

        Returns:
            Formatted string
        """
        if not results:
            return "No relevant documents found."

        context_parts = []
        context_parts.append(f"Retrieved {len(results)} relevant documents:\n")

        for i, (chunk_id, similarity, metadata) in enumerate(results, 1):
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                continue

            # Header
            context_parts.append(f"\n[Document {i}]")
            context_parts.append(f"Source: {metadata['company_id']} {metadata['form_type']} ({metadata['period_end']})")
            context_parts.append(f"Section: Item {metadata.get('section_item', 'N/A')}")
            context_parts.append(f"Relevance: Similarity={similarity:.4f}\n")

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
        Answers a question with FAISS + LLM.

        Args:
            question: user question
            include_trace: whether to include the execution trace (default: False)

        Returns:
            Dict with the answer, token counts, timing and, optionally, the trace
        """
        trace = []

        # Step 1: Encode query
        trace.append(f"[Step 1] Encoding query: '{question}'")
        query_embedding = self._encode_query(question)
        trace.append(f"[Step 1] Generated embedding (1024 dims)")

        # Step 2: FAISS search
        trace.append(f"[Step 2] FAISS search (top_k={self.top_k})")
        results = self.faiss.search(query_embedding, top_k=self.top_k)
        trace.append(f"[Step 2] Retrieved {len(results)} chunks")

        # Step 3: Format context
        context = self._format_context(results)
        trace.append(f"[Step 3] Formatted context ({len(context)} chars)")

        # Step 4: Create prompt
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

        trace.append(f"[Step 4] Created prompt ({len(prompt)} chars)")

        # Step 5: LLM generation
        trace.append("[Step 5] Calling LLM...")
        result = self.llm.generate(
            prompt=prompt,
            system_message=system_message
        )

        if result.get('error'):
            trace.append(f"[Step 5] ERROR: {result['error']}")
            return {
                'question': question,
                'answer': None,
                'error': result['error'],
                'system': 'System2_VectorOnly',
                'retrieved_chunks': len(results),
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'time_seconds': result['time_seconds'],
                'trace': trace if include_trace else None
            }

        trace.append(f"[Step 5] Generated answer ({len(result['answer'])} chars)")
        trace.append(f"[Step 5] Tokens: {result['total_tokens']} ({result['prompt_tokens']}+{result['completion_tokens']})")

        # Extract chunk texts for RAGAS
        chunk_texts = [self.chunks.get(cid, {}).get('text', '') for cid, _, _ in results]

        return {
            'question': question,
            'answer': result['answer'],
            'system': 'System2_VectorOnly',
            'retrieved_chunks': len(results),
            'chunk_ids': [cid for cid, _, _ in results],
            'chunk_texts': chunk_texts,  # used by RAGAS
            'similarities': [sim for _, sim, _ in results],
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
    """System 2 demo (run with `python -m src.systems.system2_vector_only`)."""
    print("\n" + "="*60)
    print("SYSTEM 2: Vector Only RAG")
    print("="*60)

    project_root = Path(__file__).parent.parent.parent
    faiss_path = project_root / "data/indexes/faiss_index"
    chunks_path = project_root / "data/chunks/chunks.jsonl"

    print("\n1. Initializing System 2...")
    system = System2_VectorOnly(
        faiss_index_path=faiss_path,
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
