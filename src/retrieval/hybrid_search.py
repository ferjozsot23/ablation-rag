"""
Hybrid search with RRF (Reciprocal Rank Fusion).

Combines BM25 (lexical search) and FAISS (semantic search) and re-ranks
the results with Reciprocal Rank Fusion.
"""

import json
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

from src.retrieval.bm25_index import BM25Index
from src.retrieval.faiss_index import FAISSIndex

logger = logging.getLogger(__name__)


class HybridSearch:
    """
    Hybrid search combining BM25 and FAISS with RRF.

    RRF (Reciprocal Rank Fusion):
    score_rrf(d) = Σ 1 / (k + rank_i(d))

    where:
    - rank_i(d) is the rank of document d in system i
    - k is a constant (typically 60)
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        faiss_index: FAISSIndex,
        embedder=None,  # model used to embed queries
        rrf_k: int = 60
    ):
        """
        Initializes the hybrid search.

        Args:
            bm25_index: built BM25 index
            faiss_index: built FAISS index
            embedder: embedding model (optional, loaded lazily when None)
            rrf_k: RRF constant (typically 60)
        """
        self.bm25 = bm25_index
        self.faiss = faiss_index
        self.embedder = embedder
        self.rrf_k = rrf_k

        # Full chunk text, returned with the results when requested
        self.chunks_text = {}

        logger.info(f"HybridSearch initialized (RRF k={rrf_k})")

    def load_chunks_text(self, chunks_file: Path):
        """
        Loads the full text of the chunks.

        Args:
            chunks_file: path to chunks.jsonl
        """
        logger.info(f"Loading chunk text: {chunks_file}")

        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                self.chunks_text[chunk['chunk_id']] = chunk['text']

        logger.info(f"  {len(self.chunks_text):,} chunks loaded")

    def _load_embedder_if_needed(self):
        """Loads the embedding model if it is not loaded yet."""
        if self.embedder is None:
            logger.info("Loading BGE-M3 for query embeddings...")
            from FlagEmbedding import BGEM3FlagModel

            self.embedder = BGEM3FlagModel(
                "BAAI/bge-m3",
                use_fp16=False  # CPU mode
            )
            logger.info("Model loaded")

    def search(
        self,
        query: str,
        top_k: int = 8,
        bm25_weight: float = 1.0,
        faiss_weight: float = 1.0,
        include_text: bool = False
    ) -> List[Dict]:
        """
        Hybrid search with RRF.

        Args:
            query: query string
            top_k: number of final results
            bm25_weight: BM25 weight in RRF (default: 1.0)
            faiss_weight: FAISS weight in RRF (default: 1.0)
            include_text: whether to include the full text in the results

        Returns:
            List of results sorted by RRF score
        """
        # 1. BM25 search (top-20)
        logger.debug(f"BM25 search: {query}")
        bm25_results = self.bm25.search(query, top_k=20)
        logger.debug(f"BM25 -> {len(bm25_results)} results")

        # 2. FAISS search (top-20)
        logger.debug(f"FAISS search: {query}")

        # Embed the query
        self._load_embedder_if_needed()
        embeddings = self.embedder.encode(
            [query],
            batch_size=1,
            max_length=8192
        )['dense_vecs'][0]

        # Normalize for cosine similarity
        norm = np.linalg.norm(embeddings)
        if norm > 0:
            query_embedding = embeddings / norm
        else:
            query_embedding = embeddings
        query_embedding = query_embedding.astype('float32')

        faiss_results = self.faiss.search(query_embedding, top_k=20)
        logger.debug(f"FAISS -> {len(faiss_results)} results")

        # 3. RRF fusion
        logger.debug("RRF fusion...")
        rrf_scores = self._reciprocal_rank_fusion(
            bm25_results,
            faiss_results,
            bm25_weight,
            faiss_weight
        )

        # 4. Sort by RRF score and keep the top-k
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1]['rrf_score'],
            reverse=True
        )[:top_k]

        # 5. Final format
        results = []
        for chunk_id, data in sorted_results:
            result = {
                'chunk_id': chunk_id,
                'rrf_score': data['rrf_score'],
                'bm25_rank': data['bm25_rank'],
                'faiss_rank': data['faiss_rank'],
                'bm25_score': data.get('bm25_score'),
                'faiss_score': data.get('faiss_score'),
                'metadata': data['metadata']
            }

            if include_text and chunk_id in self.chunks_text:
                result['text'] = self.chunks_text[chunk_id]

            results.append(result)

        logger.debug(f"Returning {len(results)} results")
        return results

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple],
        faiss_results: List[Tuple],
        bm25_weight: float,
        faiss_weight: float
    ) -> Dict[str, Dict]:
        """
        Applies Reciprocal Rank Fusion.

        RRF score = bm25_weight / (k + rank_bm25) + faiss_weight / (k + rank_faiss)

        Args:
            bm25_results: BM25 results [(chunk_id, score, metadata), ...]
            faiss_results: FAISS results [(chunk_id, score, metadata), ...]
            bm25_weight: BM25 weight
            faiss_weight: FAISS weight

        Returns:
            Dict chunk_id → {rrf_score, ranks, metadata}
        """
        rrf_scores = defaultdict(lambda: {
            'rrf_score': 0.0,
            'bm25_rank': None,
            'faiss_rank': None,
            'bm25_score': None,
            'faiss_score': None,
            'metadata': None
        })

        # BM25 results
        for rank, (chunk_id, score, metadata) in enumerate(bm25_results, start=1):
            rrf_scores[chunk_id]['rrf_score'] += bm25_weight / (self.rrf_k + rank)
            rrf_scores[chunk_id]['bm25_rank'] = rank
            rrf_scores[chunk_id]['bm25_score'] = score
            rrf_scores[chunk_id]['metadata'] = metadata

        # FAISS results
        for rank, (chunk_id, score, metadata) in enumerate(faiss_results, start=1):
            rrf_scores[chunk_id]['rrf_score'] += faiss_weight / (self.rrf_k + rank)
            rrf_scores[chunk_id]['faiss_rank'] = rank
            rrf_scores[chunk_id]['faiss_score'] = score

            # No BM25 metadata for this chunk: use the FAISS one
            if rrf_scores[chunk_id]['metadata'] is None:
                rrf_scores[chunk_id]['metadata'] = metadata

        return dict(rrf_scores)

    def save_config(self, output_path: Path):
        """
        Saves the hybrid search configuration.

        Args:
            output_path: path to save the config to
        """
        config = {
            'rrf_k': self.rrf_k,
            'bm25_params': {
                'k1': self.bm25.k1,
                'b': self.bm25.b
            },
            'faiss_dimension': self.faiss.dimension,
            'n_chunks': len(self.chunks_text)
        }

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Configuration saved: {output_path}")


def main():
    """Hybrid search demo (run from the project root: `python -m src.retrieval.hybrid_search`)."""
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*60)
    print("Hybrid Search (BM25 + FAISS + RRF) - Demo")
    print("="*60)

    chunks_file = Path("data/chunks/chunks.jsonl")
    bm25_path = Path("data/indexes/bm25_index")
    faiss_path = Path("data/indexes/faiss_index")

    print("\n1. Loading indexes...")
    print("   BM25...")
    bm25 = BM25Index.load(bm25_path)

    print("   FAISS...")
    faiss_idx = FAISSIndex.load(faiss_path)

    print("\n2. Initializing hybrid search...")
    hybrid = HybridSearch(
        bm25_index=bm25,
        faiss_index=faiss_idx,
        rrf_k=60
    )

    hybrid.load_chunks_text(chunks_file)

    print("\n3. Search tests...")
    test_queries = [
        "What was Apple's revenue in fiscal 2023?",
        "How much did NVIDIA's GPU sales grow?",
        "Microsoft's cloud computing performance"
    ]

    for query in test_queries:
        print(f"\n   Query: '{query}'")
        print("   " + "-"*56)

        results = hybrid.search(query, top_k=5, include_text=False)

        for i, result in enumerate(results, 1):
            meta = result['metadata']
            company = meta['company_id']
            form = meta['form_type']
            period = meta['period_end']

            rrf = result['rrf_score']
            bm25_r = result['bm25_rank'] or '-'
            faiss_r = result['faiss_rank'] or '-'

            print(f"   {i}. [{company} {form} {period}]")
            print(f"      RRF={rrf:.4f} | BM25 rank={bm25_r} | FAISS rank={faiss_r}")

    # Comparison: BM25 only vs FAISS only vs hybrid
    print("\n" + "="*60)
    print("4. Method comparison")
    print("="*60)

    query = "Apple iPhone revenue growth 2023"
    print(f"\nQuery: '{query}'\n")

    print("BM25 only (top-3):")
    bm25_results = bm25.search(query, top_k=3)
    for i, (cid, score, meta) in enumerate(bm25_results, 1):
        print(f"  {i}. [{meta['company_id']} {meta['form_type']}] score={score:.2f}")

    print("\nFAISS only (top-3):")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3", device="cpu", trust_remote_code=True)
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    faiss_results = faiss_idx.search(q_emb, top_k=3)
    for i, (cid, score, meta) in enumerate(faiss_results, 1):
        print(f"  {i}. [{meta['company_id']} {meta['form_type']}] similarity={score:.4f}")

    print("\nHybrid (RRF, top-3):")
    hybrid_results = hybrid.search(query, top_k=3)
    for i, result in enumerate(hybrid_results, 1):
        meta = result['metadata']
        print(f"  {i}. [{meta['company_id']} {meta['form_type']}] rrf_score={result['rrf_score']:.4f}")

    print("\n5. Saving configuration...")
    hybrid.save_config(Path("data/indexes/hybrid_config.json"))

    print("\n" + "="*60)
    print("Demo completed")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
