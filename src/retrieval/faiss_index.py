"""
FAISS vector index for semantic search.

Uses precomputed BGE-M3 embeddings and FAISS for efficient
nearest-neighbor search.
"""

import json
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    FAISS index for semantic vector search.

    Uses a flat inner-product index (exact nearest-neighbor search). With
    L2-normalized vectors the inner product equals cosine similarity.
    """

    def __init__(self, dimension: int = 1024):
        """
        Initializes the FAISS index.

        Args:
            dimension: embedding dimension (1024 for BGE-M3)
        """
        self.dimension = dimension
        self.index = None
        self.chunk_ids = []
        self.chunks_metadata = []

        logger.info(f"FAISSIndex initialized (dim={dimension})")

    def build(
        self,
        embeddings_file: Path,
        index_file: Path
    ):
        """
        Builds the FAISS index from precomputed embeddings.

        Args:
            embeddings_file: path to embeddings.npz
            index_file: path to index.jsonl (chunk_id → index mapping)
        """
        import faiss

        logger.info(f"Building FAISS index...")
        logger.info(f"  Embeddings: {embeddings_file}")
        logger.info(f"  Index map: {index_file}")

        # Load embeddings
        logger.info("  Loading embeddings...")
        data = np.load(embeddings_file)
        embeddings = data['embeddings']

        n_vectors, dim = embeddings.shape
        logger.info(f"  {n_vectors:,} vectors of {dim} dimensions")

        if dim != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, but embeddings are {dim}D")

        # Check normalization (embeddings must be L2-normalized)
        norms = np.linalg.norm(embeddings[:5], axis=1)
        logger.info(f"  Sample norms: {norms}")

        if not np.allclose(norms, 1.0, atol=0.01):
            logger.warning("  Embeddings are not normalized, normalizing...")
            faiss.normalize_L2(embeddings)

        # Load the index → chunk_id mapping
        logger.info("  Loading index mapping...")
        self.chunk_ids = []
        self.chunks_metadata = []

        with open(index_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                self.chunk_ids.append(entry['chunk_id'])
                self.chunks_metadata.append({
                    'chunk_id': entry['chunk_id'],
                    'document_id': entry['document_id'],
                    'company_id': entry['company_id'],
                    'form_type': entry['form_type'],
                    'period_end': entry['period_end'],
                    'section_item': entry['section_item']
                })

        if len(self.chunk_ids) != n_vectors:
            raise ValueError(
                f"Mismatch: {len(self.chunk_ids)} chunk_ids "
                f"vs {n_vectors} embeddings"
            )

        logger.info(f"  {len(self.chunk_ids):,} chunks mapped")

        # Create the FAISS index
        logger.info("  Creating FAISS index (Flat inner product)...")

        # Flat index = exact (not approximate) search.
        # Inner product on normalized vectors = cosine similarity.
        self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings.astype(np.float32))

        logger.info(f"FAISS index built ({self.index.ntotal:,} vectors)")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20
    ) -> List[Tuple[str, float, Dict]]:
        """
        Finds the chunks most similar to the query embedding.

        Args:
            query_embedding: query vector (1024 dims)
            top_k: number of results

        Returns:
            List of (chunk_id, score, metadata) sorted by similarity
        """
        import faiss

        if self.index is None:
            raise ValueError("Index not built. Call build() first.")

        # Make sure the query is normalized
        query = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)

        # scores = cosine similarity (normalized vectors + inner product)
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue

            chunk_id = self.chunk_ids[idx]
            metadata = self.chunks_metadata[idx]
            results.append((chunk_id, float(score), metadata))

        return results

    def save(self, output_path: Path):
        """
        Saves the FAISS index to disk.

        Args:
            output_path: path to save to (without extension)
        """
        import faiss

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(output_path) + ".faiss")
        logger.info(f"FAISS index saved: {output_path}.faiss")

        # Save the chunk_id mapping
        mapping = {
            'chunk_ids': self.chunk_ids,
            'chunks_metadata': self.chunks_metadata,
            'dimension': self.dimension
        }

        with open(f"{output_path}_mapping.json", 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

        logger.info(f"Mapping saved: {output_path}_mapping.json")

    @classmethod
    def load(cls, index_path: Path) -> 'FAISSIndex':
        """
        Loads the FAISS index from disk.

        Args:
            index_path: path to the index (without extension)

        Returns:
            Loaded FAISSIndex
        """
        import faiss

        index_path = Path(index_path)

        with open(f"{index_path}_mapping.json", 'r', encoding='utf-8') as f:
            mapping = json.load(f)

        instance = cls(dimension=mapping['dimension'])
        instance.chunk_ids = mapping['chunk_ids']
        instance.chunks_metadata = mapping['chunks_metadata']

        instance.index = faiss.read_index(str(index_path) + ".faiss")

        logger.info(f"FAISS index loaded: {instance.index.ntotal:,} vectors")

        return instance


def main():
    """FAISS index demo (run from the project root: `python -m src.retrieval.faiss_index`)."""
    logging.basicConfig(level=logging.INFO)

    embeddings_file = Path("data/embeddings/embeddings.npz")
    index_file = Path("data/embeddings/index.jsonl")
    output_dir = Path("data/indexes")

    if not embeddings_file.exists():
        logger.error(f"File not found: {embeddings_file}")
        return

    if not index_file.exists():
        logger.error(f"File not found: {index_file}")
        return

    print("\n" + "="*60)
    print("FAISS Vector Index - Demo")
    print("="*60)

    print("\n1. Building FAISS index...")
    index = FAISSIndex(dimension=1024)
    index.build(embeddings_file, index_file)

    print("\n2. Saving index...")
    index.save(output_dir / "faiss_index")

    print("\n3. Search test...")

    # Use a real embedding as the test query
    data = np.load(embeddings_file)
    embeddings = data['embeddings']

    # Use the embedding of chunk 100 as the query
    test_embedding = embeddings[100]

    print(f"\n   Query: chunk index 100")
    results = index.search(test_embedding, top_k=5)

    for i, (chunk_id, score, metadata) in enumerate(results, 1):
        company = metadata['company_id']
        form = metadata['form_type']
        period = metadata['period_end']
        print(f"     {i}. [{company} {form} {period}] similarity={score:.4f}")

    print(f"\n   (The first result should be the same chunk with score ≈ 1.0)")

    print("\n" + "="*60)
    print("Demo completed")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
