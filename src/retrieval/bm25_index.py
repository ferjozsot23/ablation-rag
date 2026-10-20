"""
BM25 index for lexical search.

Implements BM25 (Best Matching 25) for document retrieval based on
keyword matches.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import re

logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25 index for lexical search.

    BM25 parameters:
    - k1: controls term-frequency saturation (default: 1.5)
    - b: controls document-length normalization (default: 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initializes the BM25 index.

        Args:
            k1: TF saturation parameter (typically 1.2-2.0)
            b: length normalization parameter (typically 0.75)
        """
        self.k1 = k1
        self.b = b

        # Index data
        self.chunks = []           # list of chunks (metadata)
        self.tokenized_chunks = [] # tokenized chunks
        self.chunk_ids = []        # chunk IDs

        # Corpus statistics
        self.avgdl = 0.0          # average document length
        self.doc_freqs = {}       # document frequency per term
        self.idf = {}             # IDF scores
        self.doc_lens = []        # document lengths

        logger.info(f"BM25Index initialized (k1={k1}, b={b})")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text into terms.

        Simple tokenization:
        - Lowercase
        - Split into words
        - Strip punctuation
        - Terms of 2+ characters

        Args:
            text: text to tokenize

        Returns:
            List of tokens
        """
        # Lowercase and strip special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)

        # Split and filter
        tokens = text.split()
        tokens = [t for t in tokens if len(t) >= 2]

        return tokens

    def build(self, chunks_file: Path):
        """
        Builds the BM25 index from a chunks file.

        Args:
            chunks_file: path to chunks.jsonl
        """
        logger.info(f"Building BM25 index from: {chunks_file}")

        # Load chunks
        self.chunks = []
        self.tokenized_chunks = []
        self.chunk_ids = []

        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)

                # Keep metadata
                self.chunks.append(chunk)
                self.chunk_ids.append(chunk['chunk_id'])

                # Tokenize text
                tokens = self.tokenize(chunk['text'])
                self.tokenized_chunks.append(tokens)

        n_docs = len(self.chunks)
        logger.info(f"  {n_docs:,} chunks loaded")

        # Document lengths and avgdl
        self.doc_lens = [len(tokens) for tokens in self.tokenized_chunks]
        self.avgdl = sum(self.doc_lens) / n_docs

        logger.info(f"  Average length: {self.avgdl:.1f} tokens")

        # Document frequencies
        logger.info("  Computing document frequencies...")
        self.doc_freqs = {}

        for tokens in self.tokenized_chunks:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        vocab_size = len(self.doc_freqs)
        logger.info(f"  Vocabulary: {vocab_size:,} unique terms")

        # IDF scores
        logger.info("  Computing IDF scores...")
        self._calculate_idf(n_docs)

        logger.info(f"BM25 index built ({n_docs:,} docs, {vocab_size:,} terms)")

    def _calculate_idf(self, n_docs: int):
        """
        Computes the IDF (Inverse Document Frequency) of every term.

        IDF = log((N - df + 0.5) / (df + 0.5) + 1)

        Args:
            n_docs: total number of documents
        """
        import math

        self.idf = {}
        for term, df in self.doc_freqs.items():
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            self.idf[term] = idf

    def get_scores(self, query: str) -> List[float]:
        """
        Computes the BM25 scores of all documents for a query.

        BM25 score = Σ(IDF(qi) * (f(qi, D) * (k1 + 1)) /
                        (f(qi, D) + k1 * (1 - b + b * |D| / avgdl)))

        Args:
            query: query string

        Returns:
            List of scores (same order as self.chunks)
        """
        # Tokenize the query
        query_tokens = self.tokenize(query)

        # Score every document
        scores = [0.0] * len(self.chunks)

        for token in query_tokens:
            # Term IDF
            if token not in self.idf:
                continue

            idf = self.idf[token]

            for i, doc_tokens in enumerate(self.tokenized_chunks):
                # Term frequency in the document
                tf = doc_tokens.count(token)

                if tf == 0:
                    continue

                doc_len = self.doc_lens[i]

                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)

                scores[i] += idf * (numerator / denominator)

        return scores

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float, Dict]]:
        """
        Searches relevant chunks with BM25.

        Args:
            query: query string
            top_k: number of results to return

        Returns:
            List of (chunk_id, score, metadata) sorted by descending score
        """
        scores = self.get_scores(query)

        scored_chunks = [
            (self.chunk_ids[i], scores[i], self.chunks[i])
            for i in range(len(scores))
        ]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return scored_chunks[:top_k]

    def save(self, output_path: Path):
        """
        Saves the index to disk.

        Args:
            output_path: path to save to (without extension)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        index_data = {
            'k1': self.k1,
            'b': self.b,
            'chunks': self.chunks,
            'tokenized_chunks': self.tokenized_chunks,
            'chunk_ids': self.chunk_ids,
            'avgdl': self.avgdl,
            'doc_freqs': self.doc_freqs,
            'idf': self.idf,
            'doc_lens': self.doc_lens
        }

        with open(f"{output_path}.pkl", 'wb') as f:
            pickle.dump(index_data, f)

        logger.info(f"BM25 index saved: {output_path}.pkl")

        # Save metadata separately
        metadata = {
            'n_docs': len(self.chunks),
            'vocab_size': len(self.idf),
            'avgdl': self.avgdl,
            'k1': self.k1,
            'b': self.b
        }

        with open(f"{output_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, index_path: Path) -> 'BM25Index':
        """
        Loads the index from disk.

        Args:
            index_path: path to the index (without extension)

        Returns:
            Loaded BM25Index
        """
        index_path = Path(index_path)

        with open(f"{index_path}.pkl", 'rb') as f:
            index_data = pickle.load(f)

        index = cls(k1=index_data['k1'], b=index_data['b'])

        index.chunks = index_data['chunks']
        index.tokenized_chunks = index_data['tokenized_chunks']
        index.chunk_ids = index_data['chunk_ids']
        index.avgdl = index_data['avgdl']
        index.doc_freqs = index_data['doc_freqs']
        index.idf = index_data['idf']
        index.doc_lens = index_data['doc_lens']

        logger.info(f"BM25 index loaded: {len(index.chunks):,} docs")

        return index


def main():
    """BM25 index demo (run from the project root: `python -m src.retrieval.bm25_index`)."""
    logging.basicConfig(level=logging.INFO)

    chunks_file = Path("data/chunks/chunks.jsonl")
    output_dir = Path("data/indexes")

    if not chunks_file.exists():
        logger.error(f"File not found: {chunks_file}")
        return

    print("\n" + "="*60)
    print("BM25 Index - Demo")
    print("="*60)

    print("\n1. Building BM25 index...")
    index = BM25Index(k1=1.5, b=0.75)
    index.build(chunks_file)

    print("\n2. Saving index...")
    index.save(output_dir / "bm25_index")

    print("\n3. Search test...")
    test_queries = [
        "Apple revenue 2023",
        "NVIDIA GPU sales growth",
        "Microsoft cloud computing revenue"
    ]

    for query in test_queries:
        print(f"\n   Query: '{query}'")
        results = index.search(query, top_k=3)

        for i, (chunk_id, score, metadata) in enumerate(results, 1):
            company = metadata['company_id']
            form = metadata['form_type']
            period = metadata['period_end']
            print(f"     {i}. [{company} {form} {period}] score={score:.2f}")

    print("\n" + "="*60)
    print("Demo completed")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
