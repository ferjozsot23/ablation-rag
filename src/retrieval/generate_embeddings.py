"""
Generates BGE-M3 embeddings for all chunks.

Pipeline:
1. Loads the BGE-M3 model
2. Reads chunks from data/chunks/chunks.jsonl
3. Generates embeddings in batches
4. Saves embeddings + index in data/embeddings/
"""

import json
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import time

# Configure logging
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/embeddings.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BGEEmbedder:
    """Embedding generator based on BGE-M3."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        """
        Initializes the BGE-M3 model.

        Args:
            model_name: model name on Hugging Face
            device: device (cpu or cuda)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.dim = 1024  # BGE-M3 dimension
        self.max_length = 8192  # BGE-M3 max sequence length

    def load_model(self):
        """Loads the model and tokenizer."""
        logger.info(f"\nLoading model: {self.model_name}")
        logger.info(f"Device: {self.device}")

        start = time.time()

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True
            )

            load_time = time.time() - start

            logger.info(f"Model loaded in {load_time:.2f}s")
            logger.info(f"  Dimension: {self.dim}")
            logger.info(f"  Max length: {self.max_length}")

            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generates embeddings for a batch of texts.

        Args:
            texts: list of texts
            batch_size: batch size
            show_progress: show a progress bar

        Returns:
            Embedding array (N x dim)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 normalization for cosine similarity
            convert_to_numpy=True
        )

        return embeddings


def load_chunks(chunks_file: Path) -> Tuple[List[str], List[Dict]]:
    """
    Loads chunks from the JSONL file.

    Args:
        chunks_file: path to chunks.jsonl

    Returns:
        (texts, metadatas)
    """
    logger.info(f"\nReading chunks: {chunks_file}")

    texts = []
    metadatas = []

    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            texts.append(chunk['text'])
            metadatas.append({
                'chunk_id': chunk['chunk_id'],
                'document_id': chunk['document_id'],
                'company_id': chunk['company_id'],
                'form_type': chunk['form_type'],
                'period_end': chunk['period_end'],
                'section_item': chunk['section_item'],
                'word_count': chunk['word_count']
            })

    logger.info(f"{len(texts):,} chunks loaded")
    return texts, metadatas


def save_embeddings(
    embeddings: np.ndarray,
    metadatas: List[Dict],
    output_dir: Path
):
    """
    Saves the embeddings and their index.

    Args:
        embeddings: embedding array (N x dim)
        metadatas: list of metadata dicts
        output_dir: output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the embeddings (compressed)
    embeddings_file = output_dir / "embeddings.npz"
    np.savez_compressed(
        embeddings_file,
        embeddings=embeddings.astype(np.float32)
    )
    logger.info(f"Embeddings saved: {embeddings_file}")
    logger.info(f"  Shape: {embeddings.shape}")
    logger.info(f"  Size: {embeddings_file.stat().st_size / 1024 / 1024:.1f} MB")

    # Save the index (chunk_id → index mapping)
    index_file = output_dir / "index.jsonl"
    with open(index_file, 'w', encoding='utf-8') as f:
        for i, meta in enumerate(metadatas):
            entry = {
                'index': i,
                **meta
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    logger.info(f"Index saved: {index_file}")

    # Save the model metadata
    metadata_file = output_dir / "metadata.json"
    metadata = {
        "model": "BAAI/bge-m3",
        "dimension": int(embeddings.shape[1]),
        "num_chunks": int(embeddings.shape[0]),
        "dtype": str(embeddings.dtype),
        "normalized": True,
        "generated_at": datetime.now().isoformat()
    }
    metadata_file.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    logger.info(f"Metadata saved: {metadata_file}")


def generate_embeddings():
    """Main embedding generation pipeline."""

    logger.info(f"\n{'#'*60}")
    logger.info(f"# Embedding generation - started")
    logger.info(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*60}")

    chunks_file = Path("data/chunks/chunks.jsonl")
    output_dir = Path("data/embeddings")

    if not chunks_file.exists():
        logger.error(f"File not found: {chunks_file}")
        return

    # 1. Load chunks
    start_time = time.time()
    texts, metadatas = load_chunks(chunks_file)

    # 2. Initialize the model (use device="cuda" when a GPU is available)
    embedder = BGEEmbedder(model_name="BAAI/bge-m3", device="cpu")
    embedder.load_model()

    # 3. Generate embeddings
    logger.info(f"\n{'='*60}")
    logger.info(f"Generating embeddings...")
    logger.info(f"  Chunks: {len(texts):,}")
    logger.info(f"  Batch size: 32")
    logger.info(f"{'='*60}\n")

    embed_start = time.time()
    embeddings = embedder.encode_batch(
        texts,
        batch_size=32,
        show_progress=True
    )
    embed_time = time.time() - embed_start

    logger.info(f"\nEmbeddings generated in {embed_time:.1f}s")
    logger.info(f"  Speed: {len(texts)/embed_time:.1f} chunks/s")

    # 4. Save results
    logger.info(f"\n{'='*60}")
    logger.info(f"Saving results...")
    logger.info(f"{'='*60}\n")

    save_embeddings(embeddings, metadatas, output_dir)

    # Final report
    total_time = time.time() - start_time

    logger.info(f"\n{'#'*60}")
    logger.info(f"# Generation completed")
    logger.info(f"{'#'*60}\n")

    logger.info(f"Summary:")
    logger.info(f"  Chunks processed: {len(texts):,}")
    logger.info(f"  Dimension: {embeddings.shape[1]}")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"  Embedding time: {embed_time:.1f}s")
    logger.info(f"  Speed: {len(texts)/embed_time:.1f} chunks/s")

    logger.info(f"\nEmbeddings: {output_dir}/embeddings.npz")
    logger.info(f"Index: {output_dir}/index.jsonl")
    logger.info(f"Metadata: {output_dir}/metadata.json")
    logger.info(f"Logs: logs/embeddings.log\n")


def main():
    """Runs the embedding generation (`python -m src.retrieval.generate_embeddings`)."""
    generate_embeddings()


if __name__ == "__main__":
    main()
