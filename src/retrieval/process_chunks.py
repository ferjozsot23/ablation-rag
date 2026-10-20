"""
Processes all documents and generates chunks.

Pipeline:
1. Reads documents from data/processed/
2. Splits them with DocumentChunker
3. Saves the chunks in data/chunks/chunks.jsonl
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from src.retrieval.chunker import DocumentChunker

# Configure logging
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/chunking.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def process_all_documents():
    """Processes all documents and generates chunks."""

    processed_dir = Path("data/processed")
    chunks_dir = Path("data/chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    output_file = chunks_dir / "chunks.jsonl"

    logger.info(f"\n{'#'*60}")
    logger.info(f"# Chunk generation - started")
    logger.info(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*60}\n")

    chunker = DocumentChunker(
        target_chunk_size=800,
        min_chunk_size=600,
        max_chunk_size=1000,
        overlap_size=100
    )

    doc_files = list(processed_dir.glob("*.json"))
    logger.info(f"Documents found: {len(doc_files)}")
    logger.info(f"Output: {output_file}\n")

    stats = {
        "documents_processed": 0,
        "total_chunks": 0,
        "total_words": 0,
        "by_company": {},
        "by_form_type": {},
        "chunks_per_doc": []
    }

    start_time = datetime.now()

    with open(output_file, 'w', encoding='utf-8') as f:

        for i, doc_file in enumerate(sorted(doc_files), 1):
            logger.info(f"[{i}/{len(doc_files)}] {doc_file.name}")

            try:
                doc_data = json.loads(doc_file.read_text())

                chunks = chunker.chunk_document(doc_data)

                # Write the chunks to the JSONL file
                for chunk in chunks:
                    f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + '\n')

                # Update stats
                stats["documents_processed"] += 1
                stats["total_chunks"] += len(chunks)
                stats["total_words"] += sum(c.word_count for c in chunks)
                stats["chunks_per_doc"].append(len(chunks))

                # Per-company stats
                company = doc_data["company_id"]
                if company not in stats["by_company"]:
                    stats["by_company"][company] = {"docs": 0, "chunks": 0}
                stats["by_company"][company]["docs"] += 1
                stats["by_company"][company]["chunks"] += len(chunks)

                # Per-form-type stats
                form_type = doc_data["form_type"]
                if form_type not in stats["by_form_type"]:
                    stats["by_form_type"][form_type] = {"docs": 0, "chunks": 0}
                stats["by_form_type"][form_type]["docs"] += 1
                stats["by_form_type"][form_type]["chunks"] += len(chunks)

                logger.info(f"  {len(chunks)} chunks generated")

            except Exception as e:
                logger.error(f"  Error: {e}")

    # Final report
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info(f"\n{'#'*60}")
    logger.info(f"# Chunking completed")
    logger.info(f"{'#'*60}\n")

    logger.info(f"Statistics:")
    logger.info(f"  Documents processed: {stats['documents_processed']}")
    logger.info(f"  Total chunks: {stats['total_chunks']:,}")
    logger.info(f"  Total words: {stats['total_words']:,}")
    logger.info(f"  Chunks per document:")
    logger.info(f"    - Average: {sum(stats['chunks_per_doc'])/len(stats['chunks_per_doc']):.1f}")
    logger.info(f"    - Minimum: {min(stats['chunks_per_doc'])}")
    logger.info(f"    - Maximum: {max(stats['chunks_per_doc'])}")
    logger.info(f"  Total time: {duration:.1f}s")

    logger.info(f"\nBy document type:")
    for form_type, data in sorted(stats["by_form_type"].items()):
        avg_chunks = data["chunks"] / data["docs"]
        logger.info(
            f"  {form_type}: {data['docs']} docs, "
            f"{data['chunks']:,} chunks ({avg_chunks:.1f} chunks/doc)"
        )

    logger.info(f"\nBy company:")
    for company, data in sorted(stats["by_company"].items()):
        avg_chunks = data["chunks"] / data["docs"]
        logger.info(
            f"  {company}: {data['docs']} docs, "
            f"{data['chunks']:,} chunks ({avg_chunks:.1f} chunks/doc)"
        )

    logger.info(f"\nChunks saved: {output_file}")
    logger.info(f"Logs: logs/chunking.log\n")

    return stats


def main():
    """Runs the chunking (`python -m src.retrieval.process_chunks`)."""
    process_all_documents()


if __name__ == "__main__":
    main()
