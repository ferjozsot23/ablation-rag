"""
Entity extractor for the knowledge graph - SEC documents

Extracts entities and relations from financial chunks with an LLM.
Adapted from: https://github.com/ferjozsot23/agentic-graphrag

Entities for SEC documents:
- COMPANY: AAPL, NVDA, MSFT, GOOGL, ...
- METRIC: Revenue, NetIncome, EPS, Assets, Liabilities, ...
- PERIOD: FY2023, Q1-2024, 2023-09-30, ...
- PRODUCT: iPhone, iPad, GPU, Azure, ...
- EVENT: Earnings, Acquisition, Product_Launch, ...

Relations:
- HAS_METRIC: AAPL --[HAS_METRIC]--> Revenue
- IN_PERIOD: Revenue --[IN_PERIOD]--> FY2023
- HAS_VALUE: Revenue --[HAS_VALUE]--> "$383.3B"
- SELLS: AAPL --[SELLS]--> iPhone
- GENERATED: iPhone --[GENERATED]--> Revenue
- MENTIONED_IN: Revenue --[MENTIONED_IN]--> chunk_id

Note: the JSON field names of the extracted data (`entidades`, `relaciones`,
`nombre`, `tipo`, `descripcion`, `origen`, `destino`, `relacion`) are kept as they
are because the saved files in data/knowledge_graph/ and the graph code use them.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm


def extract_entities_batch(
    llm_client,
    chunks: List[Dict],
    batch_size: int = 100,
    temperature: float = 0.0
) -> Tuple[List[Dict], List[Dict]]:
    """
    Extracts entities and relations from a batch of chunks.

    Args:
        llm_client: configured LLMClient
        chunks: chunks to process
        batch_size: chunks per batch (default: 100)
        temperature: LLM temperature (default: 0.0)

    Returns:
        (all_entities, all_relations)
    """
    all_entities = []
    all_relations = []

    num_batches = (len(chunks) + batch_size - 1) // batch_size

    print(f"\n{'='*80}")
    print(f"ENTITY EXTRACTION: {len(chunks)} chunks → {num_batches} batches")
    print(f"{'='*80}\n")

    for i in tqdm(range(0, len(chunks), batch_size), desc="Processing batches"):
        batch = chunks[i:i + batch_size]

        prompt = _create_batch_prompt(batch)

        # Call the LLM (the prompt asks for JSON)
        result = llm_client.generate(
            prompt=prompt,
            system_message=_get_system_message(),
            temperature=temperature
        )

        if result.get('error'):
            print(f"\n  Batch {i//batch_size + 1}/{num_batches} error: {result['error']}")
            continue

        # Parse the JSON response (strip code fences if present)
        try:
            answer_text = result['answer'].strip()
            if answer_text.startswith('```'):
                lines = answer_text.split('\n')
                # Skip the first line (```json) and the last one (```)
                answer_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else answer_text

            data = json.loads(answer_text)
            entities = data.get('entidades', [])
            relations = data.get('relaciones', [])

            # Attach batch metadata to the entities
            for entity in entities:
                entity['batch_id'] = i // batch_size
                entity['batch_start'] = i
                entity['batch_end'] = min(i + batch_size, len(chunks))

            # Attach batch metadata to the relations
            for relation in relations:
                relation['batch_id'] = i // batch_size

            all_entities.extend(entities)
            all_relations.extend(relations)

            # Per-batch logging
            batch_num = i // batch_size + 1
            print(f"\n{''*80}")
            print(f" Batch {batch_num}/{num_batches}:")
            print(f"   Entities: {len(entities)} | Relations: {len(relations)}")
            print(f"   Running total: {len(all_entities)} entities, {len(all_relations)} relations")
            print(f"   Tokens: {result.get('total_tokens', 0)} ({result.get('prompt_tokens', 0)}+{result.get('completion_tokens', 0)})")
            print(f"   Time: {result.get('time_seconds', 0):.2f}s")

            # Examples of the entities extracted in this batch
            if entities:
                print(f"\n    Entity examples:")
                for entity in entities[:3]:
                    print(f"      • {entity['nombre']} ({entity['tipo']})")

            # Examples of the relations extracted in this batch
            if relations:
                print(f"\n    Relation examples:")
                for rel in relations[:3]:
                    print(f"      • {rel['origen']} --[{rel['relacion']}]--> {rel['destino']}")

        except json.JSONDecodeError as e:
            print(f"\n  Batch {i//batch_size + 1}/{num_batches} JSON decode error: {e}")
            print(f"  Response preview: {result['answer'][:200]}...")
            continue

        # Rate limiting (avoid saturating the API)
        if i + batch_size < len(chunks):
            time.sleep(0.5)

    print(f"\n{'='*80}")
    print(f" EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Total entities: {len(all_entities)}")
    print(f"Total relations: {len(all_relations)}")
    print()

    return all_entities, all_relations


def _get_system_message() -> str:
    """System message for entity extraction."""
    return """You are a financial document analyst. Extract structured entities and relationships from SEC filings (10-K, 10-Q forms).

Focus on:
- Companies (AAPL, NVDA, MSFT, ...)
- Financial metrics (Revenue, Net Income, EPS, Assets, ...)
- Time periods (FY2023, Q1-2024, dates)
- Products/Services (iPhone, GPU, Azure, ...)
- Events (Earnings, Acquisitions, Product launches, ...)

Be precise and extract ONLY what is explicitly stated in the text."""


def _create_batch_prompt(chunks: List[Dict]) -> str:
    """
    Builds the prompt that extracts entities from a batch of chunks.

    Strategy: concatenate the chunks of the batch and extract global entities.
    """
    batch_texts = []
    for idx, chunk in enumerate(chunks):
        meta = chunk.get('metadata', {})
        company = meta.get('company_id', 'UNKNOWN')
        form = meta.get('form_type', 'UNKNOWN')
        period = meta.get('period_end', 'UNKNOWN')

        batch_texts.append(
            f"[Chunk {idx+1}/{len(chunks)} | {company} {form} {period}]\n{chunk['text'][:500]}"
        )

    batch_text = "\n\n".join(batch_texts)

    prompt = f"""Analyze the following SEC filing excerpts and extract:

1. ENTITIES (max 50 per batch): Key financial entities
   Format: {{"nombre": "...", "tipo": "COMPANY|METRIC|PERIOD|PRODUCT|EVENT", "descripcion": "brief description"}}

   Types:
   - COMPANY: ticker symbols or company names (AAPL, Apple Inc., NVIDIA, etc.)
   - METRIC: financial metrics (Revenue, Net Income, Total Assets, EPS, etc.)
   - PERIOD: time periods (FY2023, Q1-2024, fiscal year 2023, etc.)
   - PRODUCT: products or services (iPhone, iPad, GPU, Cloud Services, etc.)
   - EVENT: corporate events (Earnings Release, Acquisition, Product Launch, etc.)

2. RELATIONSHIPS (max 80 per batch): Connections between entities
   Format: {{"origen": "entity1", "destino": "entity2", "relacion": "relationship_type", "descripcion": "optional context"}}

   Relationship types:
   - HAS_METRIC: company has a metric (AAPL → Revenue)
   - IN_PERIOD: metric in a period (Revenue → FY2023)
   - HAS_VALUE: metric has a value (Revenue → $383.3B)
   - SELLS: company sells product (AAPL → iPhone)
   - GENERATED: product generated revenue (iPhone → $200.6B)
   - ACQUIRED: company acquired another (MSFT → Activision)
   - REPORTED: company reported metric (AAPL → Earnings)

SEC Filing Excerpts:
{batch_text}

Respond with VALID JSON:
{{
  "entidades": [...],
  "relaciones": [...]
}}"""

    return prompt


def save_extraction_results(
    entities: List[Dict],
    relations: List[Dict],
    output_dir: Path
):
    """
    Saves entities and relations as JSONL files.

    Args:
        entities: extracted entities
        relations: extracted relations
        output_dir: output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Entities
    entities_file = output_dir / "entities.jsonl"
    with open(entities_file, 'w', encoding='utf-8') as f:
        for entity in entities:
            f.write(json.dumps(entity, ensure_ascii=False) + '\n')

    print(f" Entities saved: {entities_file} ({len(entities)} entities)")

    # Relations
    relations_file = output_dir / "relations.jsonl"
    with open(relations_file, 'w', encoding='utf-8') as f:
        for relation in relations:
            f.write(json.dumps(relation, ensure_ascii=False) + '\n')

    print(f" Relations saved: {relations_file} ({len(relations)} relations)")

    # Statistics
    stats = {
        'total_entities': len(entities),
        'total_relations': len(relations),
        'entity_types': {},
        'relation_types': {}
    }

    for entity in entities:
        tipo = entity.get('tipo', 'UNKNOWN')
        stats['entity_types'][tipo] = stats['entity_types'].get(tipo, 0) + 1

    for relation in relations:
        tipo = relation.get('relacion', 'UNKNOWN')
        stats['relation_types'][tipo] = stats['relation_types'].get(tipo, 0) + 1

    stats_file = output_dir / "extraction_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f" Stats saved: {stats_file}")
    print(f"\n Entity Types:")
    for tipo, count in sorted(stats['entity_types'].items(), key=lambda x: -x[1]):
        print(f"   {tipo}: {count}")

    print(f"\n Relation Types:")
    for tipo, count in sorted(stats['relation_types'].items(), key=lambda x: -x[1]):
        print(f"   {tipo}: {count}")


def load_chunks(chunks_file: Path) -> List[Dict]:
    """Loads chunks from chunks.jsonl."""
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def main():
    """Entity extraction run (`python -m src.knowledge_graph.entity_extractor`)."""
    from src.llm.llm_client import LLMClient

    project_root = Path(__file__).parent.parent.parent

    print("\n" + "="*80)
    print("ENTITY EXTRACTION")
    print("="*80)

    chunks_file = project_root / "data/chunks/chunks.jsonl"
    output_dir = project_root / "data/knowledge_graph"

    print(f"\n1. Loading chunks from: {chunks_file}")
    all_chunks = load_chunks(chunks_file)
    print(f"    Loaded {len(all_chunks)} chunks")

    # Slice all_chunks here (e.g. all_chunks[:100]) for a cheap trial run
    demo_chunks = all_chunks
    print(f"\n     Processing ALL {len(demo_chunks)} chunks")

    print(f"\n2. Initializing LLM client...")
    llm = LLMClient(temperature=0.0)
    print(f"    Model: {llm.model}")
    print(f"    Base URL: {llm.base_url}")

    print(f"\n3. Extracting entities and relations...")
    entities, relations = extract_entities_batch(
        llm_client=llm,
        chunks=demo_chunks,
        batch_size=100,
        temperature=0.0
    )

    print(f"\n4. Saving results...")
    save_extraction_results(entities, relations, output_dir)

    print("\n" + "="*80)
    print(" Extraction completed")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
