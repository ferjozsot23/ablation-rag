"""
Knowledge graph module (System 4 and the System 5 search tool).

Components:
- entity_extractor: extracts entities and relations from chunks
- graph_builder: builds the knowledge graph
- graph_search: search and expansion over the graph
"""

from .entity_extractor import (
    extract_entities_batch,
    save_extraction_results,
    load_chunks
)

from .graph_builder import (
    build_graph,
    save_graph,
    load_graph
)

from .graph_search import (
    find_relevant_entities,
    expand_context,
    get_subgraph,
    get_entity_context,
    graph_search_pipeline
)

__all__ = [
    # Entity extraction
    'extract_entities_batch',
    'save_extraction_results',
    'load_chunks',
    # Graph building
    'build_graph',
    'save_graph',
    'load_graph',
    # Graph search
    'find_relevant_entities',
    'expand_context',
    'get_subgraph',
    'get_entity_context',
    'graph_search_pipeline'
]
