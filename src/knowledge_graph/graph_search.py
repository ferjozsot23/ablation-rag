"""
Graph Search - search and context expansion over the knowledge graph.

Main functions:
- find_relevant_entities(): find entities for a query
- expand_context(): expand context from entities (k-hop neighbors)
- get_subgraph(): extract the relevant subgraph
"""

from typing import List, Set, Dict, Tuple
import networkx as nx
from difflib import SequenceMatcher


def find_relevant_entities(
    query: str,
    G: nx.DiGraph,
    top_k: int = 10,
    min_similarity: float = 0.3
) -> List[Tuple[str, float, str]]:
    """
    Finds relevant entities in the graph by name similarity.

    Args:
        query: user query
        G: NetworkX DiGraph
        top_k: number of entities to return
        min_similarity: minimum similarity (0-1)

    Returns:
        List of (entity_name, similarity, entity_type)
    """
    query_lower = query.lower()
    query_tokens = set(query_lower.split())

    matches = []

    for node in G.nodes():
        node_lower = node.lower()

        # Substring similarity
        if query_lower in node_lower or node_lower in query_lower:
            similarity = 0.9
        else:
            # SequenceMatcher similarity
            similarity = SequenceMatcher(None, query_lower, node_lower).ratio()

            # Bonus for shared tokens
            node_tokens = set(node_lower.split())
            if query_tokens & node_tokens:  # intersection
                token_overlap = len(query_tokens & node_tokens) / len(query_tokens)
                similarity = max(similarity, token_overlap * 0.8)

        if similarity >= min_similarity:
            node_type = G.nodes[node].get('tipo', 'UNKNOWN')
            matches.append((node, similarity, node_type))

    # Sort by similarity and keep the top_k
    matches.sort(key=lambda x: -x[1])
    return matches[:top_k]


def expand_context(
    seed_entities: List[str],
    G: nx.DiGraph,
    depth: int = 2,
    max_nodes: int = 100
) -> Set[str]:
    """
    Expands the context from seed entities (k-hop neighbors).

    Args:
        seed_entities: initial entities
        G: NetworkX DiGraph
        depth: expansion depth (number of hops)
        max_nodes: maximum number of nodes to return

    Returns:
        Set of expanded entity IDs
    """
    expanded = set(seed_entities)
    current_level = set(seed_entities)

    for hop in range(depth):
        next_level = set()

        for entity in current_level:
            if entity not in G:
                continue

            # Outgoing neighbors (successors)
            for successor in G.successors(entity):
                if successor not in expanded:
                    next_level.add(successor)
                    expanded.add(successor)

            # Incoming neighbors (predecessors)
            for predecessor in G.predecessors(entity):
                if predecessor not in expanded:
                    next_level.add(predecessor)
                    expanded.add(predecessor)

            # Limit the size
            if len(expanded) >= max_nodes:
                return expanded

        current_level = next_level

        if not current_level:
            break

    return expanded


def get_subgraph(
    entity_ids: Set[str],
    G: nx.DiGraph
) -> nx.DiGraph:
    """
    Extracts the subgraph induced by a set of entities.

    Args:
        entity_ids: set of entity IDs
        G: original NetworkX DiGraph

    Returns:
        Subgraph with only those entities and their edges
    """
    # Keep the entities that exist in G
    valid_entities = [e for e in entity_ids if e in G]

    subgraph = G.subgraph(valid_entities).copy()

    return subgraph


def get_entity_context(
    entity: str,
    G: nx.DiGraph,
    max_relations: int = 20
) -> Dict:
    """
    Gets the detailed context of an entity.

    Args:
        entity: entity ID
        G: NetworkX DiGraph
        max_relations: maximum number of relations to return

    Returns:
        Dict with the entity info and its relations
    """
    if entity not in G:
        return {
            'entity': entity,
            'exists': False
        }

    node_data = G.nodes[entity]

    # Outgoing relations
    outgoing = []
    for successor in G.successors(entity):
        edge_data = G.edges[entity, successor]
        outgoing.append({
            'target': successor,
            'relation': edge_data.get('relacion', ''),
            'target_type': G.nodes[successor].get('tipo', 'UNKNOWN')
        })

    # Incoming relations
    incoming = []
    for predecessor in G.predecessors(entity):
        edge_data = G.edges[predecessor, entity]
        incoming.append({
            'source': predecessor,
            'relation': edge_data.get('relacion', ''),
            'source_type': G.nodes[predecessor].get('tipo', 'UNKNOWN')
        })

    outgoing = outgoing[:max_relations]
    incoming = incoming[:max_relations]

    return {
        'entity': entity,
        'exists': True,
        'type': node_data.get('tipo', 'UNKNOWN'),
        'description': node_data.get('descripcion', ''),
        'degree': G.degree(entity),
        'outgoing_relations': outgoing,
        'incoming_relations': incoming,
        'num_outgoing': len(list(G.successors(entity))),
        'num_incoming': len(list(G.predecessors(entity)))
    }


def graph_search_pipeline(
    query: str,
    G: nx.DiGraph,
    top_k_entities: int = 5,
    expansion_depth: int = 2,
    max_expanded_nodes: int = 50
) -> Dict:
    """
    Full graph search pipeline.

    Args:
        query: user query
        G: NetworkX DiGraph
        top_k_entities: top K entities to look up
        expansion_depth: expansion depth
        max_expanded_nodes: maximum number of expanded nodes

    Returns:
        Dict with the search results
    """
    # 1. Find relevant entities
    relevant_entities = find_relevant_entities(query, G, top_k=top_k_entities)

    if not relevant_entities:
        return {
            'query': query,
            'found_entities': [],
            'expanded_entities': [],
            'subgraph_nodes': 0,
            'subgraph_edges': 0
        }

    # 2. Expand the context
    seed_entity_ids = [e[0] for e in relevant_entities]
    expanded_entities = expand_context(
        seed_entity_ids,
        G,
        depth=expansion_depth,
        max_nodes=max_expanded_nodes
    )

    # 3. Extract the subgraph
    subgraph = get_subgraph(expanded_entities, G)

    return {
        'query': query,
        'found_entities': relevant_entities,
        'expanded_entities': list(expanded_entities),
        'subgraph_nodes': subgraph.number_of_nodes(),
        'subgraph_edges': subgraph.number_of_edges(),
        'subgraph': subgraph
    }


def main():
    """Knowledge graph search demo (`python -m src.knowledge_graph.graph_search`)."""
    from pathlib import Path
    from src.knowledge_graph.graph_builder import load_graph

    print("\n" + "="*80)
    print("GRAPH SEARCH - DEMO")
    print("="*80)

    project_root = Path(__file__).parent.parent.parent
    graph_file = project_root / "data/knowledge_graph/knowledge_graph.pkl"

    print(f"\n1. Loading graph from {graph_file.name}...")
    G = load_graph(graph_file)

    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")

    query = "Apple revenue 2023"
    print(f"\n2. Searching for: '{query}'")

    print(f"\n   a) Finding relevant entities...")
    entities = find_relevant_entities(query, G, top_k=10)

    print(f"\n   Found {len(entities)} entities:")
    for i, (entity, similarity, entity_type) in enumerate(entities, 1):
        print(f"      {i}. {entity} ({entity_type}) - similarity: {similarity:.3f}")

    if entities:
        print(f"\n   b) Expanding context...")
        seed_ids = [e[0] for e in entities[:3]]  # top 3
        print(f"      Seed entities: {seed_ids}")

        expanded = expand_context(seed_ids, G, depth=2, max_nodes=30)
        print(f"      Expanded to {len(expanded)} entities (depth=2)")

        print(f"\n   c) Extracting subgraph...")
        subgraph = get_subgraph(expanded, G)
        print(f"      Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")

        # Context of the most relevant entity
        top_entity = entities[0][0]
        print(f"\n   d) Context of '{top_entity}':")
        context = get_entity_context(top_entity, G, max_relations=5)

        print(f"      Type: {context['type']}")
        print(f"      Degree: {context['degree']}")
        print(f"      Outgoing relations ({context['num_outgoing']} total):")
        for rel in context['outgoing_relations'][:5]:
            print(f"        → {rel['target']} ({rel['target_type']}) via {rel['relation']}")

        print(f"      Incoming relations ({context['num_incoming']} total):")
        for rel in context['incoming_relations'][:5]:
            print(f"        ← {rel['source']} ({rel['source_type']}) via {rel['relation']}")

    print("\n" + "="*80)
    print(" Graph search demo completed")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
