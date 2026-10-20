"""
Graph Builder - builds a NetworkX DiGraph from extracted entities/relations.

Creates a directed graph with:
- Nodes: entities (COMPANY, METRIC, PERIOD, PRODUCT, EVENT)
- Edges: relations (HAS_METRIC, IN_PERIOD, SELLS, etc.)

The node/edge attribute names (`tipo`, `descripcion`, `relacion`) follow the
schema produced by entity_extractor.py.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List
import networkx as nx
from tqdm import tqdm


def load_entities(entities_file: Path) -> List[Dict]:
    """Loads entities from entities.jsonl."""
    entities = []
    with open(entities_file, 'r', encoding='utf-8') as f:
        for line in f:
            entities.append(json.loads(line))
    return entities


def load_relations(relations_file: Path) -> List[Dict]:
    """Loads relations from relations.jsonl."""
    relations = []
    with open(relations_file, 'r', encoding='utf-8') as f:
        for line in f:
            relations.append(json.loads(line))
    return relations


def build_graph(
    entities: List[Dict],
    relations: List[Dict],
    verbose: bool = True
) -> nx.DiGraph:
    """
    Builds a NetworkX DiGraph from entities and relations.

    Args:
        entities: list of entities {nombre, tipo, descripcion, ...}
        relations: list of relations {origen, destino, relacion, descripcion}
        verbose: print progress and graph statistics

    Returns:
        NetworkX DiGraph with nodes (entities) and edges (relations)
    """
    G = nx.DiGraph()

    if verbose:
        print(f"\n{'='*80}")
        print(f"GRAPH CONSTRUCTION")
        print(f"{'='*80}\n")

    # Add nodes (entities)
    if verbose:
        print(f"1. Adding {len(entities)} nodes (entities)...")
    for entity in tqdm(entities, desc="  Nodes", disable=not verbose):
        # Use 'nombre' as the node id
        node_id = entity['nombre']

        # Add the node with all its attributes
        G.add_node(
            node_id,
            tipo=entity['tipo'],
            descripcion=entity.get('descripcion', ''),
            batch_id=entity.get('batch_id', -1),
            batch_start=entity.get('batch_start', -1),
            batch_end=entity.get('batch_end', -1)
        )

    # Add edges (relations)
    if verbose:
        print(f"\n2. Adding {len(relations)} edges (relations)...")
    added_edges = 0
    skipped_edges = 0

    for relation in tqdm(relations, desc="  Edges", disable=not verbose):
        origen = relation['origen']
        destino = relation['destino']

        # Only add the edge if both nodes exist
        if origen in G.nodes and destino in G.nodes:
            G.add_edge(
                origen,
                destino,
                relacion=relation['relacion'],
                descripcion=relation.get('descripcion', ''),
                batch_id=relation.get('batch_id', -1)
            )
            added_edges += 1
        else:
            skipped_edges += 1

    if not verbose:
        return G

    print(f"\n   Added: {added_edges} edges")
    print(f"   Skipped: {skipped_edges} edges (missing nodes)")

    # Graph statistics
    print(f"\n{'='*80}")
    print(f" GRAPH STATISTICS")
    print(f"{'='*80}")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.6f}")

    # Connected components (undirected view)
    G_undirected = G.to_undirected()
    num_components = nx.number_connected_components(G_undirected)
    print(f"Connected components: {num_components}")

    # Nodes by type
    print(f"\nNodes by type:")
    node_types = {}
    for node, data in G.nodes(data=True):
        tipo = data.get('tipo', 'UNKNOWN')
        node_types[tipo] = node_types.get(tipo, 0) + 1

    for tipo, count in sorted(node_types.items(), key=lambda x: -x[1]):
        print(f"  {tipo}: {count}")

    # Edges by relation type
    print(f"\nEdges by relation type:")
    edge_types = {}
    for u, v, data in G.edges(data=True):
        relacion = data.get('relacion', 'UNKNOWN')
        edge_types[relacion] = edge_types.get(relacion, 0) + 1

    for relacion, count in sorted(edge_types.items(), key=lambda x: -x[1]):
        print(f"  {relacion}: {count}")

    # Most connected nodes
    print(f"\nTop 10 most connected nodes:")
    degree_dict = dict(G.degree())
    top_nodes = sorted(degree_dict.items(), key=lambda x: -x[1])[:10]
    for node, degree in top_nodes:
        node_type = G.nodes[node].get('tipo', 'UNKNOWN')
        print(f"  {node} ({node_type}): {degree} connections")

    print(f"\n{'='*80}\n")

    return G


def save_graph(G: nx.DiGraph, output_file: Path):
    """Saves the graph as a pickle file."""
    with open(output_file, 'wb') as f:
        pickle.dump(G, f)
    print(f" Graph saved: {output_file}")
    print(f"   Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


def load_graph(graph_file: Path) -> nx.DiGraph:
    """
    Loads the graph from the pickle file.

    If the pickle does not exist, the graph is rebuilt from the entities.jsonl and
    relations.jsonl files stored next to it (the pickle itself is not versioned).
    """
    graph_file = Path(graph_file)
    if graph_file.exists():
        with open(graph_file, 'rb') as f:
            return pickle.load(f)

    entities_file = graph_file.parent / "entities.jsonl"
    relations_file = graph_file.parent / "relations.jsonl"
    if entities_file.exists() and relations_file.exists():
        return build_graph(
            load_entities(entities_file),
            load_relations(relations_file),
            verbose=False
        )

    raise FileNotFoundError(f"Graph not found: {graph_file}")


def main():
    """Builds the graph from the extracted entities/relations (`python -m src.knowledge_graph.graph_builder`)."""
    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH BUILDER")
    print("="*80)

    project_root = Path(__file__).parent.parent.parent
    kg_dir = project_root / "data/knowledge_graph"

    entities_file = kg_dir / "entities.jsonl"
    relations_file = kg_dir / "relations.jsonl"
    output_file = kg_dir / "knowledge_graph.pkl"

    print(f"\n1. Loading data...")
    entities = load_entities(entities_file)
    print(f"   Loaded {len(entities)} entities")

    relations = load_relations(relations_file)
    print(f"   Loaded {len(relations)} relations")

    print(f"\n2. Building graph...")
    G = build_graph(entities, relations)

    print(f"\n3. Saving graph...")
    save_graph(G, output_file)

    print("\n" + "="*80)
    print(" Graph construction completed")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
