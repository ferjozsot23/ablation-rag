"""
Graph Visualizer - builds an interactive HTML visualization of the knowledge graph.

Uses pyvis to create an interactive graph with:
- Nodes colored by type
- Edges labeled by relation
- Node physics for automatic layout
- Zoom, pan and hover info
"""

from pathlib import Path
import networkx as nx


def visualize_graph(
    G: nx.DiGraph,
    output_file: Path,
    max_nodes: int = 500,
    physics_enabled: bool = True
):
    """
    Creates an interactive HTML visualization of the graph.

    Args:
        G: NetworkX DiGraph
        output_file: path of the output HTML file
        max_nodes: maximum number of nodes to draw (for performance)
        physics_enabled: enable node physics
    """
    from pyvis.network import Network

    print(f"\n{'='*80}")
    print(f"GRAPH VISUALIZATION")
    print(f"{'='*80}\n")

    # If the graph is too large, keep only the most connected nodes
    if G.number_of_nodes() > max_nodes:
        print(f"Graph too large ({G.number_of_nodes()} nodes)")
        print(f"Showing top {max_nodes} most connected nodes...\n")

        degree_dict = dict(G.degree())
        top_nodes = sorted(degree_dict.items(), key=lambda x: -x[1])[:max_nodes]
        top_node_ids = [node for node, _ in top_nodes]

        # Subgraph with only those nodes
        G = G.subgraph(top_node_ids).copy()

    print(f"Visualizing {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Create the pyvis network
    net = Network(
        height='900px',
        width='100%',
        directed=True,
        notebook=False,
        cdn_resources='remote'
    )

    # Colors by node type
    type_colors = {
        'COMPANY': '#e74c3c',    # red
        'PRODUCT': '#3498db',    # blue
        'METRIC': '#2ecc71',     # green
        'PERIOD': '#f39c12',     # orange
        'EVENT': '#9b59b6',      # purple
        'UNKNOWN': '#95a5a6'     # gray
    }

    # Add nodes
    for node in G.nodes():
        node_data = G.nodes[node]
        tipo = node_data.get('tipo', 'UNKNOWN')
        descripcion = node_data.get('descripcion', '')

        # Hover title
        title = f"<b>{node}</b><br>Type: {tipo}"
        if descripcion:
            title += f"<br>{descripcion[:100]}"

        # Size based on degree
        degree = G.degree(node)
        size = 10 + min(degree * 2, 50)  # 10-60

        net.add_node(
            node,
            label=node,
            title=title,
            color=type_colors.get(tipo, '#95a5a6'),
            size=size
        )

    # Add edges
    for u, v, data in G.edges(data=True):
        relacion = data.get('relacion', '')
        descripcion = data.get('descripcion', '')

        # Edge title
        title = f"{relacion}"
        if descripcion:
            title += f": {descripcion[:50]}"

        net.add_edge(
            u,
            v,
            title=title,
            label=relacion,
            arrows='to'
        )

    # Physics configuration
    if physics_enabled:
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08,
              "damping": 0.4,
              "avoidOverlap": 0.5
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
              "enabled": true,
              "iterations": 1000,
              "updateInterval": 25
            }
          },
          "nodes": {
            "font": {
              "size": 14,
              "color": "#333333"
            },
            "borderWidth": 2,
            "borderWidthSelected": 4
          },
          "edges": {
            "font": {
              "size": 10,
              "align": "middle"
            },
            "color": {
              "color": "#848484",
              "highlight": "#000000"
            },
            "smooth": {
              "type": "continuous"
            },
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.5
              }
            }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """)

    # Save the HTML
    net.save_graph(str(output_file))

    print(f"\n Graph visualization saved: {output_file}")
    print(f"   Size: {output_file.stat().st_size / 1024:.2f} KB")
    print(f"\n{'='*80}\n")


def main():
    """Visualizes the knowledge graph (`python -m src.knowledge_graph.visualize_graph`)."""
    from src.knowledge_graph.graph_builder import load_graph

    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH VISUALIZER")
    print("="*80)

    project_root = Path(__file__).parent.parent.parent
    graph_file = project_root / "data/knowledge_graph/knowledge_graph.pkl"
    output_file = project_root / "data/knowledge_graph/graph_visualization.html"

    print(f"\n1. Loading graph from {graph_file.name}...")
    G = load_graph(graph_file)

    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")

    print(f"\n2. Creating visualization...")
    visualize_graph(
        G,
        output_file,
        max_nodes=500,  # limit for performance
        physics_enabled=True
    )

    print(f"\n3. Opening in browser...")
    import webbrowser
    webbrowser.open(str(output_file))

    print("\n" + "="*80)
    print(" Visualization complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
