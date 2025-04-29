import networkx as nx

from typing import Dict, List

class GraphUtils:

    @staticmethod
    def to_neighborhoods(graph: nx.graph) -> Dict[str, List[int]]:
        return {n-1:[int(x) - 1 for x in graph.neighbors(n)] for n in graph}
    
    @staticmethod
    def minimal_graph() -> nx.graph:
        G = nx.DiGraph()

        # Add nodes
        G.add_nodes_from([1, 2, 3])

        # Add directed edges
        G.add_edge(1, 2)
        G.add_edge(2, 1)
        G.add_edge(2, 3)
        G.add_edge(3, 1)
    
        return G