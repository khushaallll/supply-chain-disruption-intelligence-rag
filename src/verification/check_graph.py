import networkx as nx
import pickle

import pandas as pd
from graph.build_graph import find_suppliers

# Load the saved graph
with open("data/processed/graph_enriched_corrected.pkl", "rb") as f:
    g = pickle.load(f)

# Basic graph statistics
num_nodes = g.number_of_nodes()
num_edges = g.number_of_edges()

# Count companies (nodes) with the "component" attribute attached
companies_with_component = sum(
    1 for _, attrs in g.nodes(data=True)
    if "component" in attrs and attrs["component"] is not None
)

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")
print(f"Companies with component attribute: {companies_with_component}")