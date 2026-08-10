import networkx as nx
import pickle

import pandas as pd
from graph.build_graph import find_suppliers

# load the saved graph
with open("data\processed\graph_enriched_corrected.pkl", "rb") as f:
    g = pickle.load(f)

for name in ['steel', 'auto', 'unit', 'logistics']:
    if g.has_node(name):
        print(name, '->', dict(g.nodes[name]))
    else:
        print(name, '-> not actually a node')