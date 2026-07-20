import networkx as nx
import pickle

import pandas as pd
from graph import find_suppliers

# load the saved graph
with open("data/processed/graph_clean.pkl", "rb") as f:
    g = pickle.load(f)

suppliers = find_suppliers(g, "ford motor", max_tier=1)
print(len(suppliers))

df = pd.read_csv("data/raw/supplychainKG.csv")
ford_rows = df[df["Customer"].str.lower().str.contains("ford", na=False)]
print(ford_rows["Supplier"].nunique())

# see exactly which "Customer" values matched
matches = df[df["Customer"].str.lower().str.contains("ford", na=False)]["Customer"].unique()
print(matches)