import json
import pickle
import networkx as nx
import pandas as pd

with open("data/events/events_candidates.json") as f:
    events = json.load(f)

all_companies = set()
for event in events:
    all_companies.update(event.get("companies_affected", []))
    all_companies.update(event.get("two_hop_oem_reach", []))

all_companies = sorted(all_companies)

# Words that show up in institutional/administrative noise, not company names
NOISE_KEYWORDS = [
    "union", "committee", "association", "district", "reconstruction",
    "school", "adjacent", "academy", "council", "municipality",
    "ministry", "department", "authority", "commission", "foundation",
]

MAX_WORDS = 4  # real company names are almost always <= 4 words

def is_suspect(name: str) -> bool:
    word_count = len(name.split())
    has_keyword = any(kw in name.lower() for kw in NOISE_KEYWORDS)
    return word_count > MAX_WORDS or has_keyword

flagged = [name for name in all_companies if is_suspect(name)]
print(f"Flagged {len(flagged)} out of {len(all_companies)} for manual review")

# with open("data/processed/graph_clean.pkl", "rb") as f:
#     G = pickle.load(f)

# review_rows = []
# for name in flagged:
#     if not G.has_node(name):
#         continue
#     attrs = G.nodes[name]
#     review_rows.append({
#         "node": name,
#         "industry": attrs.get("industry"),
#         "country": attrs.get("country"),
#         "in_degree": G.in_degree(name),
#         "out_degree": G.out_degree(name),
#         "word_count": len(name.split()),
#         "decision": "",  # fill in by hand: keep / remove
#     })

# pd.DataFrame(review_rows).to_csv("data/review/event_noise_candidates.csv", index=False)
# print(f"Wrote {len(review_rows)} rows to data/review/event_noise_candidates.csv")