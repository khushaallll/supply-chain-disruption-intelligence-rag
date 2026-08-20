import pickle
from datetime import date
from coordinate_overrides import apply_coordinate_overrides

INPUT_PATH = "data/processed/graph_enriched.pkl"
OUTPUT_PATH = "data/processed/graph_enriched_corrected2.pkl"
LOG_PATH = "data/processed/coordinate_override_log.txt"

with open(INPUT_PATH, "rb") as f:
    G = pickle.load(f)

G, override_log = apply_coordinate_overrides(G)

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(G, f)

with open(LOG_PATH, "w") as f:
    f.write(f"Coordinate override applied {date.today()}\n")
    f.write(f"Source: {INPUT_PATH} (left unmodified)\n")
    f.write(f"Output: {OUTPUT_PATH}\n")
    f.write("Reason: location_overrides.csv from Day 3 log was never actually\n")
    f.write("created; 7 of 9 flagged companies carried raw, incorrect geocodes.\n\n")
    for company, old, new in override_log:
        f.write(f"{company}: {old} -> {new}\n")

print(f"Applied {len(override_log)} overrides.")
print(f"Saved corrected graph to {OUTPUT_PATH}")
print(f"Log written to {LOG_PATH}")