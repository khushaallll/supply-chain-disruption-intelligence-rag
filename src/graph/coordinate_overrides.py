"""
Manual coordinate corrections for companies whose raw geocode
(from data/processed/locations.csv) landed somewhere wrong.

Background: Day 3's implementation log documented a planned
`location_overrides.csv` for 9 companies whose city/country text was
narrative rather than a real address (e.g. "USA / global", "China /
Japan"). That file was never actually created — confirmed on Day 7 by
searching the project tree. As a result, graph_enriched.pkl carries the
raw, unfixed geocode for all 9 companies. Two happened to land correctly
anyway; seven did not, two of them by an entire continent
(china steel -> Connecticut, USA; xiamen tungsten -> Uganda).

This table applies the corrections from the Day 3 log directly, without
modifying graph_enriched.pkl on disk. Verified against locations.csv and
the Day 3 log on [today's date].
"""

COORDINATE_OVERRIDES = {
    "china steel":          (22.6273, 120.3014),   # Kaohsiung, Taiwan
    "xiamen tungsten":      (24.4798, 118.0894),   # Xiamen, Fujian, China
    "united co rusal international pjsc": (55.7558, 37.6173),  # Moscow, Russia
    "exxon mobil":          (30.4515, -91.1871),   # Baton Rouge, Louisiana
    "coronado global resources": (-23.5786, 148.8794),  # Blackwater, Queensland
    "korea electric power":  (35.0160, 126.7108),  # Naju, South Korea
    "nippon steel":          (39.2667, 141.8833),  # Kamaishi, Iwate, Japan
    # gazprom pjsc and state grid corp of china were already correct
    # (within 5km and 0km respectively) - no override needed
}


def apply_coordinate_overrides(G):
    """
    Apply manual coordinate corrections to an in-memory graph.
    Does NOT modify graph_enriched.pkl on disk.
    Returns the same graph object, mutated in place, plus a log of what changed.
    """
    applied = []
    for company, (lat, lon) in COORDINATE_OVERRIDES.items():
        if G.has_node(company):
            old_lat = G.nodes[company].get("lat")
            old_lon = G.nodes[company].get("lon")
            G.nodes[company]["lat"] = lat
            G.nodes[company]["lon"] = lon
            applied.append((company, (old_lat, old_lon), (lat, lon)))
        else:
            print(f"  [override] WARNING: '{company}' not found in graph, skipped")

    return G, applied