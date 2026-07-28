"""
geocode_enrichment.py

Geocodes the city/country text in master_enrichment_raw.csv using Nominatim
(OpenStreetMap, free, no API key) and splits the result into locations.csv
and components.csv for graph.py's loader.

Usage:
    python geocode_enrichment.py --input master_enrichment_raw.csv --outdir data/processed

Requires:
    pip install geopy pandas

Notes:
    - Nominatim's usage policy caps requests at 1/sec and requires a descriptive
      user agent - both are handled below. Do not remove the RateLimiter or drop
      the delay below 1 second, or your IP can get temporarily blocked.
    - Results are cached to disk (geocode_cache.json) and saved after every
      lookup, so if the script crashes or your connection drops partway through
      ~400 sequential calls, rerunning it picks up where it left off instead of
      re-querying everything.
    - Rows with no city AND no country are skipped rather than guessed.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderServiceError


def load_cache(path):
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def build_query(city, country):
    parts = [p for p in [city, country] if isinstance(p, str) and p.strip()]
    return ", ".join(parts) if parts else None


def geocode_all(df, cache_path, user_agent, min_delay_seconds=1.1, max_retries=3):
    geolocator = Nominatim(user_agent=user_agent, timeout=10)
    geocode = RateLimiter(
        geolocator.geocode,
        min_delay_seconds=min_delay_seconds,
        max_retries=max_retries,
        error_wait_seconds=5.0,
        swallow_exceptions=False,
    )

    cache = load_cache(cache_path)
    results = []
    total = len(df)

    for i, row in df.iterrows():
        company = row["company"]
        query = build_query(row.get("city"), row.get("country"))

        if query is None:
            print(f"[{i + 1}/{total}] {company}: SKIPPED (no city/country)")
            results.append({"company": company, "query": None, "lat": None, "lon": None, "status": "no_city_or_country"})
            continue

        if query in cache:
            cached = cache[query]
            print(f"[{i + 1}/{total}] {company}: cached -> {query}")
            results.append({"company": company, "query": query, "lat": cached["lat"], "lon": cached["lon"], "status": cached["status"]})
            continue

        status = "ok"
        location = None
        try:
            location = geocode(query)
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"[{i + 1}/{total}] {company}: FAILED after retries ({e}) -> {query}")
            status = "error"

        if location:
            lat, lon = location.latitude, location.longitude
            print(f"[{i + 1}/{total}] {company}: {query} -> ({lat:.4f}, {lon:.4f})")
        else:
            lat, lon = None, None
            if status != "error":
                status = "not_found"
                print(f"[{i + 1}/{total}] {company}: NOT FOUND -> {query}")

        # save after every lookup, not just at the end, so progress survives a crash
        cache[query] = {"lat": lat, "lon": lon, "status": status}
        save_cache(cache, cache_path)

        results.append({"company": company, "query": query, "lat": lat, "lon": lon, "status": status})

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Geocode master_enrichment_raw.csv into locations.csv + components.csv")
    parser.add_argument("--input", default="master_enrichment_raw.csv")
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--cache", default="geocode_cache.json")
    parser.add_argument(
        "--user-agent",
        default="supply-chain-dissertation-geocoder (contact: replace-with-your-email@example.com)",
        help="Nominatim requires a descriptive user agent identifying the app/contact - replace the placeholder email.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    required = {"company", "city", "country", "component", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing expected columns: {missing}")

    geo_df = geocode_all(df, cache_path=args.cache, user_agent=args.user_agent)
    merged = df.merge(geo_df, on="company", how="left")

    # locations.csv - what graph.py needs for node coordinates
    locations = merged[["company", "city", "country", "lat", "lon", "status"]]
    locations.to_csv(outdir / "locations.csv", index=False)

    # components.csv - what graph.py needs for node component/industry attributes
    components = merged[["company", "component", "confidence"]]
    components.to_csv(outdir / "components.csv", index=False)

    n_total = len(merged)
    n_failed = (merged["status"] != "ok").sum()
    print()
    print(f"Done. {n_total - n_failed}/{n_total} geocoded successfully.")
    print(f"Wrote {outdir / 'locations.csv'} and {outdir / 'components.csv'}")
    if n_failed:
        print(f"\n{n_failed} rows need manual attention (status != 'ok'):")
        print(merged.loc[merged["status"] != "ok", ["company", "city", "country", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
