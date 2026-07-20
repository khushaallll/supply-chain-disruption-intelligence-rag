"""Clean the raw supply chain knowledge graph CSV and generate review artifacts.

Pipeline stage: cleaning + merge-candidate generation only. No merging or
graph construction happens here — data/review/merge_candidates.csv and
data/review/conflicts_unresolved.csv are meant for manual review first.
"""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_CSV = Path("data/raw/supplychainKG.csv")
REVIEW_DIR = Path("data/review")
MERGE_CANDIDATES_CSV = REVIEW_DIR / "merge_candidates.csv"
CONFLICTS_CSV = REVIEW_DIR / "conflicts_unresolved.csv"

# Similarity threshold (0-100) for flagging a pair of company_clean values as
# a merge candidate. Tune this up/down and re-run to control recall/precision.
FUZZY_MATCH_THRESHOLD = 90

# Known bad/inconsistent country variants -> canonical form. This is
# intentionally small — extend it with entries printed by this script.
COUNTRY_MAPPING = {
    "usa": "USA",
    "united states": "USA",
    "united states of america": "USA",
    "uk": "UK",
    "united kingdom": "UK",
    "great britain": "UK",
    "uae": "UAE",
    "united arab emirates": "UAE",
    "philipines": "Philippines",  # common typo
}

LEGAL_SUFFIXES = {
    "inc", "ltd", "llc", "plc", "sa", "ag", "nv", "bv", "spa",
    "co", "corp", "gmbh", "group", "holdings",
}

# Generic corporate/industry tokens that shouldn't count toward token-set
# overlap during fuzzy scoring — stripped from the scoring input only, not
# from company_clean itself. Tuned from clusters observed in
# merge_candidates.csv: "persero"/"tbk"/"pt"/"sdn" (Indonesian/Malaysian
# corporate terms) and "steel"/"international"/"auto"/"logistics"/"asia"
# were each showing up in dozens of unrelated pairs as the sole overlap.
FUZZY_STOPWORDS = {
    "persero", "tbk", "pt", "sdn",
    "electric", "machinery", "transport", "services", "energy",
    "technology", "steel", "international", "auto", "logistics", "asia", "air",
}

# Tokens appearing in more than this fraction of unique company_clean values
# are auto-treated as generic for scoring, on top of FUZZY_STOPWORDS. Catches
# recurring-but-not-yet-listed generic words (e.g. "china", "engineering")
# without needing another manual round of "spot it in the output, add it".
AUTO_STOPWORD_DOC_FREQ_THRESHOLD = 0.012


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_country(raw):
    if pd.isna(raw):
        return raw
    stripped = str(raw).strip()
    canonical = COUNTRY_MAPPING.get(stripped.lower())
    return canonical if canonical is not None else stripped.title()


def normalize_company_name(raw):
    """Lowercase and strip trailing legal-entity suffixes (repeated, e.g.
    'Group Co Ltd' -> stripped one token at a time until none remain)."""
    if pd.isna(raw):
        return raw
    text = str(raw).lower()
    text = re.sub(r"[.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split(" ") if text else []
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
        while tokens and tokens[-1] == "&":
            tokens.pop()
    cleaned = " ".join(tokens).strip()
    return cleaned if cleaned else text


def is_punctuation_token(token):
    return not any(ch.isalnum() for ch in token)


def compute_auto_stopwords(unique_companies, threshold):
    """Tokens present in more than `threshold` fraction of unique company_clean
    values, e.g. bare country names or sector words too common to be
    distinguishing (computed from the data, not hand-picked)."""
    n = len(unique_companies)
    doc_freq = Counter()
    for name in unique_companies:
        tokens = {t for t in name.split(" ") if t and not is_punctuation_token(t)}
        doc_freq.update(tokens)
    return {t for t, cnt in doc_freq.items() if cnt / n > threshold}


def strip_scoring_stopwords(name, stopwords):
    """Scoring-only view of a company_clean value with generic corporate/
    industry tokens and bare punctuation tokens (e.g. "&") removed, so
    neither drives token-set overlap."""
    tokens = [
        t for t in name.split(" ")
        if t and t not in stopwords and not is_punctuation_token(t)
    ]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def clean_countries(df):
    raw_values = pd.concat([df["Supplier Country"], df["Customer Country"]]).dropna().unique()

    for col in ["Supplier Country", "Customer Country"]:
        df[col] = df[col].apply(normalize_country)

    unmatched = sorted(v for v in raw_values if str(v).strip().lower() not in COUNTRY_MAPPING)
    if unmatched:
        print(
            f"\n[country normalization] {len(unmatched)} raw country value(s) not covered "
            f"by COUNTRY_MAPPING (fell back to title-case). Review and extend the mapping "
            f"if any of these are inconsistent spellings rather than genuine country names:"
        )
        for v in unmatched:
            print(f"  - {v!r}")

    return df


def build_companies_table(df):
    """Long-form table: one row per (company occurrence), combining the
    Supplier side and Customer side of every relationship row."""
    supplier_side = df[["Supplier", "Supplier Industry", "Supplier Country"]].rename(
        columns={"Supplier": "company_raw", "Supplier Industry": "industry", "Supplier Country": "country"}
    )
    customer_side = df[["Customer", "Customer Industry", "Customer Country"]].rename(
        columns={"Customer": "company_raw", "Customer Industry": "industry", "Customer Country": "country"}
    )
    companies = pd.concat([supplier_side, customer_side], ignore_index=True)
    companies["company_clean"] = companies["company_raw"].apply(normalize_company_name)
    return companies


# Minimum tokens a stopword-stripped name must retain to be eligible for
# token_set_ratio scoring. Below this (e.g. "china steel" -> "china" after
# stripping "steel"), token_set_ratio's subset special-case makes a single
# short/common leftover token match dozens of unrelated companies at ~100.
MIN_TOKENS_FOR_TOKEN_SET = 2


def generate_merge_candidates(companies):
    unique_companies = sorted(companies["company_clean"].dropna().unique().tolist())
    counts = companies["company_clean"].value_counts()
    n = len(unique_companies)

    auto_stopwords = compute_auto_stopwords(unique_companies, AUTO_STOPWORD_DOC_FREQ_THRESHOLD)
    new_auto_stopwords = auto_stopwords - FUZZY_STOPWORDS
    if new_auto_stopwords:
        print(
            f"\n[fuzzy match] auto-stopwords (doc freq > {AUTO_STOPWORD_DOC_FREQ_THRESHOLD:.1%}, "
            f"not already in FUZZY_STOPWORDS): {sorted(new_auto_stopwords)}"
        )
    scoring_stopwords = FUZZY_STOPWORDS | auto_stopwords

    stripped = [strip_scoring_stopwords(c, scoring_stopwords) for c in unique_companies]
    orig_token_count = np.array([len(c.split()) for c in unique_companies])
    stripped_token_count = np.array([len(s.split()) for s in stripped])
    # Eligible for token_set_ratio unless stopword-stripping is what pushed
    # it under the token minimum (e.g. "china steel" -> "china"). A name
    # that was already a single token before stripping (e.g. "denso", which
    # contains no stopwords) stays eligible — it's a genuinely short name,
    # not a stopword-driven degenerate one.
    eligible = (stripped_token_count >= MIN_TOKENS_FOR_TOKEN_SET) | (stripped_token_count == orig_token_count)
    n_fallback = n - int(eligible.sum())

    print(
        f"\n[fuzzy match] comparing {n} unique company_clean values (threshold={FUZZY_MATCH_THRESHOLD}); "
        f"{n_fallback} degenerate to <{MIN_TOKENS_FOR_TOKEN_SET} tokens because of stopword-stripping and use "
        f"token_sort_ratio-on-original fallback instead of token_set_ratio..."
    )

    # Pairs where both sides retain enough tokens post-strip: best of
    # token_sort_ratio/token_set_ratio on the stripped strings (catches both
    # reorder/typo and superset/subset variants, stopwords excluded).
    set_matrix = process.cdist(stripped, stripped, scorer=fuzz.token_set_ratio, workers=-1)
    sort_matrix_stripped = process.cdist(stripped, stripped, scorer=fuzz.token_sort_ratio, workers=-1)
    eligible_score = np.maximum(set_matrix, sort_matrix_stripped)

    # Pairs where either side strips down to <2 tokens: token_set_ratio is
    # skipped entirely and we fall back to token_sort_ratio on the full,
    # unstripped company_clean strings (no special-cased subset matching).
    fallback_score = process.cdist(unique_companies, unique_companies, scorer=fuzz.token_sort_ratio, workers=-1)

    pair_eligible = np.outer(eligible, eligible)
    score_matrix = np.where(pair_eligible, eligible_score, fallback_score)

    company_a, company_b, scores = [], [], []
    for i in range(n):
        row = score_matrix[i, i + 1:]
        if row.size == 0:
            continue
        hits = np.nonzero(row >= FUZZY_MATCH_THRESHOLD)[0]
        for offset in hits:
            j = i + 1 + offset
            company_a.append(unique_companies[i])
            company_b.append(unique_companies[j])
            scores.append(row[offset])

    merge_candidates = pd.DataFrame({
        "company_a": company_a,
        "company_b": company_b,
        "similarity_score": scores,
        "count_a": [int(counts[c]) for c in company_a],
        "count_b": [int(counts[c]) for c in company_b],
    }).sort_values("similarity_score", ascending=False, ignore_index=True)

    return merge_candidates


def resolve_conflicts(companies):
    conflict_rows = []

    for company, group in companies.groupby("company_clean"):
        for field in ("industry", "country"):
            value_counts = group[field].dropna().value_counts()
            if len(value_counts) <= 1:
                continue
            top_count = value_counts.iloc[0]
            tied = value_counts[value_counts == top_count]
            if len(tied) > 1:
                conflict_rows.append({
                    "company_clean": company,
                    "field": field,
                    "tied_values": ", ".join(tied.index.tolist()),
                    "tied_count": int(top_count),
                    "all_values": ", ".join(f"{v} ({c})" for v, c in value_counts.items()),
                })
            # else: resolved by majority vote (most frequent value wins);
            # no merging is performed in this script, so the winner is not
            # written anywhere yet — only unresolved ties are logged.

    return pd.DataFrame(conflict_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv(RAW_CSV)
    total_rows = len(df)

    # 1. Drop unused relationship columns
    df = df.drop(columns=["Relationship2", "Relationship3"])

    # 2. Normalize country labels
    df = clean_countries(df)

    # 3. Normalize company names (company_raw / company_clean)
    companies = build_companies_table(df)
    unique_before = companies["company_raw"].nunique()
    unique_after = companies["company_clean"].nunique()

    # 4. Fuzzy match merge candidates (review only, no auto-merge)
    merge_candidates = generate_merge_candidates(companies)

    # 5. Resolve industry/country conflicts per company (majority vote; log ties)
    conflicts = resolve_conflicts(companies)

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    merge_candidates.to_csv(MERGE_CANDIDATES_CSV, index=False)
    conflicts.to_csv(CONFLICTS_CSV, index=False)

    print("\n=== Cleaning summary ===")
    print(f"Total rows: {total_rows}")
    print(f"Unique companies before cleaning: {unique_before}")
    print(f"Unique companies after cleaning: {unique_after}")
    print(f"Merge candidates generated: {len(merge_candidates)} -> {MERGE_CANDIDATES_CSV}")
    print(f"Unresolved conflicts logged: {len(conflicts)} -> {CONFLICTS_CSV}")


if __name__ == "__main__":
    main()
