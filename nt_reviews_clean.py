#!/usr/bin/env python
"""
NT Tourism — Text Cleaning & Processing Script (NO topic modeling)

What it does:
1) Normalise column names (lowercase, trim, spaces->underscores)
2) Validate required columns: place_id, place_name, region, rating, text
3) Drop empty review texts (trimmed)
4) Coerce rating to numeric and clip to [1..5]; drop rows where rating is NaN after coercion
5) Create sentiment class from rating: positive (4,5), neutral (3), negative (1,2)
6) Anonymise author_name -> author_hash = sha256(author_name)[:12]
   (drop author_name unless --keep-author-name)
7) Clean text -> text_clean (lowercase, remove URLs, keep letters/digits/spaces, compress whitespace)
8) Build stable review_id = md5(place_id|text_clean[:120]|rating|region)
9) De-duplicate by review_id
10) Write single cleaned CSV

Usage:
  python nt_reviews_clean.py --in places_reviews.csv --out nt_reviews_cleaned.csv [--keep-author-name]
"""

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


def normalise_col(col: str) -> str:
    return re.sub(r"\s+", "_", col.strip().lower())


URL_RE = re.compile(r"(http[s]?://\S+|www\.\S+)")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def clean_text_simple(t: str) -> str:
    t = ("" if t is None else str(t)).lower()
    t = URL_RE.sub(" ", t)
    t = NON_ALNUM_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def sha256_12(x) -> str | float:
    import numpy as _np
    if x is None or (isinstance(x, float) and _np.isnan(x)):
        return _np.nan
    import hashlib as _hashlib
    return _hashlib.sha256(str(x).encode("utf-8")).hexdigest()[:12]


def review_md5(place_id: str, text_clean: str, rating: float, region: str) -> str:
    slice_120 = (text_clean or "")[:120]
    s = f"{place_id}|{slice_120}|{int(rating)}|{region}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def to_sentiment_class(r: float) -> str:
    r = float(r)
    if r in (4.0, 5.0):
        return "positive"
    if r == 3.0:
        return "neutral"
    return "negative"  # 1 or 2


def run_clean(in_csv: Path, out_csv: Path, keep_author_name: bool) -> None:
    # Read
    df_raw = pd.read_csv(in_csv)

    # 1) Normalise column names
    df = df_raw.copy()
    df.columns = [normalise_col(c) for c in df.columns]

    # 2) Validate required columns
    required = ["place_id", "place_name", "region", "rating", "text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalisation: {missing}")

    # 3) Drop empty review texts
    df["text"] = df["text"].astype(str).str.strip()
    before = len(df)
    df = df[df["text"].str.len() > 0].copy()
    dropped_empty = before - len(df)

    # 4) rating -> numeric, clip to 1..5; drop NaNs after coercion
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").clip(lower=1, upper=5)
    before = len(df)
    df = df[~df["rating"].isna()].copy()
    dropped_bad_rating = before - len(df)

    # 5) sentiment classes
    df["sentiment"] = df["rating"].apply(to_sentiment_class)

    # 6) author hash + optional drop
    if "author_name" in df.columns:
        df["author_hash"] = df["author_name"].apply(sha256_12)
        if not keep_author_name:
            df = df.drop(columns=["author_name"])
    else:
        df["author_hash"] = np.nan

    # 7) text_clean
    df["text_clean"] = df["text"].apply(clean_text_simple)

    # 8) review_id
    df["review_id"] = df.apply(
        lambda r: review_md5(
            str(r.get("place_id", "")),
            str(r.get("text_clean", "")),
            float(r.get("rating", 0)),
            str(r.get("region", "")),
        ),
        axis=1,
    )

    # 9) de-duplicate on review_id
    before = len(df)
    df = df.drop_duplicates(subset=["review_id"]).copy()
    deduped = before - len(df)

    # Reorder key columns
    front = [
        "review_id",
        "place_id",
        "place_name",
        "region",
        "rating",
        "sentiment",
        "text_clean",
        "author_hash",
    ]
    front_existing = [c for c in front if c in df.columns]
    others = [c for c in df.columns if c not in front_existing]
    df_out = df[front_existing + others]

    # Save
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # Simple report
    print("=== NT Reviews Clean ===")
    print(f"Input rows:           {len(df_raw):,}")
    print(f"Dropped empty texts:  {dropped_empty:,}")
    print(f"Dropped bad ratings:  {dropped_bad_rating:,}")
    print(f"Duplicates removed:   {deduped:,}")
    print(f"Output rows:          {len(df_out):,}")
    print(f"Wrote:                {out_csv}")


def parse_args():
    ap = argparse.ArgumentParser(description="Clean NT tourism reviews (no topic modeling).")
    ap.add_argument("--in", dest="in_csv", required=True, help="Path to raw places_reviews.csv")
    ap.add_argument("--out", dest="out_csv", required=True, help="Path to write cleaned CSV")
    ap.add_argument("--keep-author-name", action="store_true", help="Keep author_name column (default: dropped)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_clean(
        in_csv=Path(args.in_csv),
        out_csv=Path(args.out_csv),
        keep_author_name=args.keep_author_name,
    )
