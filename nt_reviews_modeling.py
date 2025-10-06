#!/usr/bin/env python3
"""
NT Tourism — Modeling Script (Topics + Sentiment Score)

Takes CLEANED CSV from nt_reviews_clean.py and adds:
- topic (BERTopic label per review)
- sentiment_score (VADER compound in [-1,1])

Usage:
  python nt_reviews_modeling.py \
    --in data/processed/nt_reviews_cleaned.csv \
    --out data/processed/nt_reviews_modeled.csv \
    --embedding-model all-MiniLM-L6-v2 \
    [--save-topic-model topic_model]
    [--min-words 3]

Dependencies:
  pip install pandas numpy bertopic sentence-transformers hdbscan umap-learn nltk
  python -c "import nltk; nltk.download('vader_lexicon')"
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# UMAP for dimensionality reduction (seed here, not in BERTopic ctor)
from umap import UMAP

# VADER for sentiment score
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


def require_columns(df: pd.DataFrame, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")


def build_topic_labels(topic_model: BERTopic, topn: int = 4) -> dict[int, str]:
    labels = {}
    info = topic_model.get_topic_info()
    for _, row in info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            labels[tid] = "outliers"
            continue
        words = topic_model.get_topic(tid) or []
        phrase = " ".join([w for w, _ in words[:topn]]) if words else f"topic_{tid}"
        labels[tid] = phrase
    return labels


def run_modeling(in_csv: Path, out_csv: Path, embedding_model_name: str, save_topic_model: str | None, min_words: int, seed: int) -> None:
    df = pd.read_csv(in_csv)
    require_columns(df, ["review_id", "text_clean"])

    # Prep docs (pad very short docs)
    docs = df["text_clean"].fillna("").tolist()
    docs_proc = [d if len(d.split()) >= min_words else (d + " placeholder") for d in docs]

    # Embedding model
    embedder = SentenceTransformer(embedding_model_name)

    # Ensure VADER lexicon exists (silent no-op if already downloaded)
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    # BERTopic with UMAP seeding (version-safe; no random_state in BERTopic ctor)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed)
    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        verbose=False,
        calculate_probabilities=False,
        low_memory=True,
    )

    topics, _ = topic_model.fit_transform(docs_proc)

    # Map topic ids to readable labels
    id2label = build_topic_labels(topic_model, topn=4)
    df["_topic_id"] = topics
    df["topic"] = df["_topic_id"].map(id2label).fillna("outliers")
    df = df.drop(columns=["_topic_id"])

    # Sentiment score via VADER
    sia = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df["text_clean"].fillna("").apply(lambda t: sia.polarity_scores(t)["compound"])

    # Save outputs
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    if save_topic_model:
        topic_model.save(str(Path(save_topic_model)))

    print("=== NT Reviews Modeling ===")
    print(f"Input rows:      {len(docs):,}")
    print(f"Output rows:     {len(df):,}")
    print(f"Wrote:           {out_csv}")
    if save_topic_model:
        print(f"Saved topic model to: {save_topic_model}")


def parse_args():
    ap = argparse.ArgumentParser(description="Add BERTopic topics + VADER sentiment_score to CLEANED reviews.")
    ap.add_argument("--in", dest="in_csv", required=True, help="Path to cleaned CSV from nt_reviews_clean.py")
    ap.add_argument("--out", dest="out_csv", required=True, help="Path to write modeled CSV")
    ap.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                    help="SentenceTransformer model (e.g., all-MiniLM-L6-v2, all-mpnet-base-v2)")
    ap.add_argument("--save-topic-model", type=str, default=None, help="Optional path prefix to save the BERTopic model")
    ap.add_argument("--min-words", type=int, default=3, help="Min words per doc before padding (default: 3)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for UMAP")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_modeling(
        in_csv=Path(args.in_csv),
        out_csv=Path(args.out_csv),
        embedding_model_name=args.embedding_model,
        save_topic_model=args.save_topic_model,
        min_words=args.min_words,
        seed=args.seed,
    )
