# NT Tourism Heatmap — AI-Driven Sentiment & Policy Insights
CDU IT Code Fair 2025 — Data Science Challenge  
Author: Rafsan Rahman (S378025)

## Project Overview
NT Tourism Heatmap turns real-world Google reviews into actionable insights for the Northern Territory. The pipeline collects reviews across key regions, cleans and anonymizes the data, assigns topic labels and text sentiment, and produces a modeling-ready dataset to support evidence-based tourism policy and business investment.

### Objectives
- Collect and analyze tourist reviews across major NT regions.
- Build a sentiment heatmap and topic clusters to visualize visitor experience.
- Provide predictive inputs to simulate the impact of investments (e.g., transport, Wi-Fi, facilities).

---

## Directory Structure
```
│   .env
│   .gitignore
│   nt_regions_reviews.py
│   nt_reviews_clean.py
│   nt_reviews_modeling.py
│   NT_Tourism_Mini_Overview.html
│   README.md
│   requirements.txt
│   Data analytics
└───data
    ├───processed
    │       nt_reviews_cleaned.csv
    │       nt_reviews_modeled.csv
    │
    └───raw
            places_reviews.csv
            places_reviews.db
```

---

## Setup

### 1) Clone and create a virtual environment
```bash
git clone https://github.com/<your-username>/nt-tourism-heatmap.git
cd nt-tourism-heatmap

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Configure API credentials
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_google_places_api_key_here
```
Ensure Google Places API is enabled and billing is active.

---

## End-to-End Pipeline: Run Order

### Step 1 — Data Scraping
Collect tourism places and reviews for NT regions; writes CSV and SQLite DB.
```bash
python nt_regions_reviews.py
```
**Outputs**
- `data/raw/places_reviews.csv`
- `data/raw/places_reviews.db`

**What this does / Why**
- Queries Google Places Nearby + Details APIs across configured NT regions.
- Captures place metadata (name, address, lat/lng, types) and recent reviews.
- Provides geo-anchored review data required for mapping and policy analysis.

---

### Step 2 — Data Cleaning & Processing
Normalize columns, clean text, anonymize authors, classify rating-based sentiment, de-duplicate.
```bash
python nt_reviews_clean.py \
  --in data/raw/places_reviews.csv \
  --out data/processed/nt_reviews_cleaned.csv
# Optional: --keep-author-name
```
**Outputs**
- `data/processed/nt_reviews_cleaned.csv`

**What this does / Why**
- Normalizes column names for consistency.
- Validates required fields (`place_id`, `place_name`, `region`, `rating`, `text`) to ensure integrity.
- Trims and drops empty reviews; coerces `rating` to [1..5] to prevent scoring drift.
- Creates `sentiment` class from rating: positive (4,5), neutral (3), negative (1,2) for a clear, rules-based baseline.
- Anonymizes `author_name` to `author_hash` (SHA-256 short) for privacy compliance.
- Produces `text_clean` (lowercase, URLs removed, alphanumeric only) for stable NLP.
- Builds stable `review_id` (hash of place_id | first 120 chars of text_clean | rating | region) and de-duplicates to remove repeats.

---

### Step 3 — Topic Modeling & Text Sentiment Scoring
Assigns per-review topics (BERTopic) and VADER sentiment score.
```bash
python nt_reviews_modeling.py \
  --in data/processed/nt_reviews_cleaned.csv \
  --out data/processed/nt_reviews_modeled.csv \
  --embedding-model all-MiniLM-L6-v2
# Optional: --save-topic-model nt_reviews_modeled
# Optional: --min-words 3
# Optional: --seed 42
```
**Outputs**
- `data/processed/nt_reviews_modeled.csv`
- Optional saved BERTopic model: `nt_reviews_modeled.json` (or directory, depending on save path)

**What this does / Why**
- Uses Sentence-Transformers embeddings + BERTopic to cluster semantically similar reviews into interpretable topics (e.g., transport, amenities, staff, nature).
- Adds `topic` label per review based on top words, enabling targeted interventions.
- Computes VADER `sentiment_score` in [-1, 1] from review text to complement star ratings (captures language tone).

---

## Data Flow Diagram
```
Google Places API
       │
       ▼
[ Scraper: nt_regions_reviews.py ]
       │
       ▼
data/raw/places_reviews.(csv|db)
       │
       ▼
[ Cleaner: nt_reviews_clean.py ]
       │
       ▼
data/processed/nt_reviews_cleaned.csv
       │
       ▼
[ Modeling: nt_reviews_modeling.py ]
       │
       ▼
data/processed/nt_reviews_modeled.csv (+ optional model json)
```

---

## Rationale for Key Processing & Modeling Steps
- Normalize + validate columns: ensures stable schemas for downstream scripts and BI tools.
- Drop empty text + coerce ratings: avoids misleading rows and preserves numeric comparability.
- Rating-based sentiment class: simple, auditable baseline tied to platform semantics (1–5 stars).
- Text cleaning + anonymization: compliant with privacy expectations and necessary for quality NLP.
- Stable `review_id` + de-duplication: prevents double counting and inflating trends.
- BERTopic topics: interpretable clusters for policy actions and business recommendations.
- VADER sentiment score: captures textual tone beyond numeric ratings (praise vs complaint nuance).

---

## Brief overview of the data
Open the `NT_Tourism_Mini_Overview.html` file and upload the `nt_reviews_modeled.json` file to see a brief overview of the data.
