"""
nt_regions_reviews.py
Collect Google Places + Reviews for populated NT regions only.

Adds lat/lng to review rows and a review_date in dd-mm-YYYY format.
Output paths can be provided via CLI:
  --csv /path/to/file.csv
  --db /path/to/file.db
  --outdir /path/to/folder   (writes defaults inside)

Prereqs:
- GOOGLE_API_KEY in environment (dotenv supported)
- pip install requests pandas python-dotenv
"""

import os
from dotenv import load_dotenv
import time
import math
import requests
import pandas as pd
import sqlite3
from datetime import datetime
import argparse
from typing import Dict, Any, Iterable, List, Tuple

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set")

# ---------- Tunables ----------
DEFAULT_OUTPUT_CSV = "places_reviews.csv"
DEFAULT_OUTPUT_DB  = "places_reviews.db"

# Gentle pacing & retries
SLEEP_BETWEEN_REQUESTS = 0.9
SLEEP_BEFORE_NEXT_PAGE = 2.6
MAX_RETRIES = 5

# Place types to target (edit as you like)
PLACE_TYPES = [
    "tourist_attraction",
    "restaurant",
    "bar",
    "lodging",           # hotels
    "amusement_park",    # activity
    "park",
    "museum",
    "transit_station"    # transport hubs
]

# Fields for Place Details (reviews included by 'reviews')
DETAILS_FIELDS = "name,rating,user_ratings_total,formatted_address,geometry,reviews,types,place_id"

# Regions: center lat/lng, radius_km, grid_step_km (smaller step => denser coverage => more API calls)
REGIONS = [
    {"name": "Darwin_Palmerston", "lat": -12.4634, "lng": 130.8456, "radius_km": 30, "step_km": 7},
    {"name": "Alice_Springs",     "lat": -23.6980, "lng": 133.8807, "radius_km": 20, "step_km": 7},
    {"name": "Katherine",         "lat": -14.4669, "lng": 132.2630, "radius_km": 18, "step_km": 7},
    {"name": "Tennant_Creek",     "lat": -19.6470, "lng": 134.1900, "radius_km": 15, "step_km": 8},
    {"name": "Nhulunbuy",         "lat": -12.1840, "lng": 136.7790, "radius_km": 15, "step_km": 8},
    {"name": "Jabiru",            "lat": -12.6710, "lng": 132.8330, "radius_km": 15, "step_km": 8},
    {"name": "Yulara_Uluru",      "lat": -25.2400, "lng": 130.9880, "radius_km": 18, "step_km": 7},
]

# Nearby Search supports radius up to 50,000 m
NEARBY_RADIUS_M = 50000
# ------------------------------


def _request_json(url: str, params: Dict[str, Any], retries: int = MAX_RETRIES) -> Dict[str, Any]:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            print(f"[WARN] HTTP {resp.status_code}: {resp.text[:180]}")
        except requests.RequestException as e:
            print(f"[WARN] Request error: {e}")
        backoff = min(30, 1.5 * attempt ** 2)
        print(f"[INFO] Retry {attempt}/{retries} in {backoff:.1f}s")
        time.sleep(backoff)
    raise RuntimeError("Exceeded max retries")


def km_to_deg_lat(km: float) -> float:
    # ~111 km per degree of latitude
    return km / 111.0


def km_to_deg_lng(km: float, at_lat_deg: float) -> float:
    # longitudinal degree width shrinks by cos(latitude)
    return km / (111.0 * max(0.2, math.cos(math.radians(at_lat_deg))))


def points_in_disc(lat0: float, lng0: float, radius_km: float, step_km: float) -> List[Tuple[float, float]]:
    """Generate approx square grid over a disc (center, radius_km), spaced by step_km."""
    dlat = km_to_deg_lat(step_km)
    lat_min = lat0 - km_to_deg_lat(radius_km)
    lat_max = lat0 + km_to_deg_lat(radius_km)

    pts: List[Tuple[float, float]] = []
    lat = lat_min
    while lat <= lat_max:
        dy_km = abs((lat - lat0) * 111.0)
        if dy_km <= radius_km:
            half_chord_km = math.sqrt(max(0.0, radius_km**2 - dy_km**2))
            dlng = km_to_deg_lng(step_km, lat)
            lng_min = lng0 - km_to_deg_lng(half_chord_km, lat)
            lng_max = lng0 + km_to_deg_lng(half_chord_km, lat)
            lng = lng_min
            while lng <= lng_max:
                pts.append((round(lat, 6), round(lng, 6)))
                lng += dlng
        lat += dlat
    return pts


def nearby_search(lat: float, lng: float, ptype: str | None = None) -> List[Dict[str, Any]]:
    base = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params: Dict[str, Any] = {
        "key": API_KEY,
        "location": f"{lat},{lng}",
        "radius": NEARBY_RADIUS_M,
    }
    if ptype:
        params["type"] = ptype

    data = _request_json(base, params)
    out = data.get("results", [])
    next_token = data.get("next_page_token")

    while next_token:
        time.sleep(SLEEP_BEFORE_NEXT_PAGE)
        data = _request_json(base, {"key": API_KEY, "pagetoken": next_token})
        out.extend(data.get("results", []))
        next_token = data.get("next_page_token")

    time.sleep(SLEEP_BETWEEN_REQUESTS)
    return out


def get_place_details(place_id: str) -> Dict[str, Any] | None:
    base = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {"place_id": place_id, "fields": DETAILS_FIELDS, "key": API_KEY}
    data = _request_json(base, params)
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    return data.get("result")


def setup_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS places (
        place_id TEXT PRIMARY KEY,
        name TEXT,
        rating REAL,
        user_ratings_total INTEGER,
        formatted_address TEXT,
        lat REAL, lng REAL,
        types TEXT,
        region TEXT,
        fetched_at TEXT
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        place_id TEXT,
        author_name TEXT,
        rating INTEGER,
        text TEXT,
        time_description TEXT,
        review_date TEXT,   -- dd-mm-YYYY
        lat REAL,
        lng REAL,
        fetched_at TEXT
    )""")
    conn.commit()
    _ensure_review_columns(conn)  # if table existed from older runs, add missing cols
    return conn


def _ensure_review_columns(conn: sqlite3.Connection) -> None:
    """Adds new columns to `reviews` if an older DB exists without them."""
    need_cols = {"review_date": "TEXT", "lat": "REAL", "lng": "REAL"}
    cur = conn.cursor()
    cur.execute("PRAGMA table_info('reviews')")
    existing = {row[1] for row in cur.fetchall()}  # column names
    for col, typ in need_cols.items():
        if col not in existing:
            try:
                cur.execute(f"ALTER TABLE reviews ADD COLUMN {col} {typ}")
                conn.commit()
                print(f"[INFO] Added column reviews.{col}")
            except sqlite3.Error as e:
                print(f"[WARN] Could not add column {col}: {e}")


def parse_review_date_dmy(review_obj: Dict[str, Any]) -> str | None:
    """
    Google review objects often include 'time' (unix seconds) in addition to
    'relative_time_description'. Use 'time' for exact date if present.
    """
    ts = review_obj.get("time")
    if isinstance(ts, (int, float)):
        try:
            return datetime.utcfromtimestamp(int(ts)).strftime("%d-%m-%Y")
        except Exception:
            pass
    # Fallback: None (we avoid trying to parse the relative description)
    return None


def main():
    args = parse_args()

    output_csv = args.csv or (os.path.join(args.outdir, DEFAULT_OUTPUT_CSV) if args.outdir else DEFAULT_OUTPUT_CSV)
    output_db  = args.db  or (os.path.join(args.outdir, DEFAULT_OUTPUT_DB)  if args.outdir else DEFAULT_OUTPUT_DB)

    print("[START] Populated NT regions collection")
    print(f"[CFG] CSV -> {output_csv}")
    print(f"[CFG] DB  -> {output_db}")

    # 1) Build regional grids
    regional_points: List[Tuple[str, List[Tuple[float, float]]]] = []
    for r in REGIONS:
        pts = points_in_disc(r["lat"], r["lng"], r["radius_km"], r["step_km"])
        print(f"[INFO] {r['name']}: {len(pts)} centers (radius={r['radius_km']} km, step={r['step_km']} km)")
        regional_points.append((r["name"], pts))

    # 2) Collect place_ids (dedupe)
    place_index: Dict[str, Dict[str, Any]] = {}
    for region_name, pts in regional_points:
        for i, (lat, lng) in enumerate(pts, 1):
            for ptype in PLACE_TYPES:
                print(f"[SCAN] {region_name} {i}/{len(pts)} | type={ptype} at {lat},{lng}")
                try:
                    results = nearby_search(lat, lng, ptype)
                    for r in results:
                        pid = r.get("place_id")
                        if not pid:
                            continue
                        if pid not in place_index:
                            place_index[pid] = {
                                "name": r.get("name"),
                                "types": ",".join(r.get("types", [])[:10]),
                                "lat": (r.get("geometry", {}) or {}).get("location", {}).get("lat"),
                                "lng": (r.get("geometry", {}) or {}).get("location", {}).get("lng"),
                                "region": region_name
                            }
                except Exception as e:
                    print(f"[ERROR] Nearby search failed at {lat},{lng} ({ptype}): {e}")

    print(f"[INFO] Unique places found: {len(place_index)}")

    # 3) Details + reviews -> DB + CSV
    conn = setup_db(output_db)
    cur = conn.cursor()
    review_rows: List[Dict[str, Any]] = []

    for idx, (pid, meta) in enumerate(place_index.items(), 1):
        print(f"[DETAIL] {idx}/{len(place_index)} {meta.get('name')} ({pid})")
        try:
            det = get_place_details(pid)
            if not det:
                continue
            name = det.get("name")
            rating = det.get("rating")
            urt = det.get("user_ratings_total")
            addr = det.get("formatted_address")
            loc = (det.get("geometry", {}) or {}).get("location", {}) or {}
            place_lat = loc.get("lat", meta.get("lat"))
            place_lng = loc.get("lng", meta.get("lng"))
            types = ",".join(det.get("types", []))
            fetched_at_iso = datetime.utcnow().isoformat()

            cur.execute("""
                INSERT OR REPLACE INTO places (place_id,name,rating,user_ratings_total,formatted_address,lat,lng,types,region,fetched_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (pid, name, rating, urt, addr, place_lat, place_lng, types, meta["region"], fetched_at_iso))

            for rv in det.get("reviews", []) or []:
                review_date_dmy = parse_review_date_dmy(rv)
                cur.execute("""
                    INSERT INTO reviews (place_id,author_name,rating,text,time_description,review_date,lat,lng,fetched_at)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (
                    pid,
                    rv.get("author_name"),
                    rv.get("rating"),
                    rv.get("text"),
                    rv.get("relative_time_description"),
                    review_date_dmy,
                    place_lat,
                    place_lng,
                    fetched_at_iso
                ))

                review_rows.append({
                    "place_id": pid,
                    "place_name": name,
                    "region": meta["region"],
                    "author_name": rv.get("author_name"),
                    "rating": rv.get("rating"),
                    "text": rv.get("text"),
                    "time_description": rv.get("relative_time_description"),
                    "review_date": review_date_dmy,          # dd-mm-YYYY
                    "lat": place_lat,
                    "lng": place_lng,
                })

            conn.commit()
        except Exception as e:
            print(f"[ERROR] Details/reviews for {pid}: {e}")

    # 4) CSV for reviews
    if review_rows:
        df = pd.DataFrame(review_rows)
        # Ensure column order is friendly
        col_order = [
            "place_id", "place_name", "region",
            "author_name", "rating", "text",
            "time_description", "review_date",
            "lat", "lng"
        ]
        df = df[[c for c in col_order if c in df.columns]]
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"[SAVE] Reviews CSV -> {output_csv}")
    else:
        print("[INFO] No reviews fetched (Places Details often returns only a few).")

    conn.close()
    print("[DONE]")


def parse_args():
    p = argparse.ArgumentParser(description="Collect Google Places + Reviews for NT regions")
    p.add_argument("--csv", type=str, help="Path to output CSV of reviews")
    p.add_argument("--db", type=str, help="Path to output SQLite DB")
    p.add_argument("--outdir", type=str, help="Directory to write default outputs")
    return p.parse_args()


if __name__ == "__main__":
    main()
