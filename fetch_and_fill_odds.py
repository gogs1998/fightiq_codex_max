"""
Fetch missing 2024/2025 odds from The Odds API (v4) and fill gaps in the local dataset.
Requires: requests
API docs: https://the-odds-api.com/liveapi/guides/v4/#endpoint-9
"""
import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import requests

from train_fightiq_model import DATA_DIR, attach_odds


API_KEY = "74b4ca301791b4b4c6ebe95897ac8673"
SPORT_KEY = "mma_mixed_martial_arts"
BASE_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds-history"


def norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def name_tokens(name: str) -> set:
    return set(norm_name(name).split())


def fetch_snapshot(ts: datetime) -> dict:
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "h2h",
        "date": ts.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "oddsFormat": "decimal",
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def extract_prices(event: dict) -> Tuple[float, float]:
    home_price = away_price = None
    for bm in event.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = market.get("outcomes", [])
            for out in outcomes:
                if out.get("name") == event.get("home_team"):
                    home_price = out.get("price", home_price)
                elif out.get("name") == event.get("away_team"):
                    away_price = out.get("price", away_price)
    return home_price, away_price


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("Data/UFC_betting_odds_enriched.csv"), help="Where to write filled odds"
    )
    args = parser.parse_args()

    df = pd.read_csv(DATA_DIR / "UFC_full_data_silver.csv")
    odds = pd.read_csv(DATA_DIR / "UFC_betting_odds.csv")
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = attach_odds(df, odds)

    mask = df["event_date"].dt.year.isin([2024, 2025])
    missing_mask = df[["f_1_odds", "f_2_odds"]].isna().any(axis=1)
    missing = df.loc[mask & missing_mask]
    if missing.empty:
        print("No missing odds for 2024/2025.")
        return
    print(f"Missing odds rows: {len(missing)}")

    # Fetch snapshots per event_date (midday UTC)
    cache: Dict[pd.Timestamp, dict] = {}
    for date in missing["event_date"].dt.date.unique():
        ts = datetime.combine(date, datetime.min.time()) + timedelta(hours=12)
        try:
            cache[pd.Timestamp(date)] = fetch_snapshot(ts)
            print(f"Fetched odds snapshot for {date}")
        except Exception as e:
            print(f"Failed to fetch {date}: {e}")

    filled = []
    for idx, row in missing.iterrows():
        date_key = pd.Timestamp(row["event_date"].date())
        snap = cache.get(date_key)
        if not snap:
            continue
        best_match = None
        f1_tokens, f2_tokens = name_tokens(row["f_1_name"]), name_tokens(row["f_2_name"])
        for event in snap.get("data", []):
            home_tokens = name_tokens(event.get("home_team"))
            away_tokens = name_tokens(event.get("away_team"))
            cond1 = f1_tokens.issubset(home_tokens) and f2_tokens.issubset(away_tokens)
            cond2 = f1_tokens.issubset(away_tokens) and f2_tokens.issubset(home_tokens)
            if cond1 or cond2:
                best_match = event
                break
        if best_match:
            home_price, away_price = extract_prices(best_match)
            if home_price and away_price:
                filled.append(
                    {
                        "event_name": row["event_name"],
                        "event_date": row["event_date"].date(),
                        "f_1_name": row["f_1_name"],
                        "f_2_name": row["f_2_name"],
                        "f_1_odds": home_price,
                        "f_2_odds": away_price,
                        "source": "odds_api_snapshot",
                    }
                )

    if not filled:
        print("No odds matched.")
        return

    new_odds = pd.DataFrame(filled)
    combined = pd.concat([odds, new_odds], ignore_index=True)
    combined.to_csv(args.out, index=False)
    print(f"Added {len(new_odds)} odds rows. Saved to {args.out}")


if __name__ == "__main__":
    main()
