import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

DATA_DIR = Path(__file__).resolve().parent / "Data"
HOLDOUT_YEARS: Sequence[int] = (2024, 2025)


def _parse_finish_time_to_seconds(time_str: str) -> float:
    try:
        parts = str(time_str).split(":")
        if len(parts) != 2:
            return math.nan
        mins, secs = int(parts[0]), int(parts[1])
        return mins * 60 + secs
    except Exception:
        return math.nan


def _approx_weight_from_class(weight_class: str) -> float:
    if pd.isna(weight_class):
        return math.nan
    wc = str(weight_class)
    match = re.search(r"(\d+)", wc)
    if match:
        return float(match.group(1))
    if "Heavyweight" in wc:
        return 265.0
    if "Open" in wc:
        return 220.0
    return math.nan


def _build_stance_encoder(df: pd.DataFrame) -> Dict[str, int]:
    stances = pd.concat([df["f_1_fighter_stance"], df["f_2_fighter_stance"]]).dropna().unique()
    return {stance: i for i, stance in enumerate(sorted(stances))}


def attach_odds(df: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    odds = odds.copy()
    odds["event_date"] = pd.to_datetime(odds["event_date"])

    base_cols = ["event_name", "event_date", "f_1_name", "f_2_name", "f_1_odds", "f_2_odds"]
    odds_direct = odds[base_cols].set_index(["event_name", "event_date", "f_1_name", "f_2_name"])
    odds_swap = odds.rename(
        columns={
            "f_1_name": "f_2_name",
            "f_2_name": "f_1_name",
            "f_1_odds": "f_2_odds",
            "f_2_odds": "f_1_odds",
        }
    )[base_cols].set_index(["event_name", "event_date", "f_1_name", "f_2_name"])
    odds_all = pd.concat([odds_direct, odds_swap])
    odds_all = odds_all[~odds_all.index.duplicated(keep="last")]
    keys = df.set_index(["event_name", "event_date", "f_1_name", "f_2_name"]).index
    joined = odds_all.reindex(keys).reset_index(drop=True)
    for col in ("f_1_odds", "f_2_odds"):
        df[col] = df[col].fillna(joined[col])
    return df


def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Margin/recency-aware Elo updated chronologically.
    Margin proxy: per-minute strike diff (capped).
    Recency decay: exp(-days_since_last/400).
    """
    df = df.sort_values("event_date").copy()
    ratings: Dict[str, float] = defaultdict(lambda: 1500.0)
    last_date: Dict[str, pd.Timestamp] = {}
    elo_f1, elo_f2 = [], []
    k_base = 24.0
    for _, row in df.iterrows():
        f1, f2 = row["f_1_name"], row["f_2_name"]
        r1, r2 = ratings[f1], ratings[f2]
        elo_f1.append(r1)
        elo_f2.append(r2)

        # margin proxy
        dur = max(1.0, float(row.get("fight_duration_minutes", 15.0)))
        strike_diff = (
            row.get("f_1_total_strikes_succ", 0) - row.get("f_2_total_strikes_succ", 0)
        ) / dur
        margin_mult = np.clip(1.0 + abs(strike_diff) / 10.0, 1.0, 1.5)

        # recency decay per fighter
        today = row["event_date"]
        decay1 = math.exp(-((today - last_date.get(f1, today)).days) / 400) if f1 in last_date else 1.0
        decay2 = math.exp(-((today - last_date.get(f2, today)).days) / 400) if f2 in last_date else 1.0

        outcome = row["label"]
        score1 = outcome
        score2 = 1 - outcome
        expected1 = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400))
        expected2 = 1.0 - expected1
        finish_str = str(row.get("result", "")).lower()
        is_finish = "dec" not in finish_str
        k1 = k_base * (1.2 if is_finish else 1.0) * margin_mult * decay1
        k2 = k_base * (1.2 if is_finish else 1.0) * margin_mult * decay2
        ratings[f1] = r1 + k1 * (score1 - expected1)
        ratings[f2] = r2 + k2 * (score2 - expected2)
        last_date[f1] = today
        last_date[f2] = today
    df["elo_f1"] = elo_f1
    df["elo_f2"] = elo_f2
    return df


def build_long_fighter_rows(df: pd.DataFrame, stance_encoder: Dict[str, int]) -> pd.DataFrame:
    records: List[Dict] = []
    for fight_id, row in df.iterrows():
        finish_round = pd.to_numeric(row["finish_round"], errors="coerce")
        duration_sec = (finish_round - 1) * 5 * 60 + _parse_finish_time_to_seconds(row.get("finish_time"))
        if math.isnan(duration_sec) or duration_sec <= 0:
            duration_sec = pd.to_numeric(row["num_rounds"], errors="coerce") * 5 * 60
        duration_min = duration_sec / 60.0 if duration_sec and duration_sec > 0 else 5.0

        finish_str = str(row.get("result", "")).lower()
        is_finish = "dec" not in finish_str
        is_sub = "sub" in finish_str or "submission" in finish_str

        for side, opp in ((1, 2), (2, 1)):
            prefix, opp_prefix = f"f_{side}_", f"f_{opp}_"
            fighter = row[prefix + "name"]
            opponent = row[opp_prefix + "name"]
            winner = row["winner"]
            outcome = 1.0 if winner == fighter else 0.0

            sig_landed = row[prefix + "sig_strikes_succ"]
            sig_att = row[prefix + "sig_strikes_att"]
            sig_abs = row[opp_prefix + "sig_strikes_succ"]
            total_landed = row[prefix + "total_strikes_succ"]
            total_att = row[prefix + "total_strikes_att"]
            total_abs = row[opp_prefix + "total_strikes_succ"]
            td_succ = row[prefix + "takedown_succ"]
            td_att = row[prefix + "takedown_att"]
            td_allowed = row[opp_prefix + "takedown_succ"]
            opp_td_att = row[opp_prefix + "takedown_att"]
            ctrl_for = row[prefix + "ctrl_time_sec"]
            ctrl_against = row[opp_prefix + "ctrl_time_sec"]
            kd = row[prefix + "knockdowns"]

            sig_acc = sig_landed / sig_att if pd.notna(sig_att) and sig_att > 0 else 0.0
            total_acc = total_landed / total_att if pd.notna(total_att) and total_att > 0 else 0.0
            td_acc = td_succ / td_att if pd.notna(td_att) and td_att > 0 else 0.0
            td_def = 1.0 - (td_allowed / opp_td_att) if pd.notna(opp_td_att) and opp_td_att > 0 else 0.5
            ctrl_share = (
                ctrl_for / (ctrl_for + ctrl_against)
                if pd.notna(ctrl_for + ctrl_against) and (ctrl_for + ctrl_against) > 0
                else 0.5
            )

            record = {
                "fight_id": fight_id,
                "fighter": fighter,
                "opponent": opponent,
                "side": side,
                "event_date": row["event_date"],
                "duration_min": duration_min,
                "outcome": outcome,
            "finish": int(is_finish),
            "sub_win": int(is_sub and outcome == 1.0),
            "finish_against": int(is_finish and outcome == 0.0),
            "sig_rate_for": sig_landed / duration_min,
            "sig_rate_against": sig_abs / duration_min,
            "sig_diff_pm": (sig_landed - sig_abs) / duration_min,
            "strike_diff_pm": (total_landed - total_abs) / duration_min,
            "margin_pm": (total_landed - total_abs) / duration_min,
            "sig_acc": sig_acc,
            "total_acc": total_acc,
            "td_rate_for": td_succ / duration_min,
            "td_rate_against": td_allowed / duration_min,
                "td_acc": td_acc,
                "td_def": td_def,
            "control_share": ctrl_share,
            "kd_rate": kd / duration_min if duration_min and duration_min > 0 else 0.0,
            "kd_abs_rate": row[opp_prefix + "knockdowns"] / duration_min if duration_min and duration_min > 0 else 0.0,
                "height_cm": row[prefix + "fighter_height_cm"],
                "reach_cm": row[prefix + "fighter_reach_cm"],
                "weight_lbs": row[prefix + "fighter_weight_lbs"],
                "stance_code": stance_encoder.get(row[prefix + "fighter_stance"], -1),
            }

            dob = row.get(prefix + "fighter_dob")
            try:
                dob_dt = pd.to_datetime(dob)
                record["age"] = (row["event_date"] - dob_dt).days / 365.25
            except Exception:
                record["age"] = math.nan

            records.append(record)
    return pd.DataFrame(records)


def add_rolling_features(long_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "outcome",
        "finish",
        "sub_win",
        "finish_against",
        "sig_diff_pm",
        "strike_diff_pm",
        "margin_pm",
        "sig_acc",
        "total_acc",
        "td_acc",
        "td_def",
        "control_share",
        "kd_rate",
        "kd_abs_rate",
        "sig_rate_for",
        "sig_rate_against",
        "td_rate_for",
        "td_rate_against",
    ]
    windows = (3, 5, 10)

    def _per_fighter(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("event_date").copy()
        group["prev_fights"] = np.arange(len(group))
        group["days_since_last"] = group["event_date"].diff().dt.days
        group["last_finish_against"] = group["finish_against"].shift().fillna(0)
        for w in windows:
            group[f"win_rate_{w}"] = group["outcome"].shift().rolling(w, min_periods=1).mean()
            group[f"finish_rate_{w}"] = group["finish"].shift().rolling(w, min_periods=1).mean()
            group[f"sub_rate_{w}"] = group["sub_win"].shift().rolling(w, min_periods=1).mean()
            group[f"chin_kd_abs_{w}"] = group["kd_abs_rate"].shift().rolling(w, min_periods=1).mean()
            group[f"chin_finish_against_{w}"] = group["finish_against"].shift().rolling(w, min_periods=1).mean()
        for col in metrics:
            shifted = group[col].shift()
            for w in windows:
                group[f"{col}_ma_{w}"] = shifted.rolling(w, min_periods=1).mean()
            group[f"{col}_mean_all"] = shifted.expanding().mean()
        # rolling margin signals
        group["margin_pm_ma_3"] = group["margin_pm"].shift().rolling(3, min_periods=1).mean()
        group["margin_pm_ma_5"] = group["margin_pm"].shift().rolling(5, min_periods=1).mean()
        return group

    return long_df.groupby("fighter", group_keys=False).apply(_per_fighter)


def assemble_fight_level_features(
    df: pd.DataFrame, long_df: pd.DataFrame, fighter_cols: Iterable[str]
) -> pd.DataFrame:
    f1 = long_df[long_df["side"] == 1][["fight_id", *fighter_cols]].set_index("fight_id")
    f2 = long_df[long_df["side"] == 2][["fight_id", *fighter_cols]].set_index("fight_id")
    f1 = f1.add_prefix("f1_")
    f2 = f2.add_prefix("f2_")

    wide = df[
        [
            "fight_id",
            "event_date",
            "label",
            "weight_class",
            "gender",
            "title_fight",
            "num_rounds",
            "f_1_odds",
            "f_2_odds",
            "f_1_implied_prob",
            "f_2_implied_prob",
            "odds_logit",
            "elo_f1",
            "elo_f2",
            "weight_class_lbs",
        ]
    ].set_index("fight_id")
    wide = wide.join(f1).join(f2)
    for col in fighter_cols:
        wide[f"diff_{col}"] = wide[f"f1_{col}"] - wide[f"f2_{col}"]
    wide["elo_diff"] = wide["elo_f1"] - wide["elo_f2"]
    wide["stance_same"] = (wide["f1_stance_code"] == wide["f2_stance_code"]).astype(float)
    return wide.reset_index()


def prepare_dataset() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    path = DATA_DIR / "UFC_full_data_golden.csv"
    odds_path = DATA_DIR / "UFC_betting_odds_enriched.csv"
    if not odds_path.exists():
        odds_path = DATA_DIR / "UFC_betting_odds.csv"
    df = pd.read_csv(path)
    odds = pd.read_csv(odds_path)

    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df[(df["event_date"].dt.year >= 2010)]
    df = df.sort_values("event_date").reset_index(drop=True)
    df["fight_id"] = df.index
    df["label"] = np.where(df["winner"] == df["f_1_name"], 1.0, 0.0)
    df = df[df["winner"] != "No Contest"]

    df = attach_odds(df, odds)
    df["f_1_implied_prob"] = 1.0 / df["f_1_odds"]
    df["f_2_implied_prob"] = 1.0 / df["f_2_odds"]
    df["odds_logit"] = np.log(df["f_1_implied_prob"] / df["f_2_implied_prob"])
    df["weight_class_lbs"] = df["weight_class"].apply(_approx_weight_from_class)
    df["title_fight"] = df["title_fight"].astype(float)
    df["num_rounds"] = df["num_rounds"].astype(float)

    stance_encoder = _build_stance_encoder(df)
    long_df = build_long_fighter_rows(df, stance_encoder)
    long_df = add_rolling_features(long_df)

    fighter_cols = [
        "prev_fights",
        "days_since_last",
        "win_rate_3",
        "win_rate_5",
        "win_rate_10",
        "finish_rate_5",
        "sub_rate_5",
        "chin_kd_abs_3",
        "chin_finish_against_3",
        "sig_diff_pm_ma_3",
        "sig_diff_pm_ma_5",
        "strike_diff_pm_ma_5",
        "sig_acc_ma_5",
        "total_acc_ma_5",
        "td_acc_ma_5",
        "td_def_ma_5",
        "control_share_ma_5",
        "kd_rate_ma_5",
        "sig_rate_for_ma_5",
        "sig_rate_against_ma_5",
        "td_rate_for_ma_5",
        "td_rate_against_ma_5",
        "margin_pm_ma_3",
        "margin_pm_ma_5",
        "height_cm",
        "reach_cm",
        "weight_lbs",
        "age",
        "stance_code",
    ]

    df = compute_elo(df)
    wide = assemble_fight_level_features(df, long_df, fighter_cols)

    drop_cols = {"fight_id", "event_date", "label", "weight_class", "gender"}
    feature_cols = [c for c in wide.columns if c not in drop_cols]

    years = wide["event_date"].dt.year
    train_mask = ~years.isin(HOLDOUT_YEARS)
    test_mask = years.isin(HOLDOUT_YEARS)
    X = wide[feature_cols]
    y = wide["label"]
    return (
        X[train_mask],
        y[train_mask],
        X[test_mask],
        y[test_mask],
        wide.loc[test_mask, ["fight_id", "event_date"]],
        feature_cols,
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    if XGBClassifier is None:
        raise ImportError("xgboost required")
    model = XGBClassifier(
        n_estimators=1200,
        learning_rate=0.035,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        reg_lambda=1.2,
        gamma=0.05,
    )
    pipeline = make_pipeline(SimpleImputer(strategy="median"), model)
    pipeline.fit(X_train, y_train)
    return pipeline


def main() -> None:
    X_train, y_train, X_test, y_test, meta_test, _ = prepare_dataset()
    model = train_model(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    per_year = (
        pd.DataFrame({"event_date": meta_test["event_date"], "pred": preds, "y": y_test})
        .assign(year=lambda d: d["event_date"].dt.year)
        .groupby("year")
        .apply(lambda d: accuracy_score(d["y"], d["pred"]))
    )
    print(f"Train size: {len(y_train)}, Hold-out size: {len(y_test)}")
    print("Hold-out years:", HOLDOUT_YEARS)
    print(f"Hold-out accuracy: {acc:.4f}")
    for year, score in per_year.items():
        print(f"  {int(year)} accuracy: {score:.4f}")


if __name__ == "__main__":
    main()
