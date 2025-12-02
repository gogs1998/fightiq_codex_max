import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

# Attempt to use XGBoost if available; otherwise fall back to HistGradientBoosting.
try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_XGB = False
    XGBClassifier = None  # type: ignore


DATA_DIR = Path(__file__).resolve().parent / "Data"
HOLDOUT_YEARS: Sequence[int] = (2024, 2025)  # 2024/25 hold-out as requested


def _parse_finish_time_to_seconds(time_str: str) -> float:
    """Parse a mm:ss string into seconds; return NaN on failure."""
    try:
        parts = str(time_str).split(":")
        if len(parts) != 2:
            return math.nan
        mins, secs = int(parts[0]), int(parts[1])
        return mins * 60 + secs
    except Exception:
        return math.nan


def _approx_weight_from_class(weight_class: str) -> float:
    """Convert a weight-class label into an approximate numeric weight (lbs)."""
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
    """Fill missing odds in the fight data using the odds feed (handles swapped order)."""
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
    """Add pre-fight Elo ratings for f1 and f2 using chronological updates."""
    df = df.sort_values("event_date").copy()
    ratings: Dict[str, float] = defaultdict(lambda: 1500.0)
    elo_f1, elo_f2 = [], []
    k_base = 24.0

    for _, row in df.iterrows():
        f1, f2 = row["f_1_name"], row["f_2_name"]
        r1, r2 = ratings[f1], ratings[f2]
        elo_f1.append(r1)
        elo_f2.append(r2)

        outcome = row["label"]
        score1 = outcome
        score2 = 1 - outcome
        expected1 = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400))
        expected2 = 1.0 - expected1
        is_finish = isinstance(row.get("result"), str) and ("Dec" not in row["result"])
        k = k_base * (1.2 if is_finish else 1.0)
        ratings[f1] = r1 + k * (score1 - expected1)
        ratings[f2] = r2 + k * (score2 - expected2)

    df["elo_f1"] = elo_f1
    df["elo_f2"] = elo_f2
    return df


def build_long_fighter_rows(df: pd.DataFrame, stance_encoder: Dict[str, int]) -> pd.DataFrame:
    """Transform fight-level rows into fighter-level rows with per-fight stats."""
    records: List[Dict] = []
    for fight_id, row in df.iterrows():
        finish_round = pd.to_numeric(row["finish_round"], errors="coerce")
        duration_sec = (finish_round - 1) * 5 * 60 + _parse_finish_time_to_seconds(row.get("finish_time"))
        if math.isnan(duration_sec) or duration_sec <= 0:
            duration_sec = pd.to_numeric(row["num_rounds"], errors="coerce") * 5 * 60
        duration_min = duration_sec / 60.0 if duration_sec and duration_sec > 0 else 5.0

        finish_str = str(row.get("result", ""))
        finish_lower = finish_str.lower()
        is_finish = "dec" not in finish_lower
        is_sub = "sub" in finish_lower or "submission" in finish_lower

        for side, opp in ((1, 2), (2, 1)):
            prefix, opp_prefix = f"f_{side}_", f"f_{opp}_"
            fighter = row[prefix + "name"]
            opponent = row[opp_prefix + "name"]
            winner = row["winner"]
            outcome = (
                1.0 if winner == fighter else 0.5 if winner not in {fighter, opponent} else 0.0
            )

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
            td_def = (
                1.0 - (td_allowed / opp_td_att) if pd.notna(opp_td_att) and opp_td_att > 0 else 0.5
            )
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
                "sig_rate_for": sig_landed / duration_min,
                "sig_rate_against": sig_abs / duration_min,
                "sig_diff_pm": (sig_landed - sig_abs) / duration_min,
                "strike_diff_pm": (total_landed - total_abs) / duration_min,
                "sig_acc": sig_acc,
                "total_acc": total_acc,
                "td_rate_for": td_succ / duration_min,
                "td_rate_against": td_allowed / duration_min,
                "td_acc": td_acc,
                "td_def": td_def,
                "control_share": ctrl_share,
                "kd_rate": kd / duration_min if duration_min and duration_min > 0 else 0.0,
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
    """Compute rolling/expanding stats per fighter (shifted to avoid leakage)."""
    metrics = [
        "outcome",
        "finish",
        "sub_win",
        "sig_diff_pm",
        "strike_diff_pm",
        "sig_acc",
        "total_acc",
        "td_acc",
        "td_def",
        "control_share",
        "kd_rate",
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
        for w in windows:
            group[f"win_rate_{w}"] = group["outcome"].shift().rolling(w, min_periods=1).mean()
            group[f"finish_rate_{w}"] = group["finish"].shift().rolling(w, min_periods=1).mean()
            group[f"sub_rate_{w}"] = group["sub_win"].shift().rolling(w, min_periods=1).mean()
        for col in metrics:
            shifted = group[col].shift()
            for w in windows:
                group[f"{col}_ma_{w}"] = shifted.rolling(w, min_periods=1).mean()
            group[f"{col}_mean_all"] = shifted.expanding().mean()
        return group

    return long_df.groupby("fighter", group_keys=False).apply(_per_fighter)


def assemble_fight_level_features(
    df: pd.DataFrame, long_df: pd.DataFrame, fighter_cols: Iterable[str]
) -> pd.DataFrame:
    """Join fighter-level rolling features back to fight rows with diffs."""
    f1 = long_df[long_df["side"] == 1][["fight_id", *fighter_cols]].set_index("fight_id")
    f2 = long_df[long_df["side"] == 2][["fight_id", *fighter_cols]].set_index("fight_id")
    f1 = f1.add_prefix("f1_")
    f2 = f2.add_prefix("f2_")

    wide = df[["fight_id", "event_date", "label", "weight_class", "gender", "title_fight", "num_rounds",
               "f_1_odds", "f_2_odds", "f_1_implied_prob", "f_2_implied_prob",
               "odds_logit", "elo_f1", "elo_f2", "weight_class_lbs"]].set_index("fight_id")
    wide = wide.join(f1).join(f2)

    for col in fighter_cols:
        wide[f"diff_{col}"] = wide[f"f1_{col}"] - wide[f"f2_{col}"]
    wide["elo_diff"] = wide["elo_f1"] - wide["elo_f2"]
    wide["stance_same"] = (wide["f1_stance_code"] == wide["f2_stance_code"]).astype(float)
    return wide.reset_index()


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    if _HAS_XGB:
        model = XGBClassifier(
            n_estimators=1400,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",
            reg_lambda=1.0,
            gamma=0.05,
            min_child_weight=1.0,
        )
    else:  # pragma: no cover - fallback
        model = HistGradientBoostingClassifier(
            learning_rate=0.08, max_depth=6, max_iter=450, l2_regularization=0.05
        )
    pipeline = make_pipeline(SimpleImputer(strategy="median"), model)
    pipeline.fit(X_train, y_train)
    return pipeline


def train_odds_baseline(X_train: pd.DataFrame, y_train: pd.Series):
    odds_cols = [c for c in ["f_1_implied_prob", "f_2_implied_prob", "odds_logit"] if c in X_train.columns]
    model = make_pipeline(SimpleImputer(strategy="median"), LogisticRegression(max_iter=500))
    model.fit(X_train[odds_cols], y_train)
    return model, odds_cols


def prepare_dataset() -> Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, List[str]
]:
    wide, feature_cols = _build_full_dataset()
    X = wide[feature_cols]
    y = wide["label"]

    years = wide["event_date"].dt.year
    train_mask = ~years.isin(HOLDOUT_YEARS)
    test_mask = years.isin(HOLDOUT_YEARS)
    return (
        X[train_mask],
        y[train_mask],
        X[test_mask],
        y[test_mask],
        wide.loc[test_mask, ["fight_id", "event_date"]],
        feature_cols,
    )


def _build_full_dataset() -> Tuple[pd.DataFrame, List[str]]:
    silver_path = DATA_DIR / "UFC_full_data_silver.csv"
    odds_path = DATA_DIR / "UFC_betting_odds_enriched.csv"
    if not odds_path.exists():
        odds_path = DATA_DIR / "UFC_betting_odds.csv"
    df = pd.read_csv(silver_path)
    odds = pd.read_csv(odds_path)

    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df[df["event_date"].dt.year >= 2010]
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

    # Optional enrichment from golden file: pre-built differential aggregates.
    golden_path = DATA_DIR / "UFC_full_data_golden.csv"
    def _golden_filter(col: str) -> bool:
        if col == "fight_url":
            return True
        if not col.startswith("diff_") or "_r" in col:
            return False
        keep_targets = ("_3", "_5")
        return col in {"diff_age", "diff_fight_number", "diff_odds"} or any(k in col for k in keep_targets)

    golden = pd.read_csv(golden_path, usecols=_golden_filter)
    df = df.merge(golden, on="fight_url", how="left")
    golden_cols = [c for c in df.columns if c.startswith("diff_") and "_r" not in c]

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
        "height_cm",
        "reach_cm",
        "weight_lbs",
        "age",
        "stance_code",
    ]

    df = compute_elo(df)
    wide = assemble_fight_level_features(df, long_df, fighter_cols)
    wide = wide.merge(df[["fight_id"] + golden_cols], on="fight_id", how="left")
    wide = wide.merge(df[["fight_id", "result", "finish_round"]], on="fight_id", how="left")

    stat_cols = [
        "fighter_SlpM",
        "fighter_SApM",
        "fighter_Str_Acc",
        "fighter_Str_Def",
        "fighter_TD_Avg",
        "fighter_TD_Acc",
        "fighter_TD_Def",
        "fighter_Sub_Avg",
        "fighter_w",
        "fighter_l",
        "fighter_nc_dq",
    ]
    extra = pd.DataFrame({"fight_id": df["fight_id"]})
    for col in stat_cols:
        c1, c2 = f"f_1_{col}", f"f_2_{col}"
        extra[f"f1_{col}"] = df[c1]
        extra[f"f2_{col}"] = df[c2]
        extra[f"diff_{col}"] = df[c1] - df[c2]
    wide = wide.merge(extra, on="fight_id", how="left")

    # Select numeric feature columns (drop identifiers and label).
    drop_cols = {"fight_id", "event_date", "label", "weight_class", "gender", "result", "finish_round"}
    feature_cols = [c for c in wide.columns if c not in drop_cols]
    return wide, feature_cols


def main() -> None:
    X_train, y_train, X_test, y_test, meta_test, _ = prepare_dataset()
    main_model = train_model(X_train, y_train)
    odds_model, odds_cols = train_odds_baseline(X_train, y_train)

    p_main = main_model.predict_proba(X_test)[:, 1]
    p_odds = odds_model.predict_proba(X_test[odds_cols])[:, 1]
    p_blend = 0.7 * p_main + 0.3 * p_odds

    preds_main = (p_main >= 0.5).astype(int)
    preds_blend = (p_blend >= 0.5).astype(int)

    def year_scores(preds):
        return (
            pd.DataFrame({"event_date": meta_test["event_date"], "pred": preds, "y": y_test})
            .assign(year=lambda d: d["event_date"].dt.year)
            .groupby("year")
            .apply(lambda d: accuracy_score(d["y"], d["pred"]))
        )

    acc_main = accuracy_score(y_test, preds_main)
    acc_blend = accuracy_score(y_test, preds_blend)
    per_year_main = year_scores(preds_main)
    per_year_blend = year_scores(preds_blend)

    print(f"Train size: {len(y_train)}, Hold-out size: {len(y_test)}")
    print("Hold-out years:", HOLDOUT_YEARS)
    print(f"Main model accuracy: {acc_main:.4f}")
    for year, score in per_year_main.items():
        print(f"  Main {int(year)} accuracy: {score:.4f}")
    print(f"Blended (0.7 main / 0.3 odds) accuracy: {acc_blend:.4f}")
    for year, score in per_year_blend.items():
        print(f"  Blend {int(year)} accuracy: {score:.4f}")


if __name__ == "__main__":
    main()
