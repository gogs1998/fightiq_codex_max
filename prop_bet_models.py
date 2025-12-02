"""
Train and evaluate prop bet models: winner, method, round, and trifecta (winner+method+round).
Reuses the feature set from train_fightiq_model.py (_build_full_dataset).
Outputs accuracy on 2024/2025 hold-out.
"""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from train_fightiq_model import HOLDOUT_YEARS, _build_full_dataset


def method_bucket(res: str) -> str:
    s = str(res).lower()
    if "ko" in s:
        return "KO/TKO"
    if "sub" in s:
        return "SUB"
    if "decision" in s:
        return "DEC"
    return "OTHER"


def build_targets(wide: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    y_win = wide["label"].fillna(-1).astype(int)
    y_method = wide["result"].apply(method_bucket).fillna("OTHER")
    y_round = pd.to_numeric(wide["finish_round"], errors="coerce").fillna(-1).astype(int).astype(str)
    trifecta = y_win.astype(str) + "_" + y_method + "_" + y_round
    return y_win, y_method, y_round, trifecta


def make_model(num_class: int = 2):
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.8,
        tree_method="hist",
        reg_lambda=1.0,
        gamma=0.05,
        objective="multi:softprob" if num_class > 2 else "binary:logistic",
        num_class=num_class if num_class > 2 else None,
        eval_metric="mlogloss" if num_class > 2 else "logloss",
    )
    return make_pipeline(SimpleImputer(strategy="median"), model)


def encode_labels(y: pd.Series):
    classes = sorted(pd.unique(y))
    mapping = {c: i for i, c in enumerate(classes)}
    return y.map(mapping), mapping


def evaluate_model(X_train, y_train, X_test, y_test):
    y_train_enc, mapping = encode_labels(y_train)
    y_test_enc = y_test.map(mapping)
    num_class = len(mapping)
    model = make_model(num_class=num_class)
    model.fit(X_train, y_train_enc)
    preds = model.predict(X_test)
    return accuracy_score(y_test_enc, preds), model, mapping


def main():
    wide, feature_cols = _build_full_dataset()
    wide = wide.sort_values("event_date")
    X = wide[feature_cols]
    y_win, y_method, y_round, y_trifecta = build_targets(wide)

    years = wide["event_date"].dt.year
    train_mask = ~years.isin(HOLDOUT_YEARS)
    test_mask = years.isin(HOLDOUT_YEARS)

    # downsample train for faster sanity metrics
    train_idx = np.where(train_mask)[0]
    if len(train_idx) > 2500:
        rng = np.random.default_rng(0)
        keep = set(rng.choice(train_idx, size=2500, replace=False))
        mask_array = np.array([i in keep for i in range(len(wide))])
        train_mask = mask_array

    # bucket rare trifecta classes to avoid unseen-test mapping issues
    train_counts = y_trifecta[train_mask].value_counts()
    keep_classes = set(train_counts[train_counts >= 20].index)
    y_trifecta = y_trifecta.where(y_trifecta.isin(keep_classes), "OTHER_TRIFECTA")

    splits = {
        "winner": y_win,
        "method": y_method,
        "round": y_round,
        "trifecta": y_trifecta,
    }

    print("Hold-out years:", HOLDOUT_YEARS)
    for name, target in splits.items():
        acc, model, mapping = evaluate_model(
            X[train_mask], target[train_mask], X[test_mask], target[test_mask]
        )
        preds = model.predict(X[test_mask])
        per_year = (
            pd.DataFrame(
                {"event_date": wide.loc[test_mask, "event_date"], "pred": preds, "y": target[test_mask].map(mapping)}
            )
            .assign(year=lambda d: d["event_date"].dt.year)
            .groupby("year")
            .apply(lambda d: accuracy_score(d["y"], d["pred"]))
        )
        print(f"{name.capitalize()} accuracy: {acc:.4f}")
        for yr, score in per_year.items():
            print(f"  {int(yr)}: {score:.4f}")


if __name__ == "__main__":
    main()
