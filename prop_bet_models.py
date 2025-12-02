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

try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

    XGBClassifier = None  # type: ignore

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
    y_win = wide["label"]
    y_method = wide["result"].apply(method_bucket)
    y_round = pd.to_numeric(wide["finish_round"], errors="coerce").fillna(-1).astype(int).astype(str)
    trifecta = y_win.astype(str) + "_" + y_method + "_" + y_round
    return y_win, y_method, y_round, trifecta


def make_model(num_class: int = 2):
    if _HAS_XGB:
        params = dict(
            n_estimators=700,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.8,
            tree_method="hist",
            reg_lambda=1.0,
            gamma=0.05,
        )
        if num_class > 2:
            params["objective"] = "multi:softprob"
            params["num_class"] = num_class
            params["eval_metric"] = "mlogloss"
        else:
            params["eval_metric"] = "logloss"
        model = XGBClassifier(**params)
    else:  # pragma: no cover
        model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_depth=6)
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
