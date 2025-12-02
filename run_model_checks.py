"""
Lightweight validation suite for leakage detection and sanity checks.
Runs:
1) Feature-name scan for forbidden leakage terms.
2) Target correlation check (>0.95 absolute).
3) Monkey tests: random labels and random features.
4) Walk-forward evaluation (2010-2023 train vs 2024/25 hold-out).
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

try:
    from xgboost import XGBClassifier  # type: ignore

    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False
    XGBClassifier = None  # type: ignore

from train_fightiq_model import prepare_dataset, train_model, train_odds_baseline


def feature_name_scan(feature_names):
    forbidden = ["winner", "outcome", "result", "decision", "finish_round", "finish_time"]
    hits = [f for f in feature_names if any(term in f.lower() for term in forbidden)]
    print("\n🕵️ Feature Name Scan")
    if hits:
        print("Suspicious feature names found:", hits)
    else:
        print("OK: no forbidden terms detected.")


def target_correlation_check(X_train: pd.DataFrame, y_train: pd.Series):
    print("\n📈 Target Correlation Check")
    corr = X_train.copy()
    # simple median fill to avoid NaNs in correlation
    corr = corr.fillna(corr.median())
    corrs = corr.corrwith(y_train).abs()
    offenders = corrs[corrs > 0.95].sort_values(ascending=False)
    if len(offenders):
        print("High-correlation features found:")
        print(offenders)
    else:
        top = corrs.sort_values(ascending=False).head(5)
        print("OK: no feature |corr| > 0.95. Top correlations:")
        print(top)


def _make_quick_model():
    if _HAS_XGB:
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",
        )
    else:  # pragma: no cover
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_depth=6)
    return make_pipeline(SimpleImputer(strategy="median"), model)


def monkey_test_random_labels(X_train: pd.DataFrame, y_train: pd.Series, X_test, y_test):
    print("\n🐒 Monkey Test (Random Labels)")
    y_perm = y_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
    X_sub, _, y_sub, _ = train_test_split(X_train, y_perm, train_size=0.5, random_state=42, stratify=y_perm)
    model = _make_quick_model()
    model.fit(X_sub, y_sub)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    baseline = max(y_train.mean(), 1 - y_train.mean())
    print(f"Accuracy with shuffled labels: {acc:.3f} (baseline {baseline:.3f})")


def monkey_test_random_features(X_train: pd.DataFrame, y_train: pd.Series, X_test, y_test):
    print("\n🎲 Monkey Test (Random Features)")
    rng = np.random.default_rng(0)
    X_train_noise = pd.DataFrame(rng.standard_normal(X_train.shape), columns=X_train.columns)
    X_test_noise = pd.DataFrame(rng.standard_normal(X_test.shape), columns=X_test.columns)
    X_sub, _, y_sub, _ = train_test_split(X_train_noise, y_train, train_size=0.5, random_state=42, stratify=y_train)
    model = _make_quick_model()
    model.fit(X_sub, y_sub)
    preds = model.predict(X_test_noise)
    acc = accuracy_score(y_test, preds)
    baseline = max(y_train.mean(), 1 - y_train.mean())
    print(f"Accuracy with random features: {acc:.3f} (baseline {baseline:.3f})")


def walk_forward_validation():
    print("\n⏳ Walk-Forward Validation")
    X_train, y_train, X_test, y_test, meta_test, _ = prepare_dataset()
    main_model = train_model(X_train, y_train)
    odds_model, odds_cols = train_odds_baseline(X_train, y_train)

    p_main = main_model.predict_proba(X_test)[:, 1]
    p_odds = odds_model.predict_proba(X_test[odds_cols])[:, 1]
    p_blend = 0.7 * p_main + 0.3 * p_odds

    preds_main = (p_main >= 0.5).astype(int)
    preds_blend = (p_blend >= 0.5).astype(int)

    def _scores(preds):
        df_year = (
            pd.DataFrame({"event_date": meta_test["event_date"], "pred": preds, "y": y_test})
            .assign(year=lambda d: d["event_date"].dt.year)
            .groupby("year")
            .apply(lambda d: accuracy_score(d["y"], d["pred"]))
        )
        acc = accuracy_score(y_test, preds)
        return acc, df_year

    acc_main, df_main = _scores(preds_main)
    acc_blend, df_blend = _scores(preds_blend)

    print(f"Main model accuracy (2024/25): {acc_main:.4f}")
    for year, score in df_main.items():
        print(f"  Main {int(year)} accuracy: {score:.4f}")
    print(f"Blended (0.7 main / 0.3 odds) accuracy: {acc_blend:.4f}")
    for year, score in df_blend.items():
        print(f"  Blend {int(year)} accuracy: {score:.4f}")


def main():
    X_train, y_train, X_test, y_test, _, feature_names = prepare_dataset()
    feature_name_scan(feature_names)
    target_correlation_check(X_train, y_train)
    monkey_test_random_labels(X_train, y_train, X_test, y_test)
    monkey_test_random_features(X_train, y_train, X_test, y_test)
    walk_forward_validation()


if __name__ == "__main__":
    main()
