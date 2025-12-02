"""
ROI backtest on 2024/2025 hold-out using the winner model.
Strategies:
1) flat_ev_pos: 1u stake when best side has positive edge (model_p * odds > 1).
2) flat_ev_pos_2pct: 1u stake when edge > 2%.
3) kelly_cap25: Kelly stake capped at 25% of bankroll per bet.
4) kelly_half_cap10: Half-Kelly stake capped at 10% of bankroll per bet.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from train_fightiq_model import HOLDOUT_YEARS, _build_full_dataset, train_model


@dataclass
class BetResult:
    n_bets: int
    hit_rate: float
    total_staked: float
    profit: float
    roi: float
    bankroll_end: float


def select_best_side(p1: float, odds1: float, odds2: float) -> Tuple[int, float, float]:
    """Return side (1 or 2), edge, and odds for the side with highest positive edge."""
    p2 = 1.0 - p1
    edge1 = p1 * odds1 - 1.0
    edge2 = p2 * odds2 - 1.0
    if edge1 > edge2:
        return (1, edge1, odds1)
    else:
        return (2, edge2, odds2)


def run_strategy(
    df: pd.DataFrame,
    probs: np.ndarray,
    name: str,
    rule: Callable[[float, float, float, float], float],
    initial_bankroll: float = 1000.0,
) -> BetResult:
    bankroll = initial_bankroll
    total_staked = 0.0
    profit = 0.0
    wins = 0
    bets = 0

    for (_, row), p1 in zip(df.iterrows(), probs):
        # Skip if odds missing
        if pd.isna(row["f_1_odds"]) or pd.isna(row["f_2_odds"]):
            continue
        side, edge, sel_odds = select_best_side(p1, row["f_1_odds"], row["f_2_odds"])
        stake = rule(edge, sel_odds, bankroll, p1)
        if stake <= 0:
            continue
        bets += 1
        total_staked += stake
        outcome_win = (row["label"] == 1 and side == 1) or (row["label"] == 0 and side == 2)
        if outcome_win:
            wins += 1
            profit += stake * (sel_odds - 1.0)
            bankroll += stake * (sel_odds - 1.0)
        else:
            profit -= stake
            bankroll -= stake

    hit_rate = wins / bets if bets else 0.0
    roi = profit / total_staked if total_staked > 0 else 0.0
    return BetResult(bets, hit_rate, total_staked, profit, roi, bankroll)


def main():
    wide, feature_cols = _build_full_dataset()
    wide = wide.sort_values("event_date")

    # Prepare split
    X = wide[feature_cols]
    y = wide["label"]
    years = wide["event_date"].dt.year
    train_mask = ~years.isin(HOLDOUT_YEARS)
    test_mask = years.isin(HOLDOUT_YEARS)

    model = train_model(X[train_mask], y[train_mask])
    probs = model.predict_proba(X[test_mask])[:, 1]

    test_df = wide.loc[test_mask, ["label", "f_1_odds", "f_2_odds"]].reset_index(drop=True)

    strategies: Dict[str, Callable[[float, float, float, float], float]] = {
        "flat_ev_pos": lambda edge, odds, bankroll, p1: 1.0 if edge > 0 else 0.0,
        "flat_ev_pos_2pct": lambda edge, odds, bankroll, p1: 1.0 if edge > 0.02 else 0.0,
        "kelly_cap25": lambda edge, odds, bankroll, p1: max(0.0, min(bankroll * edge / (odds - 1.0), bankroll * 0.25))
        if edge > 0
        else 0.0,
        "kelly_half_cap10": lambda edge, odds, bankroll, p1: max(
            0.0, min(bankroll * 0.5 * edge / (odds - 1.0), bankroll * 0.10)
        )
        if edge > 0
        else 0.0,
    }

    print("Hold-out years:", HOLDOUT_YEARS)
    for name, rule in strategies.items():
        res = run_strategy(test_df, probs, name, rule)
        print(
            f"{name}: bets={res.n_bets}, hit={res.hit_rate:.3f}, staked={res.total_staked:.1f}, "
            f"profit={res.profit:.1f}, ROI={res.roi:.3f}, bankroll_end={res.bankroll_end:.1f}"
        )


if __name__ == "__main__":
    main()
