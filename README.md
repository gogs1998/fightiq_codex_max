# FightIQ Codex Models

End-to-end UFC fight prediction and prop models built from raw fight data plus odds. Includes leakage checks, odds enrichment via The Odds API, and betting backtests on a 2024/25 hold-out.

## Data
- Raw CSVs in `Data/`: `UFC_full_data_silver.csv`, `UFC_full_data_golden.csv`, `UFC_betting_odds.csv`.
- Optional enriched odds: `UFC_betting_odds_enriched.csv` produced by `fetch_and_fill_odds.py` (uses The Odds API v4, sport `mma_mixed_martial_arts`).

## Core scripts
- `train_fightiq_model.py`: Leak-free winner model (rolling stats + Elo + odds). Current hold-out (2024/25) accuracy: **0.6418** (2024: 0.6511, 2025: 0.6300).
- `train_fightiq_model_golden.py`: Rebuild using per-fight stats from the golden file with enhanced Elo (margin/recency) and chin-health rollups (KD absorbed, finishes suffered); no static career stats. Current hold-out (2024/25) accuracy: **0.6506** (2024: 0.6667, 2025: 0.6300).
- `run_model_checks.py`: Leakage/robustness suite (feature-name scan, high-corr check, monkey tests, walk-forward accuracy).
- `prop_bet_models.py`: Prop baselines (Winner, Method, Round, Trifecta) on leak-free features. Hold-out (2024/25): Winner 0.6550; Method 0.5345; Round 0.5849; Trifecta 0.3308.
- `siamese_fight_model.py`: Reference PyTorch Siamese encoder (sequence model) for winner prediction using per-fight histories. Provided as a starting point; not executed by default.
- `fetch_and_fill_odds.py`: Pulls additional odds (2024/25) from The Odds API and writes `Data/UFC_betting_odds_enriched.csv`. Improved missing-odds coverage: 2024 missing f1/f2 ≈ 5–6%, 2025 ≈ 7–8%.
- `backtest_betting.py`: Simple ROI backtest on 2024/25 winner bets with several staking rules. Flat EV-positive staking yields ~24–25% ROI on hold-out. Kelly variants are illustrative only; use caps for realism.

## Quickstart
```bash
python fetch_and_fill_odds.py          # optional: enrich odds via API
python train_fightiq_model.py          # train/eval winner model
python run_model_checks.py             # leakage + walk-forward
python prop_bet_models.py              # prop-model metrics
python backtest_betting.py             # winner betting ROI backtest
```

## Betting backtest (winner, hold-out 2024/25)
- Flat EV>0 (1u): ROI ~0.246, profit ~189.6 on 772 bets (start bankroll 1000).
- Flat EV>2% (1u): ROI ~0.247, profit ~181.5 on 734 bets.
- Fractional Kelly variants are available but compound aggressively; use caps for safety.

## Notes on coverage
- 2025 odds coverage after enrichment: missing f_1_odds ≈ 6.7%, f_2_odds ≈ 7.7% (slightly higher than 2024 but acceptable).

## Pushing to GitHub
Repo initialized locally. To push to your GitHub repo:
```bash
git remote add origin https://github.com/gogs1998/fightiq_codex_max.git
git add .
git commit -m "Add fight prediction models, props, odds enrichment, backtests, and leakage checks"
git push origin main   # or master, depending on your default branch
```
