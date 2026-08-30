---
name: dataset-validator
description: Validates MT5 Strategy Tester historical CSV datasets for Net Liquid Profit labeling invariants, chronological ordering, and feature dimension parity.
---

# Dataset & Labeling Verification Protocol

Use this skill to audit generated datasets or verify data contracts.

## Steps

1. Run automated dataset tests:
   ```powershell
   pytest tests/test_dataset_manager.py tests/test_feature_schema.py -v
   ```
2. Check quantitative invariants:
   - **Net Liquid Profit Rule**: Trades with profit + swap + commission <= 0.0 must be labeled `0.0f`.
   - **Strict Partitioning**: `<Symbol>_<TF>_buy.csv` contains exclusively BUY rows; `<Symbol>_<TF>_sell.csv` contains exclusively SELL rows.
   - **Feature Parity**: CSV columns must equal $N_{	ext{active\_base\_features}} 	imes (	ext{Lookback} + 1) + 1 	ext{ (label)}$.
