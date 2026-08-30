---
name: pipeline-runner
description: Executes the full automated Python MLOps pipeline (cleanup, collection, training, ONNX export, preset generation, compilation).
---

# Full Automated MLOps Pipeline Runner

Use this skill to run the complete machine learning lifecycle for a symbol and timeframe.

## Steps

1. Verify that `.env` contains the desired `SYMBOL`, `TIMEFRAME`, and feature flags.
2. Run the orchestrator:
   ```powershell
   # Default execution (loads .env):
   python run_pipeline.py

   # Or explicitly specifying the configuration file:
   python run_pipeline.py .env
   ```
3. Follow pipeline stages:
   - Scoped artifact cleanup (`src/cleaner.py`).
   - Strategy Tester data collection with `DMatrix-EA.mq5`.
   - Dual XGBoost training with Optuna Bayesian optimization and early stopping (`src/trainer.py`).
   - Flat 1D Float ONNX export without ZipMap (`src/onnx_exporter.py`).
   - Native `.set` preset and `.tpl` template generation.
   - Live inference EA compilation with `LiveONNX-EA.mq5`.
