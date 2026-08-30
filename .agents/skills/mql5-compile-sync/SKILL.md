---
name: mql5-compile-sync
description: Synchronizes MQL5 Expert Advisors and Include libraries to the MT5 terminal and compiles via MetaEditor CLI.
---

# MQL5 Synchronization and Compilation

Use this skill to sync MQL5 files and compile Expert Advisors without modifying existing ONNX models or CSV datasets.

## Steps

1. Run the compile-only command:
   ```powershell
   # Default execution (loads .env):
   python run_pipeline.py --compile-only

   # Or explicitly specifying the configuration file:
   python run_pipeline.py .env --compile-only
   ```
2. Verify MetaEditor CLI output:
   - Check that `Compilation SUCCESS for DMatrix-EA.mq5 (0 errors)` is reported.
   - Check that `Compilation SUCCESS for LiveONNX-EA.mq5 (0 errors)` is reported.
