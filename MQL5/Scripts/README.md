# MQL5 Scripts Directory (`MQL5/Scripts/`)

This directory contains standalone, on-demand MQL5 scripts (`.mq5`) designed for utility tasks, verification, and automated test execution.

---

## 📂 Subdirectories & Scripts

### [`Tests/RunAllMQL5UnitTests.mq5`](Tests/RunAllMQL5UnitTests.mq5)
- **Role**: Native Master Test Runner.
- **Execution Mode**: Script program (`#property script_show_inputs`, runs via `OnStart()`).
- **Dependencies**: Includes `MqlTestFramework.mqh`, `TestGarchEngine.mqh`, `TestOrderTracker.mqh`, and `TestFeatureExtractor.mqh`.
- **Functionality**:
  - Instantiates `CMqlTestFramework`.
  - Sequentially triggers test suites for GARCH volatility modeling, in-memory order tracking, and multi-lag feature extraction.
  - Aggregates and logs assertion pass/fail metrics with detailed stack diagnostics.
  - Compiles directly via MetaEditor CLI and is verified in CI/CD via Python test harness (`tests/test_mql5_unit_suite.py`).
