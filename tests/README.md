# Automated Test Suite (`tests/`)

This directory contains the comprehensive automated unit, integration, and regression test suite verifying mathematical parity, tensor contracts, configuration strictness, dataset integrity, and file isolation.

---

## 🧪 Test Matrix

| Test Module | Coverage Scope | Primary Invariants Verified |
|---|---|---|
| [`test_config.py`](test_config.py) | `src/config.py` | Strict environment variable parsing, missing key detection, invalid type validation, boolean truthy/falsy evaluation, immutable dataclass enforcement, pandemic blackout keys (`AVOID_PANDEMICTIME`, `PANDEMIC_START_DATE`, `PANDEMIC_END_DATE`), directional evaluation & sensitivity grid parameters (`EVAL_*`, `OPTUNA_OBJECTIVE_METRIC`), and `.env` vs `.env.example` parity with isolation of LiveONNX-only parameters. |
| [`test_cleanup_and_isolation.py`](test_cleanup_and_isolation.py) | `src/cleaner.py` & `src/mt5_client.py` | Scoped artifact cleanup, MQL5 sync code-only filtering (`.mq5`, `.mqh`, `.set`, `.mq4`), markdown/documentation exclusion, stray README.md purging in terminal/common, workspace document preservation, and multi-symbol/timeframe isolation. |
| [`test_dataset_manager.py`](test_dataset_manager.py) | `src/dataset_manager.py` | Dataset search across terminal and common folders, label column validation, missing file handling, and metadata parsing. |
| [`test_feature_schema.py`](test_feature_schema.py) | `src/preset_generator.py` | Feature vector dimension parity across all toggle permutations, native `.set` key verification (including `InpMagicNumber`, `InpRiskGarchHorizon`, and DMatrix pandemic blackout parameters), and chronological time-series splitting (zero leakage). |
| [`test_garch_math.py`](test_garch_math.py) | `MQL5/Include/GarchEngine.mqh` | Analytical GARCH(1,1) Bollerslev (1986) recursion parity between Python reference and MQL5 formulas, multi-step aggregation monotonicity, dynamic econometric features (`vol_ratio`, `vol_trend`), and stationarity bounds. |
| [`test_macro_agent.py`](test_macro_agent.py) | `macro_agent/` | SQLite database schema initialization, CRUD operations on `calendar_events` and `news_events` with `trailing_points`, active time window queries, expired event purging, currency extraction, and architectural boundary isolation. |
| [`test_macro_calendar.py`](test_macro_calendar.py) | `src/tools/macro_calendar.py` | Economic calendar event dataclass immutability, open economic feed parsing, high-impact currency classification, and CLI execution. |
| [`test_mt5_tester_guard.py`](test_mt5_tester_guard.py) | `src/mt5_client.py` | Strategy Tester institutional `.ini` parameters (`Deposit=1000000000000000`, `Leverage=500`, `Model=4`, `ProfitInPips=0`, pandemic blackout parameters `InpAvoidPandemicTime`, `InpPandemicStartTime`, `InpPandemicEndTime`), dynamic UTF-16/UTF-8 log streaming with byte offset tracking, non-fatal warning whitelist tolerance (`[WARMUP]`, `[WARNING]`, `invalid stops`, market closed, tick absence/discard), and fatal error interception aborts. |
| [`test_onnx_pipeline.py`](test_onnx_pipeline.py) | `src/onnx_exporter.py` | XGBoost to flat ONNX conversion without ZipMap, 1D float tensor graph contract `[None, num_features] -> [None, 2]`, and batch inference shape consistency. |
| [`test_skip_dataset_pipeline.py`](test_skip_dataset_pipeline.py) | `run_pipeline.py` | Pipeline execution with `SKIP_DATASET_GENERATION` and `--skip-dataset` flag, Strategy Tester bypass with existing CSVs, and fallback on missing datasets. |
| [`test_template_generator.py`](test_template_generator.py) | `src/template_generator.py` | MT5 chart template generation (`.tpl`), color scheme configuration, and active indicator overlay filtering. |
| [`test_mql5_unit_suite.py`](test_mql5_unit_suite.py) | `MQL5/Scripts/Tests/` & `MQL5/Include/Tests/` | Native MQL5 assertion framework existence, black-box unit test suites (`TestGarchEngine`, `TestOrderTracker`, `TestFeatureExtractor`), and automated MetaEditor CLI compilation with 0 errors. |
| [`test_mt5_mcp_server.py`](test_mt5_mcp_server.py) | `src/tools/mt5_mcp_server.py` & `.agents/mcp_config.json` | Registration of `mt5-local` MCP server, JSON-RPC 2.0 handshake/discovery/error protocol compliance, schema validity across 9 tools, and live MT5 terminal diagnostic queries. |
| [`test_triple_barrier_and_garch_features.py`](test_triple_barrier_and_garch_features.py) | `src/config.py`, `src/preset_generator.py`, `src/mt5_client.py` | Triple Barrier labeling configuration defaults and fallbacks, GARCH feature inclusion in dataset schema, `.set` preset parity, and Strategy Tester `.ini` input generation. |

---

## 🚀 Running the Tests

```powershell
# Run full test suite with verbose output
pytest tests/ -v

# Run specific test module
pytest tests/test_garch_math.py -v
```
