# MLOps Pipeline Operational & Deployment Guide

This operational guide provides practical instructions for quantitative researchers, data scientists, and algorithmic traders to configure, train, tune, and deploy the **MetaTrader 5 MLOps Pipeline**.

---

## 1. Quick Start & Execution Modes

The orchestrator (`run_pipeline.py`) provides three execution modes:

### Mode 1: Full End-to-End Automated Pipeline
```powershell
python run_pipeline.py
```
**Execution Sequence**:
1. **Scoped Cleanup**: Atomically deletes previous artifacts for target symbol/timeframe.
2. **MT5 Initialization**: Validates API connection and market symbol properties.
3. **MQL5 Synchronization & Compilation**: Syncs Include files and compiles `DMatrix-EA.mq5` via MetaEditor CLI.
4. **Strategy Tester Backtest**: Launches background tester simulation with Watchdog monitoring to collect historical feature vectors.
5. **Dataset Discovery & Validation**: Discovers and validates `<Symbol>_<TF>_buy.csv` and `sell.csv` in Common Files.
6. **Dual XGBoost Training & Optuna Optimization**: Splits chronologically, tunes hyperparameters with Optuna Bayesian search, and trains BUY and SELL models with early stopping and real-time execution telemetry (start/completion timestamps and elapsed wall-clock duration formatted as `HH:MM:SS`).
7. **Pure 1D Float ONNX Export**: Converts boosters to ONNX graphs (`[None, num_features] -> [None, 2]`), removes ZipMap, and verifies tensor inference with ONNX Runtime.
8. **Artifact Deployment & Presets**: Deploys `.onnx` models, generates native `.set` presets, and creates chart templates (`.tpl`).
9. **Live EA Compilation**: Compiles `LiveONNX-EA.mq5` via MetaEditor CLI (0 errors).

### Mode 2: Reusing Existing Datasets (Skip Strategy Tester)
```powershell
python run_pipeline.py --skip-dataset
```
*(Or by setting `SKIP_DATASET_GENERATION=1` in `.env`)*
Skips Strategy Tester simulation and `DMatrix-EA` compilation if existing `<Symbol>_<TF>_buy.csv` and `sell.csv` datasets are present, proceeding directly to XGBoost training, Optuna tuning, ONNX export/deploy, preset generation, and `LiveONNX-EA.mq5` compilation.

### Mode 3: Compile-Only Mode
```powershell
python run_pipeline.py --compile-only
```
Synchronizes workspace MQL5 files to the terminal directory, updates native presets and chart templates, and compiles both `DMatrix-EA.mq5` and `LiveONNX-EA.mq5` via MetaEditor CLI.
> **Dataset & Model Preservation Guarantee**: This mode strictly **preserves** all pre-existing `.onnx` machine learning models and historical `.csv` training datasets across workspace and terminal folders. They are **never deleted or overwritten** in compile-only mode.

---

## 2. Complete Configuration Reference (`.env`)

All 80 user-configurable parameters supported by the pipeline:

| Category | Parameter | Type | Default | Description |
|---|---|---|---|---|
| **MT5 Paths** | `MT5_PATH` | `Path` | `C:\Program Files\MetaTrader 5\terminal64.exe` | Path to MetaTrader 5 terminal executable |
| | `METAEDITOR_PATH` | `Path` | `C:\Program Files\MetaTrader 5\metaeditor64.exe` | Path to MetaEditor compiler executable |
| | `MT5_DATA_PATH` | `Path` | *(Auto-discovered)* | Explicit terminal data directory path |
| | `MT5_COMMON_PATH` | `Path` | *(Auto-discovered)* | Explicit shared terminal common folder path |
| **Backtest** | `SYMBOL` | `str` | `EURUSD` | Trading asset pair symbol |
| | `TIMEFRAME` | `str` | `H1` | Chart timeframe (`M1`, `M5`, `M15`, `M30`, `H1`, `H4`, `D1`) |
| | `MAGIC_NUMBER` | `int` | `222100` | EA magic number for position routing and order isolation |
| | `FROM_DATE` | `str` | `2012.01.01` | Backtest collection start date (`YYYY.MM.DD`) |
| | `TO_DATE` | `str` | `2026.05.03` | Backtest collection end date (`YYYY.MM.DD`) |
| | `SHUTDOWN_TERMINAL`| `int` | `1` | Auto-close terminal upon test completion (`1`: Yes, `0`: No) |
| | `BACKTEST_TIMEOUT` | `int` | `0` | Watchdog timeout in seconds (`0`: Infinite / manual control) |
| | `WATCHDOG_POLL_INTERVAL` | `int` | `10` | Watchdog polling interval in seconds |
| | `SKIP_DATASET_GENERATION` | `bool` | `0` | Skip MT5 Strategy Tester and reuse existing datasets (`1`: Yes, `0`: No) |
| | `AVOID_PANDEMICTIME` | `bool` | `0` | Blackout / skip pandemic anomaly period in DMatrix-EA (`1`: Yes, `0`: No) |
| | `PANDEMIC_START_DATE` | `str` | `2020.01.01 00:00:00` | Blackout window start date/time in EET/EEST Server Time (inclusive) |
| | `PANDEMIC_END_DATE` | `str` | `2021.06.01 00:00:00` | Blackout window end date/time in EET/EEST Server Time (exclusive) |
| **Lookback & Barriers**| `FEATURE_LOOKBACK`| `int` | `4` | Number of lagged bars $[t..t-N]$ in feature vector |
| | `LABEL_HORIZON_BARS` | `int` | `12` | Vertical barrier: Maximum holding bars before timeout (closes at market) |
| | `LABEL_MIN_POINTS` | `int` | `150` | Upper barrier: Minimum favorable points for positive label (OPEN) |
| | `LABEL_MAX_ADVERSE_POINTS` | `int` | `150` | Lower barrier: Maximum adverse points for label invalidation (NOT_OPEN) |
| **XGBoost ML** | `XGB_MAX_DEPTH` | `int` | `4` | Maximum tree depth |
| | `XGB_ETA` | `float` | `0.015` | Learning rate (shrinkage factor) |
| | `XGB_SUBSAMPLE` | `float` | `0.70` | Subsample ratio of training instances per tree |
| | `XGB_COLSAMPLE_BYTREE` | `float` | `0.55` | Subsample ratio of columns per tree |
| | `XGB_MIN_CHILD_WEIGHT` | `float` | `6.0` | Minimum sum of instance weight in a child node |
| | `XGB_LAMBDA` | `float` | `4.0` | L2 regularization term on leaf weights |
| | `XGB_ALPHA` | `float` | `1.5` | L1 regularization term on leaf weights |
| | `XGB_ROUNDS` | `int` | `800` | Maximum boosting rounds |
| | `XGB_EARLY_STOPPING_ROUNDS`| `int`| `25` | Early stopping patience rounds on validation set |
| | `VALIDATION_PERCENTAGE` | `float` | `0.10` | Chronological time-series validation split ratio |
| **Optimization** | `OPTUNA_TRIALS` | `int` | `100` | Number of hyperparameter search trials per direction |
| | `OPTUNA_OBJECTIVE_METRIC` | `str` | `logloss` | Metric optimized during Optuna Bayesian search (`logloss`, `roc_auc`, `precision`, `f1`) |
| **Directional Evaluation** | `EVAL_CLASSIFICATION_THRESHOLD` | `float` | `0.50` | Primary conditional probability threshold $\theta \in (0, 1)$ for directional metrics |
| | `EVAL_ENABLE_THRESHOLD_GRID` | `bool` | `1` | Enable ASCII parametric sensitivity grid output across threshold span |
| | `EVAL_THRESHOLD_MIN` | `float` | `0.40` | Minimum threshold for parametric evaluation grid |
| | `EVAL_THRESHOLD_MAX` | `float` | `0.70` | Maximum threshold for parametric evaluation grid |
| | `EVAL_THRESHOLD_STEP` | `float` | `0.02` | Step increment for parametric evaluation grid |
| **Feature Toggles**| `USE_ADX` .. `USE_SPREAD` | `bool` | `1/0` | Toggles for technical indicators |
| | `USE_GARCH_FEATURES` | `bool` | `1` | Include GARCH volatility features ($\omega, \text{vol\_ratio}, \text{vol\_trend}, \sigma_{\text{cond}}, \sigma_{\text{agg}}$) in dataset |
| **GARCH & Indicators** | `GARCH_HORIZON` | `int` | `8` | Forecast horizon in bars $[t+1..t+H]$ for volatility aggregation |
| | `PRICE_SIZE` | `int` | `500` | Historical sample window (bars) for GARCH variance fitting |
| | `GARCH_ALPHA` | `float` | `0.05` | ARCH shock parameter ($\alpha > 0$) |
| | `GARCH_BETA` | `float` | `0.92` | GARCH persistence parameter ($\beta > 0, \alpha + \beta < 1$) |
| | `ADX_PERIOD` .. `STOCH_PRICE_FIELD` | `int/float` | *(Standard)* | Periods, shifts, methods, and applied prices |
 
---

### 2.1 Directional ML Evaluation & Acceptance Threshold Calibration

The pipeline models BUY and SELL setups as two independent binary classification problems. The trainer evaluates model performance out-of-sample over the validation set:

1. **Dataset Class Balance Distribution**:
   - Emits total sample count, positive samples ($y=1$, reached target within barrier), and negative samples ($y=0$, adverse move or timeout), split between training and validation sets.
   - Separate telemetry is provided for `[BUY]` and `[SELL]`.

2. **Directional Validation Metrics (at `EVAL_CLASSIFICATION_THRESHOLD`)**:
   - **ROC-AUC**: Probability that a positive setup scores higher than a negative setup.
   - **LogLoss**: Out-of-sample cross-entropy calibration loss.
   - **Accuracy**: Overall classification accuracy at threshold $\theta$.
   - **Directional Precision (Win Rate)**: Percentage of predicted signals that actually reached profit target.
   - **Momentum Recall**: Percentage of profitable market opportunities captured by the model.
   - **F1-Score**: Harmonic mean of Precision and Recall.
   - **Active Signals Count & Frequency**: Number of bars triggering a signal and market participation percentage.

3. **Parametric Directional Sensitivity Grid**:
   - Evaluates the model across a fine-grained threshold grid (`EVAL_THRESHOLD_MIN` to `EVAL_THRESHOLD_MAX` in `EVAL_THRESHOLD_STEP` increments).
   - Shows the exact empirical trade-off between **Selectivity (Precision / Win Rate)** and **Signal Volume (Frequency / Recall)**.
   - Researchers directly inspect this table to choose calibrated values for `InpMinimalLevelAcceptedBuy` and `InpMinimalLevelAcceptedSell` in `LiveONNX-EA.mq5`.

---

## 3. Deploying Models to Live MetaTrader 5 Charts

### 3.1 Automated Artifact Placement
Artifacts are automatically deployed into the active MT5 terminal directory:
- **ONNX Models**: `MQL5\Files\Models\<Symbol>_<Timeframe>_model_*.onnx`
- **Native Presets**: `MQL5\Presets\LiveONNX-EA_<Symbol>_<Timeframe>.set`
- **Chart Templates**: `MQL5\Profiles\Templates\<Symbol>_<Timeframe>.tpl`
- **Compiled EA**: `MQL5\Experts\LiveONNX-EA.ex5`

### 3.2 Attaching to Chart in MT5
1. Open a chart for the target symbol (e.g., `EURUSD`) and timeframe (e.g., `H1`).
2. In the **Navigator** window $\rightarrow$ Expand **Expert Advisors** $\rightarrow$ Drag **`LiveONNX-EA`** onto the chart.
3. In the EA Properties Dialog:
   - Click **Inputs** tab $\rightarrow$ Click **Load** button.
   - Select **`LiveONNX-EA_EURUSD_H1.set`** from the `Presets` folder.
   - Click **Open**. All parameters, feature lookbacks, indicator periods, and model paths load automatically with zero train-serving skew.
4. Set **Trade Direction** (`InpTradeDirection`):
   - `0 (DIRECTION_BOTH)`: Evaluates both BUY and SELL models.
   - `1 (DIRECTION_ONLY_BUY)`: Executes exclusively long positions.
   - `2 (DIRECTION_ONLY_SELL)`: Executes exclusively short positions.
5. Dynamic Risk Sizing:
   - Evaluates real-time analytical GARCH(1,1) multi-step dynamic TP/SL (`InpKTP` and `InpKSL`).
6. Click **OK** and ensure the **Algo Trading** button in the MT5 top toolbar is active (Green).

---

## 4. Live Log Streaming & Training Telemetry Architecture

### 4.1 Strategy Tester Live Log Streaming & Error Interception
During backtest data collection (`DMatrix-EA.mq5`), the Python orchestrator streams Strategy Tester logs in real time while guarding against silent failures and corrupt datasets:

1. **Byte Offset Tailing & Encoding Auto-Detection**:
   - Tails active `.log` files in `Tester/logs` and agent sandboxes via byte offset tracking (`file_offsets`).
   - Dynamically detects UTF-16 (with null byte `\x00` delimiters) vs. UTF-8 encodings to ensure cross-platform compatibility.
2. **Non-Fatal Warning & Warmup Whitelist**:
   - Benign logs — including warmup notices (`[FeatureExtractor] [WARMUP]`), market closed warnings (`[DMatrix-EA] [WARNING]`), tick sync notices (`no real ticks`, `real ticks discarded/absent`), and history buffer loading — are streamed cleanly as `[MT5] <line>` without terminating the process.
3. **Fatal Error Interception**:
   - Critical errors — such as `[DMatrix-EA] [ERROR]`, `critical runtime error`, `cannot load expert`, `zero divide`, `array out of range`, and `pointer cannot be used` — immediately trigger subprocess termination (`proc.kill()`), kill zombie MT5 instances, and raise a descriptive `RuntimeError`.

### 4.2 Machine Learning Training Telemetry & Duration Tracking
During dual XGBoost training and Optuna hyperparameter optimization (`DualXGBoostTrainer`), the orchestrator captures real-time wall-clock execution metrics:
- **Start Timestamp**: Recorded at the onset of optimization/training for each direction (`BUY` and `SELL`): `[YYYY-MM-DD HH:MM:SS] [*] Optimizing and Training XGBoost <DIR> Model...`.
- **Completion Timestamp & Elapsed Duration**: Formatted as `HH:MM:SS` upon final estimator fit:
  `[*] Training completed at: [YYYY-MM-DD HH:MM:SS] (Elapsed: HH:MM:SS)`.
- **Performance Diagnostics**: Directly reports validation `ROC-AUC`, `Accuracy`, `LogLoss`, and early-stopping `best_iteration`.

---

## 5. Troubleshooting & Operational FAQ

### Q1: MetaEditor compilation fails with "File not found"
- **Resolution**: Verify `METAEDITOR_PATH` in `.env` and execute `python run_pipeline.py --compile-only`. Inspect compilation logs in `MT5_DATA_PATH/logs/compile_*.log`.

### Q2: Strategy Tester exits immediately without generating datasets
- **Resolution**: Open MT5 $\rightarrow$ Press `F2` $\rightarrow$ Download historical bars for the symbol and timeframe. Ensure no zombie `terminal64.exe` processes are running.

### Q3: Strategy Tester logs show `[DMatrix-EA] [WARNING] Market is closed or trade disabled`
- **Resolution**: This is expected when testing spans weekend market gaps or off-session holiday periods (`TRADE_RETCODE_MARKET_CLOSED` [10018], `TRADE_RETCODE_OFFQUOTES` [10004], `TRADE_RETCODE_PRICE_OFF` [10021], or `TRADE_RETCODE_TRADE_DISABLED` [10017]). `DMatrix-EA` gracefully skips the bar and continues data collection without interrupting the backtest.

### Q4: Strategy Tester aborts with `RuntimeError: Critical error detected during MT5 Strategy Tester execution`
- **Resolution**: The live log streaming guard detected a fatal error log (e.g., `[DMatrix-EA] [ERROR]`, `critical runtime error`, `cannot load expert`). Inspect the printed `[MT5]` log lines immediately preceding the abort to determine the exact retcode, order ticket, or `GetLastError()` error. Common causes include insufficient testing deposit (resolved by setting `Deposit=1000000000000000`), excessive lot sizing, or missing symbol trading permissions.

### Q5: Tester logs show `[FeatureExtractor] [WARMUP] Insufficient historical rates` or `Tester: real ticks discarded`
- **Resolution**: These lines are part of the normal terminal startup and indicator history buffer initialization. The live log streamer whitelists these messages and processes them without halting execution.

### Q6: ONNX error: "Input tensor shape mismatch"
- **Resolution**: Always load the auto-generated `.set` preset (`LiveONNX-EA_<Symbol>_<TF>.set`) rather than modifying indicator toggles manually.

### Q7: Achieving Sub-Millisecond Execution Latency
- Static tensor shapes are configured once during `OnInit()`.
- Single-precision `vectorf` arrays pass directly to `OnnxRun(..., ONNX_NO_CONVERSION, ...)`.
- Feature extraction executes only once per bar open (`IsNewBar()`).

---

## 6. Serverless Quantitative Analytics & Model Context Protocol (MCP)

The repository uses a 100% serverless, zero-container analytical architecture powered by **DuckDB** and specialized Model Context Protocol (MCP) servers. Analytical queries, dataset audits, and experiment history run directly on disk with zero memory or container overhead.

### 6.1 Embedded Analytics with DuckDB (`quant_analytics.duckdb`)
- **Direct CSV Querying**: DuckDB executes high-performance analytical SQL queries directly over Strategy Tester CSV exports (`MQL5/Files/*.csv`) without requiring external database services.
- **Embedded Experiment Store**: Historical performance metrics, Optuna Bayesian trials, and dataset class distributions are stored locally inside `quant_analytics.duckdb`.
- **Zero Overhead**: No background daemons, no Docker containers, and no open database ports.

### 6.2 Model Context Protocol (MCP) Servers (`.agents/mcp_config.json` and `.agy/settings.json`)
The AI agent is equipped with 5 specialized, high-performance MCP servers:
1. **`duckdb`**: Executes instant analytical OLAP SQL queries directly on Strategy Tester CSV datasets, trade logs, and local tables (`quant_analytics.duckdb`) without token overhead.
2. **`memory`**: Persistent knowledge graph preserving pair-specific market nuances, session liquidity rules, and quantitative constraints across sessions.
3. **`duckduckgo-search`**: Zero-credential web search engine for academic financial literature (SSRN, ArXiv) and historical macroeconomic calendar release audits.
4. **`fetch`**: Versatile, structured web extraction engine used to retrieve content from any web source relevant to the project—including technical API documentation, MQL5 community articles, financial news feeds, EA research resources, and macroeconomic data.
5. **`economic-calendar`**: Native, dedicated Python Stdio MCP server (`src.tools.macro_calendar --mcp`) implementing tools for the **[MQL5 Economic Calendar](https://www.mql5.com/en/economic-calendar)** (`get_mql5_economic_calendar`), open economic news feeds (`get_economic_news`), high-impact catalyst lookups (`get_high_impact_catalysts`), and automated backtest anomaly correlation (`audit_backtest_anomaly`).
