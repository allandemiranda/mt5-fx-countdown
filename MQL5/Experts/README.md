# MQL5 Expert Advisors (`MQL5/Experts/`)

This directory contains the executable Expert Advisors (`.mq5`) that operate inside the MetaTrader 5 terminal.

---

## 🤖 Expert Advisors

### 1. [`DMatrix-EA.mq5`](DMatrix-EA.mq5) (Data Collection Engine)
- **Operating Environment**: MetaTrader 5 Strategy Tester.
- **Workflow**:
  1. `OnInit()`: Initializes `CFeatureExtractor` (including GARCH features if `InpUseGarchFeatures` is enabled), sets up indicator handles, and prepares `COrderTracker`.
  2. `OnTick()`: Detects new bar open (`IsNewBar()`).
     - **Vertical Barrier Check**: Calls `COrderTracker::CheckTimeouts(InpLabelHorizonBars, g_trade)`, closing active positions exceeding `InpLabelHorizonBars` and labeling them `0.0f` (`NOT_OPEN`).
     - **Anomaly & Pandemic Blackout Filter**: Evaluates `InpAvoidPandemicTime`. If active and `baseTimestamp >= InpPandemicStartTime && baseTimestamp < InpPandemicEndTime` (evaluated strictly in MT5 Server Time: EET/EEST), skips candidate order opening completely without logging (allowing existing trades to timeout or exit cleanly).
     - **Daily Schedule Filter**: Evaluates `IsTradeScheduleAllowed(baseTimestamp)` in MT5 Server Time; skips candidate entry if the current day or time window is disabled.
     - Extracts flattened feature vector $\mathbf{x}_t$.
     - Computes fixed Triple Barrier price levels (`InpLabelMinPoints` and `InpLabelMaxAdversePoints`) bounded by broker stop levels and spread.
     - Attempts simultaneous BUY and SELL positions with configurable `InpMagicNumber` (default `222100`):
       - **Stop Calculation Rules**:
         - BUY: Execution at Ask, closes at Bid $\rightarrow$ `buySL = NormalizeDouble(bid - slDist, digits)`, `buyTP = NormalizeDouble(ask + tpDist, digits)`.
         - SELL: Execution at Bid, closes at Ask $\rightarrow$ `sellSL = NormalizeDouble(ask + slDist, digits)`, `sellTP = NormalizeDouble(bid - tpDist, digits)`.
         - Stop buffers: Pure theoretical targets `slDist = targetSL = InpLabelMaxAdversePoints * point` and `tpDist = targetTP = InpLabelMinPoints * point`. If broker constraints `(stopsLevel + spread + 5) * point` exceed `targetTP` or `targetSL`, the bar is skipped to prevent label distortion.
     - **Non-Fatal Market Conditions (`[WARNING]`)**: If order placement fails due to normal market closures, disabled sessions, or invalid stops (`TRADE_RETCODE_MARKET_CLOSED`, `TRADE_RETCODE_OFFQUOTES`, `TRADE_RETCODE_PRICE_OFF`, `TRADE_RETCODE_TRADE_DISABLED`, `TRADE_RETCODE_INVALID_STOPS`), emits `[DMatrix-EA] [WARNING]` with full diagnostic prices/stops and skips the bar gracefully.
     - **Fatal Order Execution Failures (`[ERROR]`)**: If order placement fails due to genuine unexpected execution errors (e.g., account margin exhaustion, parameter errors), emits `[DMatrix-EA] [ERROR]` with retcode, deal, order ticket, description, and `GetLastError()`, triggering immediate pipeline abort.
  3. Registers successful position tickets (`DEAL_POSITION_ID`) and feature vectors into RAM via `COrderTracker::RegisterPosition()`.
  4. `OnTradeTransaction()`: Intercepts deal closure events and records labeled samples ($1.0f$ / $0.0f$) with the Golden Rule.
  5. `OnDeinit()`: Labels unresolved positions as `0.0f` (`NOT_OPEN`), sorts chronologically using index-based QuickSort, and exports `<Symbol>_<TF>_buy.csv` and `sell.csv`.

#### Order Execution Error & Warning Classification

| Severity | Log Prefix | Retcodes Handled | Description & Pipeline Action |
|---|---|---|---|
| **Non-Fatal** | `[DMatrix-EA] [WARNING]`<br/>`[LiveONNX-EA] [WARNING]` | `TRADE_RETCODE_MARKET_CLOSED` (10018)<br/>`TRADE_RETCODE_OFFQUOTES` (10004)<br/>`TRADE_RETCODE_PRICE_OFF` (10021)<br/>`TRADE_RETCODE_TRADE_DISABLED` (10017)<br/>`TRADE_RETCODE_INVALID_STOPS` (10016) | **Transient Market State**: Emits warning with full diagnostic parameters (Ask, Bid, SL, TP, StopsLevel, Spread), skips order registration for the current bar, and continues execution uninterrupted. |
| **Fatal** | `[DMatrix-EA] [ERROR]`<br/>`[LiveONNX-EA] [ERROR]` | All other failure retcodes (e.g., `TRADE_RETCODE_NO_MONEY`, `TRADE_RETCODE_LIMIT_ORDERS`) | **Genuine Execution Error**: Logs full diagnostic details (Ask, Bid, SL, TP, StopsLevel, Spread, retcode, ticket, deal, description, `GetLastError()`). For DMatrix-EA, the Python live log monitor intercepts this log line and aborts the backtest immediately. |

### 2. [`LiveONNX-EA.mq5`](LiveONNX-EA.mq5) (Live Inference & Trading Engine)
- **Operating Environment**: Live / Demo Chart Execution in MT5 Terminal.
- **Workflow**:
  1. `OnInit()`: Loads native `.set` preset, initializes `CFeatureExtractor` (with strict feature parity to `DMatrix-EA`), resolves ONNX model files (`model_buy.onnx` and `model_sell.onnx`), configures static 1D float tensor shapes `[1, num_features]` and `[1, 2]`, and initializes the GARCH execution risk engine with `InpRiskGarchHorizon`.
  2. `OnTick()`:
     - On new bar open, evaluates `IsTradeScheduleAllowed(barTime)` in MT5 Server Time; skips bar if day or time window is not permitted.
     - Extracts real-time feature vector into native `vectorf` (guaranteeing zero train-serving skew with `DMatrix-EA`).
     - Calls `OnnxRun(hModel, ONNX_NO_CONVERSION, vectorf, outProb)` for microsecond inference without dynamic memory allocations.
     - Applies direction filter (`BOTH`, `ONLY_BUY`, `ONLY_SELL`) and probability thresholds (`InpMinimalLevelAcceptedBuy` / `InpMinimalLevelAcceptedSell`).
     - Computes real-time exit levels: Calculates dynamic GARCH(1,1) Stop Loss and Take Profit levels as the permanent volatility envelope. When `InpEnableSRSnapping=true`, refines these levels by snapping to real structural fractal swing pivots (Swing Highs / Swing Lows with radius `InpSRPivotStrength`) within `InpSRLookbackBars`: pulls TP closer to open price (before the zone) to secure execution and pushes SL further away (behind the zone) by `InpSROffsetPoints` to prevent sweeps, strictly clamped within the GARCH Stop Loss boundary to prevent risk expansion.
     - Evaluates pre-trade Risk & Margin Governance (Viability Filter & Position Sizing): When `InpEnableRiskFilter=true`, validates 3 protection gates via `CheckTradeViability` before dispatching: ensures projected Margin Level $\ge \text{brokerCall} \cdot \text{InpMarginSafetyMultiplier` (e.g. $100\% \cdot 1.5 = 150\%$), limits Asymmetry Ratio SL/TP $\le$ `InpMaxRiskRewardRatio` (e.g. 1.5), and enforces maximum adverse trade loss in account currency $\le$ `InpMaxTradeRiskPct` of equity (e.g. 3.0%). When `InpEnableDynamicLotSizing=true`, analytically downsizes starting volume from `InpMaxLotSize` to fit within risk budget and margin capacity. Rejects unviable orders cleanly without fatal aborts.
     - Evaluates Macroeconomic Calendar & News Governance (SQLite): Connects to static `Common/Files/macro_governance.db`. Automatically queries symbol currency components (e.g. EURUSD checks EUR, USD, EURUSD, and GLOBAL) and converts MT5 server time to UTC. When `InpEnableNewsFilter=true` (Live only) or `InpEnableCalendarFilter=true` (Live and Strategy Tester), applies protection actions (`BLOCK_ENTRIES`, `TRAILING_STOP`, `BREAKEVEN`, `CLOSE_ALL`, `ADVISORY_ONLY`). If trailing points are 0 or if modification fails, immediately closes positions for capital defense.
     - Submits market orders via `CTrade` with broker-adaptive filling modes (`FOK`, `IOC`, `RETURN`).
     - Position exits are managed strictly by real Support/Resistance and GARCH Stop Loss and Take Profit levels at the broker level.
     - **Conflicting Signals Suppression (`InpIgnoreConflictingSignals`)**: When enabled (`true`), suppresses order opening if both BUY and SELL models fire on the same bar. When disabled (`false`), removes directional dominance restrictions, permitting concurrent hedging.
     - **Consecutive Signal Management (`CConsecutiveManager`)**: When consecutive candles generate signals in the same direction, the EA executes according to `InpConsecutiveMode`:
       - `CONSECUTIVE_MODE_LEGACY_INDEPENDENT` (0): Baseline independent orders.
       - `CONSECUTIVE_MODE_SINGLE_HURDLE_RATCHET` (1): Single position; ratchets Stop Loss to protect profits only after `InpHurdleProfitPct` of initial target is achieved.
       - `CONSECUTIVE_MODE_SINGLE_CHAIN_LINK` (2): Single position; anchors Stop Loss to previous bar close with `InpAntiChopMinDisplacement` filter against consolidation whipsaws.
       - `CONSECUTIVE_MODE_UNIFIED_BASKET` (3): Scales in volume up to `InpMaxConsecutiveOrders` with synchronized Take Profit and Stop Loss across the entire basket.
       - `CONSECUTIVE_MODE_PYRAMIDING_STEP_LOCK` (4): Opens subsequent orders only when prior positions are fully secured at breakeven/profit.
       - **Swap Amortization**: Automatically converts accrued overnight swap and commissions into price points, ensuring guaranteed non-negative net financial outcomes upon breakeven exit.
     - **Opposing Regime Defense Filter (`InpEnableOpposingRegimeFilter`)**: When active positions face $N$ consecutive adverse ML predictions (`InpOpposingStreakThreshold`), executes the selected defensive strategy:
       - `OPPOSING_ACTION_CLOSE_IF_PROFIT` (0): Liquidates positions in net profit ahead of the adverse move.
       - `OPPOSING_ACTION_CLOSE_IMMEDIATE` (1): Liquidates positions immediately upon statistical thesis invalidation.
       - `OPPOSING_ACTION_TRAILING_DEFENSIVE` (2): Trailing Take-Profit / tight trailing stop (`InpOpposingTrailingPoints`).
       - `OPPOSING_ACTION_BREAKEVEN_NET` (3): Moves SL to net breakeven factoring accrued negative swap.
       - `OPPOSING_ACTION_RECALCULATE_DEFENSIVE` (4): Pulls TP closer by ratio (`InpOpposingRecalculateRatio`) and locks net breakeven SL.
       - `OPPOSING_ACTION_STOP_AND_REVERSE` (5): Closes active positions and reverses to the opposing direction.
      - **Execution & Prediction Audit Engine (`CExecutionAuditor`)**:
        - Institutional logging engine operating on closed candles and trade deal closures (toggleable via `InpIgnoreAudit`, default `false` = active).
        - When `InpIgnoreAudit = true`, completely bypasses database creation and skips all logging.
        - Creates isolated database in `Common/Files/AuditLogs/<Symbol>_<TF>_<YYYYMMDD_HHMMSS>.db` (SQLite 3 with WAL mode and 5000 ms busy timeout).
        - **Collision & Tester Protection (`.bkp` Rollover)**: Automatically creates a backup copy `<db_path>.bkp`, purges existing DB and WAL/SHM artifacts, and initializes an empty fresh database (*zerado*) if the file already exists.
        - Tri-pillar schema architecture:
          1. `candle_telemetry` (view `prediction_audit_logs`): Unbroken 38-metric time series logging raw probabilities, Shannon entropy, conviction delta, conflicting signals, GARCH volatility, S&R coordinates, viability gates, account equity/balance, broker order latency (ms), and entry slippage (points).
          2. `system_events_log`: Granular incident recording for non-fatal broker warnings (10004, 10016, 10021), execution errors, copy-rates failures, and database lock contention.
          3. `trade_lifecycle_log`: Closed-loop trade attribution triggered via `OnTradeTransaction` (`DEAL_ENTRY_OUT` / `DEAL_ENTRY_OUT_BY`), logging entry/exit prices, volume, slippage, MAE/MFE excursions, holding duration, gross/swap/commission/net liquid profit, and exit reasons (`TP`, `SL`, `TRAILING_STOP`, `OPPOSING_REGIME`, `MACRO_NEWS`, `TIMEOUT`).
   3. `OnTradeTransaction()`:
      - Intercepts deal closure transactions (`TRADE_TRANSACTION_DEAL_ADD` where deal entry is `DEAL_ENTRY_OUT` or `DEAL_ENTRY_OUT_BY`).
      - Retrieves active trade metadata (`SActiveTradeMetadata`), computes final holding duration in bars, pulls accumulated Maximum Adverse Excursion (`mae_points`) and Maximum Favorable Excursion (`mfe_points`), extracts net financial outcomes, and calls `g_auditor.RecordTradeExit()`.
   4. `OnDeinit()`: Closes the execution auditor database handle cleanly (`g_auditor.Close()`).
   5. Implements robust non-fatal warning handling (`[WARNING]`) and error logging (`[ERROR]`) matching `DMatrix-EA`.
