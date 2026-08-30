# MQL5 Shared Include Libraries (`MQL5/Include/`)

This directory contains the modular, highly optimized MQL5 header files (`.mqh`) that implement the core mathematical, econometric, and feature engineering logic of the pipeline.

---

## 📚 Libraries Overview

### 1. [`FeatureExtractor.mqh`](FeatureExtractor.mqh)
- **Primary Class**: `CFeatureExtractor`
- **Core Functionality**:
  - Initializes, manages, and safely releases handles for technical indicators across 14 toggleable groups (ADX, ATR, Bands, MACD, Fast MA, Slow MA, RSI, Stochastic, Candlestick, Weekday, Quarter Day, Open Markets, Spread, and GARCH(1,1) Volatility Features).
  - Integrates `CGarchEngine` directly to compute 5 econometric volatility metrics per bar:
    $$\mathbf{f}_{\text{garch}} = [\omega, \text{vol\_ratio}, \text{vol\_trend}, \sigma_{\text{cond}}, \sigma_{\text{agg}}]$$
    where $\text{vol\_ratio} = \sigma_{\text{cond}} / \sqrt{s^2}$ and $\text{vol\_trend} = \sigma_{\text{agg}} / (\sqrt{H} \sigma_{\text{cond}})$.
  - Flattens multi-lag historical sequences across lookback horizon $[t, t-1, \dots, t-H]$ into a continuous 1D feature vector:
    $$\mathbf{x}_t = [F_1(t), \dots, F_M(t), F_1(t-1), \dots, F_M(t-H)]$$
  - Emits non-fatal `[WARMUP]` log alerts during initial history buffer loading without triggering pipeline aborts.
  - Provides column naming metadata (`GetCSVHeader()`) and dynamic dimension calculations (`GetTotalFeatureCount()`).

### 2. [`GarchEngine.mqh`](GarchEngine.mqh)
- **Primary Class**: `CGarchEngine`
- **Core Functionality**:
  - Computes continuously compounded log returns $r_t = \ln(P_t / P_{t-1})$ and sample variance strictly over closed historical bars $[t-1 .. t-(N+1)]$.
  - Implements Bollerslev (1986) GARCH(1,1) forward conditional variance recursion:
    $$\sigma_t^2 = \omega + \alpha (r_t - \mu)^2 + \beta \sigma_{t-1}^2$$
  - Exposes `ComputeGarchMetrics(...)` returning dynamic metrics $\omega, \text{vol\_ratio}, \text{vol\_trend}, \sigma_{\text{cond}}, \sigma_{\text{agg}}$ to eliminate zero-variance constants.
  - Aggregates multi-step forward variance across horizon $H = \text{GARCH\_HORIZON}$:
    $$\sigma_{\text{agg}} = \sqrt{\sum_{h=1}^H \left( V_L + (\alpha + \beta)^h (\sigma_t^2 - V_L) \right)}$$
  - Computes dynamic price risk using last closed bar price (bar 1): $\text{TP}_{\text{points}} = K_{\text{TP}} \cdot \text{RiskPoints}$ and $\text{SL}_{\text{points}} = K_{\text{SL}} \cdot \text{RiskPoints}$.

### 3. [`OrderTracker.mqh`](OrderTracker.mqh)
- **Primary Class**: `COrderTracker`
- **Core Functionality**:
  - Maps active MT5 position tickets (`DEAL_POSITION_ID`) to in-memory feature vectors in RAM during Strategy Tester simulation.
  - Enforces Triple Barrier vertical holding time limits via `CheckTimeouts(int maxBars, CTrade &trade)`, closing positions that exceed horizon bars.
  - Ingests `OnTradeTransaction` deal closure events (`DEAL_ENTRY_OUT` / `DEAL_ENTRY_OUT_BY`).
  - Evaluates the **Golden Rule of Net Liquid Profit**:
    $$\text{NetLiquidProfit} = \text{DEAL\_PROFIT} + \text{DEAL\_SWAP} + \text{DEAL\_COMMISSION}$$
    $$\text{Label } y = \begin{cases} 1.0f \; (\text{OPEN}), & \text{if } \text{TP reached} \land \text{NetLiquidProfit} > 0.0 \\ 0.0f \; (\text{NOT\_OPEN}), & \text{if } \text{SL reached} \lor \text{Timeout} \lor \text{NetLiquidProfit} \le 0.0 \end{cases}$$
  - Evaluates deinitialization rules (`OnDeinit`), labeling unresolved open trades as `0.0f` (`NOT_OPEN`) per the Triple Barrier vertical horizon.
  - Sorts datasets chronologically via an optimized index-based QuickSort (`m_sortIndices[]`) by `baseTimestamp` (avoiding expensive deep struct copying) and exports `<Symbol>_<TF>_buy.csv` and `sell.csv` to `Common/Files`.

### 4. [`ConsecutiveManager.mqh`](ConsecutiveManager.mqh)
- **Primary Class**: `CConsecutiveManager`
- **Core Functionality**:
  - Encapsulates execution and stop management when consecutive prediction signals occur in the same direction (`BUY` / `SELL`).
  - Supports 5 modular modes: `LEGACY_INDEPENDENT` (0), `SINGLE_HURDLE_RATCHET` (1), `SINGLE_CHAIN_LINK` (2), `UNIFIED_BASKET` (3), and `PYRAMIDING_STEP_LOCK` (4).
  - Implements dynamic **Swap Amortization** (`CalculateSwapAmortizationPoints`), converting accumulated negative overnight swap into exact price points to guarantee $\text{NetLiquidProfit} \ge 0.0$ upon breakeven stop-out:
    $$\text{SwapPoints} = \frac{|\text{AccruedSwap}|}{\text{Volume} \cdot (\text{TickValue} / \text{TickSize}) \cdot \text{Point}}$$
  - Implements **Anti-Chop Displacement Filtering** (`InpAntiChopMinDisplacement`) preventing position suffocation and whipsaws during consolidation ranges.
  - Synchronizes multi-order baskets and guarantees that pyramiding orders only scale when preceding positions are fully locked in profit.
  - Implements **Opposing Regime Defense Filter** (`CheckAndProcessOpposingRegime`), tracking consecutive adverse ML predictions ($N$ bars streak) against active positions and executing configurable institutional defensive actions: Close If Net Profit (0), Immediate Liquidation (1), Defensive Trailing Stop (2), Net-Breakeven Swap Lock (3), Recalculate Target Barrier (4), and Stop & Reverse (5).

### 5. [`ExecutionAuditor.mqh`](ExecutionAuditor.mqh) (alias: [`PredictionAuditor.mqh`](PredictionAuditor.mqh))
- **Primary Class**: `CExecutionAuditor` (aliased as `CPredictionAuditor`)
- **Core Functionality**:
  - Institutional-grade, non-configurable, mandatory execution and prediction telemetry engine operating inside `LiveONNX-EA.mq5`.
  - Automatically initializes and manages an isolated, high-performance SQLite 3 database in the shared terminal filesystem:
    $$\text{Path} = \text{Common/Files/AuditLogs/}\langle\text{Symbol}\rangle\_\langle\text{TF}\rangle\_\langle\text{YYYYMMDD\_HHMMSS}\rangle\text{.db}$$
  - Enforces WAL journal mode (`PRAGMA journal_mode=WAL;`), normal synchronization (`PRAGMA synchronous=NORMAL;`), and a 5,000 ms busy timeout (`PRAGMA busy_timeout=5000;`) for non-blocking sub-microsecond writing under heavy tick traffic.
  - **Tri-Pillar Relational SQLite Architecture**:
    1. **`candle_telemetry`** (aliased by view `prediction_audit_logs`): Unbroken, bar-by-bar chronological time series capturing 38 columns:
       - Timestamps and execution latencies: inference latency (`execution_latency_us`) and broker order routing roundtrip latency (`order_latency_ms`).
       - Raw probabilities: $P(\text{BUY})$ and $P(\text{SELL})$.
       - **Leading Uncertainty Indicators**: Shannon entropy $H = -\sum p \log p$ (`shannon_entropy`), conviction delta $|P_{\text{BUY}} - P_{\text{SELL}}|$ (`conviction_delta`), and conflicting signal flags (`has_conflicting_signals`).
       - Volatility and risk metrics: GARCH conditional $\sigma_{\text{cond}}$ and aggregate $\sigma_{\text{agg}}$, dynamic TP/SL points, and fractal Support/Resistance snapping coordinates.
       - Viability gating: gate passed boolean, rejected gate ID (1=Margin, 2=R:R, 3=Loss budget), account equity, balance, and margin level.
       - Execution profiling: execution action, order ticket, requested vs. executed fill prices, entry slippage in points (`slippage_points`), spread, and broker return codes.
    2. **`system_events_log`**: Structured operational event and incident log for non-fatal broker warnings (requotes 10004, invalid stops 10016, price off 10021), fatal execution errors, database lock states, and copy-rates failures, tracking `severity` (`INFO`, `WARNING`, `ERROR`, `CRITICAL`), `subsystem`, `event_code`, `message`, and context.
    3. **`trade_lifecycle_log`**: Comprehensive trade lifecycle and financial outcome attribution recorded via `OnTradeTransaction`:
       - Position ID, symbol, direction, entry/exit timestamps, entry/exit prices, volume, and holding duration in bars.
       - Execution friction: entry slippage points and exit slippage points.
       - **Excursion Profiling**: Maximum Adverse Excursion (`mae_points`) and Maximum Favorable Excursion (`mfe_points`) tracked dynamically candle-by-candle during trade life.
       - Detailed PnL accounting: gross profit, swap drag, broker commissions, and final Net Liquid Profit ($\text{NetLiquidProfit} = \text{Profit} + \text{Swap} + \text{Commission}$).
       - Exit attribution: `exit_reason` (`TP`, `SL`, `TRAILING_STOP`, `OPPOSING_REGIME`, `MACRO_NEWS`, `TIMEOUT`, `MANUAL`) and deal ticket.
  - **Collision & Strategy Tester Protection (`.bkp` Rollover)**:
    - If a `.db` file with the exact target name already exists (e.g. consecutive backtest simulations in MT5 Tester starting at identical simulated bar timestamps), `CExecutionAuditor` automatically duplicates the existing database to `<db_path>.bkp` (`FILE_REWRITE`), purges the original `.db` along with `-wal` and `-shm` files, and re-initializes an empty, brand-new database (*zerado*).
  - Provides quantitative auditors with the requisite leading indicators to detect ML model decay, feature distribution drift, and broker execution toxicity long before lagging capital balance reflects degradation.

---

## 🧪 Native MQL5 Unit Test Suites (`MQL5/Include/Tests/`)

This directory contains native unit test classes and assertion harnesses for black-box verification of MQL5 core libraries:

- [`Tests/MqlTestFramework.mqh`](Tests/MqlTestFramework.mqh): Assertion framework (`AssertTrue`, `AssertFalse`, `AssertEqualInt`, `AssertEqualDouble`, `AssertEqualString`, `AssertGreater`, `AssertLess`) and test statistics aggregator.
- [`Tests/TestGarchEngine.mqh`](Tests/TestGarchEngine.mqh): Black-box unit tests for `CGarchEngine` verifying parameter clamping, covariance stationarity constraints ($\alpha + \beta < 1.0$), variance targeting equations, and zero-divide protections.
- [`Tests/TestOrderTracker.mqh`](Tests/TestOrderTracker.mqh): Black-box unit tests for `COrderTracker` verifying dynamic buffer expansion (+512 chunks), unresolved trade zero-labeling, chronological QuickSort, and Golden Rule profit evaluations.
- [`Tests/TestFeatureExtractor.mqh`](Tests/TestFeatureExtractor.mqh): Black-box unit tests for `CFeatureExtractor` verifying default state, session hour mappings in EET/EEST, handle release idempotency, and uninitialized failure guards.
- [`Tests/TestConsecutiveManager.mqh`](Tests/TestConsecutiveManager.mqh): Unit tests for `CConsecutiveManager` verifying constructor defaults, config mutation, swap point formulas, empty-book zero counts, mode & opposing action enumeration invariants, and disabled filter guards.
