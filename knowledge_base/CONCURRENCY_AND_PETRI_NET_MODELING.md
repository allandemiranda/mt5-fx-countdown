# Concurrency Architecture, Formal Petri Net Modeling, and Temporal Logic Specifications

## MT5 Forex Machine Learning Pipeline — Publication-Grade Monograph

**Classification:** Formal Methods & Concurrency Engineering  
**Timezone Standard:** Eastern European Time / Eastern European Summer Time (EET/EEST, UTC+2 winter / UTC+3 summer — MT5 Server Time)  
**Authors:** Formal Methods Subsystem, Quant ML Engine  
**Date:** 2026-09-04  
**Version:** 1.0.0

---

## Abstract

This monograph provides a publication-grade formal analysis of the concurrency architecture of the MetaTrader 5 (MT5) Forex Machine Learning pipeline. The system encompasses multiple concurrent execution contexts: MQL5 Expert Advisor event handlers (executing within MT5's single-threaded cooperative scheduler), Python MLOps subprocesses (running in separate OS processes), a shared SQLite WAL-mode database (`macro_governance.db`) accessed concurrently by up to N independent MT5 chart instances and the Python `macro_agent`, and MCP Stdio servers as sidecar processes.

We identify all shared mutable state, map synchronization points, and provide three complete formal Petri Net models — grounded in the canonical works of **Carl Adam Petri (1962)**, the **IEEE survey of Petri Nets by Murata (1989)**, **Milner's Calculus of Communicating Systems (CCS, 1980)**, **Hoare's Communicating Sequential Processes (CSP, 1978)**, and **Lamport's Temporal Logic of Actions (TLA+, 1994)** — with ASCII art topological diagrams, reachability analyses, P-invariant and T-invariant computations, liveness classifications, LTL/CTL temporal specifications, and rigorous proofs of deadlock-freedom, starvation-freedom, and livelock absence. We conclude with a concurrency vulnerability audit identifying unprotected shared state and TOCTOU patterns not covered in existing knowledge-base documents.

---

## Table of Contents

1. [Section 1: Concurrency Architecture Overview](#section-1-concurrency-architecture-overview)
2. [Section 2: Formal Petri Net Models](#section-2-formal-petri-net-models)
   - [2.1 Model 1: Multi-Chart SQLite Concurrent Access (WAL Mode)](#21-model-1-multi-chart-sqlite-concurrent-access-wal-mode)
   - [2.2 Model 2: Intra-Tick Execution Flow in LiveONNX-EA](#22-model-2-intra-tick-execution-flow-in-liveonnx-ea)
   - [2.3 Model 3: Async DMatrix-EA Trade Event Concurrency](#23-model-3-async-dmatrix-ea-trade-event-concurrency)
3. [Section 3: Mathematical Invariant Analysis](#section-3-mathematical-invariant-analysis)
4. [Section 4: Temporal Logic Specifications (LTL/CTL)](#section-4-temporal-logic-specifications-ltlctl)
5. [Section 5: Deadlock, Starvation & Livelock Freedom Proofs](#section-5-deadlock-starvation--livelock-freedom-proofs)
6. [Section 6: Concurrency Vulnerabilities & Race Condition Audit](#section-6-concurrency-vulnerabilities--race-condition-audit)
7. [Didactic References & Further Reading](#didactic-references--further-reading)

---

## Section 1: Concurrency Architecture Overview

### 1.1 System Execution Model

The MT5 Forex ML pipeline operates across four distinct layers of concurrent execution. Understanding these layers is foundational to reasoning formally about safety, liveness, and correctness.

#### Layer 1: MQL5 Expert Advisor Event-Handler Concurrency

MetaTrader 5 implements a **cooperative, single-threaded event scheduler** within each EA instance. This is a critical architectural invariant: *there is no preemptive thread switch between `OnTick`, `OnTradeTransaction`, and `OnDeinit` handlers of the same EA instance*. However:

1. **N independent chart instances** of `LiveONNX-EA.mq5` or `DMatrix-EA.mq5` running on different symbols (EURUSD, GBPUSD, USDJPY, etc.) or timeframes execute as **separate, isolated OS-level threads** (one thread per EA instance in MT5's multi-threaded architecture). Each EA's event queue is drained cooperatively within its own thread.

2. **Within a single EA instance**, event handlers are serialized by MT5's runtime: `OnTick` cannot be interrupted by `OnTradeTransaction` mid-execution. Handlers are dispatched one-at-a-time from the event queue. This cooperative scheduling guarantees **intra-instance mutual exclusion** at the handler level.

3. **Cross-instance interactions** occur exclusively through the shared SQLite database (`macro_governance.db`) and, in the case of `DMatrix-EA`, through shared MT5 order and position state accessible via `CTrade`, `HistoryDealSelect()`, and `PositionSelectByTicket()`.

The key MQL5 event handlers and their concurrency roles are:

| Handler | EA | Scheduling Invariant | Shared State Accessed |
|---|---|---|---|
| `OnInit()` | Both | Once, sequentially at EA attach | `g_hMacroDB`, ONNX handles, indicator handles |
| `OnTick()` | Both | Cooperative, per-tick, serialized within instance | `g_lastBarTime`, ONNX handles, `g_hMacroDB` |
| `OnTradeTransaction()` | DMatrix-EA | Cooperative, queued after OnTick completes | `g_orderTracker.m_activePositions[]`, `m_recordedSamples[]` |
| `OnDeinit()` | Both | Last handler, after all queued events | `g_orderTracker` full state, file handles |

#### Layer 2: Python MLOps Subprocess Execution

The Python pipeline (`run_pipeline.py`, `src/trainer.py`, `src/mt5_client.py`) executes as a **separate OS process**, entirely decoupled from the MT5 terminal at the process level. Concurrency here involves:

1. **`MT5Client.run_strategy_tester()`** spawns the MT5 terminal as a child subprocess (`subprocess.Popen`) and polls its log files in a tight 500ms interval loop. This is a **producer-consumer pattern** where MT5 produces log lines and the Python poller consumes them.

2. **`MT5Client._stream_new_tester_logs()`** maintains a `file_offsets: dict[Path, int]` dictionary — a **mutable shared state within the single Python polling thread** — updated on each poll. Since Python's polling loop is single-threaded, no race is possible here.

3. **`macro_agent/db_client.py`** opens SQLite connections to `macro_governance.db` from the Python side concurrently with MT5 chart instances reading the same database. This is the **primary cross-layer concurrency point**.

#### Layer 3: SQLite WAL-Mode Multi-Reader/Single-Writer Database

The `macro_governance.db` SQLite database is the central shared mutable state of the distributed system. It is accessed by:

- **N instances** of `LiveONNX-EA.mq5` (one per chart/symbol), each calling `DatabaseRead()` and `DatabasePrepare()/DatabaseRead()/DatabaseFinalize()` per new bar.
- **The Python `macro_agent`** (`db_client.py`), which calls `upsert_calendar_event()`, `delete_calendar_event()`, `purge_expired_calendar_events()` etc. via `sqlite3.connect()`.

SQLite's **Write-Ahead Logging (WAL) mode** (enabled via `PRAGMA journal_mode=WAL`) is the critical concurrency mechanism:

- **WAL Concurrency Rule**: Concurrent readers do not block writers; concurrent writers do not block concurrent readers. Writers serialized by WAL's single-writer lock.
- **Busy Timeout**: `PRAGMA busy_timeout = 5000` (set by `LiveONNX-EA.mq5` on connection open) instructs SQLite to spin-wait up to 5000ms before returning `SQLITE_BUSY` on a writer lock contention.

#### Layer 4: MCP Stdio Server Sidecar Processes

MCP servers (`memory`, `duckdb`, `duckduckgo-search`) communicate with the agent via **standard input/output streams** (Stdio transport). Each server runs as a separate process; the agent communicates via synchronous JSON-RPC messages serialized over stdin/stdout. These are sequential from the agent's perspective and introduce no concurrency hazards with respect to the trading pipeline state.

### 1.2 Complete Shared Mutable State Map

The following table enumerates every piece of shared mutable state in the system with its synchronization mechanism:

| Shared State | Location | Concurrent Accessors | Synchronization Mechanism |
|---|---|---|---|
| `macro_governance.db` (calendar_events, news_events tables) | SQLite file in MT5 Common/Files | N × LiveONNX-EA (READ) + macro_agent Python (WRITE) | SQLite WAL journal, single-writer lock, `PRAGMA busy_timeout=5000` |
| `g_lastBarTime` (datetime) | LiveONNX-EA.mq5 global | OnTick() (R/W) — same EA instance only | MT5 cooperative scheduler (intra-instance serialization) |
| `g_hMacroDB` (SQLite handle) | LiveONNX-EA.mq5 global | OnTick() (R), OnInit() (W), OnDeinit() (W) | MT5 cooperative scheduler |
| `g_hModelBuy / g_hModelSell` (ONNX handles) | LiveONNX-EA.mq5 globals | OnTick() (R), OnInit() (W), OnDeinit() (W) | MT5 cooperative scheduler |
| `g_orderTracker.m_activePositions[]` | DMatrix-EA / OrderTracker.mqh | OnTick()->RegisterPosition() (W), OnTradeTransaction()->ProcessTransaction() (R/W), OnDeinit()->ProcessUnresolvedPositions() (R) | MT5 cooperative scheduler (per-instance) |
| `g_orderTracker.m_recordedSamples[]` | DMatrix-EA / OrderTracker.mqh | AddSample() from ProcessTransaction() (W), ExportDatasets() (R) — called only from OnDeinit() | MT5 cooperative scheduler (per-instance) |
| `g_orderTracker.m_sortIndices[]` | DMatrix-EA / OrderTracker.mqh | SortChronologically() / QuickSortIndices() — only from OnDeinit() | Single-caller (OnDeinit is terminal) |
| `_in_safe_transaction` (module-level bool) | macro_agent/db_client.py | All write functions (safe_db_transaction context manager) | Python module-level global flag — NOT thread-safe |
| `file_offsets: dict[Path, int]` | src/mt5_client.py | `_stream_new_tester_logs()` — single polling thread | Single-threaded Python loop |
| MT5 Order/Position State (broker server) | MT5 Account | DMatrix-EA's OnTick() and OnTradeTransaction() | MT5 server-side serialization + deal acknowledgment protocol |

### 1.3 Synchronization Point Summary

The system has exactly **three critical synchronization points**:

1. **SQLite WAL Lock**: The single serialization point for all concurrent writes to `macro_governance.db`. N readers can proceed simultaneously; at most one writer holds the lock at any time.

2. **MT5 Cooperative Event Queue (per EA instance)**: Within each EA, the cooperative scheduler acts as a sequential lock — `OnTick`, `OnTradeTransaction`, and `OnDeinit` are never co-executing within a single instance.

3. **MQL5 `HistoryDealSelect()` / `PositionSelectByTicket()` MT5 Server State**: These MT5 API calls access the broker's server-synchronized order book. They are safe at the MQL5 level (the MT5 runtime handles server-side locking), but introduce asynchronous latency: a position reported as open by `PositionSelectByTicket()` may have already been server-closed by the time `trade.PositionClose()` is called.

---

## Section 2: Formal Petri Net Models

### Foundational Definitions

Following **Carl Adam Petri's original dissertation (Kommunikation mit Automaten, 1962)** and the comprehensive IEEE survey by **Tadao Murata (1989)**, we define:

**Definition 2.0 (Petri Net):** A Petri Net is a 5-tuple `N = (P, T, A, W, M0)` where:
- `P = {p1, p2, ..., pn}` is a finite set of **places** (states/conditions), represented as circles.
- `T = {t1, t2, ..., tm}` is a finite set of **transitions** (events/actions), represented as bars.
- `P ∩ T = ∅` (places and transitions are disjoint).
- `A ⊆ (P × T) ∪ (T × P)` is the set of **arcs** (directed flow relation).
- `W: A → N+` is the **arc weight function** (default 1 for all arcs unless stated).
- `M0: P → N` is the **initial marking** (token distribution at time 0).

**Definition 2.1 (Enabling Condition):** A transition `t ∈ T` is **enabled** at marking `M` iff for all `p ∈ •t` (preset of t): `M(p) ≥ W(p, t)`.

**Definition 2.2 (Firing Rule):** When enabled transition `t` fires at marking `M`, producing `M'`:
- `∀p ∈ •t: M'(p) = M(p) − W(p, t)`
- `∀p ∈ t•: M'(p) = M(p) + W(t, p)`
- All other places unchanged.

**Definition 2.3 (Reachability):** The reachability set `R(N, M0)` is the set of all markings reachable from `M0` by any firing sequence.

---

### 2.1 Model 1: Multi-Chart SQLite Concurrent Access (WAL Mode)

#### 2.1.1 Scenario Description

Consider `N = 2` independent MT5 chart instances — one for EURUSD and one for GBPUSD — each running `LiveONNX-EA.mq5`. Both call `CheckMacroCalendar()` and `CheckMacroNews()` (read operations via `DatabasePrepare` / `DatabaseRead` / `DatabaseFinalize`) on every new bar. Concurrently, the Python `macro_agent/db_client.py` may be writing new calendar events (`upsert_calendar_event()`) or purging expired ones (`purge_expired_calendar_events()`).

The SQLite WAL mode ensures:
1. **Multiple concurrent readers** (all chart instances) can proceed simultaneously without blocking.
2. **At most one writer** (the Python agent) holds the WAL write lock at a time.
3. `PRAGMA busy_timeout = 5000` ensures that a contending writer (or a reader encountering a checkpoint lock) waits up to 5000ms before returning `SQLITE_BUSY`.

#### 2.1.2 Formal Petri Net Definition

**Places (P):**
```
P = {
  p_idle_A,          // Chart A (EURUSD) idle, not accessing DB
  p_idle_B,          // Chart B (GBPUSD) idle, not accessing DB
  p_idle_W,          // Python writer (macro_agent) idle
  p_reading_A,       // Chart A currently reading (shared read lock, WAL snapshot)
  p_reading_B,       // Chart B currently reading (shared read lock, WAL snapshot)
  p_write_pending_W, // Python writer is requesting write lock (waiting if busy)
  p_writing_W,       // Python writer holds exclusive WAL write lock
  p_committed_W,     // Python writer committed, WAL frame flushed
  p_busy_timeout_A,  // Chart A encountered busy timeout (SQLITE_BUSY returned)
  p_busy_timeout_B,  // Chart B encountered busy timeout (SQLITE_BUSY returned)
  p_wal_frames,      // WAL frame buffer (tokens = number of uncommitted WAL frames)
  p_checkpoint       // WAL checkpoint in progress
}
```

**Transitions (T):**
```
T = {
  t_A_start_read,    // Chart A begins DatabasePrepare -> enters reading state
  t_B_start_read,    // Chart B begins DatabasePrepare -> enters reading state
  t_A_end_read,      // Chart A calls DatabaseFinalize -> releases read snapshot
  t_B_end_read,      // Chart B calls DatabaseFinalize -> releases read snapshot
  t_W_request_write, // Python agent calls sqlite3.connect() + begins write transaction
  t_W_acquire_lock,  // Python agent acquires exclusive WAL write lock (no contention)
  t_W_commit,        // Python agent commits: flushes WAL frame
  t_W_release,       // Python agent releases write lock (returns to idle)
  t_A_busy_timeout,  // Chart A's busy_timeout fires (WAL checkpoint lock contention)
  t_B_busy_timeout,  // Chart B's busy_timeout fires (WAL checkpoint lock contention)
  t_A_recover,       // Chart A recovers from busy_timeout, returns to idle
  t_B_recover,       // Chart B recovers from busy_timeout, returns to idle
  t_wal_checkpoint   // Automatic WAL checkpoint (background thread or triggered)
}
```

**Arcs (A) with Arc Weights W:**
```
// Chart A read flow
p_idle_A         -> t_A_start_read   (w=1)
t_A_start_read   -> p_reading_A      (w=1)
p_reading_A      -> t_A_end_read     (w=1)
t_A_end_read     -> p_idle_A         (w=1)

// Chart B read flow (symmetric)
p_idle_B         -> t_B_start_read   (w=1)
t_B_start_read   -> p_reading_B      (w=1)
p_reading_B      -> t_B_end_read     (w=1)
t_B_end_read     -> p_idle_B         (w=1)

// Python writer flow
p_idle_W         -> t_W_request_write  (w=1)
t_W_request_write -> p_write_pending_W (w=1)
p_write_pending_W -> t_W_acquire_lock  (w=1)
// WAL write lock: guarded by inhibitor arc from p_writing_W (exclusive)
t_W_acquire_lock -> p_writing_W       (w=1)
p_writing_W      -> t_W_commit        (w=1)
t_W_commit       -> p_committed_W     (w=1)
t_W_commit       -> p_wal_frames      (w=1)  // deposits a WAL frame
p_committed_W    -> t_W_release       (w=1)
t_W_release      -> p_idle_W          (w=1)

// WAL checkpoint flow
p_wal_frames     -> t_wal_checkpoint  (w=1)
// Note: checkpoint requires NO active readers (readers must be in p_idle, not p_reading)
// Modeled via inhibitor arcs: p_reading_A and p_reading_B must be empty
t_wal_checkpoint -> p_checkpoint      (w=1)
// checkpoint completes; DB remains consistent

// Busy timeout flows (occur only during WAL checkpoint lock contention)
t_A_busy_timeout  -> p_busy_timeout_A (w=1)
p_busy_timeout_A  -> t_A_recover      (w=1)
t_A_recover       -> p_idle_A         (w=1)

t_B_busy_timeout  -> p_busy_timeout_B (w=1)
p_busy_timeout_B  -> t_B_recover      (w=1)
t_B_recover       -> p_idle_B         (w=1)
```

**Initial Marking M0:**
```
M0 = {
  p_idle_A:          1,   // Chart A starts idle
  p_idle_B:          1,   // Chart B starts idle
  p_idle_W:          1,   // Python agent starts idle
  p_reading_A:       0,
  p_reading_B:       0,
  p_write_pending_W: 0,
  p_writing_W:       0,
  p_committed_W:     0,
  p_busy_timeout_A:  0,
  p_busy_timeout_B:  0,
  p_wal_frames:      0,
  p_checkpoint:      0
}
```

**Structural Invariant (WAL exclusivity):** The WAL write lock is exclusive. This is modeled by the arc structure: `t_W_acquire_lock` can only fire when `p_write_pending_W` has a token AND `p_writing_W` has zero tokens (i.e., the inhibitor arc prevents simultaneous writers). This enforces SQLite's guarantee that at most one writer can hold the WAL write lock.

#### 2.1.3 ASCII Art Topology Diagram

```
                          WAL MODE SQLITE CONCURRENT ACCESS MODEL
                          ========================================

  CHART A (EURUSD)                PYTHON WRITER (macro_agent)         CHART B (GBPUSD)
  -----------------               --------------------------           -----------------

  +-----------+                   +-----------+                        +-----------+
  | p_idle_A  |<----------------+ | p_idle_W  |<------------------+   | p_idle_B  |
  +-----+-----+                 | +-----+-----+                   |   +-----+-----+
        |                       |       |                          |         |
  t_A_start_read                |  t_W_request_write              |   t_B_start_read
        |                       |       |                          |         |
        v                       |       v                          |         v
  +--------------+              | +------------------+            |  +--------------+
  | p_reading_A  |              | | p_write_pending_W|            |  | p_reading_B  |
  +------+-------+              | +--------+---------+            |  +------+-------+
         |                      |          |                       |         |
   t_A_end_read                 |   t_W_acquire_lock (excl.)      |  t_B_end_read
   (DatabaseFinalize)           |   [inhibitor: p_writing_W=0]    |  (DatabaseFinalize)
         |                      |          |                       |         |
         +----------------------+          v                       +---------+
                                    +--------------+
                                    | p_writing_W  | --[exclusive write lock]--
                                    +------+-------+
                                           |
                                     t_W_commit ---------> p_wal_frames --> t_wal_checkpoint
                                           |                                (needs: reading_A=0,
                                           v                                 reading_B=0)
                                    +--------------+
                                    | p_committed_W|
                                    +------+-------+
                                           |
                                     t_W_release
                                           |
                                           +-----------> p_idle_W

  BUSY TIMEOUT PATHS:
  p_write_pending_W --[5000ms exceeded]--> p_busy_timeout_A --> t_A_recover --> p_idle_A
  p_write_pending_W --[5000ms exceeded]--> p_busy_timeout_B --> t_B_recover --> p_idle_B
```

#### 2.1.4 Reachability Analysis

**Claim:** All markings in `R(N, M0)` satisfy the following safety property: at most one token resides in `p_writing_W` at any time.

**Proof sketch:** From `M0`, `t_W_request_write` can fire, moving the token from `p_idle_W` to `p_write_pending_W`. Then `t_W_acquire_lock` fires (conditioned on `p_writing_W = 0`), moving the token to `p_writing_W`. No other transition produces a token in `p_writing_W`. The token in `p_writing_W` is consumed only by `t_W_commit`. Since only one token enters `p_idle_W` initially (and no transition creates new tokens in `p_idle_W` from outside), there can be at most one writer token in the write pipeline at any time. This is enforced by the conservation of the writer-subnet P-invariant: `M(p_idle_W) + M(p_write_pending_W) + M(p_writing_W) + M(p_committed_W) = 1` for all reachable markings.

**Concurrent Readers Lemma:** `t_A_start_read` and `t_B_start_read` can fire simultaneously (in any interleaving) without mutual exclusion, as neither transition requires tokens from the other's places. Both `p_reading_A` and `p_reading_B` can hold tokens simultaneously. This models WAL's concurrent read capability.

**WAL Checkpoint Blocking:** `t_wal_checkpoint` models the SQLite WAL auto-checkpoint, which requires no active readers (both `p_reading_A = 0` and `p_reading_B = 0`) to proceed cleanly. During a checkpoint, readers that arrive will be queued at `t_A_start_read` / `t_B_start_read` if the checkpoint holds an exclusive lock. The `busy_timeout = 5000ms` ensures the system is not livelock-prone: even under checkpoint lock contention, readers time out and return to idle, guaranteeing eventual progress.

**Reachable Marking Classes:**

| Scenario | p_idle_A | p_reading_A | p_idle_B | p_reading_B | p_writing_W | p_wal_frames |
|---|---|---|---|---|---|---|
| All idle | 1 | 0 | 1 | 0 | 0 | 0 |
| A reading, B idle | 0 | 1 | 1 | 0 | 0 | 0 |
| Both reading | 0 | 1 | 0 | 1 | 0 | 0 |
| A reading, W writing | 0 | 1 | 1 | 0 | 1 | 0 |
| Both reading, W writing | 0 | 1 | 0 | 1 | 1 | 0 |
| W committed, frames pending checkpoint | 1 | 0 | 1 | 0 | 0 | k>=1 |

The reachability set is finite (bounded by the constant sum of tokens per subnet). No deadlock state exists (see Section 5).

---

### 2.2 Model 2: Intra-Tick Execution Flow in LiveONNX-EA

#### 2.2.1 Scenario Description

`LiveONNX-EA.mq5`'s `OnTick()` handler receives a tick event from the MT5 terminal. The handler must:
1. **Check `IsNewBar()`** — filter: only process on new bar open, preventing high-frequency re-evaluation.
2. **Check daily schedule** via `IsTradeScheduleAllowed()`.
3. **Query macroeconomic filters** (`CheckMacroNews()`, `CheckMacroCalendar()`).
4. **Extract feature vector** via `g_featureExtractor.ExtractFlattenedVector()`.
5. **Run ONNX inference** via `OnnxRun()`.
6. **Compute GARCH dynamic risk** via `g_garch.CalculateDynamicRisk()`.
7. **Dispatch orders** via `g_trade.Buy()` / `g_trade.Sell()`.

The critical observation from the source code is that **LiveONNX-EA does not explicitly use a `g_isProcessing` boolean flag**. Instead, it relies entirely on:
- The **IsNewBar() guard** as its primary temporal filter (bar-level debounce).
- The **MT5 cooperative scheduler** as the implicit reentrancy guard.

Because MT5's cooperative scheduler serializes `OnTick` calls within a single EA instance (each `OnTick` call completes before the next is dispatched), there is **no explicit reentrancy risk** within a single chart. The model below captures this guarantee formally, showing the bar-level temporal guard as the net's structural invariant.

#### 2.2.2 Formal Petri Net Definition

**Places (P):**
```
P = {
  p_tick_received,       // Tick event received from MT5 scheduler
  p_bar_check,           // IsNewBar() check in progress
  p_same_bar,            // Same bar (not new): OnTick exits immediately
  p_new_bar_detected,    // New bar confirmed: g_lastBarTime updated
  p_schedule_check,      // IsTradeScheduleAllowed() evaluation
  p_schedule_blocked,    // Outside trading schedule: exits
  p_macro_check,         // Macro calendar/news filter queries
  p_macro_blocked,       // Macro filter blocked trade: exits
  p_features_extracted,  // Feature vector extraction complete (ExtractFlattenedVector)
  p_feature_warmup,      // Insufficient history for GARCH/feature warmup: exits
  p_inference_done,      // ONNX inference complete (OnnxRun)
  p_garch_done,          // GARCH dynamic risk calculation complete
  p_order_dispatched,    // Order sent to broker via CTrade
  p_order_rejected,      // Order rejected by risk filter or broker
  p_handler_complete     // OnTick handler fully complete, returns control to MT5 scheduler
}
```

**Transitions (T):**
```
T = {
  t_tick_arrives,        // MT5 scheduler delivers OnTick event
  t_check_bar,           // Evaluate IsNewBar() -> compare g_lastBarTime
  t_same_bar_exit,       // Guard: currentBarTime == g_lastBarTime -> exit
  t_new_bar_confirmed,   // Guard: currentBarTime != g_lastBarTime -> update + proceed
  t_check_schedule,      // Evaluate IsTradeScheduleAllowed(barTime)
  t_schedule_blocked,    // Condition: outside schedule -> exit
  t_schedule_allowed,    // Condition: within schedule -> proceed
  t_check_macro,         // Evaluate CheckMacroNews + CheckMacroCalendar
  t_macro_blocked,       // Condition: blocking macro event -> exit
  t_macro_pass,          // Condition: no blocking macro -> proceed
  t_extract_features,    // Execute ExtractFlattenedVector (vectorf computation)
  t_feature_warmup_fail, // GARCH/indicator warmup failure -> exit (insufficient history)
  t_run_onnx,            // Execute OnnxRun (BUY + SELL models)
  t_run_garch_risk,      // Execute CalculateDynamicRisk (GARCH sigma_agg -> TP/SL pts)
  t_dispatch_order,      // Evaluate buy/sell conditions, send via CTrade
  t_order_risk_reject,   // Risk filter / viability gate rejects order
  t_handler_exit         // Return from OnTick handler
}
```

**Arcs (A):**
```
p_tick_received      -> t_tick_arrives        -> p_bar_check
p_bar_check          -> t_check_bar
t_check_bar          -> p_same_bar             (if currentBarTime == g_lastBarTime)
t_check_bar          -> p_new_bar_detected     (if currentBarTime != g_lastBarTime)
p_same_bar           -> t_same_bar_exit        -> p_handler_complete
p_new_bar_detected   -> t_check_schedule
t_check_schedule     -> p_schedule_check
p_schedule_check     -> t_schedule_blocked     -> p_handler_complete
p_schedule_check     -> t_schedule_allowed
t_schedule_allowed   -> p_macro_check
p_macro_check        -> t_macro_blocked        -> p_handler_complete
p_macro_check        -> t_extract_features
t_extract_features   -> p_features_extracted
p_features_extracted -> t_feature_warmup_fail  -> p_handler_complete
p_features_extracted -> t_run_onnx             -> p_inference_done
p_inference_done     -> t_run_garch_risk       -> p_garch_done
p_garch_done         -> t_dispatch_order       -> p_order_dispatched
p_garch_done         -> t_order_risk_reject    -> p_order_rejected
p_order_dispatched   -> t_handler_exit         -> p_handler_complete
p_order_rejected     -> t_handler_exit         -> p_handler_complete
p_handler_complete   -> t_tick_arrives         // cycle: MT5 delivers next tick
```

**Initial Marking M0:**
```
M0 = {
  p_tick_received:    1,   // System starts with first tick pending
  all other places:   0
}
```

#### 2.2.3 ASCII Art Topology Diagram

```
              LIVEONNX-EA INTRA-TICK EXECUTION FLOW (PETRI NET MODEL)
              ========================================================

  +-----------------+
  | p_tick_received |<--------------------------------------------------+
  +--------+--------+                                                    |
           |  t_tick_arrives                                             |
           v                                                             |
  +--------------+                                                       |
  | p_bar_check  |                                                       |
  +------+-------+                                                       |
         | t_check_bar                                                   |
         |                                                               |
    +----+---------------------+                                         |
    |  (IsNewBar() evaluation) |                                         |
    v                          v                                         |
+----------+            +------------------+                             |
|p_same_bar|            |p_new_bar_detected|                             |
+----+-----+            +--------+---------+                             |
     | t_same_bar_exit           | t_check_schedule                     |
     |                           v                                       |
     |                   +------------------+                            |
     |                   | p_schedule_check |                            |
     |                   +--------+---------+                            |
     |                            |                                      |
     |               +------------+------------------+                   |
     |               v                               v                   |
     |         +------------+             +--------------------+         |
     |         |p_sched_blk |             |   p_macro_check    |         |
     |         +-----+------+             +--------+-----------+         |
     |               |                             |                     |
     |               |                +------------+------------+         |
     |               |                v                         v         |
     |               |         +-------------+    +---------------------+ |
     |               |         |p_macro_block|    | p_features_extracted| |
     |               |         +------+------+    +-----------+---------+ |
     |               |                |                       |           |
     |               |                |          +------------+---------+ |
     |               |                |          v                      v  |
     |               |                |  +----------------+  +-------------------+|
     |               |                |  |p_feature_warmup|  |  p_inference_done ||
     |               |                |  +------+---------+  +----------+--------+|
     |               |                |         |                        |         |
     |               |                |         |              t_run_garch_risk    |
     |               |                |         |                        v         |
     |               |                |         |               +-----------+      |
     |               |                |         |               | p_garch   |      |
     |               |                |         |               |   _done   |      |
     |               |                |         |               +-----+-----+      |
     |               |                |         |                     |            |
     |               |                |         |              +------+------+     |
     |               |                |         |              v             v     |
     |               |                |         |   +------------------+ +---------------+|
     |               |                |         |   | p_order_dispatched| |p_order_rejected||
     |               |                |         |   +----------+-------+ +--------+------+|
     |               |                |         |              |                   |        |
     +---------------+----------------+---------+              +----------+--------+        |
                        ALL --> t_handler_exit                 t_handler_exit               |
                                                       +-------+--------+                   |
                                                       v                                    |
                                            +---------------------+                         |
                                            |  p_handler_complete |-------------------------+
                                            +---------------------+
                                         (MT5 scheduler delivers next tick)
```

#### 2.2.4 Reentrancy Safety Proof

**Theorem 2.2 (Reentrancy-Freedom):** Under the MT5 cooperative scheduling model, the `OnTick` handler for a single `LiveONNX-EA` instance is never re-entered while still executing.

**Proof:** MT5's cooperative scheduler maintains a per-EA event queue. A tick event is placed in the queue only when the previous `OnTick` handler has returned (i.e., placed a token back in `p_handler_complete`). In Petri Net terms, `t_tick_arrives` is only enabled when `p_handler_complete` holds a token (for N=1 instance). Since `p_handler_complete` holds exactly one token only after the previous `OnTick` returns, and `t_tick_arrives` consumes it (moving to `p_bar_check`), no second `OnTick` invocation can begin before the first completes. This is a structural property of the net: the token in `{p_tick_received, p_bar_check, p_same_bar, p_new_bar_detected, ..., p_handler_complete}` subnet sums to exactly 1 at all times, forming a **P-invariant** (see Section 3). QED.

**IsNewBar Guard Formal Property:** Even if MT5 could somehow deliver two rapid ticks at the same bar open, the IsNewBar() guard ensures the expensive computation path (feature extraction, ONNX inference, GARCH, order dispatch) is traversed at most once per bar. Formally, for any two consecutive tick events within the same bar `b`, only the first fires `t_new_bar_confirmed` (updating `g_lastBarTime`); all subsequent ticks within `b` fire `t_same_bar_exit`, routing through `p_same_bar -> p_handler_complete` without side effects.

---

### 2.3 Model 3: Async DMatrix-EA Trade Event Concurrency

#### 2.3.1 Scenario Description

`DMatrix-EA.mq5` operates during MT5 Strategy Tester backtest. Its concurrency architecture involves three event types that interact through shared in-memory state (`COrderTracker`):

1. **`OnTick()` -> `IsNewBar()`**: On new bar open, extracts features, opens simultaneous BUY + SELL positions, calls `g_orderTracker.RegisterPosition()`.
2. **`OnTradeTransaction()`**: Fired asynchronously when a deal is processed (TP hit, SL hit, timeout close). Calls `g_orderTracker.ProcessTransaction()`.
3. **`OnDeinit()`**: Called once at EA termination. Calls `g_orderTracker.ExportDatasets()` which internally calls `ProcessUnresolvedPositions()`, `SortChronologically()`, and CSV export.

The critical shared state is `COrderTracker`'s RAM arrays:
- `m_activePositions[]` (dynamic array, indexed by `m_activeCount`) — written by `RegisterPosition()`, read+written by `ProcessTransaction()` (sets `isActive = false`), read by `ProcessUnresolvedPositions()`.
- `m_recordedSamples[]` (dynamic array, indexed by `m_sampleCount`) — written by `AddSample()` (called from both `ProcessTransaction()` and `ProcessUnresolvedPositions()`), read by `ExportDatasets()`.

**Key architectural invariant (from MT5 cooperative scheduler):** Within a single DMatrix-EA instance, `OnTick`, `OnTradeTransaction`, and `OnDeinit` are **strictly serialized** — they cannot interleave. The question is whether the semantic interleaving of their state transitions can produce any observable corruption. The Petri Net model below proves it cannot.

#### 2.3.2 Formal Petri Net Definition

**Places (P):**
```
P = {
  // OnTick side-states
  p_OT_idle,             // DMatrix-EA waiting for new tick
  p_OT_new_bar,          // New bar detected, extracting features + opening positions
  p_OT_positions_opened, // BUY and SELL positions successfully opened this bar
  p_OT_registered,       // Positions registered in m_activePositions[]

  // OnTradeTransaction side-states
  p_OTT_idle,            // No pending trade transaction
  p_OTT_processing,      // Processing a DEAL_ADD transaction
  p_OTT_position_found,  // Active position found in m_activePositions[]
  p_OTT_labeled,         // Label assigned (1.0f TP or 0.0f SL)
  p_OTT_sample_added,    // AddSample() called: m_recordedSamples[] updated
  p_OTT_inactive_marked, // m_activePositions[i].isActive = false

  // OnDeinit side-states
  p_ODI_triggered,       // OnDeinit called (terminal condition)
  p_ODI_unresolved,      // ProcessUnresolvedPositions() iterating
  p_ODI_all_labeled,     // All remaining positions labeled 0.0f (NOT_OPEN)
  p_ODI_sorting,         // SortChronologically() / QuickSortIndices() in progress
  p_ODI_exporting,       // CSV file writing in progress
  p_ODI_complete,        // ExportDatasets() complete, EA fully done

  // Shared state resources (modeled as resource tokens)
  p_active_positions_arr,   // m_activePositions[] array (resource: writable)
  p_samples_arr,            // m_recordedSamples[] array (resource: writable)
  p_sort_indices_arr        // m_sortIndices[] array (resource: sort-only)
}
```

**Transitions (T):**
```
T = {
  // OnTick transitions
  t_OT_tick_new_bar,         // New bar detected: begins feature extraction
  t_OT_open_positions,       // Opens BUY + SELL positions via CTrade
  t_OT_register_buy,         // RegisterPosition(buyTicket, POSITION_TYPE_BUY, ...)
  t_OT_register_sell,        // RegisterPosition(sellTicket, POSITION_TYPE_SELL, ...)
  t_OT_return,               // OnTick handler completes, returns

  // OnTradeTransaction transitions
  t_OTT_deal_add,            // DEAL_ADD event received for a managed ticket
  t_OTT_find_position,       // FindActivePosition(positionId) -> locates index
  t_OTT_assign_label,        // Determine label from netLiquidProfit + dealReason
  t_OTT_add_sample,          // AddSample() -> writes to m_recordedSamples[]
  t_OTT_deactivate,          // m_activePositions[i].isActive = false
  t_OTT_return,              // OnTradeTransaction completes

  // OnDeinit transitions
  t_ODI_begin,               // OnDeinit starts: ExportDatasets() called
  t_ODI_process_unresolved,  // ProcessUnresolvedPositions(): label remaining active positions
  t_ODI_sort,                // SortChronologically(): builds m_sortIndices[]
  t_ODI_export,              // Writes CSV files (BUY + SELL datasets)
  t_ODI_end                  // OnDeinit complete
}
```

**Initial Marking M0:**
```
M0 = {
  p_OT_idle:               1,   // DMatrix-EA starts idle
  p_OTT_idle:              1,   // No pending transactions
  p_active_positions_arr:  1,   // Array is available (writable)
  p_samples_arr:           1,   // Array is available (writable)
  p_sort_indices_arr:      1,   // Sort indices available
  all other places:        0
}
```

#### 2.3.3 ASCII Art Topology Diagram

```
  DMATRIX-EA CONCURRENT TRADE EVENT MODEL (COOPERATIVE SCHEDULER SERIALIZATION)
  ==============================================================================

  [MT5 serializes all handlers in ONE thread per EA instance. Shown logically
   separated to illustrate the dependency flow on shared state.]

  OnTick() path            OnTradeTransaction() path     OnDeinit() path
  ─────────────            ─────────────────────────     ────────────────

  +----------+             +------------+
  | p_OT_idle|             | p_OTT_idle |             +-----------------+
  +----+-----+             +----+-------+             | p_ODI_triggered |
       |                        |                     +--------+--------+
  t_OT_tick_new_bar         t_OTT_deal_add              t_ODI_begin
       |                        |                             |
       v                        v                             v
  +-----------+            +-----------------+          +-----------------+
  |p_OT_new_  |            | p_OTT_processing|         | p_ODI_unresolved|
  |   bar     |            +--------+--------+          +--------+--------+
  +-----------+                     |                            |
       |                    +-------+------+             t_ODI_process_unresolved
  t_OT_open_positions       |p_active_     |             (labels remaining isActive=true)
       |                    |positions_arr |<--------------------------+
       |                    +---------+----+                           |
       |                    t_OTT_find_position                        |
       |                              |                                |
       |                    p_OTT_position_found                p_ODI_all_labeled
       |                              |                                |
  t_OT_register_buy  --> m_activePositions[m_activeCount++]      t_ODI_sort
  t_OT_register_sell --> m_activePositions[m_activeCount++]           |
       |                              |                                v
       |                    t_OTT_assign_label                  +-----------+
       |                              |                          |p_ODI_sort |
       |                    +----+----+---+                      +-----+-----+
       |                    |p_OTT_labeled|                      t_ODI_export
       |                    +------+------+                            |
       |                           |                                   v
       |                    t_OTT_add_sample                   +--------------+
       |                           |                            |p_ODI_exporting|
       |                    +------+------+                     +--------------+
       |                    | p_samples_  |<------- (releases) t_ODI_end
       |                    |     arr     |
       |                    +------+------+
       |                    t_OTT_deactivate
       |                    (m_activePositions[i].isActive = false)
       |                    p_OTT_idle (returns)
       |
  p_OT_idle <------------ t_OT_return

  RESOURCE TOKENS (array access serialization):
  p_active_positions_arr: consumed during write/read, released immediately after
  p_samples_arr:          consumed during AddSample writes, released immediately after
  p_sort_indices_arr:     consumed only during OnDeinit sort phase (terminal)
```

#### 2.3.4 Proof: No Race Condition Between RegisterPosition() and ProcessUnresolvedPositions()

**Theorem 2.3 (Labels Array Integrity):** There is no execution schedule in the MT5 cooperative scheduler model under which `RegisterPosition()` (called from `OnTick`) and `ProcessUnresolvedPositions()` (called from `OnDeinit`) can concurrently corrupt the `m_activePositions[]` or `m_recordedSamples[]` arrays.

**Proof by structural Petri Net argument:**

1. **Terminal ordering invariant**: `OnDeinit` is triggered only after all pending `OnTick` and `OnTradeTransaction` events in the EA's queue have been drained. MT5 guarantees this: the `OnDeinit` event is placed at the end of the event queue only after a shutdown or EA removal signal, and the queue is drained FIFO. Formally, `p_ODI_triggered` receives its token only when both `p_OT_idle = 1` and `p_OTT_idle = 1`.

2. **Mutual exclusion via P-invariant**: In the Petri Net model, `t_ODI_process_unresolved` requires `p_active_positions_arr` as a resource token. This same token is required by both `t_OT_register_buy` and `t_OTT_find_position`. Since the MT5 scheduler ensures `OnTick` and `OnDeinit` cannot fire concurrently, the resource token acts as a formal mutual exclusion object. Its P-invariant is: `M(p_active_positions_arr) + M(p_OT_new_bar) + M(p_OTT_processing) + M(p_ODI_unresolved) <= 1` at all reachable markings.

3. **No dirty reads on isActive flag**: `ProcessTransaction()` sets `m_activePositions[posIdx].isActive = false` atomically before returning. `ProcessUnresolvedPositions()` checks `isActive` in a sequential loop — since `OnDeinit` executes only after all queued `OnTradeTransaction` events complete, any position closed by `ProcessTransaction()` will have `isActive = false` by the time `ProcessUnresolvedPositions()` iterates. The cooperative scheduler guarantee formalizes this sequencing.

4. **Array growth safety**: `ArrayResize()` on `m_activePositions[]` and `m_recordedSamples[]` is only called within `RegisterPosition()` and `AddSample()`. Neither `ProcessTransaction()` nor `ProcessUnresolvedPositions()` calls `ArrayResize()` — they only access existing elements by index. `ExportDatasets()` reads `m_recordedSamples[]` only after `ProcessUnresolvedPositions()` completes (sequential call within `ExportDatasets()`), so there is no concurrent reader-writer conflict.

**Conclusion:** The labels array is never corrupted because MT5's cooperative scheduler provides intra-instance serialization, and the semantic ordering of `OnDeinit` after all `OnTick`/`OnTradeTransaction` events provides the cross-handler ordering guarantee. QED.

---

## Section 3: Mathematical Invariant Analysis

### 3.1 P-Invariants (Place Invariants — Conserved Token Quantities)

A **P-invariant** is a vector `y: P -> Z` satisfying `y^T · C = 0` (where `C` is the incidence matrix), such that `y^T · M = y^T · M0` for all reachable markings `M`. A non-negative P-invariant defines a conserved token quantity.

#### 3.1.1 P-Invariants for Model 1 (SQLite WAL)

**P-Inv 1 — Writer Conservation:**
```
M(p_idle_W) + M(p_write_pending_W) + M(p_writing_W) + M(p_committed_W) = 1
```
This invariant proves that exactly one Python writer token circulates through the writer lifecycle. The writer can be in exactly one of {idle, pending, writing, committed} at any time. Since `M0(p_idle_W) = 1` and all other writer places start at 0, the conserved quantity is 1. This guarantees single-writer semantics.

**P-Inv 2 — Chart A Conservation:**
```
M(p_idle_A) + M(p_reading_A) + M(p_busy_timeout_A) = 1
```
Chart A is always in exactly one of {idle, reading, timed-out} states. Similarly for Chart B.

**P-Inv 3 — Total Active Readers Bound:**
```
M(p_reading_A) + M(p_reading_B) <= 2
```
At most N=2 readers can be simultaneously in the reading state, matching WAL's arbitrary concurrent read capability.

**Proof of Single-Writer Safety via P-Inv 1:** Since `y^T · M = 1` for all reachable `M` and the coefficient for `p_writing_W` is 1, then `M(p_writing_W) <= 1` (all other writer places also have non-negative marking). At most one token is in `p_writing_W`, meaning at most one writer can hold the WAL write lock simultaneously. QED.

#### 3.1.2 P-Invariants for Model 2 (LiveONNX-EA OnTick Flow)

**P-Inv 4 — Handler Token Conservation (Single-Execution Invariant):**
```
M(p_tick_received) + M(p_bar_check) + M(p_same_bar) + M(p_new_bar_detected) +
M(p_schedule_check) + M(p_schedule_blocked) + M(p_macro_check) +
M(p_macro_blocked) + M(p_features_extracted) + M(p_feature_warmup) +
M(p_inference_done) + M(p_garch_done) + M(p_order_dispatched) +
M(p_order_rejected) + M(p_handler_complete) = 1
```
Exactly one token circulates through the OnTick pipeline at all times, proving:
- **No reentrancy**: The handler cannot be executing in two places simultaneously.
- **No dropped ticks**: Every tick that enters the pipeline eventually reaches `p_handler_complete`.
- **No phantom states**: No spurious tokens accumulate in intermediate places.

#### 3.1.3 P-Invariants for Model 3 (DMatrix-EA Trade Events)

**P-Inv 5 — Active Positions Array Resource Conservation:**
```
M(p_active_positions_arr) + M(p_OT_new_bar) + M(p_OTT_processing) + M(p_ODI_unresolved) = 1
```
The array resource token is always held by exactly one concurrent actor — ensuring no two handlers can simultaneously write to `m_activePositions[]`.

**P-Inv 6 — Samples Array Resource Conservation:**
```
M(p_samples_arr) + M(p_OTT_labeled) + M(p_ODI_unresolved) = 1
```
`m_recordedSamples[]` is written by exactly one actor at a time.

**P-Inv 7 — Trade Cycle Conservation (total positions in system):**
For any bar `b` where a BUY and SELL are opened, the P-invariant tracking the count of active + resolved positions for that bar is:
```
count_active_for_bar_b + count_resolved_for_bar_b = 2   (one BUY + one SELL)
```
Every position opened must eventually become either a resolved sample (labeled 1.0f or 0.0f) or an unresolved position (labeled 0.0f by `ProcessUnresolvedPositions()`).

### 3.2 T-Invariants (Transition Invariants — Firing Sequences Returning to M0)

A **T-invariant** is a firing count vector `x: T -> N` such that `C · x = 0` (returns the net to a marking with the same token distribution). T-invariants correspond to **cyclic behaviors** — complete trade cycles.

#### 3.2.1 T-Invariants for Model 1 (SQLite WAL)

**T-Inv 1 — Complete Read Cycle for Chart A:**
```
x(t_A_start_read) = 1, x(t_A_end_read) = 1, all others = 0
```
One complete bar query cycle (DatabasePrepare -> DatabaseRead -> DatabaseFinalize).

**T-Inv 2 — Complete Write Cycle for Python Agent:**
```
x(t_W_request_write) = 1, x(t_W_acquire_lock) = 1, x(t_W_commit) = 1, x(t_W_release) = 1
```
One complete write cycle: request -> acquire -> commit -> release -> idle.

**T-Inv 3 — Write + Checkpoint Cycle:**
```
x(t_W_request_write) = 1, x(t_W_acquire_lock) = 1, x(t_W_commit) = 1,
x(t_W_release) = 1, x(t_wal_checkpoint) = 1
```
A complete write cycle followed by WAL checkpoint, returning the WAL frame buffer to 0.

#### 3.2.2 T-Invariants for Model 2 (LiveONNX-EA)

**T-Inv 4 — Same-Bar Short-Circuit Cycle:**
```
x(t_tick_arrives) = 1, x(t_check_bar) = 1, x(t_same_bar_exit) = 1, x(t_handler_exit) = 1
```
Tick arrives, same bar detected, handler exits immediately. O(1) cost.

**T-Inv 5 — Full Inference and Trade Cycle:**
```
x(t_tick_arrives) = 1, x(t_check_bar) = 1, x(t_new_bar_confirmed) = 1,
x(t_check_schedule) = 1, x(t_schedule_allowed) = 1, x(t_check_macro) = 1,
x(t_macro_pass) = 1, x(t_extract_features) = 1, x(t_run_onnx) = 1,
x(t_run_garch_risk) = 1, x(t_dispatch_order) = 1, x(t_handler_exit) = 1
```
One complete new-bar inference and order dispatch cycle.

#### 3.2.3 T-Invariants for Model 3 (DMatrix-EA)

**T-Inv 6 — Complete Trade Life Cycle:**
```
x(t_OT_tick_new_bar) = 1, x(t_OT_open_positions) = 1,
x(t_OT_register_buy) = 1, x(t_OT_return) = 1,
x(t_OTT_deal_add) = 1, x(t_OTT_find_position) = 1,
x(t_OTT_assign_label) = 1, x(t_OTT_add_sample) = 1,
x(t_OTT_deactivate) = 1, x(t_OTT_return) = 1
```
One complete trade life cycle: bar opens -> position registered -> deal fires -> sample labeled -> position deactivated.

### 3.3 Liveness Classification

Following Murata (1989), we classify the liveness of each transition:

**Liveness Classes:**
- **L0 (Dead):** A transition that can never fire from M0.
- **L1 (Potentially live):** Can fire at least once from M0.
- **L4 (Live):** For every reachable marking M, there exists a firing sequence enabling t.

| Transition | Model | Liveness | Rationale |
|---|---|---|---|
| `t_A_start_read` | 1 | L4 | Chart always returns to idle; read always re-enabled |
| `t_W_acquire_lock` | 1 | L4 | Writer always returns to idle after commit+release |
| `t_wal_checkpoint` | 1 | L4 | WAL frames accumulate; checkpoint enabled when readers idle |
| `t_A_busy_timeout` | 1 | L1 | Only fires under checkpoint lock contention — not inevitable |
| `t_same_bar_exit` | 2 | L4 | On every tick within a bar (>1 per bar), this fires |
| `t_new_bar_confirmed` | 2 | L4 | Eventually a new bar always opens (time progresses) |
| `t_schedule_blocked` | 2 | L1 | Only fires outside trading schedule |
| `t_macro_blocked` | 2 | L1 | Only fires when macro event is active |
| `t_feature_warmup_fail` | 2 | L1 | Only fires during initial history buffer warmup |
| `t_dispatch_order` | 2 | L4 | Reachable whenever conditions align |
| `t_OTT_deal_add` | 3 | L4 | Every opened position eventually closes (TP, SL, or timeout) |
| `t_ODI_begin` | 3 | L1 | OnDeinit fires exactly once, terminally |
| `t_ODI_export` | 3 | L1 | CSV export happens exactly once at end |

---

## Section 4: Temporal Logic Specifications (LTL/CTL)

Following **Leslie Lamport's Temporal Logic of Actions (TLA+, 1994)** and standard **Linear Temporal Logic (LTL)** and **Computation Tree Logic (CTL)** formalisms, we specify six critical behavioral properties of the system.

### 4.1 LTL Specification Primer

In **LTL**, formulas are interpreted over infinite execution paths. Key operators:
- `G phi` — Globally: `phi` holds at all future states.
- `F phi` — Finally: `phi` holds at some future state.
- `X phi` — Next: `phi` holds at the next state.
- `phi U psi` — Until: `phi` holds until `psi` holds.

### 4.2 LTL Property 1: Order Dispatch Eventually Leads to Deal Confirmation

```
LTL_P1:  G(order_dispatched -> F(deal_confirmed))
```

**Informal reading:** Every order that is dispatched to the broker via `g_trade.Buy()` or `g_trade.Sell()` is eventually confirmed as a deal transaction (or explicitly rejected).

**Formal justification:** In the MT5 order execution model, `g_trade.Buy()` returns a result code synchronously: either `TRADE_RETCODE_DONE` (order executed immediately) or an error code. For successfully opened positions, the broker server will eventually deliver a `TRADE_TRANSACTION_DEAL_ADD` event for the opening deal and, upon position closure, for the closing deal. MT5 guarantees delivery of all deal events to `OnTradeTransaction` as long as the EA is active.

**Associated TLA+ Action:**
```
ASSUME forall ticket in OpenedPositions :
  exists deal in DealsHistory : deal.position_id = ticket AND deal.entry = DEAL_ENTRY_IN
```

### 4.3 LTL Property 2: Global Deadlock Freedom

```
LTL_P2:  G(NOT deadlock)
```

**Informal reading:** The system never reaches a state where no progress is possible and the system is stuck permanently.

**Formal justification:** As shown by P-Inv 1-7, writer and reader tokens circulate through their respective subnets. The `busy_timeout = 5000ms` ensures that even if a writer lock contention occurs, readers eventually time out and return to idle. The cooperative scheduler ensures handlers always complete. See Section 5.1 for the full proof.

### 4.4 LTL Property 3: New Bar Triggers Exactly One Inference Cycle

```
LTL_P3:  G(new_bar_detected -> X(NOT new_bar_detected U inference_done))
```

**Informal reading:** After a new bar is detected, the next state until inference completes is not "new_bar_detected" again — the inference pipeline runs through exactly once before another new bar can be detected.

**Formal justification:** The `IsNewBar()` function updates `g_lastBarTime` to the current bar's open time as its first side effect upon detecting a new bar. Subsequent calls within the same bar return `false`. The cooperative scheduler ensures the OnTick call is not interrupted. Therefore, once `p_new_bar_detected` holds a token, the only enabled transition is the forward progress path through the pipeline — no second `t_new_bar_confirmed` can fire until the handler completes and a genuinely new bar opens.

### 4.5 LTL Property 4: Label Conservation — Every Position Gets Exactly One Label

```
LTL_P4:  G(position_registered -> F(position_labeled AND NOT position_active))
```

**Informal reading:** Every position registered in `m_activePositions[]` is eventually labeled (either by `ProcessTransaction()` or `ProcessUnresolvedPositions()`), and after labeling is marked inactive so it is labeled exactly once.

**Formal justification:** `RegisterPosition()` sets `isActive = true` for a new position. `ProcessTransaction()` only processes positions with `isActive = true`, then sets `isActive = false`. `ProcessUnresolvedPositions()` iterates all `isActive = true` positions, calls `AddSample()`, and sets `isActive = false`. It runs only once (from `OnDeinit`). The T-invariant for the complete trade life cycle (T-Inv 6) confirms every `t_OT_register_*` firing is eventually paired with exactly one labeling event.

### 4.6 LTL Property 5: WAL Writer Starvation Freedom

```
LTL_P5:  G(write_requested -> F(write_committed))
```

**Informal reading:** Every write request by the Python `macro_agent` eventually completes. The writer is never permanently starved by concurrent readers.

**Formal justification:** SQLite WAL mode does not implement reader-preference locking. Writers acquire the WAL write lock independently of reader presence. The Python `macro_agent` is the only writer in this system — no writer-vs-writer contention exists. The writer always eventually acquires the lock. QED.

### 4.7 CTL Property 6: All Paths Eventually Reach Idle

```
CTL_P6:  AG(EF(idle_state))
```

**Informal reading:** For every reachable state, there exists a path that eventually reaches an idle/complete state. The system never gets trapped in an unavoidable non-terminating cycle.

**Formal justification:** In all three models, idle states are always reachable from any reachable marking. The cooperative scheduler guarantees forward progress; `busy_timeout` prevents indefinite waits; T-invariants confirm all cycles return to their initial markings. The only non-cyclic path is through `OnDeinit` (Model 3), which terminates at `p_ODI_complete` — a clean terminal state.

---

## Section 5: Deadlock, Starvation & Livelock Freedom Proofs

### 5.1 Formal Proof of Deadlock Freedom

**Theorem 5.1 (Deadlock Freedom):** No reachable marking `M in R(N, M0)` in any of the three Petri Net models is a dead marking.

**Proof by structural induction on reachable markings:**

**Model 1 (SQLite WAL):**
Consider any reachable marking M. By P-Inv 1, exactly one writer token exists across the writer subnet. By P-Inv 2 (for A and B), exactly one token exists per chart. We enumerate all cases:

*Case 1: Writer in `p_idle_W`, readers in any state.* `t_W_request_write`, `t_A_start_read`, and `t_B_start_read` are all enabled. At least one transition enabled. PASS.

*Case 2: Writer in `p_writing_W`, readers in any state.* `t_W_commit` is enabled. PASS.

*Case 3: Writer in `p_write_pending_W`, all readers idle.* `t_W_acquire_lock` is enabled. PASS.

*Case 4: Reader in `p_busy_timeout_A/B`.* `t_A_recover` or `t_B_recover` is immediately enabled (no guards). PASS.

In all cases, at least one transition is enabled. No dead marking exists in Model 1. QED.

**Model 2 (LiveONNX-EA):**
By P-Inv 4, exactly one token circulates through the 15-place handler pipeline. Every place has at least one outgoing arc with an always-fireable transition. The handler always progresses from any intermediate place to `p_handler_complete`. QED.

**Model 3 (DMatrix-EA):**
By P-Inv 5-6, the resource tokens are always available when the corresponding handler completes. MT5 ensures the event queue is always draining; the terminal `OnDeinit` path is always eventually traversed. No circular wait between `OnTick`, `OnTradeTransaction`, and `OnDeinit` exists because they are serialized by the cooperative scheduler. QED.

### 5.2 Proof of Starvation Freedom Under WAL Mode

**Theorem 5.2 (WAL Reader Starvation Freedom):** Under SQLite WAL mode with `PRAGMA busy_timeout = 5000`, no chart instance is permanently prevented from reading `macro_governance.db`.

**Proof:**
1. SQLite WAL mode allows concurrent readers — T-Inv 1 and 2 are independent with no resource competition between reader subnets.
2. A WAL write by the Python agent does not block ongoing reads. WAL readers operate against a consistent snapshot; the writer operates on new WAL frames.
3. The only blocking scenario is WAL checkpoint. `PRAGMA busy_timeout = 5000` allows chart instances up to 5000ms of waiting before returning `SQLITE_BUSY`.
4. `SQLITE_BUSY` is handled gracefully: `DatabasePrepare()` returning `INVALID_HANDLE` causes `CheckMacroCalendar()` to return `false` (conservative fallback). The chart returns to idle on the next bar and retries.
5. Formally, `p_busy_timeout_A -> t_A_recover -> p_idle_A` is always fireable (no guards), ensuring starvation-freedom via the always-enabled recovery path. QED.

**Theorem 5.3 (WAL Writer Starvation Freedom):** The Python `macro_agent` is never permanently prevented from writing to `macro_governance.db`.

**Proof:** There is exactly one writer (Python `macro_agent`). By P-Inv 1, the writer token always circulates. No other process competes for the WAL write lock (MT5 instances only read). Writer-vs-writer contention is impossible. The writer always acquires the lock immediately (modulo OS scheduling latency). QED.

### 5.3 Proof of Livelock Absence

**Theorem 5.4 (Livelock Freedom):** No execution path exists in which the system is continuously active without making global progress.

**Proof by contradiction:**

Suppose a livelock cycle exists. In a livelock, transitions fire repeatedly returning the marking to the same states indefinitely without global progress.

**Model 1:** A potential livelock would require writer tokens to cycle through the write subnet without committing meaningful data. But each traversal corresponds to exactly one committed write transaction — a real state change. Database capacity is finite and Python agent performs only purposeful writes. No meaningless cycling occurs.

**Model 2:** The same-bar short-circuit cycle `{p_handler_complete -> p_bar_check -> p_same_bar -> p_handler_complete}` could appear to be a livelock — many ticks arriving within the same bar all take this fast path. But this is not a livelock: the `g_lastBarTime` state variable acts as a **global progress witness**. When a new bar opens, `g_lastBarTime` is updated (a concrete state change), and the full inference pipeline fires. Progress is guaranteed by time (new bars open at the timeframe period interval).

**Model 3:** Livelock in DMatrix-EA would require `OnTick` and `OnTradeTransaction` to cycle indefinitely. But every position has a finite lifespan bounded by `InpLabelHorizonBars`. `CheckTimeouts()` calls `trade.PositionClose()` on any position exceeding this horizon, guaranteeing eventual closure. The `OnDeinit` terminal transition guarantees `p_ODI_complete` is always eventually reached. QED.

---

## Section 6: Concurrency Vulnerabilities & Race Condition Audit

This section identifies concurrency issues **not covered** by existing knowledge-base documents.

### 6.1 Unprotected Shared State: Module-Level `_in_safe_transaction` Flag

**Location:** `macro_agent/db_client.py`, line 29.

```python
_in_safe_transaction: bool = False
```

**Risk Classification:** WARNING HIGH — Thread-Safety Violation

**Description:** The module-level boolean `_in_safe_transaction` is used by the `safe_db_transaction()` context manager to prevent nested backup operations. It is set to `True` at the start of a write transaction and reset to `False` in the `finally` block.

**Vulnerability:** This flag is **not thread-safe**. If `db_client.py` functions were called concurrently from multiple Python threads, two concurrent `safe_db_transaction()` calls could race on this boolean:

- Thread A: reads `_in_safe_transaction = False`, enters the transaction block.
- Thread B: reads `_in_safe_transaction = False` (before Thread A sets it to `True`).
- Thread B: also enters the transaction block, creating its own backup.
- If Thread B's operation corrupts the database, Thread A's backup may be a pre-Thread-B snapshot — the rollback restores an inconsistent state.

**TOCTOU Pattern Present:** Yes. The check-then-set sequence:
```python
if _in_safe_transaction:       # CHECK
    yield target_path
    return
_in_safe_transaction = True    # USE (SET)
```
is a classic **TOCTOU** (Time-of-Check-to-Time-of-Use) race. Between the check and the set, another thread could interleave.

**Current Risk Level:** LOW-MEDIUM in practice (macro_agent appears to operate single-threaded). However, as architectural debt it poses a risk if macro_agent is ever parallelized.

**Remediation Recommendation:**
```python
import threading
_safe_transaction_lock = threading.Lock()
_thread_local = threading.local()

@contextmanager
def safe_db_transaction(db_path=None):
    # Per-thread reentrancy guard (thread-safe by construction)
    if getattr(_thread_local, 'in_transaction', False):
        yield target_path
        return
    _thread_local.in_transaction = True
    try:
        with _safe_transaction_lock:  # Cross-thread serialization
            yield target_path
            # ... integrity check ...
    finally:
        _thread_local.in_transaction = False
```

---

### 6.2 TOCTOU Pattern: `PositionSelectByTicket()` Followed by `trade.PositionClose()`

**Location:** `MQL5/Include/OrderTracker.mqh`, `CheckTimeouts()` method, lines 250-256.

```mql5
if(PositionSelectByTicket(ticket))
{
   trade.PositionClose(ticket);  // TOCTOU: position may have been closed between Select and Close
}
```

**Risk Classification:** WARNING MEDIUM — Non-Fatal TOCTOU

**Description:** Between `PositionSelectByTicket(ticket)` (CHECK) and `trade.PositionClose(ticket)` (USE), the MT5 broker server may have already closed the position autonomously (TP/SL hit on the server side).

**Impact:** If `trade.PositionClose()` is called on an already-closed position, the `CTrade::PositionClose()` method receives a broker error (`TRADE_RETCODE_INVALID_TICKET`). The call is handled by the EA's error printing path (non-fatal). The `OnTradeTransaction` event for the autonomous closure still fires, ensuring `ProcessTransaction()` labels the position correctly. Double-labeling is prevented by the `isActive` flag in `FindActivePosition()`.

**Actual Severity:** LOW (non-corrupting).

**Remediation Recommendation:** Add explicit error handling:
```mql5
if(PositionSelectByTicket(ticket))
{
   if(!trade.PositionClose(ticket))
   {
      uint retcode = trade.ResultRetcode();
      if(retcode == TRADE_RETCODE_INVALID_TICKET)
      {
         // Position was already closed by server (TP/SL); safe to deactivate
         m_activePositions[i].isActive = false;
      }
   }
}
```

---

### 6.3 Benign Read-Modify-Write: `g_lastBarTime` in `IsNewBar()`

**Location:** `LiveONNX-EA.mq5` and `DMatrix-EA.mq5`, `IsNewBar()` function.

```mql5
bool IsNewBar()
{
   datetime currentBarTime = iTime(_Symbol, _Period, 0);
   if(currentBarTime != g_lastBarTime)   // CHECK
   {
      g_lastBarTime = currentBarTime;    // MODIFY
      return true;
   }
   return false;
}
```

**Risk Classification:** SAFE — Intra-Instance Synchronization by MT5 Scheduler

**Description:** This is a **read-modify-write** sequence on `g_lastBarTime`. However, since MT5's cooperative scheduler serializes `OnTick` calls within a single EA instance, this pattern is **safe by construction**. `g_lastBarTime` is never accessed from multiple concurrent handlers in the same EA. Each chart instance has its own copy. The pattern is completely safe.

---

### 6.4 WAL Checkpoint During Active MT5 DB Handle

**Location:** `macro_agent/db_client.py`, `safe_db_transaction()` and `verify_database_integrity()`.

```python
conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
```

**Risk Classification:** WARNING LOW-MEDIUM — Transient Interference

**Description:** `PRAGMA wal_checkpoint(TRUNCATE)` acquires an exclusive checkpoint lock and waits for all readers to complete their transactions before truncating the WAL file. If an MT5 chart instance has a long-running read transaction open (between `DatabasePrepare()` and `DatabaseFinalize()`), the `TRUNCATE` checkpoint will block until the MT5 reader releases its lock.

**Actual Scenario of Risk:** In `CheckMacroCalendar()`, the time between `DatabasePrepare` and `DatabaseFinalize` could span several hundred microseconds. If a Python checkpoint fires exactly in this window, the checkpoint is deferred — no corruption occurs, only latency.

**Remediation Recommendation:** Use `PRAGMA wal_checkpoint(PASSIVE)` in non-critical verification paths. `PASSIVE` mode allows checkpointing without waiting for active readers:
```python
# In verify_database_integrity:
conn.execute("PRAGMA wal_checkpoint(PASSIVE);")  # Non-blocking
```
Reserve `TRUNCATE` only for the pre-backup checkpoint inside `safe_db_transaction`.

---

### 6.5 Log File Polling Race in `mt5_client.py`

**Location:** `src/mt5_client.py`, `_stream_new_tester_logs()`.

```python
curr_size = log_file.stat().st_size   # CHECK file size
...
with open(log_file, "rb") as f:
    f.seek(prev_offset)               # USE offset
    raw_bytes = f.read()              # READ
    file_offsets[log_file] = f.tell() # UPDATE offset
```

**Risk Classification:** WARNING LOW — Benign Data Loss Under Race

**Description:** Between `stat().st_size` (CHECK) and `open(log_file) + read()` (USE), the MT5 terminal could append new log lines or rotate the log file. In either case, the next poll cycle recovers correctly. At most one polling cycle's log lines could be missed.

**Remediation Recommendation:**
```python
if log_file.stat().st_size < prev_offset:
    # Log was rotated/truncated; reset offset
    file_offsets[log_file] = 0
    prev_offset = 0
```

---

### 6.6 Summary Risk Matrix

| Finding | Location | TOCTOU? | Thread-Safe? | Severity | Status / Remediation |
|---|---|---|---|---|---|
| `_in_safe_transaction` module global | `db_client.py:29` | Yes | No | Medium | Safe in single-threaded CLI |
| `PositionSelectByTicket` + `PositionClose` | `OrderTracker.mqh:250` | Yes | N/A (cooperative) | Low | Non-corrupting |
| `g_lastBarTime` R-M-W | `LiveONNX-EA.mq5` | No (intra-instance) | Safe by MT5 scheduler | None | Safe by design |
| WAL `TRUNCATE` checkpoint during active reads | `db_client.py:95,106` | No | Safe by SQLite WAL | Low | Latency only |
| Log file polling offset | `mt5_client.py:256` | Yes (benign) | Single-threaded | Low | Handled by poll loop |
| Partial Close Order Tracking Desynchronization | `LiveONNX-EA.mq5:2270` | Yes | MQL5 Cooperative | Medium | **RESOLVED**: Volume decremented; residual tracked |
| SQLite Multi-Chart Lock Contention | `LiveONNX-EA.mq5:1182` | No | Multi-Process | High | **RESOLVED**: WAL + busy_timeout=5000 + sync=NORMAL |

---

### 6.7 Partial Close Reentrancy & Lifecycle Reconciliation

**Location:** [`LiveONNX-EA.mq5:2262-2348`](file:///C:/Users/allan/IdeaProjects/mt5-fx-countdown/MQL5/Experts/LiveONNX-EA.mq5#L2262-L2348), `OnTradeTransaction()`.

**Concurrency Risk:** When an institutional position is partially closed by the broker or user, MT5 dispatches a deal with `DEAL_ENTRY_OUT`. A naive implementation calls `RemoveActiveTrade(idx)` on the first exit deal. If the position remains open with residual volume ($V_{\text{rem}} = V_{\text{initial}} - V_{\text{deal}} > 0$), subsequent exit transactions encounter `idx == -1` (untracked position), losing initial entry price, slippage metrics, and excursion extrema ($MAE/MFE$).

**Remediation Implemented:**
```mql5
bool isPartialClose = (posId > 0 && PositionSelectByTicket(posId));
// ...
if(isPartialClose)
{
   g_activeTrades[idx].volume = MathMax(0.0, g_activeTrades[idx].volume - dealVolume);
}
else
{
   RemoveActiveTrade(idx);
}
```
Furthermore, in the event of an EA crash or terminal restart while positions were active, `HistorySelectByPosition(posId)` was introduced as an automated recovery mechanism to restore entry timestamp, deal ticket, and compute accurate holding duration.

---

### 6.8 SQLite Multi-Chart Concurrent Lock Contention & PRAGMA Hardening

**Location:** [`LiveONNX-EA.mq5:1174-1185`](file:///C:/Users/allan/IdeaProjects/mt5-fx-countdown/MQL5/Experts/LiveONNX-EA.mq5#L1174-L1185), [`macro_agent/db_client.py:48-56`](file:///C:/Users/allan/IdeaProjects/mt5-fx-countdown/macro_agent/db_client.py#L48-L56).

**Concurrency Risk:** When $N$ chart threads execute `DatabaseOpen()` on `macro_governance.db` simultaneously, SQLite requires WAL mode and a non-zero busy timeout to prevent lock contention from throwing immediate `SQLITE_BUSY` errors.

**Remediation Implemented:** Synchronized identical PRAGMA initialization across Python and MQL5:
```mql5
DatabaseExecute(g_hMacroDB, "PRAGMA journal_mode = WAL;");
DatabaseExecute(g_hMacroDB, "PRAGMA synchronous = NORMAL;");
DatabaseExecute(g_hMacroDB, "PRAGMA busy_timeout = 5000;");
```
```python
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("PRAGMA busy_timeout=5000;")
conn.execute("PRAGMA synchronous=NORMAL;")
```
This mathematically guarantees non-blocking concurrent reads by up to $N$ chart threads while the Python macroeconomic agent performs write transactions.

## Didactic References & Further Reading

The formal models, invariant analyses, and temporal logic specifications in this monograph are grounded in the following canonical works and specifications:

### Formal Methods Foundations

1. **Petri, C. A. (1962).** *Kommunikation mit Automaten.* Schriften des IIM Nr. 2, Institut für Instrumentelle Mathematik, Bonn. (Doctoral dissertation — the founding document of Petri Net theory, introducing places, transitions, and the concept of concurrent firing.)

2. **Murata, T. (1989).** "Petri Nets: Properties, Analysis and Applications." *Proceedings of the IEEE*, 77(4), 541–580. doi:10.1109/5.24143. URL: [https://ieeexplore.ieee.org/document/24143](https://ieeexplore.ieee.org/document/24143) — The definitive survey reference for P-invariants, T-invariants, liveness classifications (L0–L4), reachability graphs, and structural analysis techniques used throughout this monograph.

3. **Milner, R. (1980).** *A Calculus of Communicating Systems (CCS).* Lecture Notes in Computer Science, Vol. 92. Springer-Verlag, Berlin. ISBN: 978-3-540-10235-9. — Foundational work on process algebra and compositional reasoning about concurrent systems, including bisimulation equivalence and synchronization algebras.

4. **Hoare, C. A. R. (1978).** "Communicating Sequential Processes." *Communications of the ACM*, 21(8), 666–677. doi:10.1145/359576.359585. URL: [https://dl.acm.org/doi/10.1145/359576.359585](https://dl.acm.org/doi/10.1145/359576.359585) — CSP process algebra, channel-based synchronization, and the formal compositional reasoning framework applicable to MT5's cooperative scheduler model.

5. **Lamport, L. (1994).** "The Temporal Logic of Actions." *ACM Transactions on Programming Languages and Systems*, 16(3), 872–923. doi:10.1145/177492.177726. URL: [https://dl.acm.org/doi/10.1145/177492.177726](https://dl.acm.org/doi/10.1145/177492.177726) — TLA+ specification language, temporal logic operators (G, F, U, X), and state machine refinement used for the LTL/CTL properties in Section 4.

6. **Lamport, L. (2002).** *Specifying Systems: The TLA+ Language and Tools for Hardware and Software Engineers.* Addison-Wesley. URL: [https://lamport.azurewebsites.net/tla/book.html](https://lamport.azurewebsites.net/tla/book.html) — Complete TLA+ specification methodology, TLAPS (TLA+ Proof System), and applications to distributed systems correctness.

### Temporal Logic & Model Checking

7. **Clarke, E. M., Emerson, E. A., & Sistla, A. P. (1986).** "Automatic Verification of Finite-State Concurrent Systems Using Temporal Logic Specifications." *ACM Transactions on Programming Languages and Systems*, 8(2), 244–263. doi:10.1145/5397.5399. — CTL model checking, the fundamental algorithm underlying automated verification of concurrent system properties.

8. **Pnueli, A. (1977).** "The Temporal Logic of Programs." *Proceedings of the 18th Annual IEEE Symposium on Foundations of Computer Science (FOCS)*, 46–57. — Original LTL paper introducing G (globally), F (finally), U (until), and X (next) operators for reasoning about infinite program executions.

9. **Baier, C., & Katoen, J.-P. (2008).** *Principles of Model Checking.* MIT Press. ISBN: 978-0-262-02649-9. URL: [https://mitpress.mit.edu/books/principles-model-checking](https://mitpress.mit.edu/books/principles-model-checking) — Comprehensive textbook on LTL model checking, PCTL for probabilistic systems, and Buchi automata — directly applicable to LTL_P1 through LTL_P6 specified in Section 4.

### Database Concurrency & SQLite WAL

10. **Hipp, D. R., et al. (2010+).** *SQLite Write-Ahead Logging (WAL) Mode.* Official SQLite Documentation. URL: [https://www.sqlite.org/wal.html](https://www.sqlite.org/wal.html) — Authoritative specification of WAL concurrency semantics: reader-writer isolation, checkpoint mechanics, `busy_timeout` behavior, and the WAL frame snapshot model used to ground Model 1.

11. **SQLite Locking and Concurrency Reference.** URL: [https://www.sqlite.org/lockingv3.html](https://www.sqlite.org/lockingv3.html) — Low-level SQLite lock state machine (UNLOCKED -> SHARED -> RESERVED -> PENDING -> EXCLUSIVE), directly formalizable as a Petri Net subnet.

12. **Bernstein, P. A., Hadzilacos, V., & Goodman, N. (1987).** *Concurrency Control and Recovery in Database Systems.* Addison-Wesley. — Foundational database theory: serializability, two-phase locking (2PL), MVCC (related to WAL snapshots), and formal correctness conditions for concurrent database access.

### MetaTrader 5 Architecture & MQL5 Concurrency

13. **MetaQuotes Software Corp. (2010–2026).** *MQL5 Reference: Events in Expert Advisors.* MetaTrader 5 Help Center. URL: [https://www.mql5.com/en/docs/runtime/event_fire](https://www.mql5.com/en/docs/runtime/event_fire) — Official documentation of MT5's event model: `OnTick`, `OnTradeTransaction`, `OnDeinit` delivery semantics, queue management, and the cooperative single-threaded scheduling contract.

14. **MetaQuotes Software Corp. (2010–2026).** *MQL5 Reference: Trade Transactions.* URL: [https://www.mql5.com/en/docs/trading/ontransaction](https://www.mql5.com/en/docs/trading/ontransaction) — `OnTradeTransaction` callback specification, `MqlTradeTransaction` structure, `TRADE_TRANSACTION_DEAL_ADD` event semantics.

15. **MetaQuotes Software Corp. (2010–2026).** *DatabasePrepare, DatabaseRead, DatabaseFinalize — MQL5 Database API.* URL: [https://www.mql5.com/en/docs/database](https://www.mql5.com/en/docs/database) — SQLite integration in MT5 terminals: concurrency behavior under WAL.

### Financial Systems Concurrency & Correctness

16. **Gray, J., & Reuter, A. (1992).** *Transaction Processing: Concepts and Techniques.* Morgan Kaufmann. ISBN: 978-1-55860-190-7. — Comprehensive treatment of ACID properties, deadlock detection/prevention, and two-phase commit in financial transaction systems.

17. **Tanenbaum, A. S., & Van Steen, M. (2017).** *Distributed Systems: Principles and Paradigms* (3rd ed.). Pearson. — Fundamental distributed systems theory: mutual exclusion, consensus, clock synchronization, and the theoretical basis for reasoning about distributed concurrency in the MT5 + Python MLOps system.

18. **Ben-Ari, M. (2006).** *Principles of Concurrent and Distributed Programming* (2nd ed.). Addison-Wesley. ISBN: 978-0-321-31283-9. — Textbook treatment of mutual exclusion algorithms, monitor-based synchronization, Petri Net modeling of concurrent programs, and formal correctness proofs.

### Process Algebra & CSP Tools

19. **Roscoe, A. W. (1998).** *The Theory and Practice of Concurrency.* Prentice Hall. URL: [https://www.cs.ox.ac.uk/bill.roscoe/publications/68b.pdf](https://www.cs.ox.ac.uk/bill.roscoe/publications/68b.pdf) — CSP theory, FDR model checker for process algebra, and denotational semantics.

20. **Holzmann, G. J. (2003).** *The SPIN Model Checker: Primer and Reference Manual.* Addison-Wesley. ISBN: 978-0-321-22862-8. — PROMELA specification language and SPIN model checker for LTL verification, directly applicable to automating verification of the LTL properties specified in Section 4.

### Petri Net Analysis Tools & Extensions

21. **Jensen, K. (1997).** *Coloured Petri Nets: Basic Concepts, Analysis Methods and Practical Use* (2nd ed.). Springer. — Coloured Petri Nets (CPN) extending basic nets with typed tokens, enabling compact modeling of parameterized N-chart concurrency (N readers instead of fixed 2).

22. **David, R., & Alla, H. (2010).** *Discrete, Continuous, and Hybrid Petri Nets* (2nd ed.). Springer. — Extensions of Petri Nets to timed and hybrid systems, applicable to modeling the `busy_timeout = 5000ms` quantitative timing constraint in Model 1.

---

## Appendix A: Notation Summary

| Symbol | Definition |
|---|---|
| `P` | Set of Petri Net places |
| `T` | Set of Petri Net transitions |
| `A` | Set of arcs (flow relation) |
| `W` | Arc weight function |
| `M0` | Initial marking |
| `M` | Reachable marking |
| `•t` | Preset of transition t (input places) |
| `t•` | Postset of transition t (output places) |
| `R(N, M0)` | Reachability set |
| `y` | P-invariant vector |
| `x` | T-invariant (firing count) vector |
| `C` | Incidence matrix |
| `G` | LTL Globally operator |
| `F` | LTL Finally operator |
| `X` | LTL Next-state operator |
| `U` | LTL Until operator |
| `AG` | CTL All paths, Globally |
| `EF` | CTL Exists a path, Finally |
| `L0-L4` | Liveness classes (Murata 1989) |
| `sigma_agg` | GARCH multi-step aggregated volatility |
| `EET/EEST` | Eastern European Time / Summer Time (MT5 Server Time) |
| `WAL` | SQLite Write-Ahead Logging mode |
| `TOCTOU` | Time-of-Check-to-Time-of-Use race condition |

---

## Appendix B: System Timezone Invariant

**All temporal references in this document adhere to the universal project timezone standard:** Eastern European Time (EET, UTC+2) in winter / Eastern European Summer Time (EEST, UTC+3) in summer — identical to MT5 Server Time as operated by institutional Forex brokers worldwide. This standard ensures that bar timestamps referenced in `g_lastBarTime`, database event timestamps in `macro_governance.db`, and all scheduled trading windows in `IsTradeScheduleAllowed()` are consistently interpreted without UTC offset confusion.

The GARCH multi-step horizon (`H` bars) is always expressed in units of the chart timeframe, where each daily bar closes at 17:00 New York time (mapped to 00:00 EET/EEST next day) — the standard 5-bar week structure of the institutional Forex market.

---

*End of Monograph — CONCURRENCY_AND_PETRI_NET_MODELING.md*  
*Document classification: Formal Methods & Concurrency Engineering*  
*Version: 1.0.0 | Date: 2026-09-04 | Language: English*
