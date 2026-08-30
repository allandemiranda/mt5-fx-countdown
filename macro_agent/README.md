# Macro Agent & Macroeconomic Governance Sub-Project (`macro_agent/`)

> [!IMPORTANT]
> **ARCHITECTURAL BOUNDARY & ISOLATION NOTICE**:
> This directory represents an **independent, auxiliary sub-project** designed strictly for on-demand execution by an AI CLI session (such as Google Antigravity CLI or an interactive assistant).
> **DO NOT** use the instructions, scripts, or schemas in this directory to alter, override, or configure the main quantitative machine learning pipeline (`run_pipeline.py`, `src/trainer.py`, `MQL5/Experts/DMatrix-EA.mq5`, `AGENTS.md`, or `.env`).
> The main trading EA (`LiveONNX-EA.mq5`) acts strictly as a **read-only consumer** of the SQLite database populated by this sub-project.

---

## 1. Overview & Objective

Financial currency markets (Forex) are frequently disrupted by scheduled macroeconomic releases (e.g. Non-Farm Payrolls, FOMC interest rate decisions, CPI prints) and unexpected breaking geopolitical/economic news.

This sub-project equips an AI agent operating via the CLI to:
1. **Fetch fresh macro data and news** using dedicated Python collectors (`fetcher.py`);
2. **Reason and evaluate impact** for specific currency pairs (e.g. `EURUSD`, `GBPUSD`);
3. **Govern the central SQLite database** (`macro_governance.db` located in MT5's `Common/Files` directory);
4. **Direct the live Expert Advisor (`LiveONNX-EA.mq5`)** to take protective actions (`BLOCK_ENTRIES`, `TRAILING_STOP`, `BREAKEVEN`, `CLOSE_ALL`, or `ADVISORY_ONLY`).

---

## 2. Directory Structure

```
macro_agent/
├── README.md                      # This boundary and operational guide
├── db_client.py                   # Python SQLite manager for macro_governance.db
├── fetcher.py                     # Collector for economic calendar feeds & news headlines
└── prompts/
    ├── UPDATE_ECONOMIC_CALENDAR.md # CLI Agent runbook for calendar event evaluation
    └── UPDATE_NEWS_GOVERNANCE.md   # CLI Agent runbook for breaking news blacklist evaluation
```

---

## 3. Database Schema (`macro_governance.db`)

The database is stored permanently in the MetaTrader 5 Common Files folder:
`%APPDATA%\MetaQuotes\Terminal\Common\Files\macro_governance.db`

It contains two clean, relational tables:

### A. `calendar_events` (Time-Windowed Scheduled Events)
*Active in both Live Trading and Strategy Tester backtests.*

| Column | Type | Description |
| :--- | :---: | :--- |
| `id` | `INTEGER PRIMARY KEY` | Auto-incrementing identifier |
| `symbol` | `TEXT` | Target pair (e.g. `EURUSD`), base/quote currency (e.g. `EUR`, `USD`), or `GLOBAL` |
| `title` | `TEXT` | Event title (e.g. `US Non-Farm Payrolls`) |
| `description` | `TEXT` | Rationale and impact description |
| `start_time` | `TEXT` | Event start window in MT5 Server Time / EET-EEST (format: `YYYY-MM-DD HH:MM:SS`) |
| `end_time` | `TEXT` | Event end window in MT5 Server Time / EET-EEST (format: `YYYY-MM-DD HH:MM:SS`) |
| `action` | `TEXT` | Protection action (`BLOCK_ENTRIES`, `TRAILING_STOP`, `BREAKEVEN`, `CLOSE_ALL`, `ADVISORY_ONLY`) |
| `trailing_points` | `INTEGER` | Trailing stop distance in broker points (default `0`). If $\le 0$ during `TRAILING_STOP`, triggers immediate position closure |

### B. `news_events` (Active Global Breaking News)
*Active in Live Trading only (automatically bypassed during Strategy Tester backtests).*

| Column | Type | Description |
| :--- | :---: | :--- |
| `symbol` | `TEXT PRIMARY KEY` | Target pair (e.g. `EURUSD`), currency (`USD`), or `GLOBAL` |
| `title` | `TEXT` | News headline |
| `description` | `TEXT` | Detailed explanation of the market threat |
| `action` | `TEXT` | Protection action (`BLOCK_ENTRIES`, `TRAILING_STOP`, `BREAKEVEN`, `CLOSE_ALL`, `ADVISORY_ONLY`) |
| `trailing_points` | `INTEGER` | Trailing stop distance in broker points (default `0`). If $\le 0$ during `TRAILING_STOP`, triggers immediate position closure |

---

## 4. Supported EA Actions (`action`)

When the EA evaluates a bar and finds an active record for its symbol, it executes the corresponding quantitative protective action:

1. **`BLOCK_ENTRIES`** *(Default)*: Prohibits opening new orders on this symbol. Existing positions are left undisturbed with their native GARCH/S&R stops.
2. **`TRAILING_STOP`**: Prohibits opening new orders. For existing open positions in profit, tightens stop levels by `trailing_points`. If `trailing_points <= 0` (or if broker stop modification fails), immediately executes market closure for safety.
3. **`BREAKEVEN`**: Prohibits opening new orders. For existing open positions in profit, moves Stop Loss directly to entry price (`price_open`) ensuring zero financial downside risk. If distance violates broker stops level or modification fails, immediately executes market closure.
4. **`CLOSE_ALL`**: Prohibits opening new orders and immediately executes a market close on all open positions for this symbol (used for catastrophic/unpredictable events like emergency rate hikes or war escalations).
5. **`ADVISORY_ONLY`**: Does not block orders and does not alter positions. Emits an advisory warning log in the MT5 Experts log for operator awareness.

---

## 5. How to Run via AI CLI Terminal

When the user requests an update, instruct the AI in the CLI to follow the dedicated runbook:

### To Update Economic Calendar:
```powershell
# Instruct the CLI agent to read and follow:
macro_agent/prompts/UPDATE_ECONOMIC_CALENDAR.md
```

### To Update Breaking News Blacklist:
```powershell
# Instruct the CLI agent to read and follow:
macro_agent/prompts/UPDATE_NEWS_GOVERNANCE.md
```

---

## 6. Defensive Transaction Governance & Automated Backups

To guarantee database integrity and prevent accidental corruption or incomplete writes:
1. **Pre-Modification Backup**: Before any modifying operation (`init`, `add-cal`, `del-cal`, `add-news`, `del-news`, `purge`), the client automatically creates a physical backup copy named `macro_governance.db.<YYYYMMDD_HHMMSS_ffffff>.bkp`.
2. **Post-Modification Integrity Check**: After executing DDL or DML statements, SQLite executes `PRAGMA integrity_check;` to verify that all B-Trees, page headers, and indexes are 100% valid.
3. **Automatic Rollback & Restoration**: If an unhandled exception occurs OR if SQLite integrity check fails, the client immediately restores the database file from the pre-modification `.bkp` file, ensuring zero data loss.
4. **Manual Inspection & Restoration**:
   ```powershell
   # Verify SQLite integrity
   python macro_agent/db_client.py verify

   # List available timestamped backups
   python macro_agent/db_client.py list-backups

   # Restore from a specific backup
   python macro_agent/db_client.py restore --file path/to/backup.bkp
   ```
