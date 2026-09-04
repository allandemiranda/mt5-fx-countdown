# Macro Agent Tools (`macro_agent/tools/`)

This directory contains standalone CLI utilities and generation scripts supporting the Macro Agent and Macroeconomic Governance Sub-Project.

---

## 1. `generate_calendar_dataset.py`

### Why This Tool Was Created
In MetaTrader 5, native MQL5 economic calendar functions (such as `CalendarValueHistory`) return empty arrays or are entirely disabled during Strategy Tester backtests. Consequently, quantitative machine learning models and trading Expert Advisors (such as `LiveONNX-EA.mq5`) cannot evaluate event risk or enforce defensive stop/blackout actions during historical simulations without an external, persistent relational database.

To eliminate this limitation and maintain institutional backtest integrity, this tool compiles and synthesizes the full calendar of macroeconomic announcements across the **8 major global currency jurisdictions**:
* **USD** (United States Dollar)
* **EUR** (Euro)
* **GBP** (British Pound)
* **JPY** (Japanese Yen)
* **AUD** (Australian Dollar)
* **CAD** (Canadian Dollar)
* **CHF** (Swiss Franc)
* **NZD** (New Zealand Dollar)

### Strict Invariants & Ex-Ante Action Taxonomy
1. **Zero Lookahead Bias (Ex-Ante Only)**:
   Every event description is formulated using only metrics available *prior* to the announcement (Prior reading, Market Consensus / Forecast, and qualitative market risk context). No post-event data or price reaction commentary is included.
2. **Ex-Ante Action Calibration**:
   * **`TRAILING_STOP` (`trailing_points = 120`)**:
     Applied strictly to Central Bank Rate Decisions (FOMC, ECB, BOE, BOJ, RBA, BOC, SNB, RBNZ). Prohibits new entries and tightens stop loss on open profitable positions by 120 points to lock in accumulated profits before extreme central bank volatility.
   * **`BREAKEVEN`**:
     Applied strictly to US Non-Farm Payrolls (NFP). Prohibits new entries and moves stop loss on profitable positions directly to entry price (`price_open`), eliminating downside risk against erratic two-way labor revisions.
   * **`BLOCK_ENTRIES`**:
     Applied to critical inflation prints (CPI, Core PCE), GDP releases, and key labor reports. Prohibits new market orders during spread widening while allowing open trend trades to continue running under native dynamic GARCH risk management.
   * **`ADVISORY_ONLY`**:
     Applied to moderate sentiment surveys and commodity auctions (e.g. Flash PMIs, Global Dairy Trade). Emits informational audit logs in the MT5 Experts journal without blocking order execution.
3. **Universal Timezone Standard (EET/EEST - MT5 Server Time)**:
   All timestamps are converted to MT5 Server Time (`Europe/Athens`: UTC+2 in winter / UTC+3 in summer) with automatic European Daylight Saving Time (DST) transitions, matching chart bar timestamps exactly.
4. **Relational Database Parity**:
   Writes exclusively to `macro_governance.db` in `%APPDATA%\MetaQuotes\Terminal\Common\Files`. Populates `calendar_events` and creates the `news_events` table strictly empty (0 records) for Strategy Tester parity. Does not pollute `Common/Files` with CSV files.

---

## 2. How to Execute

### Default Execution (2025-01-01 to 2026-09-01 into MT5 Common Files)
Run from the repository root:
```powershell
python macro_agent/tools/generate_calendar_dataset.py
```

### Custom Date Range or Target Database Path
```powershell
# Custom start and end dates
python macro_agent/tools/generate_calendar_dataset.py --start 2025-01-01 --end 2026-09-01

# Custom SQLite database location
python macro_agent/tools/generate_calendar_dataset.py --db-path "C:/path/to/custom_macro_governance.db"
```

---

## 3. Output Database Schema

* **File Location**: `%APPDATA%\MetaQuotes\Terminal\Common\Files\macro_governance.db`
* **Table `calendar_events`**: Populated with scheduled ex-ante records and calibrated actions.
* **Table `news_events`**: Created with full DDL schema, kept strictly empty (0 records) for Strategy Tester parity.
