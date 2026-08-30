# AI Runbook: Update Macroeconomic Calendar Events

> [!NOTE]
> **Sub-Project Context**: This runbook instructs an AI CLI agent to evaluate macroeconomic announcements and update the persistent `calendar_events` table in `macro_governance.db`.

---

## 1. Objective

Given a target trading symbol (e.g. `EURUSD`) and a time window (e.g. from current time to 7 days in the future):
1. Collect fresh macroeconomic release data using `macro_agent/fetcher.py`.
2. Inspect the current SQLite database records using `macro_agent/db_client.py status`.
3. Filter and reason about high-impact volatility catalysts for the symbol's base and quote currencies.
4. Insert new events, update existing ones, or purge expired past events.

---

## 2. Step-by-Step Instructions for the AI CLI Agent

### Step 1: Check Current Database State
Run the database client to list existing events:
```powershell
python macro_agent/db_client.py status
```

### Step 2: Fetch Fresh Economic Calendar Releases
Run the fetcher for the target symbol:
```powershell
python macro_agent/fetcher.py --symbol EURUSD --calendar
```
*(Optionally include any specific date range or additional currencies).*

### Step 3: Analyze & Reason About Catalysts
Evaluate each event returned:
- **Currency Match**: Does it affect the base or quote currency (e.g. EUR or USD)?
- **Impact Level**: Is it high or critical volatility (NFP, FOMC, CPI, Interest Rate decisions, GDP)?
- **Window Definition (MT5 Server Time: EET/EEST)**:
  - Timestamps must strictly follow the institutional MT5 Server Time (EET: UTC+2 winter / EEST: UTC+3 summer).
  - `start_time`: Usually 30 minutes before the scheduled release.
  - `end_time`: Usually 30 to 60 minutes after the release (or until volatility normalizes).
- **Determine Action (`action`)**:
  - `BLOCK_ENTRIES`: High risk of slippage; no new orders allowed.
  - `TRAILING_STOP`: Critical event where open winning positions must defend profit.
  - `BREAKEVEN`: Move SL to entry price to eliminate drawdown risk.
  - `CLOSE_ALL`: Extreme uncertainty (emergency central bank meetings, rate shock).
  - `ADVISORY_ONLY`: Moderate event; log advisory warning without blocking.

### Step 4: Update SQLite Database
For each identified event, execute:
```powershell
python macro_agent/db_client.py add-cal --symbol EURUSD --title "Event Title" --desc "Rationale description" --start "YYYY-MM-DD HH:MM:00" --end "YYYY-MM-DD HH:MM:00" --action BLOCK_ENTRIES
```

### Step 5: Purge Expired Past Events
Clean up past records to keep the database optimal:
```powershell
python macro_agent/db_client.py purge
```

### Step 6: Verify Final State & Report to User
Run `python macro_agent/db_client.py status` and output a clean markdown summary table of the active calendar schedule to the user.
