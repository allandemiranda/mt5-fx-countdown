# AI Runbook: Update Breaking News Governance Blacklist

> [!NOTE]
> **Sub-Project Context**: This runbook instructs an AI CLI agent to evaluate breaking macroeconomic, geopolitical, and financial news, governing the persistent `news_events` table in `macro_governance.db`.

---

## 1. Objective

Given a list of actively traded symbols (e.g. `EURUSD`, `GBPUSD`, `USDJPY`):
1. Collect recent breaking news headlines using `macro_agent/fetcher.py --news`.
2. Inspect the current SQLite news blacklist records using `macro_agent/db_client.py status`.
3. Evaluate whether active geopolitical conflicts, central bank emergency statements, banking crises, or unscheduled market shocks pose tail risk.
4. Update the `news_events` table:
   - Add new symbols to the blacklist if threatened;
   - Update descriptions or actions if conditions changed;
   - Remove symbols from the blacklist once market panic subsides.

---

## 2. Step-by-Step Instructions for the AI CLI Agent

### Step 1: Check Current News Blacklist State
Run the database client to inspect currently blacklisted symbols:
```powershell
python macro_agent/db_client.py status
```

### Step 2: Fetch Breaking News Headlines
Run the fetcher for the traded currencies (or inject external HTML/text provided by the user from Bloomberg, Reuters, etc.):
```powershell
python macro_agent/fetcher.py --currency USD --currency EUR --currency JPY --news
```

### Step 3: Analyze & Reason About Active Threats
For each actively traded symbol:
- **Exogenous Threat Check**: Is there an ongoing war, currency intervention, flash crash risk, or emergency rate decision?
- **Action Determination**:
  - `BLOCK_ENTRIES`: Cease opening new positions until market stabilizes.
  - `CLOSE_ALL`: Immediate emergency liquidation due to catastrophic risk (e.g. military escalation).
  - `BREAKEVEN`: Lock current open positions at break-even.
  - `TRAILING_STOP`: Trail existing profits.
- **Pruning Check**: Has a previously blacklisted event concluded or normalized? If so, mark it for removal.

### Step 4: Update SQLite Database
- To add or update a symbol blacklist:
  ```powershell
  python macro_agent/db_client.py add-news --symbol EURUSD --title "Geopolitical Escalation" --desc "Active conflict escalating in Eastern Europe threatening Euro stability" --action BLOCK_ENTRIES
  ```
- To unblock a symbol whose threat has normalized:
  ```powershell
  python macro_agent/db_client.py del-news --symbol EURUSD
  ```

### Step 5: Verify Final State & Report to User
Run `python macro_agent/db_client.py status` and output a clean markdown summary table of currently blacklisted/governed symbols and reasons.
