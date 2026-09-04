"""Institutional Economic Calendar Generator for MT5 Strategy Tester & Live Trading.

Why this script was created:
-----------------------------
In MetaTrader 5, the built-in economic calendar functions (e.g. CalendarValueHistory)
are disabled or return empty arrays during Strategy Tester backtests. Furthermore,
quantitative execution engines (LiveONNX-EA) require an unbiased, zero-lookahead,
ex-ante calendar to accurately simulate how protective mechanisms and blackout
filters function in real-time.

This tool compiles and synthesizes the full calendar of macroeconomic releases across
the 8 major global currency jurisdictions: USD, EUR, GBP, JPY, AUD, CAD, CHF, and NZD
for the testing period: 2025-01-01 to 2026-09-01.

Action Taxonomy (Ex-Ante Calibration):
--------------------------------------
1. TRAILING_STOP (trailing_points=120):
   Assigned strictly to Central Bank Rate Decisions (FOMC, ECB, BOE, BOJ, RBA, BOC,
   SNB, RBNZ). Prohibits new entries and tightens stop loss on open profitable positions
   by 120 points to lock in accumulated profits before extreme central bank volatility.
2. BREAKEVEN:
   Assigned strictly to US Non-Farm Payrolls (NFP). Prohibits new entries and moves
   stop loss on profitable positions directly to entry price (price_open), eliminating
   downside financial risk against erratic two-way employment revisions.
3. BLOCK_ENTRIES:
   Assigned to critical inflation prints (CPI, Core PCE), GDP releases, and key labor
   reports. Prohibits new market orders during spread widening while allowing open trend
   trades to continue running under dynamic GARCH risk management.
4. ADVISORY_ONLY:
   Assigned to moderate sentiment surveys and commodity auctions (e.g. Flash PMIs,
   Global Dairy Trade). Emits informational audit logs in the MT5 Experts journal
   without blocking order execution.

Database Output:
----------------
Populates exclusively the central SQLite database (macro_governance.db) located in the
MT5 Common Files directory: %APPDATA%\\MetaQuotes\\Terminal\\Common\\Files\\macro_governance.db.
Creates the news_events table and leaves it strictly empty (0 records) as news cannot
be backtested. Does NOT create unneeded CSV files in Common/Files.

How to execute:
---------------
Run directly via Python CLI from the project root:
    python macro_agent/tools/generate_calendar_dataset.py

Custom options:
    python macro_agent/tools/generate_calendar_dataset.py --start 2025-01-01 --end 2026-09-01
    python macro_agent/tools/generate_calendar_dataset.py --db-path "path/to/custom_macro.db"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import os
from pathlib import Path
import sqlite3
from typing import List
from zoneinfo import ZoneInfo

MT5_TZ = ZoneInfo("Europe/Athens")
UTC_TZ = ZoneInfo("UTC")


@dataclass(frozen=True)
class CalendarEventRecord:
    """Represents a scheduled macroeconomic release formatted for LiveONNX-EA."""

    symbol: str
    title: str
    description: str
    start_time: str
    end_time: str
    action: str
    trailing_points: int = 0


def get_default_mt5_common_path() -> Path:
    """Resolve the MetaTrader 5 Common Files directory path."""
    env_common = os.getenv("MT5_COMMON_PATH")
    if env_common and Path(env_common).exists():
        return Path(env_common) / "Files"

    appdata = os.getenv("APPDATA", "")
    if appdata:
        return Path(appdata) / "MetaQuotes" / "Terminal" / "Common" / "Files"
    return Path.home() / ".mt5_common" / "Files"


def format_ex_ante_desc(
    currency: str,
    event_name: str,
    prior: str,
    consensus: str,
    rationale: str,
    action_note: str,
) -> str:
    """Construct an ex-ante event description without lookahead bias."""
    return (
        f"[{currency}] {event_name} | Prior: {prior} | Consensus: {consensus}. "
        f"Ex-ante analysis: {rationale} "
        f"Protective rule: {action_note}"
    )


def convert_utc_to_mt5_time(
    dt_utc: datetime,
    buffer_before_minutes: int = 30,
    buffer_after_minutes: int = 45,
) -> tuple[str, str]:
    """Convert a UTC event datetime to MT5 Server Time (EET/EEST) with protective window."""
    dt_mt5 = dt_utc.astimezone(MT5_TZ)
    start_dt = dt_mt5 - timedelta(minutes=buffer_before_minutes)
    end_dt = dt_mt5 + timedelta(minutes=buffer_after_minutes)
    return (
        start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        end_dt.strftime("%Y-%m-%d %H:%M:%S"),
    )


def get_nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """Get the n-th occurrence of a weekday in a given month (0=Mon, ..., 4=Fri)."""
    first_day = date(year, month, 1)
    first_weekday = first_day.weekday()
    offset = (weekday - first_weekday) % 7
    day = 1 + offset + (n - 1) * 7
    return date(year, month, day)


def get_last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """Get the last occurrence of a weekday in a given month."""
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    last_day = next_month - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def generate_central_bank_events(start_date: date, end_date: date) -> List[CalendarEventRecord]:
    """Compile official Central Bank scheduled meetings and rate decisions.
    
    Ex-ante Action: TRAILING_STOP (120 points).
    Blocks new entries and tightens stop loss on open profitable positions to lock in
    accrued equity before monetary policy statements and rate surprise shocks.
    """
    events: List[CalendarEventRecord] = []
    trailing_pts = 120

    # FOMC (Federal Reserve - USD) Rate Decisions (Wednesdays at 18:00 UTC / 20:00/21:00 MT5)
    fomc_dates = [
        # 2025
        date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7), date(2025, 6, 18),
        date(2025, 7, 30), date(2025, 9, 17), date(2025, 10, 29), date(2025, 12, 10),
        # 2026
        date(2026, 1, 28), date(2026, 3, 18), date(2026, 5, 6), date(2026, 6, 17),
        date(2026, 7, 29), date(2026, 9, 16),
    ]
    for d in fomc_dates:
        if start_date <= d <= end_date:
            utc_dt = datetime.combine(d, time(18, 0), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, buffer_before_minutes=30, buffer_after_minutes=90)
            events.append(CalendarEventRecord(
                symbol="USD",
                title="US FOMC Interest Rate Decision & Press Conference",
                description=format_ex_ante_desc(
                    "USD", "FOMC Rate Decision", "Current Fed Funds Target", "Market consensus priced in",
                    "High liquidity shock expected; Powell press conference induces two-way volatility.",
                    "TRAILING_STOP (120 pts) enforced to defend open gains; new entries prohibited.",
                ),
                start_time=st,
                end_time=et,
                action="TRAILING_STOP",
                trailing_points=trailing_pts,
            ))

    # ECB (European Central Bank - EUR) Decisions (Thursdays at 12:15 UTC / 14:15/15:15 MT5)
    ecb_dates = [
        # 2025
        date(2025, 1, 30), date(2025, 3, 6), date(2025, 4, 17), date(2025, 6, 5),
        date(2025, 7, 24), date(2025, 9, 11), date(2025, 10, 30), date(2025, 12, 18),
        # 2026
        date(2026, 1, 29), date(2026, 3, 12), date(2026, 4, 30), date(2026, 6, 11),
        date(2026, 7, 23),
    ]
    for d in ecb_dates:
        if start_date <= d <= end_date:
            utc_dt = datetime.combine(d, time(12, 15), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, buffer_before_minutes=30, buffer_after_minutes=90)
            events.append(CalendarEventRecord(
                symbol="EUR",
                title="ECB Monetary Policy Decision & Lagarde Press Conference",
                description=format_ex_ante_desc(
                    "EUR", "ECB Rate Decision", "Deposit Facility Rate", "Policy rate hold/cut baseline",
                    "Disinflation trajectory in Eurozone evaluated alongside staff macroeconomic projections.",
                    "TRAILING_STOP (120 pts) enforced to defend open gains; new entries prohibited.",
                ),
                start_time=st,
                end_time=et,
                action="TRAILING_STOP",
                trailing_points=trailing_pts,
            ))

    # BOE (Bank of England - GBP) Decisions (Thursdays at 12:00 UTC / 14:00/15:00 MT5)
    boe_dates = [
        # 2025
        date(2025, 2, 6), date(2025, 3, 20), date(2025, 5, 8), date(2025, 6, 19),
        date(2025, 8, 7), date(2025, 9, 18), date(2025, 11, 6), date(2025, 12, 18),
        # 2026
        date(2026, 2, 5), date(2026, 3, 19), date(2026, 5, 7), date(2026, 6, 18),
        date(2026, 8, 6),
    ]
    for d in boe_dates:
        if start_date <= d <= end_date:
            utc_dt = datetime.combine(d, time(12, 0), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, buffer_before_minutes=30, buffer_after_minutes=60)
            events.append(CalendarEventRecord(
                symbol="GBP",
                title="BOE Official Bank Rate & Monetary Policy Summary",
                description=format_ex_ante_desc(
                    "GBP", "BOE Rate Decision", "Official Bank Rate", "MPC vote split anticipated",
                    "Services inflation and wage pressures under scrutiny by MPC committee.",
                    "TRAILING_STOP (120 pts) enforced to defend open gains; new entries prohibited.",
                ),
                start_time=st,
                end_time=et,
                action="TRAILING_STOP",
                trailing_points=trailing_pts,
            ))

    # BOJ (Bank of Japan - JPY) Decisions (03:00-04:30 UTC / 05:00-07:30 MT5)
    boj_dates = [
        # 2025
        date(2025, 1, 24), date(2025, 3, 19), date(2025, 5, 1), date(2025, 6, 13),
        date(2025, 7, 31), date(2025, 9, 19), date(2025, 10, 31), date(2025, 12, 19),
        # 2026
        date(2026, 1, 23), date(2026, 3, 19), date(2026, 4, 28), date(2026, 6, 16),
        date(2026, 7, 31),
    ]
    for d in boj_dates:
        if start_date <= d <= end_date:
            utc_dt = datetime.combine(d, time(3, 30), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, buffer_before_minutes=30, buffer_after_minutes=90)
            events.append(CalendarEventRecord(
                symbol="JPY",
                title="BOJ Monetary Policy Statement, Rate Decision & Outlook",
                description=format_ex_ante_desc(
                    "JPY", "BOJ Rate Decision", "Overnight Call Rate target", "Policy rate normalization path",
                    "Ueda press conference and yield curve control commentary drive sharp JPY volatility.",
                    "TRAILING_STOP (120 pts) enforced to defend open gains; new entries prohibited.",
                ),
                start_time=st,
                end_time=et,
                action="TRAILING_STOP",
                trailing_points=trailing_pts,
            ))

    # RBA (Reserve Bank of Australia - AUD) Decisions (Tuesdays at 03:30 UTC / 05:30/06:30 MT5)
    rba_dates = [
        # 2025
        date(2025, 2, 18), date(2025, 4, 1), date(2025, 5, 20), date(2025, 7, 8),
        date(2025, 8, 12), date(2025, 9, 30), date(2025, 11, 4), date(2025, 12, 9),
        # 2026
        date(2026, 2, 17), date(2026, 3, 31), date(2026, 5, 19), date(2026, 7, 7),
        date(2026, 8, 11),
    ]
    for d in rba_dates:
        if start_date <= d <= end_date:
            utc_dt = datetime.combine(d, time(3, 30), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, buffer_before_minutes=30, buffer_after_minutes=60)
            events.append(CalendarEventRecord(
                symbol="AUD",
                title="RBA Cash Rate Decision & Monetary Policy Statement",
                description=format_ex_ante_desc(
                    "AUD", "RBA Cash Rate", "Current Official Cash Rate", "Market expectation priced",
                    "Australian labor resilience vs inflation moderation evaluated by RBA board.",
                    "TRAILING_STOP (120 pts) enforced to defend open gains; new entries prohibited.",
                ),
                start_time=st,
                end_time=et,
                action="TRAILING_STOP",
                trailing_points=trailing_pts,
            ))

    # BOC (Bank of Canada - CAD) Decisions (Wednesdays at 13:45 UTC / 15:45/16:45 MT5)
    boc_dates = [
        # 2025
        date(2025, 1, 29), date(2025, 3, 12), date(2025, 4, 16), date(2025, 6, 4),
        date(2025, 7, 30), date(2025, 9, 10), date(2025, 10, 29), date(2025, 12, 10),
        # 2026
        date(2026, 1, 28), date(2026, 3, 11), date(2026, 4, 15), date(2026, 6, 3),
        date(2026, 7, 29),
    ]
    for d in boc_dates:
        if start_date <= d <= end_date:
            utc_dt = datetime.combine(d, time(13, 45), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, buffer_before_minutes=30, buffer_after_minutes=60)
            events.append(CalendarEventRecord(
                symbol="CAD",
                title="BOC Overnight Rate Decision & Monetary Policy Report",
                description=format_ex_ante_desc(
                    "CAD", "BOC Rate Decision", "Target Overnight Rate", "Monetary easing/neutral expectation",
                    "Canadian mortgage renewal cycle and headline CPI trends dictate policy.",
                    "TRAILING_STOP (120 pts) enforced to defend open gains; new entries prohibited.",
                ),
                start_time=st,
                end_time=et,
                action="TRAILING_STOP",
                trailing_points=trailing_pts,
            ))

    # SNB (Swiss National Bank - CHF) Quarterly Assessments (Thursdays at 07:30 UTC / 09:30/10:30 MT5)
    snb_dates = [
        # 2025
        date(2025, 3, 20), date(2025, 6, 19), date(2025, 9, 18), date(2025, 12, 11),
        # 2026
        date(2026, 3, 19), date(2026, 6, 18),
    ]
    for d in snb_dates:
        if start_date <= d <= end_date:
            utc_dt = datetime.combine(d, time(7, 30), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, buffer_before_minutes=30, buffer_after_minutes=60)
            events.append(CalendarEventRecord(
                symbol="CHF",
                title="SNB Monetary Policy Assessment & Policy Rate",
                description=format_ex_ante_desc(
                    "CHF", "SNB Policy Rate", "SNB Policy Rate", "Low inflation regime adjustments",
                    "SNB intervention readiness against franc overvaluation analyzed.",
                    "TRAILING_STOP (120 pts) enforced to defend open gains; new entries prohibited.",
                ),
                start_time=st,
                end_time=et,
                action="TRAILING_STOP",
                trailing_points=trailing_pts,
            ))

    # RBNZ (Reserve Bank of New Zealand - NZD) Decisions (Wednesdays at 01:00 UTC / 03:00/04:00 MT5)
    rbnz_dates = [
        # 2025
        date(2025, 2, 19), date(2025, 4, 9), date(2025, 5, 28), date(2025, 7, 9),
        date(2025, 8, 13), date(2025, 10, 8), date(2025, 11, 26),
        # 2026
        date(2026, 2, 18), date(2026, 4, 8), date(2026, 5, 27), date(2026, 7, 8),
        date(2026, 8, 12),
    ]
    for d in rbnz_dates:
        if start_date <= d <= end_date:
            utc_dt = datetime.combine(d, time(1, 0), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, buffer_before_minutes=30, buffer_after_minutes=60)
            events.append(CalendarEventRecord(
                symbol="NZD",
                title="RBNZ Official Cash Rate (OCR) & Monetary Policy Statement",
                description=format_ex_ante_desc(
                    "NZD", "RBNZ OCR", "Official Cash Rate", "Policy rate baseline",
                    "New Zealand economic growth and domestic inflation capacity reviewed.",
                    "TRAILING_STOP (120 pts) enforced to defend open gains; new entries prohibited.",
                ),
                start_time=st,
                end_time=et,
                action="TRAILING_STOP",
                trailing_points=trailing_pts,
            ))

    return events


def generate_recurring_macro_events(start_date: date, end_date: date) -> List[CalendarEventRecord]:
    """Generate recurring macroeconomic releases with ex-ante action classification."""
    events: List[CalendarEventRecord] = []

    cur_year = start_date.year
    cur_month = start_date.month

    while True:
        m_start = date(cur_year, cur_month, 1)
        if m_start > end_date:
            break

        # -------------------------------------------------------------
        # 1. UNITED STATES (USD)
        # -------------------------------------------------------------
        # US Non-Farm Payrolls & Unemployment Rate (1st Friday at 12:30 UTC)
        # Ex-ante Action: BREAKEVEN (moves winning trade SL to entry price to eliminate drawdown)
        nfp_date = get_nth_weekday_of_month(cur_year, cur_month, weekday=4, n=1)
        if start_date <= nfp_date <= end_date:
            utc_dt = datetime.combine(nfp_date, time(12, 30), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, 30, 60)
            events.append(CalendarEventRecord(
                symbol="USD",
                title="US Non-Farm Payrolls (NFP) & Unemployment Rate",
                description=format_ex_ante_desc(
                    "USD", "Non-Farm Payrolls", "+180K to +210K historical baseline", "Consensus: +160K to +190K",
                    "Highest monthly volatility catalyst in FX; extreme two-way whipsaw risk.",
                    "BREAKEVEN enforced to move profitable positions to entry price; new entries prohibited.",
                ),
                start_time=st,
                end_time=et,
                action="BREAKEVEN",
            ))

        # US ISM Manufacturing PMI (1st business day of month at 14:00 UTC) -> BLOCK_ENTRIES
        ism_mfg_date = date(cur_year, cur_month, 1)
        while ism_mfg_date.weekday() >= 5:
            ism_mfg_date += timedelta(days=1)
        if start_date <= ism_mfg_date <= end_date:
            utc_dt = datetime.combine(ism_mfg_date, time(14, 0), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, 15, 45)
            events.append(CalendarEventRecord(
                symbol="USD",
                title="US ISM Manufacturing PMI",
                description=format_ex_ante_desc(
                    "USD", "ISM Manufacturing PMI", "Prior reading ~49-51", "Consensus ~50.0",
                    "Leading indicator of US industrial economic health and new orders.",
                    "BLOCK_ENTRIES enforced to prevent execution slippage during release.",
                ),
                start_time=st,
                end_time=et,
                action="BLOCK_ENTRIES",
            ))

        # US CPI (Headline & Core m/m, y/y) (Mid-month Wednesday at 12:30 UTC) -> BLOCK_ENTRIES
        cpi_date = get_nth_weekday_of_month(cur_year, cur_month, weekday=2, n=2)
        if start_date <= cpi_date <= end_date:
            utc_dt = datetime.combine(cpi_date, time(12, 30), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, 30, 60)
            events.append(CalendarEventRecord(
                symbol="USD",
                title="US Consumer Price Index (CPI) & Core CPI",
                description=format_ex_ante_desc(
                    "USD", "CPI y/y & Core CPI", "Prior ~2.8%-3.2%", "Consensus expected at ~0.2%-0.3% MoM",
                    "Key inflation benchmark guiding Fed interest rate expectations.",
                    "BLOCK_ENTRIES enforced to preserve open GARCH stops while halting new entries.",
                ),
                start_time=st,
                end_time=et,
                action="BLOCK_ENTRIES",
            ))

        # US Core PCE Price Index (Last Friday of month at 12:30 UTC) -> BLOCK_ENTRIES
        pce_date = get_last_weekday_of_month(cur_year, cur_month, weekday=4)
        if start_date <= pce_date <= end_date:
            utc_dt = datetime.combine(pce_date, time(12, 30), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, 20, 45)
            events.append(CalendarEventRecord(
                symbol="USD",
                title="US Core PCE Price Index (Fed's Preferred Inflation Gauge)",
                description=format_ex_ante_desc(
                    "USD", "Core PCE m/m & y/y", "Prior ~2.6%-2.9%", "Consensus ~0.2% MoM",
                    "Federal Reserve target metric; directly informs FOMC trajectory.",
                    "BLOCK_ENTRIES enforced to avoid entry slippage during print.",
                ),
                start_time=st,
                end_time=et,
                action="BLOCK_ENTRIES",
            ))

        # -------------------------------------------------------------
        # 2. EUROZONE (EUR)
        # -------------------------------------------------------------
        # German Prelim CPI m/m & y/y (Around 28th-30th at 13:00 UTC) -> BLOCK_ENTRIES
        ger_cpi_day = min(29, 28 if cur_month == 2 else 29)
        ger_cpi_date = date(cur_year, cur_month, ger_cpi_day)
        while ger_cpi_date.weekday() >= 5:
            ger_cpi_date -= timedelta(days=1)
        if start_date <= ger_cpi_date <= end_date:
            utc_dt = datetime.combine(ger_cpi_date, time(13, 0), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, 20, 45)
            events.append(CalendarEventRecord(
                symbol="EUR",
                title="German Preliminary CPI m/m & y/y",
                description=format_ex_ante_desc(
                    "EUR", "German Prelim CPI", "Prior headline German inflation", "Consensus estimate",
                    "Leading component for Eurozone aggregate inflation print.",
                    "BLOCK_ENTRIES enforced to avoid liquidity gap entries.",
                ),
                start_time=st,
                end_time=et,
                action="BLOCK_ENTRIES",
            ))

        # Eurozone Flash CPI Estimate (1st week of month at 10:00 UTC) -> BLOCK_ENTRIES
        ez_cpi_date = date(cur_year, cur_month, min(3, 28))
        while ez_cpi_date.weekday() >= 5:
            ez_cpi_date += timedelta(days=1)
        if start_date <= ez_cpi_date <= end_date:
            utc_dt = datetime.combine(ez_cpi_date, time(10, 0), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, 20, 45)
            events.append(CalendarEventRecord(
                symbol="EUR",
                title="Eurozone CPI Flash Estimate y/y & Core CPI",
                description=format_ex_ante_desc(
                    "EUR", "Eurozone Flash CPI", "Prior ~2.2%-2.6%", "Consensus expected at target range",
                    "Primary catalyst for ECB rate path reassessment across European sessions.",
                    "BLOCK_ENTRIES enforced to protect against execution spread widening.",
                ),
                start_time=st,
                end_time=et,
                action="BLOCK_ENTRIES",
            ))

        # Eurozone Flash PMIs (Around the 22nd at 08:30 UTC) -> ADVISORY_ONLY
        # Moderate business survey; non-blocking advisory notification in MT5 experts journal.
        pmi_date = date(cur_year, cur_month, 22)
        while pmi_date.weekday() >= 5:
            pmi_date += timedelta(days=1)
        if start_date <= pmi_date <= end_date:
            utc_dt = datetime.combine(pmi_date, time(8, 30), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, 15, 30)
            events.append(CalendarEventRecord(
                symbol="EUR",
                title="Eurozone S&P Global Flash Manufacturing & Services PMI",
                description=format_ex_ante_desc(
                    "EUR", "Eurozone Flash PMIs", "Prior Manufacturing/Services", "Consensus ~48-52",
                    "Comprehensive European business activity barometer.",
                    "ADVISORY_ONLY logged for trader surveillance; execution unblocked.",
                ),
                start_time=st,
                end_time=et,
                action="ADVISORY_ONLY",
            ))

        # -------------------------------------------------------------
        # 3. UNITED KINGDOM (GBP)
        # -------------------------------------------------------------
        # UK CPI y/y & Core CPI (3rd Wednesday at 07:00 UTC) -> BLOCK_ENTRIES
        uk_cpi_date = get_nth_weekday_of_month(cur_year, cur_month, weekday=2, n=3)
        if start_date <= uk_cpi_date <= end_date:
            utc_dt = datetime.combine(uk_cpi_date, time(7, 0), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, 20, 45)
            events.append(CalendarEventRecord(
                symbol="GBP",
                title="UK Consumer Price Index (CPI) y/y & Core CPI",
                description=format_ex_ante_desc(
                    "GBP", "UK CPI y/y", "Prior ~2.5%-3.5%", "Consensus forecast",
                    "Key inflation driver for Bank of England Monetary Policy Committee.",
                    "BLOCK_ENTRIES enforced during European London open liquidity gap.",
                ),
                start_time=st,
                end_time=et,
                action="BLOCK_ENTRIES",
            ))

        # UK Employment Change & Claimant Count (2nd Tuesday at 07:00 UTC) -> BLOCK_ENTRIES
        uk_jobs_date = get_nth_weekday_of_month(cur_year, cur_month, weekday=1, n=2)
        if start_date <= uk_jobs_date <= end_date:
            utc_dt = datetime.combine(uk_jobs_date, time(7, 0), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, 20, 45)
            events.append(CalendarEventRecord(
                symbol="GBP",
                title="UK Labour Market Report (Claimant Count & Average Earnings)",
                description=format_ex_ante_desc(
                    "GBP", "UK Labour Data", "Prior wage growth ~4.5%-5.5%", "Consensus expected moderation",
                    "Persistent UK wage growth impacts BOE rate cut timeline.",
                    "BLOCK_ENTRIES enforced to prevent orders on initial wage release spikes.",
                ),
                start_time=st,
                end_time=et,
                action="BLOCK_ENTRIES",
            ))

        # -------------------------------------------------------------
        # 4. JAPAN (JPY)
        # -------------------------------------------------------------
        # National Core CPI y/y (3rd Friday at 23:30 UTC) -> BLOCK_ENTRIES
        jp_cpi_date = get_nth_weekday_of_month(cur_year, cur_month, weekday=4, n=3)
        if start_date <= jp_cpi_date <= end_date:
            utc_dt = datetime.combine(jp_cpi_date, time(23, 30), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, 20, 45)
            events.append(CalendarEventRecord(
                symbol="JPY",
                title="Japan National Core Consumer Price Index (CPI) y/y",
                description=format_ex_ante_desc(
                    "JPY", "Japan Core CPI", "Prior ~2.5%-2.8%", "Consensus expected above 2% target",
                    "Crucial for BOJ policy rate hike expectations and JPY carry trade dynamics.",
                    "BLOCK_ENTRIES enforced during Asian session liquidity thinness.",
                ),
                start_time=st,
                end_time=et,
                action="BLOCK_ENTRIES",
            ))

        # -------------------------------------------------------------
        # 5. AUSTRALIA (AUD)
        # -------------------------------------------------------------
        # Australia Employment Change & Unemployment Rate (3rd Thursday at 01:30 UTC) -> BLOCK_ENTRIES
        aud_jobs_date = get_nth_weekday_of_month(cur_year, cur_month, weekday=3, n=3)
        if start_date <= aud_jobs_date <= end_date:
            utc_dt = datetime.combine(aud_jobs_date, time(1, 30), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, 20, 45)
            events.append(CalendarEventRecord(
                symbol="AUD",
                title="Australia Employment Change & Unemployment Rate",
                description=format_ex_ante_desc(
                    "AUD", "Australia Employment Change", "Prior ~+25K to +45K", "Consensus ~+20K",
                    "Tight Australian labor conditions dictate RBA restrictive stance.",
                    "BLOCK_ENTRIES enforced to avoid thin Asian market slippage.",
                ),
                start_time=st,
                end_time=et,
                action="BLOCK_ENTRIES",
            ))

        # Australia Monthly CPI Indicator (Last Wednesday at 01:30 UTC) -> BLOCK_ENTRIES
        aud_cpi_date = get_last_weekday_of_month(cur_year, cur_month, weekday=2)
        if start_date <= aud_cpi_date <= end_date:
            utc_dt = datetime.combine(aud_cpi_date, time(1, 30), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, 20, 45)
            events.append(CalendarEventRecord(
                symbol="AUD",
                title="Australia Monthly CPI Indicator y/y",
                description=format_ex_ante_desc(
                    "AUD", "Australia CPI Indicator", "Prior ~3.5%-3.8%", "Consensus forecast",
                    "Monthly inflation update directly shifting Australian bond yields and AUD pairs.",
                    "BLOCK_ENTRIES enforced during early Asian session window.",
                ),
                start_time=st,
                end_time=et,
                action="BLOCK_ENTRIES",
            ))

        # -------------------------------------------------------------
        # 6. CANADA (CAD)
        # -------------------------------------------------------------
        # Canada Employment Change & Unemployment Rate (1st or 2nd Friday at 12:30 UTC) -> BLOCK_ENTRIES
        cad_jobs_date = get_nth_weekday_of_month(cur_year, cur_month, weekday=4, n=1)
        if start_date <= cad_jobs_date <= end_date:
            utc_dt = datetime.combine(cad_jobs_date, time(12, 30), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, 20, 45)
            events.append(CalendarEventRecord(
                symbol="CAD",
                title="Canada Employment Change & Unemployment Rate",
                description=format_ex_ante_desc(
                    "CAD", "Canada Labour Market", "Prior ~+15K to +35K", "Consensus estimate",
                    "Labour force participation and wage growth analyzed alongside US NFP.",
                    "BLOCK_ENTRIES enforced alongside concurrent US economic releases.",
                ),
                start_time=st,
                end_time=et,
                action="BLOCK_ENTRIES",
            ))

        # Canada CPI m/m & y/y (3rd Tuesday at 12:30 UTC) -> BLOCK_ENTRIES
        cad_cpi_date = get_nth_weekday_of_month(cur_year, cur_month, weekday=1, n=3)
        if start_date <= cad_cpi_date <= end_date:
            utc_dt = datetime.combine(cad_cpi_date, time(12, 30), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, 20, 45)
            events.append(CalendarEventRecord(
                symbol="CAD",
                title="Canada Consumer Price Index (CPI) m/m & Trimmed CPI",
                description=format_ex_ante_desc(
                    "CAD", "Canada CPI", "Prior ~2.5%-2.9%", "Consensus expectation",
                    "Core median and trimmed mean inflation guide BOC policy trajectory.",
                    "BLOCK_ENTRIES enforced to avoid North American pre-market spread widening.",
                ),
                start_time=st,
                end_time=et,
                action="BLOCK_ENTRIES",
            ))

        # -------------------------------------------------------------
        # 7. SWITZERLAND (CHF)
        # -------------------------------------------------------------
        # Swiss CPI m/m & y/y (1st week of month at 06:30 UTC) -> BLOCK_ENTRIES
        chf_cpi_date = date(cur_year, cur_month, min(4, 28))
        while chf_cpi_date.weekday() >= 5:
            chf_cpi_date += timedelta(days=1)
        if start_date <= chf_cpi_date <= end_date:
            utc_dt = datetime.combine(chf_cpi_date, time(6, 30), tzinfo=UTC_TZ)
            st, et = convert_utc_to_mt5_time(utc_dt, 20, 45)
            events.append(CalendarEventRecord(
                symbol="CHF",
                title="Switzerland Consumer Price Index (CPI) m/m & y/y",
                description=format_ex_ante_desc(
                    "CHF", "Switzerland CPI", "Prior ~1.1%-1.4%", "Consensus expected ~0.0%-0.1% MoM",
                    "Low Swiss inflation profile maintains SNB dovish bias.",
                    "BLOCK_ENTRIES enforced to protect against early morning CHF illiquidity.",
                ),
                start_time=st,
                end_time=et,
                action="BLOCK_ENTRIES",
            ))

        # -------------------------------------------------------------
        # 8. NEW ZEALAND (NZD)
        # -------------------------------------------------------------
        # Global Dairy Trade (GDT) Price Index (1st & 3rd Tuesday at 15:00 UTC) -> ADVISORY_ONLY
        # Commodity auction with gradual multi-hour settlement; logged for awareness without blocking.
        gdt_1 = get_nth_weekday_of_month(cur_year, cur_month, weekday=1, n=1)
        gdt_2 = get_nth_weekday_of_month(cur_year, cur_month, weekday=1, n=3)
        for gdt_d in (gdt_1, gdt_2):
            if start_date <= gdt_d <= end_date:
                utc_dt = datetime.combine(gdt_d, time(15, 0), tzinfo=UTC_TZ)
                st, et = convert_utc_to_mt5_time(utc_dt, 15, 30)
                events.append(CalendarEventRecord(
                    symbol="NZD",
                    title="Global Dairy Trade (GDT) Price Index Event",
                    description=format_ex_ante_desc(
                        "NZD", "GDT Price Index", "Prior dairy auction change", "Auction price consensus",
                        "Primary agricultural export of New Zealand affecting terms of trade.",
                        "ADVISORY_ONLY logged for informational auditing; orders unblocked.",
                    ),
                    start_time=st,
                    end_time=et,
                    action="ADVISORY_ONLY",
                ))

        # Advance to next month
        if cur_month == 12:
            cur_year += 1
            cur_month = 1
        else:
            cur_month += 1

    return events


def populate_sqlite_database(
    events: List[CalendarEventRecord],
    db_path: Path,
) -> None:
    """Initialize schema, insert calendar events, and ensure empty news table."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA synchronous=NORMAL;")

        # Create calendar_events table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS calendar_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                action TEXT NOT NULL DEFAULT 'BLOCK_ENTRIES',
                trailing_points INTEGER NOT NULL DEFAULT 0
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cal_lookup
            ON calendar_events (symbol, start_time, end_time);
        """)

        # Create news_events table (kept strictly empty for Strategy Tester parity)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_events (
                symbol TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                action TEXT NOT NULL DEFAULT 'BLOCK_ENTRIES',
                trailing_points INTEGER NOT NULL DEFAULT 0
            );
        """)

        # Clear existing calendar events to ensure clean state
        conn.execute("DELETE FROM calendar_events;")
        conn.execute("DELETE FROM news_events;")

        # Insert events sorted chronologically
        events_sorted = sorted(events, key=lambda e: e.start_time)
        insert_rows = [
            (e.symbol, e.title, e.description, e.start_time, e.end_time, e.action, e.trailing_points)
            for e in events_sorted
        ]
        conn.executemany(
            """
            INSERT INTO calendar_events
            (symbol, title, description, start_time, end_time, action, trailing_points)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            insert_rows,
        )
        conn.commit()


def verify_database_integrity(db_path: Path) -> bool:
    """Run PRAGMA integrity_check on the database."""
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.execute("PRAGMA integrity_check;")
        res = cursor.fetchone()
        return bool(res and res[0] == "ok")


def main() -> None:
    """CLI runner to generate and populate the institutional economic calendar."""
    parser = argparse.ArgumentParser(description="Institutional Economic Calendar Generator for MT5")
    parser.add_argument("--start", type=str, default="2025-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-09-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--db-path", type=str, default=None, help="Custom target SQLite DB path")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    target_db = Path(args.db_path) if args.db_path else get_default_mt5_common_path() / "macro_governance.db"

    print("=" * 80)
    print(f"[*] Generating Institutional Economic Calendar: {start_date} to {end_date}")
    print(f"[*] Target Currencies: EUR, USD, JPY, GBP, AUD, CAD, CHF, NZD (8 G8 currencies)")
    print(f"[*] Destination Database: {target_db}")
    print(f"[*] Timezone: MT5 Server Time (EET/EEST - Europe/Athens)")
    print("=" * 80)

    # 1. Generate Central Bank decisions (Action: TRAILING_STOP 120 pts)
    cb_events = generate_central_bank_events(start_date, end_date)
    print(f"[+] Central Bank Events Generated (TRAILING_STOP 120 pts): {len(cb_events)}")

    # 2. Generate Recurring Macroeconomic Catalysts (BREAKEVEN, BLOCK_ENTRIES, ADVISORY_ONLY)
    macro_events = generate_recurring_macro_events(start_date, end_date)
    
    action_counts = {}
    for ev in cb_events + macro_events:
        action_counts[ev.action] = action_counts.get(ev.action, 0) + 1
    for act, cnt in action_counts.items():
        print(f"    - Action '{act}': {cnt} events")

    all_events = cb_events + macro_events
    print(f"[+] Total Ex-Ante Scheduled Releases: {len(all_events)}")

    # 3. Populate SQLite database
    populate_sqlite_database(all_events, target_db)
    print(f"[+] Successfully written to SQLite database: {target_db}")

    # 4. Integrity Check
    is_valid = verify_database_integrity(target_db)
    print(f"[*] SQLite Integrity Verification: {'VALID (ok)' if is_valid else 'CORRUPTED'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
