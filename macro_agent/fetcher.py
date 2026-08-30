"""Macroeconomic Data & News Collector for AI CLI Agent.

Queries open macroeconomic calendar feeds and financial news sources,
formatting the raw events into structured JSON for the AI agent to evaluate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
from typing import Any, Dict, List, Optional
import urllib.request


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MT5-FX-Quant/1.0"

HIGH_IMPACT_CATALYSTS: Dict[str, List[str]] = {
    "USD": [
        "Non-Farm Payrolls", "FOMC Rate Decision", "CPI", "Core PCE",
        "GDP", "ISM Manufacturing", "Jackson Hole"
    ],
    "EUR": [
        "ECB Rate Decision", "CPI Flash Estimate", "German Prelim CPI",
        "Monetary Policy Statement", "Eurozone GDP"
    ],
    "GBP": [
        "BOE Official Bank Rate", "CPI y/y", "Monetary Policy Summary", "GDP m/m"
    ],
    "JPY": [
        "BOJ Policy Rate", "BOJ Monetary Policy Statement", "National Core CPI"
    ],
    "CHF": ["SNB Policy Rate", "CPI m/m"],
    "CAD": ["BOC Rate Decision", "Employment Change", "CPI m/m"],
    "AUD": ["RBA Cash Rate", "Employment Change", "CPI q/q"],
    "NZD": ["RBNZ Official Cash Rate", "CPI q/q"],
}


def extract_currencies_from_symbol(symbol: str) -> List[str]:
    """Split symbol (e.g. 'EURUSD') into individual currency components ['EUR', 'USD']."""
    sym = symbol.upper().replace("/", "").replace(".", "").replace("-", "").strip()
    if len(sym) >= 6:
        return [sym[:3], sym[3:6]]
    return [sym]


def fetch_mql5_calendar(currencies: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Fetch live economic calendar events from the public MQL5 Economic Calendar."""
    url = "https://www.mql5.com/en/economic-calendar"
    events: List[Dict[str, Any]] = []
    curr_filter = [c.upper() for c in currencies] if currencies else None

    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=8) as response:
            html = response.read().decode("utf-8", errors="ignore")
            # Parse tabular calendar rows: YYYY.MM.DD HH:MM, CUR, Event Name
            pattern = r"(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}),\s*([A-Z]{3}),\s*([^,\n<]+)"
            matches = re.findall(pattern, html)
            for match in matches:
                event_time, event_curr, event_name = match[0].strip(), match[1].strip(), match[2].strip()
                if curr_filter and event_curr not in curr_filter:
                    continue

                # Check if matches high impact definition
                catalysts = HIGH_IMPACT_CATALYSTS.get(event_curr, [])
                is_high = any(c.lower() in event_name.lower() for c in catalysts)

                # Standardize datetime into YYYY-MM-DD HH:MM:00
                iso_time = event_time.replace(".", "-") + ":00"

                events.append({
                    "datetime": iso_time,
                    "currency": event_curr,
                    "event_name": event_name,
                    "importance": "HIGH" if is_high else "MEDIUM",
                    "source": "mql5.com/en/economic-calendar",
                })
    except Exception as err:
        events.append({"error": f"Failed to fetch MQL5 Calendar: {err}"})

    return events


def fetch_open_news_headlines(currencies: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Fetch recent market news headlines from public RSS feeds."""
    feed_url = "https://www.dailyfx.com/feeds/forex-market-news"
    headlines: List[Dict[str, Any]] = []
    curr_filter = [c.upper() for c in currencies] if currencies else None

    try:
        req = urllib.request.Request(feed_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=6) as response:
            content = response.read().decode("utf-8", errors="ignore")
            # Simple title tag regex for RSS
            item_pattern = r"<item>(.*?)</item>"
            items = re.findall(item_pattern, content, re.DOTALL)
            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            for item in items:
                title_match = (
                    re.search(r"<title><!\[CDATA\[(.*?)\]\]></title>", item)
                    or re.search(r"<title>(.*?)</title>", item)
                )
                desc_match = (
                    re.search(r"<description><!\[CDATA\[(.*?)\]\]></description>", item)
                    or re.search(r"<description>(.*?)</description>", item)
                )

                if title_match:
                    title_text = title_match.group(1).strip()
                    desc_text = desc_match.group(1).strip() if desc_match else ""

                    # Check currency relevance
                    matched_currencies = []
                    search_space = f"{title_text} {desc_text}".upper()
                    all_currs = curr_filter or list(HIGH_IMPACT_CATALYSTS.keys())
                    for c in all_currs:
                        if c in search_space:
                            matched_currencies.append(c)

                    if matched_currencies or not curr_filter:
                        headlines.append({
                            "title": title_text,
                            "description": desc_text[:200] + ("..." if len(desc_text) > 200 else ""),
                            "matched_currencies": matched_currencies,
                            "fetched_at": now_iso,
                            "source": "dailyfx.com/feeds/forex-market-news",
                        })
    except Exception as err:
        headlines.append({"error": f"Failed to fetch News RSS: {err}"})

    return headlines


def main() -> None:
    """CLI runner returning JSON structured macro & news metadata."""
    parser = argparse.ArgumentParser(description="Macroeconomic Calendar & News Collector")
    parser.add_argument("--symbol", type=str, help="Symbol pair (e.g. EURUSD)")
    parser.add_argument("--currency", action="append", help="Currency code filter (e.g. USD, EUR)")
    parser.add_argument("--calendar", action="store_true", help="Fetch economic calendar events")
    parser.add_argument("--news", action="store_true", help="Fetch breaking market news")
    parser.add_argument("--all", action="store_true", help="Fetch both calendar and news")

    args = parser.parse_args()

    currencies: List[str] = []
    if args.symbol:
        currencies.extend(extract_currencies_from_symbol(args.symbol))
    if args.currency:
        currencies.extend([c.upper() for c in args.currency])

    currencies = list(set(currencies)) if currencies else ["USD", "EUR"]

    output: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "target_currencies": currencies,
    }

    fetch_both = args.all or (not args.calendar and not args.news)

    if fetch_both or args.calendar:
        output["calendar_events"] = fetch_mql5_calendar(currencies)
    if fetch_both or args.news:
        output["news_headlines"] = fetch_open_news_headlines(currencies)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
