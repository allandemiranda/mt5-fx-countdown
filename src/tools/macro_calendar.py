"""Macroeconomic & Economic Calendar Diagnostic Utility & Native MCP Server.

Provides date-based economic event lookups and an institutional Stdio MCP server
to cross-reference MT5 Strategy Tester backtest drawdowns, volatility shocks,
and execution slippage against the MQL5 Economic Calendar.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
import sys
from typing import Any, Dict, List, Optional
import urllib.request


@dataclass(frozen=True)
class EconomicEvent:
    """Historical or scheduled macroeconomic event."""

    event_id: str
    date: str
    time_utc: str
    currency: str
    event_name: str
    impact: str  # 'HIGH', 'MEDIUM', 'LOW'
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None


class MacroeconomicCalendar:
    """Utility to query macroeconomic announcements for Forex volatility analysis."""

    # Key recurring high-impact Forex releases by currency
    HIGH_IMPACT_EVENTS = {
        "USD": ["Non-Farm Payrolls", "FOMC Rate Decision", "CPI m/m", "CPI y/y", "Jackson Hole", "Core PCE"],
        "EUR": ["ECB Rate Decision", "CPI Flash Estimate", "Monetary Policy Statement", "German Prelim CPI"],
        "GBP": ["BOE Official Bank Rate", "CPI y/y", "Monetary Policy Summary", "GDP m/m"],
        "JPY": ["BOJ Policy Rate", "BOJ Monetary Policy Statement", "National Core CPI y/y"],
        "AUD": ["RBA Cash Rate", "Employment Change", "CPI q/q"],
        "CAD": ["BOC Rate Decision", "Employment Change", "CPI m/m"],
        "CHF": ["SNB Policy Rate", "CPI m/m"],
        "NZD": ["RBNZ Official Cash Rate", "Employment Change q/q", "CPI q/q"],
    }

    def __init__(self) -> None:
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MT5-FX-Quant/1.0"

    def fetch_open_economic_feed(self, currency: Optional[str] = None) -> List[EconomicEvent]:
        """Fetch macroeconomic news and economic calendar events from open public RSS feeds."""
        events: List[EconomicEvent] = []
        feed_url = "https://www.dailyfx.com/feeds/forex-market-news"
        try:
            req = urllib.request.Request(feed_url, headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req, timeout=5) as response:
                content = response.read().decode("utf-8", errors="ignore")
                currencies_to_check = [currency] if currency else list(self.HIGH_IMPACT_EVENTS.keys())
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

                for curr in currencies_to_check:
                    for key_event in self.HIGH_IMPACT_EVENTS.get(curr, []):
                        if key_event.lower() in content.lower():
                            events.append(
                                EconomicEvent(
                                    event_id=f"{curr}_{key_event}_{now_str}",
                                    date=now_str,
                                    time_utc="12:00",
                                    currency=curr,
                                    event_name=key_event,
                                    impact="HIGH",
                                )
                            )
        except Exception:
            pass
        return events

    def get_high_impact_definitions(self, currency: str) -> List[str]:
        """Return list of major volatility catalyst events for the given currency."""
        return self.HIGH_IMPACT_EVENTS.get(currency.upper(), [])

    def fetch_mql5_calendar_events(self, currency: Optional[str] = None) -> List[Dict[str, str]]:
        """Fetch live economic calendar events from the authoritative MQL5 Economic Calendar."""
        url = "https://www.mql5.com/en/economic-calendar"
        events: List[Dict[str, str]] = []
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req, timeout=6) as response:
                html = response.read().decode("utf-8", errors="ignore")
                # Extract event rows from MQL5 calendar table pattern: YYYY.MM.DD HH:MM, CUR, Event Name
                pattern = r"(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}),\s*([A-Z]{3}),\s*([^,\n]+)"
                matches = re.findall(pattern, html)
                for match in matches:
                    event_time, event_curr, event_name = match[0], match[1], match[2].strip()
                    if currency and event_curr.upper() != currency.upper():
                        continue
                    events.append({
                        "datetime_utc": event_time,
                        "currency": event_curr,
                        "event": event_name,
                        "source": "mql5.com/en/economic-calendar",
                    })
        except Exception as err:
            events.append({"error": f"Failed to fetch MQL5 Economic Calendar: {err}"})
        return events

    def audit_backtest_anomaly(self, timestamp: str, symbol_or_currency: str) -> Dict[str, Any]:
        """Correlate a backtest drawdown timestamp with high-impact macroeconomic event windows."""
        clean_curr = symbol_or_currency.upper().replace("/", "")[:3]
        target_ts = timestamp.strip()
        probable_catalysts = []

        # Analyze recurring institutional release patterns by day of week and UTC hour
        try:
            # Parse timestamp if formatted as YYYY-MM-DD HH:MM or YYYY.MM.DD HH:MM
            normalized_ts = target_ts.replace(".", "-")
            dt = datetime.fromisoformat(normalized_ts[:16])
            weekday = dt.weekday()  # 0: Mon, ..., 4: Fri
            day_of_month = dt.day
            hour_utc = dt.hour

            # 1. Non-Farm Payrolls (NFP): 1st Friday of month between 12:00 and 14:00 UTC
            if clean_curr in ("USD", "EUR") and weekday == 4 and 1 <= day_of_month <= 7 and 12 <= hour_utc <= 14:
                probable_catalysts.append({
                    "event": "US Non-Farm Payrolls (NFP)",
                    "impact": "HIGH",
                    "window": "First Friday 12:30/13:30 UTC",
                })

            # 2. FOMC Rate Decision: Wednesdays between 18:00 and 20:00 UTC
            if clean_curr in ("USD", "EUR") and weekday == 2 and 18 <= hour_utc <= 20:
                probable_catalysts.append({
                    "event": "US FOMC Interest Rate Decision & Press Conference",
                    "impact": "HIGH",
                    "window": "Wednesday 18:00-19:30 UTC",
                })

            # 3. ECB Rate Decision: Thursdays between 12:00 and 14:00 UTC
            if clean_curr in ("EUR", "USD") and weekday == 3 and 12 <= hour_utc <= 14:
                probable_catalysts.append({
                    "event": "ECB Monetary Policy Decision & Press Conference",
                    "impact": "HIGH",
                    "window": "Thursday 12:15-13:30 UTC",
                })

            # 4. US CPI releases: Mid-month weekdays at 12:30 UTC
            if clean_curr in ("USD", "EUR") and 10 <= day_of_month <= 16 and 12 <= hour_utc <= 14:
                probable_catalysts.append({
                    "event": "US Consumer Price Index (CPI)",
                    "impact": "HIGH",
                    "window": "Mid-month 12:30/13:30 UTC",
                })
        except Exception:
            pass

        if probable_catalysts:
            classification = "Class A: Exogenous Macro Shock"
        else:
            classification = "Class B: Endogenous Model Degradation"

        return {
            "timestamp": timestamp,
            "currency": clean_curr,
            "classification": classification,
            "matched_catalysts": probable_catalysts,
            "reference_calendar": "https://www.mql5.com/en/economic-calendar",
            "recommendation": (
                "Verify spread widening at timestamp. "
                "If spread expanded > 3x, EA executed normally under liquidity shock."
                if probable_catalysts
                else "No high-impact release matched. Review XGBoost probability thresholds and dynamic GARCH risk."
            ),
        }


def handle_jsonrpc_request(req: Dict[str, Any], calendar: MacroeconomicCalendar) -> Optional[Dict[str, Any]]:
    """Process single JSON-RPC 2.0 MCP request and return response object."""
    req_id = req.get("id")
    method = req.get("method", "")
    params = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "economic-calendar",
                    "version": "1.0.0",
                },
            },
        }

    if method == "notifications/initialized":
        return None

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "get_mql5_economic_calendar",
                        "description": (
                            "Fetch live events from the authoritative MQL5 Economic Calendar "
                            "(mql5.com/en/economic-calendar)."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "currency": {"type": "string", "description": "Currency code (e.g. USD, EUR, GBP, JPY)"}
                            },
                        },
                    },
                    {
                        "name": "get_economic_news",
                        "description": "Fetch recent economic news and open market feed headlines.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "currency": {
                                    "type": "string",
                                    "description": "Optional currency filter (e.g. USD, EUR)",
                                }
                            },
                        },
                    },
                    {
                        "name": "get_high_impact_catalysts",
                        "description": "Get predefined high-impact volatility catalyst events for a given currency.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "currency": {"type": "string", "description": "Currency code (e.g. USD, EUR, GBP, JPY)"}
                            },
                            "required": ["currency"],
                        },
                    },
                    {
                        "name": "audit_backtest_anomaly",
                        "description": (
                            "Cross-reference a backtest drawdown timestamp with high-impact economic releases "
                            "to classify exogenous shock vs model degradation."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "timestamp": {
                                    "type": "string",
                                    "description": "Anomaly timestamp (e.g. 2024-05-03 12:30)",
                                },
                                "currency": {
                                    "type": "string",
                                    "description": "Currency code or pair (e.g. USD, EURUSD)",
                                },
                            },
                            "required": ["timestamp", "currency"],
                        },
                    },
                ]
            },
        }

    if method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "get_mql5_economic_calendar":
            res = calendar.fetch_mql5_calendar_events(args.get("currency"))
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}], "isError": False},
            }

        if tool_name == "get_economic_news":
            events = calendar.fetch_open_economic_feed(args.get("currency"))
            res_data = [
                {"currency": e.currency, "event": e.event_name, "impact": e.impact, "date": e.date}
                for e in events
            ]
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(res_data, indent=2)}], "isError": False},
            }

        if tool_name == "get_high_impact_catalysts":
            curr = args.get("currency", "USD")
            catalysts = calendar.get_high_impact_definitions(curr)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({"currency": curr, "catalysts": catalysts}, indent=2),
                    }],
                    "isError": False,
                },
            }

        if tool_name == "audit_backtest_anomaly":
            audit = calendar.audit_backtest_anomaly(args.get("timestamp", ""), args.get("currency", "USD"))
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(audit, indent=2)}], "isError": False},
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Tool not found: {tool_name}"},
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def run_mcp_server() -> None:
    """Run persistent JSON-RPC 2.0 MCP server listening over Stdio."""
    calendar = MacroeconomicCalendar()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_jsonrpc_request(req, calendar)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
        except Exception as err:
            err_resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(err)}}
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


def run_cli() -> None:
    """CLI runner or MCP entrypoint for Macroeconomic Calendar tool."""
    if "--mcp" in sys.argv:
        run_mcp_server()
        return

    calendar = MacroeconomicCalendar()
    if len(sys.argv) > 1:
        currency = sys.argv[1].upper()
        events = calendar.get_high_impact_definitions(currency)
        print(json.dumps({"currency": currency, "high_impact_catalysts": events}, indent=2))
    else:
        print(json.dumps({"currencies": list(MacroeconomicCalendar.HIGH_IMPACT_EVENTS.keys())}, indent=2))


if __name__ == "__main__":
    run_cli()
