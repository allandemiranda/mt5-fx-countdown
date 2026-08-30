"""MetaTrader 5 (MT5) Local Diagnostic Utility & Native MCP Server.

Provides direct, low-latency, read-only diagnostic inspection of the local
MetaTrader 5 terminal, trading account, market rates, ticks, active positions,
and order viability via the Model Context Protocol (MCP) over Stdio JSON-RPC 2.0.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import sys
from typing import Any, Dict, List, Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from src.config import AppConfig


TIMEFRAME_MAP: Dict[str, int] = {}
if mt5 is not None:
    TIMEFRAME_MAP = {
        "M1": mt5.TIMEFRAME_M1,
        "M2": mt5.TIMEFRAME_M2,
        "M3": mt5.TIMEFRAME_M3,
        "M4": mt5.TIMEFRAME_M4,
        "M5": mt5.TIMEFRAME_M5,
        "M6": mt5.TIMEFRAME_M6,
        "M10": mt5.TIMEFRAME_M10,
        "M12": mt5.TIMEFRAME_M12,
        "M15": mt5.TIMEFRAME_M15,
        "M20": mt5.TIMEFRAME_M20,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H2": mt5.TIMEFRAME_H2,
        "H3": mt5.TIMEFRAME_H3,
        "H4": mt5.TIMEFRAME_H4,
        "H6": mt5.TIMEFRAME_H6,
        "H8": mt5.TIMEFRAME_H8,
        "H12": mt5.TIMEFRAME_H12,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }


class LocalMT5Client:
    """Local wrapper for MetaTrader 5 terminal interaction and diagnostic queries."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig.from_env()
        self._connected = False

    def ensure_connected(self) -> bool:
        """Ensure connection to the local MT5 terminal."""
        if mt5 is None:
            return False

        term_info = mt5.terminal_info()
        if term_info is not None and term_info.connected:
            self._connected = True
            return True

        # Initialize with configured executable path if available
        init_kwargs: Dict[str, Any] = {}
        if self.config.mt5_path and self.config.mt5_path.exists():
            init_kwargs["path"] = str(self.config.mt5_path)

        self._connected = bool(mt5.initialize(**init_kwargs))
        return self._connected

    def get_terminal_info(self) -> Dict[str, Any]:
        """Query terminal version, build, paths, and connectivity status."""
        if not self.ensure_connected():
            return {"error": "MT5 terminal not available or initialization failed", "connected": False}

        info = mt5.terminal_info()
        if info is None:
            return {"error": "Failed to retrieve terminal info", "connected": False}

        version = mt5.version()
        return {
            "connected": bool(info.connected),
            "trade_allowed": bool(info.trade_allowed),
            "version": version[0] if version else None,
            "build": version[1] if version else None,
            "release_date": version[2] if version else None,
            "path": info.path,
            "data_path": info.data_path,
            "commondata_path": info.commondata_path,
            "company": info.company,
            "name": info.name,
            "ping_last": info.ping_last,
        }

    def get_account_info(self) -> Dict[str, Any]:
        """Query account balance, equity, leverage, margin, and server details."""
        if not self.ensure_connected():
            return {"error": "MT5 terminal not available or initialization failed"}

        acc = mt5.account_info()
        if acc is None:
            return {"error": "Failed to retrieve account info"}

        return {
            "login": acc.login,
            "trade_mode": "DEMO" if acc.trade_mode == 0 else ("REAL" if acc.trade_mode == 2 else "CONTEST"),
            "server": acc.server,
            "currency": acc.currency,
            "leverage": acc.leverage,
            "balance": acc.balance,
            "equity": acc.equity,
            "profit": acc.profit,
            "margin": acc.margin,
            "margin_free": acc.margin_free,
            "margin_level": acc.margin_level,
            "margin_so_call": acc.margin_so_call,
            "margin_so_so": acc.margin_so_so,
            "company": acc.company,
            "name": acc.name,
        }

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Query market specifications, bid/ask prices, spread, and sizing rules."""
        if not self.ensure_connected():
            return {"error": "MT5 terminal not available"}

        sym = symbol.strip().upper()
        if not mt5.symbol_select(sym, True):
            return {"error": f"Symbol '{sym}' could not be selected"}

        info = mt5.symbol_info(sym)
        if info is None:
            return {"error": f"Failed to retrieve info for symbol '{sym}'"}

        return {
            "symbol": info.name,
            "bid": info.bid,
            "ask": info.ask,
            "spread": info.spread,
            "digits": info.digits,
            "point": info.point,
            "trade_mode": info.trade_mode,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "contract_size": info.trade_contract_size,
            "currency_base": info.currency_base,
            "currency_profit": info.currency_profit,
            "currency_margin": info.currency_margin,
            "margin_initial": info.margin_initial,
            "session_deals": info.session_deals,
            "session_buy_orders": info.session_buy_orders,
            "session_sell_orders": info.session_sell_orders,
        }

    def get_rates(self, symbol: str, timeframe: str = "H1", count: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent historical OHLCV bars formatted with ISO and MT5 server timestamps."""
        if not self.ensure_connected():
            return []

        sym = symbol.strip().upper()
        tf_code = TIMEFRAME_MAP.get(timeframe.strip().upper(), mt5.TIMEFRAME_H1 if mt5 else 16385)
        rates = mt5.copy_rates_from_pos(sym, tf_code, 0, min(max(1, count), 1000))
        if rates is None or len(rates) == 0:
            return []

        result: List[Dict[str, Any]] = []
        for r in rates:
            dt = datetime.fromtimestamp(r["time"], tz=timezone.utc)
            result.append({
                "time": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "tick_volume": int(r["tick_volume"]),
                "spread": int(r["spread"]),
            })
        return result

    def get_ticks(self, symbol: str, count: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent market price ticks for high-frequency microstructure inspection."""
        if not self.ensure_connected():
            return []

        sym = symbol.strip().upper()
        time_from = datetime.now(timezone.utc) - timedelta(minutes=10)
        ticks = mt5.copy_ticks_from(sym, time_from, min(count, 500), mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            return []

        result: List[Dict[str, Any]] = []
        for t in ticks[-count:]:
            dt = datetime.fromtimestamp(t["time"], tz=timezone.utc)
            result.append({
                "time": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "bid": float(t["bid"]),
                "ask": float(t["ask"]),
                "last": float(t["last"]),
                "volume": float(t["volume"]),
                "flags": int(t["flags"]),
            })
        return result

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve open positions with live profit, stop levels, and magic numbers."""
        if not self.ensure_connected():
            return []

        positions = mt5.positions_get(symbol=symbol.strip().upper()) if symbol else mt5.positions_get()
        if positions is None:
            return []

        result: List[Dict[str, Any]] = []
        for p in positions:
            dt_open = datetime.fromtimestamp(p.time, tz=timezone.utc)
            result.append({
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "BUY" if p.type == 0 else "SELL",
                "volume": p.volume,
                "price_open": p.price_open,
                "sl": p.sl,
                "tp": p.tp,
                "price_current": p.price_current,
                "profit": p.profit,
                "swap": p.swap,
                "magic": p.magic,
                "comment": p.comment,
                "time_open": dt_open.strftime("%Y-%m-%d %H:%M:%S"),
            })
        return result

    def get_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve active pending orders."""
        if not self.ensure_connected():
            return []

        orders = mt5.orders_get(symbol=symbol.strip().upper()) if symbol else mt5.orders_get()
        if orders is None:
            return []

        result: List[Dict[str, Any]] = []
        for o in orders:
            dt_setup = datetime.fromtimestamp(o.time_setup, tz=timezone.utc)
            result.append({
                "ticket": o.ticket,
                "symbol": o.symbol,
                "type": o.type,
                "volume_initial": o.volume_initial,
                "volume_current": o.volume_current,
                "price_open": o.price_open,
                "sl": o.sl,
                "tp": o.tp,
                "magic": o.magic,
                "comment": o.comment,
                "time_setup": dt_setup.strftime("%Y-%m-%d %H:%M:%S"),
            })
        return result

    def get_history_deals(self, days: int = 7, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve closed trade deals with Golden Rule Net Liquid Profit auditing."""
        if not self.ensure_connected():
            return []

        from_date = datetime.now(timezone.utc) - timedelta(days=max(1, days))
        to_date = datetime.now(timezone.utc) + timedelta(days=1)

        deals = (
            mt5.history_deals_get(from_date, to_date, group=symbol.strip().upper())
            if symbol
            else mt5.history_deals_get(from_date, to_date)
        )
        if deals is None:
            return []

        result: List[Dict[str, Any]] = []
        for d in deals:
            # DEAL_ENTRY_OUT (1) indicates a closing deal
            if d.entry == 1:
                dt = datetime.fromtimestamp(d.time, tz=timezone.utc)
                net_profit = float(d.profit + d.swap + d.commission)
                result.append({
                    "ticket": d.ticket,
                    "order": d.order,
                    "position_id": d.position_id,
                    "time": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": d.symbol,
                    "type": "BUY" if d.type == 0 else "SELL",
                    "volume": d.volume,
                    "price": d.price,
                    "profit": d.profit,
                    "swap": d.swap,
                    "commission": d.commission,
                    "net_liquid_profit": net_profit,
                    "outcome": "WIN" if net_profit > 0.0 else "LOSS",
                    "magic": d.magic,
                    "comment": d.comment,
                })
        return result

    def check_viability(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Perform dry-run pre-trade viability check (margin requirements and projected outcomes)."""
        if not self.ensure_connected():
            return {"error": "MT5 terminal not available"}

        sym = symbol.strip().upper()
        sym_info = mt5.symbol_info(sym)
        if sym_info is None:
            return {"error": f"Symbol '{sym}' not found"}

        acc_info = mt5.account_info()
        if acc_info is None:
            return {"error": "Account info not available"}

        is_buy = order_type.strip().upper() == "BUY"
        mt5_order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
        exec_price = price if price and price > 0.0 else (sym_info.ask if is_buy else sym_info.bid)

        # 1. Calculate Required Margin
        margin_required = mt5.order_calc_margin(mt5_order_type, sym, volume, exec_price)
        if margin_required is None:
            margin_required = 0.0

        free_margin = acc_info.margin_free
        margin_ok = margin_required < (free_margin * 0.95)

        # 2. Calculate Projected Profit / Loss if SL and TP provided
        profit_tp = None
        loss_sl = None

        if tp and tp > 0.0:
            profit_tp = mt5.order_calc_profit(mt5_order_type, sym, volume, exec_price, tp)

        if sl and sl > 0.0:
            loss_sl = mt5.order_calc_profit(mt5_order_type, sym, volume, exec_price, sl)

        return {
            "symbol": sym,
            "order_type": "BUY" if is_buy else "SELL",
            "volume": volume,
            "exec_price": exec_price,
            "free_margin": free_margin,
            "margin_required": margin_required,
            "margin_viable": margin_ok,
            "cushion_pct": ((free_margin - margin_required) / free_margin * 100.0) if free_margin > 0 else 0.0,
            "potential_profit_tp": profit_tp,
            "potential_loss_sl": loss_sl,
        }


# ------------------------------------------------------------------
# Model Context Protocol (MCP) JSON-RPC 2.0 Request Dispatcher
# ------------------------------------------------------------------
def handle_mcp_request(req: Dict[str, Any], client: LocalMT5Client) -> Optional[Dict[str, Any]]:
    """Process incoming JSON-RPC 2.0 MCP request and route to appropriate tool."""
    method = req.get("method")
    req_id = req.get("id")
    params = req.get("params", {})

    # Ping / Liveness
    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    # Initialize Handshake
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mt5-local", "version": "1.0.0"},
            },
        }

    # Tool Discovery
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "mt5_get_terminal_info",
                        "description": "Inspect local MT5 terminal state, connection status, build, and paths.",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                    {
                        "name": "mt5_get_account_info",
                        "description": "Inspect trading account balance, equity, leverage, free margin, and server.",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                    {
                        "name": "mt5_get_symbol_info",
                        "description": "Query market specifications, bid/ask prices, spread, and lot rules.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string", "description": "Currency pair (e.g. EURUSD, GBPUSD)"}
                            },
                            "required": ["symbol"],
                        },
                    },
                    {
                        "name": "mt5_get_rates",
                        "description": "Retrieve recent historical OHLCV bars (open, high, low, close, volume).",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string", "description": "Currency pair (e.g. EURUSD)"},
                                "timeframe": {
                                    "type": "string",
                                    "description": "Timeframe code (M1, M5, M15, M30, H1, H2, D1)",
                                    "default": "H1",
                                },
                                "count": {
                                    "type": "integer",
                                    "description": "Number of bars to fetch (max 1000)",
                                    "default": 50,
                                },
                            },
                            "required": ["symbol"],
                        },
                    },
                    {
                        "name": "mt5_get_ticks",
                        "description": "Retrieve recent real-time bid/ask market ticks for microstructure inspection.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string", "description": "Currency pair (e.g. EURUSD)"},
                                "count": {"type": "integer", "description": "Number of ticks to fetch", "default": 20},
                            },
                            "required": ["symbol"],
                        },
                    },
                    {
                        "name": "mt5_get_positions",
                        "description": "List active open positions with tickets, lots, SL, TP, profit, and magic.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string", "description": "Optional filter by currency pair"}
                            },
                        },
                    },
                    {
                        "name": "mt5_get_orders",
                        "description": "List active pending orders in MT5.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string", "description": "Optional filter by currency pair"}
                            },
                        },
                    },
                    {
                        "name": "mt5_get_history_deals",
                        "description": "Retrieve closed trade deals with audited Net Liquid Profit.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "days": {"type": "integer", "description": "Lookback window in days", "default": 7},
                                "symbol": {"type": "string", "description": "Optional filter by currency pair"},
                            },
                        },
                    },
                    {
                        "name": "mt5_check_viability",
                        "description": "Perform pre-trade dry-run checking broker margin requirement and profit/loss.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string", "description": "Currency pair (e.g. EURUSD)"},
                                "order_type": {
                                    "type": "string",
                                    "description": "BUY or SELL",
                                    "enum": ["BUY", "SELL"],
                                },
                                "volume": {"type": "number", "description": "Trade volume in lots (e.g. 0.05)"},
                                "price": {"type": "number", "description": "Optional execution price"},
                                "sl": {"type": "number", "description": "Optional Stop Loss price"},
                                "tp": {"type": "number", "description": "Optional Take Profit price"},
                            },
                            "required": ["symbol", "order_type", "volume"],
                        },
                    },
                ]
            },
        }

    # Tool Execution
    if method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        handlers = {
            "mt5_get_terminal_info": lambda: client.get_terminal_info(),
            "mt5_get_account_info": lambda: client.get_account_info(),
            "mt5_get_symbol_info": lambda: client.get_symbol_info(args.get("symbol", "EURUSD")),
            "mt5_get_rates": lambda: client.get_rates(
                args.get("symbol", "EURUSD"),
                args.get("timeframe", "H1"),
                args.get("count", 50),
            ),
            "mt5_get_ticks": lambda: client.get_ticks(
                args.get("symbol", "EURUSD"),
                args.get("count", 20),
            ),
            "mt5_get_positions": lambda: client.get_positions(args.get("symbol")),
            "mt5_get_orders": lambda: client.get_orders(args.get("symbol")),
            "mt5_get_history_deals": lambda: client.get_history_deals(
                args.get("days", 7),
                args.get("symbol"),
            ),
            "mt5_check_viability": lambda: client.check_viability(
                args.get("symbol", "EURUSD"),
                args.get("order_type", "BUY"),
                args.get("volume", 0.01),
                args.get("price"),
                args.get("sl"),
                args.get("tp"),
            ),
        }

        if tool_name in handlers:
            try:
                res_data = handlers[tool_name]()
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(res_data, indent=2)}],
                        "isError": False,
                    },
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps({"error": str(exc)}, indent=2)}],
                        "isError": True,
                    },
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
    """Run persistent JSON-RPC 2.0 Stdio MCP server."""
    client = LocalMT5Client()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_mcp_request(req, client)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
        except Exception as err:
            err_resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(err)}}
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


def run_cli() -> None:
    """CLI runner for direct terminal diagnostics or MCP entrypoint."""
    if "--mcp" in sys.argv:
        run_mcp_server()
        return

    client = LocalMT5Client()
    if "--account" in sys.argv:
        print(json.dumps(client.get_account_info(), indent=2))
        return

    if "--terminal" in sys.argv:
        print(json.dumps(client.get_terminal_info(), indent=2))
        return

    if "--positions" in sys.argv:
        print(json.dumps(client.get_positions(), indent=2))
        return

    if "--deals" in sys.argv:
        print(json.dumps(client.get_history_deals(days=7), indent=2))
        return

    sym = "EURUSD"
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            sym = arg.upper()
            break

    print(f"[*] Querying MT5 Symbol Info for '{sym}':")
    print(json.dumps(client.get_symbol_info(sym), indent=2))


if __name__ == "__main__":
    run_cli()
