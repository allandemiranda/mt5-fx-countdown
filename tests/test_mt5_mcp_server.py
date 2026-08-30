"""Automated Test Suite for Local MT5 Model Context Protocol (MCP) Server.

Verifies:
1. Registration of 'mt5-local' in .agents/mcp_config.json.
2. JSON-RPC 2.0 protocol compliance (initialize, ping, tools/list, error handling).
3. Tool schemas and parameter validation across all 9 diagnostic tools.
4. Live local client integration when MT5 terminal is reachable.
"""

from pathlib import Path
import json
import pytest
from src.tools.mt5_mcp_server import LocalMT5Client, handle_mcp_request


def test_mcp_config_registers_mt5_local() -> None:
    """Verify that .agents/mcp_config.json registers the 'mt5-local' MCP server."""
    workspace_root = Path(__file__).resolve().parent.parent
    config_path = workspace_root / ".agents" / "mcp_config.json"
    assert config_path.exists(), f"MCP config missing: {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    assert "mcpServers" in config
    assert "mt5-local" in config["mcpServers"]

    server_entry = config["mcpServers"]["mt5-local"]
    assert server_entry["command"] == "python"
    assert "-m" in server_entry["args"]
    assert "src.tools.mt5_mcp_server" in server_entry["args"]
    assert "--mcp" in server_entry["args"]


def test_mcp_jsonrpc_initialize() -> None:
    """Verify MCP initialize handshake returns valid protocol version and server info."""
    client = LocalMT5Client()
    req = {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
    resp = handle_mcp_request(req, client)

    assert resp is not None
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert "result" in resp
    assert resp["result"]["serverInfo"]["name"] == "mt5-local"
    assert "protocolVersion" in resp["result"]


def test_mcp_jsonrpc_ping() -> None:
    """Verify MCP ping returns empty result dictionary."""
    client = LocalMT5Client()
    req = {"jsonrpc": "2.0", "id": 42, "method": "ping"}
    resp = handle_mcp_request(req, client)

    assert resp is not None
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 42
    assert resp["result"] == {}


def test_mcp_jsonrpc_tools_list() -> None:
    """Verify tools/list exposes all 9 local MT5 diagnostic tools with valid schemas."""
    client = LocalMT5Client()
    req = {"jsonrpc": "2.0", "id": 100, "method": "tools/list"}
    resp = handle_mcp_request(req, client)

    assert resp is not None
    assert resp["jsonrpc"] == "2.0"
    assert "result" in resp
    tools = resp["result"]["tools"]
    assert len(tools) == 9

    tool_names = {t["name"] for t in tools}
    expected_tools = {
        "mt5_get_terminal_info",
        "mt5_get_account_info",
        "mt5_get_symbol_info",
        "mt5_get_rates",
        "mt5_get_ticks",
        "mt5_get_positions",
        "mt5_get_orders",
        "mt5_get_history_deals",
        "mt5_check_viability",
    }
    assert expected_tools.issubset(tool_names)

    for tool in tools:
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool
        assert tool["inputSchema"]["type"] == "object"


def test_mcp_jsonrpc_unknown_method() -> None:
    """Verify that unsupported JSON-RPC methods return standard -32601 error."""
    client = LocalMT5Client()
    req = {"jsonrpc": "2.0", "id": 999, "method": "unsupported/method"}
    resp = handle_mcp_request(req, client)

    assert resp is not None
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 999
    assert "error" in resp
    assert resp["error"]["code"] == -32601


def test_mcp_jsonrpc_unknown_tool() -> None:
    """Verify that unsupported tool calls return standard -32601 error."""
    client = LocalMT5Client()
    req = {
        "jsonrpc": "2.0",
        "id": 888,
        "method": "tools/call",
        "params": {"name": "non_existent_tool", "arguments": {}},
    }
    resp = handle_mcp_request(req, client)

    assert resp is not None
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 888
    assert "error" in resp
    assert resp["error"]["code"] == -32601


def test_mcp_local_client_live_queries() -> None:
    """Verify live MT5 client queries when local terminal is available."""
    client = LocalMT5Client()
    if not client.ensure_connected():
        pytest.skip("Local MT5 terminal is not available in this test environment.")

    term_info = client.get_terminal_info()
    assert "connected" in term_info
    assert term_info["connected"] is True

    acc_info = client.get_account_info()
    assert "balance" in acc_info
    assert "equity" in acc_info

    sym_info = client.get_symbol_info("EURUSD")
    assert "symbol" in sym_info
    assert sym_info["symbol"] == "EURUSD"
    assert "bid" in sym_info
    assert "ask" in sym_info
