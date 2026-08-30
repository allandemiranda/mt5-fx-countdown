import unittest
from unittest.mock import MagicMock, patch
from src.tools.macro_calendar import MacroeconomicCalendar, EconomicEvent, run_cli


class TestMacroeconomicCalendar(unittest.TestCase):
    """Test suite for macroeconomic calendar and catalyst lookup tool."""

    def setUp(self) -> None:
        self.calendar = MacroeconomicCalendar()

    def test_high_impact_definitions_known_currencies(self) -> None:
        usd_events = self.calendar.get_high_impact_definitions("USD")
        self.assertIn("Non-Farm Payrolls", usd_events)
        self.assertIn("FOMC Rate Decision", usd_events)

        eur_events = self.calendar.get_high_impact_definitions("EUR")
        self.assertIn("ECB Rate Decision", eur_events)

        jpy_events = self.calendar.get_high_impact_definitions("jpy")
        self.assertIn("BOJ Policy Rate", jpy_events)

    def test_high_impact_definitions_unknown_currency(self) -> None:
        unknown = self.calendar.get_high_impact_definitions("XYZ")
        self.assertEqual(unknown, [])

    @patch("urllib.request.urlopen")
    def test_fetch_open_economic_feed_success(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.read.return_value = b"Breaking: FOMC Rate Decision and Non-Farm Payrolls update for USD"
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        events = self.calendar.fetch_open_economic_feed("USD")
        self.assertTrue(len(events) >= 2)
        event_names = [e.event_name for e in events]
        self.assertIn("FOMC Rate Decision", event_names)
        self.assertIn("Non-Farm Payrolls", event_names)

    @patch("urllib.request.urlopen")
    def test_fetch_open_economic_feed_network_error_handled_gracefully(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = Exception("Connection timed out")
        events = self.calendar.fetch_open_economic_feed("EUR")
        self.assertEqual(events, [])

    def test_economic_event_dataclass_immutability(self) -> None:
        event = EconomicEvent(
            event_id="USD_NFP_2026-09-02",
            date="2026-09-02",
            time_utc="12:30",
            currency="USD",
            event_name="Non-Farm Payrolls",
            impact="HIGH",
        )
        self.assertEqual(event.currency, "USD")
        with self.assertRaises(AttributeError):
            event.currency = "EUR"  # type: ignore

    @patch("sys.argv", ["macro_calendar.py", "USD"])
    @patch("builtins.print")
    def test_run_cli_with_currency_arg(self, mock_print: MagicMock) -> None:
        run_cli()
        mock_print.assert_called_once()
        output_str = mock_print.call_args[0][0]
        self.assertIn("Non-Farm Payrolls", output_str)

    @patch("sys.argv", ["macro_calendar.py"])
    @patch("builtins.print")
    def test_run_cli_without_args(self, mock_print: MagicMock) -> None:
        run_cli()
        mock_print.assert_called_once()
        output_str = mock_print.call_args[0][0]
        self.assertIn("USD", output_str)
        self.assertIn("EUR", output_str)

    def test_audit_backtest_anomaly_nfp_match(self) -> None:
        # First Friday of May 2024 (2024-05-03) at 12:30 UTC
        res = self.calendar.audit_backtest_anomaly("2024-05-03 12:30", "EURUSD")
        self.assertEqual(res["classification"], "Class A: Exogenous Macro Shock")
        self.assertTrue(len(res["matched_catalysts"]) > 0)
        self.assertIn("Non-Farm Payrolls", res["matched_catalysts"][0]["event"])

    def test_audit_backtest_anomaly_no_match(self) -> None:
        # Random non-catalyst timestamp (e.g. Sunday midnight)
        res = self.calendar.audit_backtest_anomaly("2024-05-05 01:00", "EURUSD")
        self.assertEqual(res["classification"], "Class B: Endogenous Model Degradation")
        self.assertEqual(res["matched_catalysts"], [])

    @patch("urllib.request.urlopen")
    def test_fetch_mql5_calendar_events_success(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.read.return_value = (
            b"2026.08.30 23:50, JPY, Retail Sales m/m, Actual: 2.4%\n"
            b"2026.08.31 12:30, USD, Non-Farm Payrolls, Actual: 180K\n"
        )
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        events = self.calendar.fetch_mql5_calendar_events(currency="USD")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["currency"], "USD")
        self.assertIn("Non-Farm Payrolls", events[0]["event"])

    def test_handle_jsonrpc_request_lifecycle(self) -> None:
        from src.tools.macro_calendar import handle_jsonrpc_request

        # initialize
        init_resp = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, self.calendar)
        self.assertIsNotNone(init_resp)
        self.assertEqual(init_resp["result"]["serverInfo"]["name"], "economic-calendar")

        # tools/list
        list_resp = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, self.calendar)
        tool_names = [t["name"] for t in list_resp["result"]["tools"]]
        self.assertIn("get_mql5_economic_calendar", tool_names)
        self.assertIn("audit_backtest_anomaly", tool_names)

        # tools/call audit_backtest_anomaly
        call_resp = handle_jsonrpc_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "audit_backtest_anomaly",
                "arguments": {"timestamp": "2024-05-03 12:30", "currency": "USD"},
            },
        }, self.calendar)
        self.assertFalse(call_resp["result"]["isError"])
        self.assertIn("Class A: Exogenous Macro Shock", call_resp["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
