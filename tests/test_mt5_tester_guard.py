"""Unit tests for MT5 Strategy Tester guard, static ini generation, and log stream monitoring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from src.config import AppConfig
from src.mt5_client import MT5Client


@pytest.fixture(autouse=True)
def setup_default_env():
    """Ensure baseline environment variables are loaded."""
    AppConfig.from_env()


def test_tester_ini_contains_verified_institutional_defaults(tmp_path: Path):
    """Verify that generate_tester_ini writes the exact institutional high-liquidity parameters."""
    config = AppConfig.from_env()
    client = MT5Client(config, tmp_path)
    client.terminal_data_path = tmp_path

    ini_file = client.generate_tester_ini()
    assert ini_file.exists()

    content = ini_file.read_text(encoding="ascii")
    assert "Expert=DMatrix-EA.ex5" in content
    assert f"Symbol={config.symbol}" in content
    assert f"Period={config.timeframe}" in content
    assert "Optimization=0" in content
    assert "Model=4" in content
    assert f"FromDate={config.from_date}" in content
    assert f"ToDate={config.to_date}" in content
    assert "ForwardMode=0" in content
    assert "Deposit=1000000000000000" in content
    assert "Currency=USD" in content
    assert "ProfitInPips=0" in content
    assert "Leverage=500" in content
    assert "ExecutionMode=0" in content
    assert "OptimizationCriterion=0" in content
    assert "Visual=0" in content
    assert f"ShutdownTerminal={config.shutdown_terminal}" in content
    assert "InpLotSize=0.01" in content
    assert f"InpUseADX={'true' if config.use_adx else 'false'}" in content
    assert f"InpUseATR={'true' if config.use_atr else 'false'}" in content
    assert f"InpUseBands={'true' if config.use_bands else 'false'}" in content
    assert f"InpUseMACD={'true' if config.use_macd else 'false'}" in content
    assert f"InpUseFastMA={'true' if config.use_fast_ma else 'false'}" in content
    assert f"InpUseSlowMA={'true' if config.use_slow_ma else 'false'}" in content
    assert f"InpUseRSI={'true' if config.use_rsi else 'false'}" in content
    assert f"InpUseStochastic={'true' if config.use_stochastic else 'false'}" in content
    assert f"InpUseCandlestick={'true' if config.use_candlestick else 'false'}" in content
    assert f"InpUseTimestampWeek={'true' if config.use_timestamp_week else 'false'}" in content
    assert f"InpUseTimestampDay={'true' if config.use_timestamp_day else 'false'}" in content
    assert f"InpUseOpenMarkets={'true' if config.use_open_markets else 'false'}" in content
    assert f"InpUseSpread={'true' if config.use_spread else 'false'}" in content
    assert f"InpAvoidPandemicTime={'true' if config.avoid_pandemictime else 'false'}" in content
    assert f"InpPandemicStartTime={config.pandemic_start_date}" in content
    assert f"InpPandemicEndTime={config.pandemic_end_date}" in content


def test_tester_ini_overwrites_existing_file(tmp_path: Path):
    """Verify that generate_tester_ini safely removes and recreates any existing .ini file."""
    config = AppConfig.from_env()
    client = MT5Client(config, tmp_path)
    client.terminal_data_path = tmp_path

    ini_file = tmp_path / f"tester_{config.symbol}_{config.clean_timeframe}.ini"
    ini_file.write_text("OLD_STALE_CONTENT", encoding="ascii")
    assert ini_file.read_text(encoding="ascii") == "OLD_STALE_CONTENT"

    new_ini = client.generate_tester_ini()
    assert new_ini == ini_file
    assert "OLD_STALE_CONTENT" not in new_ini.read_text(encoding="ascii")
    assert "[Tester]" in new_ini.read_text(encoding="ascii")


def test_get_tester_log_directories_resolution_and_deduplication(tmp_path: Path):
    """Verify that _get_tester_log_directories finds standard, agent, and deduplicates directories."""
    config = AppConfig.from_env()
    client = MT5Client(config, tmp_path)
    client.terminal_data_path = tmp_path

    # Create nested agent logs directory
    agent_logs = tmp_path / "Tester" / "Agent-127.0.0.1-3000" / "logs"
    agent_logs.mkdir(parents=True, exist_ok=True)
    tester_logs = tmp_path / "Tester" / "logs"
    tester_logs.mkdir(parents=True, exist_ok=True)
    terminal_logs = tmp_path / "logs"
    terminal_logs.mkdir(parents=True, exist_ok=True)

    dirs = client._get_tester_log_directories()
    assert tester_logs in dirs
    assert terminal_logs in dirs
    assert agent_logs in dirs
    # Check deduplication
    assert len(dirs) == len(set(dirs))


def test_stream_new_tester_logs_utf16_and_utf8(tmp_path: Path):
    """Verify that _stream_new_tester_logs tails new lines across UTF-16 and UTF-8 encoded log files."""
    config = AppConfig.from_env()
    client = MT5Client(config, tmp_path)
    client.terminal_data_path = tmp_path

    tester_logs_dir = tmp_path / "Tester" / "logs"
    tester_logs_dir.mkdir(parents=True, exist_ok=True)

    log1 = tester_logs_dir / "20260902.log"
    # Write UTF-16 encoded content
    log1.write_text("Line 1 initial\nLine 2 initial\n", encoding="utf-16")

    file_offsets = {log1: log1.stat().st_size}

    # No new lines yet
    new_lines = client._stream_new_tester_logs(file_offsets)
    assert new_lines == []

    # Append new lines in UTF-16
    with open(log1, "a", encoding="utf-16") as f:
        f.write("[DMatrix-EA] Initialized successfully\n[WARMUP] Insufficient rates for EURUSD\n")

    new_lines = client._stream_new_tester_logs(file_offsets)
    assert len(new_lines) == 2
    assert "[DMatrix-EA] Initialized successfully" in new_lines[0]
    assert "[WARMUP] Insufficient rates for EURUSD" in new_lines[1]


def test_stream_new_tester_logs_ignores_excluded_and_compiler_logs(tmp_path: Path):
    """Verify that _stream_new_tester_logs skips metaeditor, diagnostic, and compile logs."""
    config = AppConfig.from_env()
    client = MT5Client(config, tmp_path)
    client.terminal_data_path = tmp_path

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    meta_log = logs_dir / "metaeditor.log"
    meta_log.write_text("metaeditor error message\n", encoding="utf-8")

    compile_log = logs_dir / "compile_DMatrix-EA.log"
    compile_log.write_text("compile log line\n", encoding="utf-8")

    diag_log = logs_dir / "favorites_diagnostic.log"
    diag_log.write_text("diagnostic log line\n", encoding="utf-8")

    valid_log = logs_dir / "terminal.log"
    valid_log.write_text("Valid terminal log line\n", encoding="utf-8")

    file_offsets: dict[Path, int] = {}
    lines = client._stream_new_tester_logs(file_offsets)
    assert len(lines) == 1
    assert "Valid terminal log line" in lines[0]


@pytest.mark.parametrize(
    "warning_line",
    [
        "[WARMUP] Insufficient historical rates. Copied: 10, Needed: 50",
        "[warmup] historical buffer warming up",
        "[FeatureExtractor] [WARMUP] Waiting for 200 bars",
        "[WARNING] Market closed for trading",
        "[warning] spread widened",
        "Tester: market closed for current session",
        "Tester: MARKET CLOSED on Sunday",
        "Tester: real ticks absent in history storage",
        "Tester: REAL TICKS ABSENT for selected symbol",
        "Tester: real ticks discarded for test period",
        "Tester: Real Ticks Discarded due to timestamp gap",
        "Tester: no real ticks found in database",
        "Tester: NO REAL TICKS generated",
        "Tester: insufficient rates available to build timeframe",
        "Tester: INSUFFICIENT RATES for bar computation",
        "Tester: insufficient historical rates from broker",
        "Tester: waiting for history buffer synchronization",
        "Tester: waiting for history buffer",
        "Tester: order send failed: invalid stops",
        "Tester: invalid stops (sl=1.0500, tp=1.0600)",
        "Tester: TRADE FAILED: INVALID STOPS",
        "order rejected due to invalid stops",
    ],
)
def test_run_strategy_tester_tolerates_all_non_fatal_warnings(
    tmp_path: Path, warning_line: str
):
    """Verify that run_strategy_tester tolerates all non-fatal warning patterns without aborting."""
    from unittest.mock import patch

    config = AppConfig.from_env()
    client = MT5Client(config, tmp_path)
    client.terminal_data_path = tmp_path

    ini_path = client.generate_tester_ini()

    tester_logs_dir = tmp_path / "Tester" / "logs"
    tester_logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = tester_logs_dir / "tester.log"
    log_file.write_text("", encoding="utf-8")

    mock_proc = MagicMock()
    mock_proc.pid = 9999
    mock_proc.poll.side_effect = [None, 0]
    mock_proc.kill = MagicMock()

    written = False

    def fake_sleep(seconds: float):
        nonlocal written
        if not written:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{warning_line}\n")
            written = True

    with patch("subprocess.Popen", return_value=mock_proc):
        with patch("time.sleep", side_effect=fake_sleep):
            with patch.object(client, "_terminate_running_mt5"):
                result = client.run_strategy_tester(ini_path)

    assert result is True
    assert mock_proc.kill.call_count == 0


@pytest.mark.parametrize(
    "fatal_line,expected_fragment",
    [
        ("[DMatrix-EA] [ERROR] Model inference failed", "[dmatrix-ea] [error]"),
        ("[dmatrix-ea] [error] Failed to initialize weights", "[dmatrix-ea] [error]"),
        ("[DMatrix-EA] [Error] Cannot allocate memory", "[dmatrix-ea] [error]"),
        ("Tester: critical runtime error 5001 in OnTick", "critical runtime error"),
        ("Tester: CRITICAL RUNTIME ERROR: panic", "critical runtime error"),
        ("Tester: Critical Runtime Error during bar processing", "critical runtime error"),
        ("Tester: cannot load expert 'DMatrix-EA'", "cannot load expert"),
        ("Tester: CANNOT LOAD EXPERT file corrupted", "cannot load expert"),
        ("Tester: Cannot Load Expert binary", "cannot load expert"),
        ("Tester: zero divide in GarchEngine.mqh:120", "zero divide"),
        ("Tester: ZERO DIVIDE in volatility calculation", "zero divide"),
        ("Tester: Zero Divide encountered", "zero divide"),
        ("Tester: array out of range in IndicatorBuffer.mqh:45", "array out of range"),
        ("Tester: ARRAY OUT OF RANGE index 10 >= 5", "array out of range"),
        ("Tester: Array Out Of Range exception", "array out of range"),
        ("Tester: pointer cannot be used in CGarchEngine::Release", "pointer cannot be used"),
        ("Tester: POINTER CANNOT BE USED null pointer", "pointer cannot be used"),
        ("Tester: Pointer Cannot Be Used after deletion", "pointer cannot be used"),
    ],
)
def test_run_strategy_tester_triggers_fatal_exception_on_critical_errors(
    tmp_path: Path, fatal_line: str, expected_fragment: str
):
    """Verify that run_strategy_tester strictly raises RuntimeError and terminates on fatal errors."""
    from unittest.mock import patch

    config = AppConfig.from_env()
    client = MT5Client(config, tmp_path)
    client.terminal_data_path = tmp_path

    ini_path = client.generate_tester_ini()

    tester_logs_dir = tmp_path / "Tester" / "logs"
    tester_logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = tester_logs_dir / "tester.log"
    log_file.write_text("", encoding="utf-8")

    mock_proc = MagicMock()
    mock_proc.pid = 9999
    mock_proc.poll.side_effect = [None, None, 0]
    mock_proc.kill = MagicMock()

    written = False

    def fake_sleep(seconds: float):
        nonlocal written
        if not written:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{fatal_line}\n")
            written = True

    with patch("subprocess.Popen", return_value=mock_proc):
        with patch("time.sleep", side_effect=fake_sleep):
            with patch.object(client, "_terminate_running_mt5") as mock_term:
                with pytest.raises(RuntimeError) as exc_info:
                    client.run_strategy_tester(ini_path)

    assert fatal_line in str(exc_info.value)
    assert mock_proc.kill.called
    assert mock_term.called


def test_mixed_log_stream_non_fatal_then_fatal(tmp_path: Path):
    """Verify that non-fatal warnings stream fine until a fatal error abruptly aborts execution."""
    from unittest.mock import patch

    config = AppConfig.from_env()
    client = MT5Client(config, tmp_path)
    client.terminal_data_path = tmp_path

    ini_path = client.generate_tester_ini()

    tester_logs_dir = tmp_path / "Tester" / "logs"
    tester_logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = tester_logs_dir / "tester.log"
    log_file.write_text("", encoding="utf-8")

    mock_proc = MagicMock()
    mock_proc.pid = 9999
    mock_proc.poll.side_effect = [None, None, 0]
    mock_proc.kill = MagicMock()

    step = 0

    def fake_sleep(seconds: float):
        nonlocal step
        if step == 0:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("[FeatureExtractor] [WARMUP] Warmup bar 1\n")
                f.write("[WARNING] Market closed for trading\n")
                f.write("Tester: real ticks absent\n")
                f.write("Tester: no real ticks found\n")
                f.write("Tester: insufficient rates available\n")
            step += 1
        elif step == 1:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("Tester: zero divide in calculate_stops\n")
            step += 1

    with patch("subprocess.Popen", return_value=mock_proc):
        with patch("time.sleep", side_effect=fake_sleep):
            with patch.object(client, "_terminate_running_mt5") as mock_term:
                with pytest.raises(RuntimeError) as exc_info:
                    client.run_strategy_tester(ini_path)

    assert "zero divide" in str(exc_info.value).lower()
    assert mock_proc.kill.called
    assert mock_term.called


def test_fatal_error_in_final_flush_triggers_exception(tmp_path: Path):
    """Verify that a fatal error occurring right before process exit (flushed at retcode) raises RuntimeError."""
    from unittest.mock import patch

    config = AppConfig.from_env()
    client = MT5Client(config, tmp_path)
    client.terminal_data_path = tmp_path

    ini_path = client.generate_tester_ini()

    tester_logs_dir = tmp_path / "Tester" / "logs"
    tester_logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = tester_logs_dir / "tester.log"
    log_file.write_text("", encoding="utf-8")

    mock_proc = MagicMock()
    mock_proc.pid = 9999
    mock_proc.poll.return_value = 0
    mock_proc.kill = MagicMock()

    written = False

    def fake_sleep(seconds: float):
        nonlocal written
        if not written:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("[DMatrix-EA] [ERROR] Model inference memory fault at shutdown\n")
            written = True

    with patch("subprocess.Popen", return_value=mock_proc):
        with patch("time.sleep", side_effect=fake_sleep):
            with patch.object(client, "_terminate_running_mt5") as mock_term:
                with pytest.raises(RuntimeError) as exc_info:
                    client.run_strategy_tester(ini_path)

    assert "[DMatrix-EA] [ERROR]" in str(exc_info.value)
    assert mock_proc.kill.called
    assert mock_term.called


def test_run_strategy_tester_timeout(tmp_path: Path):
    """Verify that run_strategy_tester aborts when execution exceeds backtest_timeout."""
    from unittest.mock import patch

    config = AppConfig.from_env()
    client = MT5Client(config, tmp_path)
    client.terminal_data_path = tmp_path
    client.config = AppConfig(**{**config.__dict__, "backtest_timeout": 5})

    ini_path = client.generate_tester_ini()

    mock_proc = MagicMock()
    mock_proc.pid = 9999
    mock_proc.poll.return_value = None
    mock_proc.kill = MagicMock()

    # Mock time.time to simulate elapsed time exceeding timeout
    time_mock = MagicMock(side_effect=[1000.0, 1000.0, 1006.0, 1006.0])

    with patch("subprocess.Popen", return_value=mock_proc):
        with patch("time.time", time_mock):
            with patch("time.sleep"):
                with patch.object(client, "_terminate_running_mt5") as mock_term:
                    result = client.run_strategy_tester(ini_path)

    assert result is False
    assert mock_proc.kill.called
    assert mock_term.called


def test_run_strategy_tester_keyboard_interrupt(tmp_path: Path):
    """Verify that run_strategy_tester catches KeyboardInterrupt, kills subprocess, and returns False."""
    from unittest.mock import patch

    config = AppConfig.from_env()
    client = MT5Client(config, tmp_path)
    client.terminal_data_path = tmp_path

    ini_path = client.generate_tester_ini()

    mock_proc = MagicMock()
    mock_proc.pid = 9999
    mock_proc.kill = MagicMock()

    with patch("subprocess.Popen", return_value=mock_proc):
        with patch("time.sleep", side_effect=KeyboardInterrupt):
            with patch.object(client, "_terminate_running_mt5") as mock_term:
                result = client.run_strategy_tester(ini_path)

    assert result is False
    assert mock_proc.kill.called
    assert mock_term.called


def test_run_strategy_tester_subprocess_exception(tmp_path: Path):
    """Verify that run_strategy_tester handles subprocess spawn failure gracefully."""
    from unittest.mock import patch

    config = AppConfig.from_env()
    client = MT5Client(config, tmp_path)
    client.terminal_data_path = tmp_path

    ini_path = client.generate_tester_ini()

    with patch("subprocess.Popen", side_effect=OSError("Executable not found")):
        with patch.object(client, "_terminate_running_mt5"):
            result = client.run_strategy_tester(ini_path)

    assert result is False


def test_run_strategy_tester_nonzero_exit_code_without_fatal_logs(tmp_path: Path):
    """Verify that a non-zero exit code without fatal log patterns returns True (with warning)."""
    from unittest.mock import patch

    config = AppConfig.from_env()
    client = MT5Client(config, tmp_path)
    client.terminal_data_path = tmp_path

    ini_path = client.generate_tester_ini()

    mock_proc = MagicMock()
    mock_proc.pid = 9999
    mock_proc.poll.side_effect = [None, 1]
    mock_proc.kill = MagicMock()

    with patch("subprocess.Popen", return_value=mock_proc):
        with patch("time.sleep"):
            with patch.object(client, "_terminate_running_mt5"):
                result = client.run_strategy_tester(ini_path)

    assert result is True
    assert mock_proc.kill.call_count == 0


def test_terminate_running_mt5_and_shutdown(tmp_path: Path):
    """Verify that _terminate_running_mt5 and shutdown handle mt5 module exceptions gracefully."""
    from unittest.mock import patch

    config = AppConfig.from_env()
    client = MT5Client(config, tmp_path)

    with patch("subprocess.run") as mock_run:
        client._terminate_running_mt5()
        assert mock_run.called

    # Shutdown with mt5 mock
    with patch("src.mt5_client.mt5") as mock_mt5:
        mock_mt5.shutdown = MagicMock()
        client.shutdown()
        assert mock_mt5.shutdown.called
