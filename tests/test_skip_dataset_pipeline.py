"""Unit and integration tests for the SKIP_DATASET_GENERATION pipeline feature."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from run_pipeline import main, run_full_pipeline
from src.config import AppConfig


@pytest.fixture(autouse=True)
def setup_default_env():
    """Ensure baseline environment variables are loaded."""
    AppConfig.from_env()


def _create_synthetic_datasets(directory: Path, symbol: str, timeframe: str) -> tuple[Path, Path]:
    """Helper to generate valid synthetic BUY and SELL CSV datasets."""
    directory.mkdir(parents=True, exist_ok=True)
    num_features = 10
    num_samples = 60
    rng = np.random.default_rng(42)
    x = rng.standard_normal((num_samples, num_features), dtype=np.float32)
    y_buy = (x[:, 0] > 0).astype(int)
    y_sell = (x[:, 1] > 0).astype(int)

    cols = [f"f{i}" for i in range(num_features)]
    df_buy = pd.DataFrame(x, columns=cols)
    df_buy["label"] = y_buy
    df_sell = pd.DataFrame(x, columns=cols)
    df_sell["label"] = y_sell

    buy_path = directory / f"{symbol}_{timeframe}_buy.csv"
    sell_path = directory / f"{symbol}_{timeframe}_sell.csv"
    df_buy.to_csv(buy_path, index=False)
    df_sell.to_csv(sell_path, index=False)
    return buy_path, sell_path


def test_run_full_pipeline_skip_dataset_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that run_full_pipeline skips Strategy Tester and DMatrix-EA compilation when datasets exist."""
    monkeypatch.setenv("OPTUNA_TRIALS", "1")
    monkeypatch.setenv("XGB_ROUNDS", "5")
    monkeypatch.setenv("XGB_EARLY_STOPPING_ROUNDS", "2")
    monkeypatch.setenv("VALIDATION_PERCENTAGE", "0.20")
    monkeypatch.setenv("SKIP_DATASET_GENERATION", "1")

    config = AppConfig.from_env()
    ws_dir = tmp_path / "workspace"
    term_dir = tmp_path / "terminal"
    comm_dir = tmp_path / "common"

    files_dir = term_dir / "MQL5" / "Files"
    _create_synthetic_datasets(files_dir, config.symbol, config.clean_timeframe)

    with patch("run_pipeline.MT5Client") as mock_mt5_cls:
        mock_client = MagicMock()
        mock_client.initialize.return_value = True
        mock_client.terminal_data_path = term_dir
        mock_client.common_path = comm_dir
        mock_client.compile_ea.return_value = True
        mock_mt5_cls.return_value = mock_client

        success = run_full_pipeline(config, ws_dir)

        assert success is True
        # DMatrix-EA should NOT be compiled
        compiled_eas = [call.args[0] for call in mock_client.compile_ea.call_args_list]
        assert "DMatrix-EA.mq5" not in compiled_eas
        assert "LiveONNX-EA.mq5" in compiled_eas
        # run_strategy_tester should NOT be called
        mock_client.run_strategy_tester.assert_not_called()
        mock_client.shutdown.assert_called_once()


def test_run_full_pipeline_skip_dataset_fallback_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that run_full_pipeline falls back to Strategy Tester when datasets are missing."""
    monkeypatch.setenv("OPTUNA_TRIALS", "1")
    monkeypatch.setenv("XGB_ROUNDS", "5")
    monkeypatch.setenv("XGB_EARLY_STOPPING_ROUNDS", "2")
    monkeypatch.setenv("VALIDATION_PERCENTAGE", "0.20")
    monkeypatch.setenv("SKIP_DATASET_GENERATION", "1")

    config = AppConfig.from_env()
    ws_dir = tmp_path / "workspace"
    term_dir = tmp_path / "terminal"
    comm_dir = tmp_path / "common"

    files_dir = term_dir / "MQL5" / "Files"

    with patch("run_pipeline.MT5Client") as mock_mt5_cls:
        mock_client = MagicMock()
        mock_client.initialize.return_value = True
        mock_client.terminal_data_path = term_dir
        mock_client.common_path = comm_dir
        mock_client.compile_ea.return_value = True
        mock_client.generate_tester_ini.return_value = ws_dir / "tester.ini"

        def fake_run_tester(ini_path):
            # Create the datasets as the Strategy Tester would
            _create_synthetic_datasets(files_dir, config.symbol, config.clean_timeframe)
            return True

        mock_client.run_strategy_tester.side_effect = fake_run_tester
        mock_mt5_cls.return_value = mock_client

        success = run_full_pipeline(config, ws_dir, skip_dataset_override=True)

        assert success is True
        # DMatrix-EA should be compiled because fallback triggered
        compiled_eas = [call.args[0] for call in mock_client.compile_ea.call_args_list]
        assert "DMatrix-EA.mq5" in compiled_eas
        assert "LiveONNX-EA.mq5" in compiled_eas
        mock_client.run_strategy_tester.assert_called_once()


def test_run_full_pipeline_skip_dataset_override_with_config_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that skip_dataset_override=True overrides config.skip_dataset_generation=False and preserves datasets."""
    monkeypatch.setenv("OPTUNA_TRIALS", "1")
    monkeypatch.setenv("XGB_ROUNDS", "5")
    monkeypatch.setenv("XGB_EARLY_STOPPING_ROUNDS", "2")
    monkeypatch.setenv("VALIDATION_PERCENTAGE", "0.20")
    monkeypatch.setenv("SKIP_DATASET_GENERATION", "0")

    config = AppConfig.from_env()
    assert config.skip_dataset_generation is False

    ws_dir = tmp_path / "workspace"
    term_dir = tmp_path / "terminal"
    comm_dir = tmp_path / "common"

    files_dir = term_dir / "MQL5" / "Files"
    _create_synthetic_datasets(files_dir, config.symbol, config.clean_timeframe)

    with patch("run_pipeline.MT5Client") as mock_mt5_cls:
        mock_client = MagicMock()
        mock_client.initialize.return_value = True
        mock_client.terminal_data_path = term_dir
        mock_client.common_path = comm_dir
        mock_client.compile_ea.return_value = True
        mock_mt5_cls.return_value = mock_client

        success = run_full_pipeline(config, ws_dir, skip_dataset_override=True)

        assert success is True
        # DMatrix-EA should NOT be compiled because override=True took effect
        compiled_eas = [call.args[0] for call in mock_client.compile_ea.call_args_list]
        assert "DMatrix-EA.mq5" not in compiled_eas
        assert "LiveONNX-EA.mq5" in compiled_eas
        mock_client.run_strategy_tester.assert_not_called()
        mock_client.shutdown.assert_called_once()


def test_run_full_pipeline_normal_execution_when_skip_dataset_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that run_full_pipeline compiles DMatrix-EA and runs Strategy Tester when skip_dataset is False."""
    monkeypatch.setenv("OPTUNA_TRIALS", "1")
    monkeypatch.setenv("XGB_ROUNDS", "5")
    monkeypatch.setenv("XGB_EARLY_STOPPING_ROUNDS", "2")
    monkeypatch.setenv("VALIDATION_PERCENTAGE", "0.20")
    monkeypatch.setenv("SKIP_DATASET_GENERATION", "0")

    config = AppConfig.from_env()
    assert config.skip_dataset_generation is False

    ws_dir = tmp_path / "workspace"
    term_dir = tmp_path / "terminal"
    comm_dir = tmp_path / "common"

    files_dir = term_dir / "MQL5" / "Files"

    with patch("run_pipeline.MT5Client") as mock_mt5_cls:
        mock_client = MagicMock()
        mock_client.initialize.return_value = True
        mock_client.terminal_data_path = term_dir
        mock_client.common_path = comm_dir
        mock_client.compile_ea.return_value = True
        mock_client.generate_tester_ini.return_value = ws_dir / "tester.ini"

        def fake_run_tester(ini_path):
            _create_synthetic_datasets(files_dir, config.symbol, config.clean_timeframe)
            return True

        mock_client.run_strategy_tester.side_effect = fake_run_tester
        mock_mt5_cls.return_value = mock_client

        success = run_full_pipeline(config, ws_dir, skip_dataset_override=None)

        assert success is True
        compiled_eas = [call.args[0] for call in mock_client.compile_ea.call_args_list]
        assert "DMatrix-EA.mq5" in compiled_eas
        assert "LiveONNX-EA.mq5" in compiled_eas
        mock_client.run_strategy_tester.assert_called_once()
        mock_client.shutdown.assert_called_once()


def test_run_full_pipeline_cli_skip_dataset_flag(monkeypatch: pytest.MonkeyPatch):
    """Verify CLI parses --skip-dataset and triggers skip_dataset_override=True."""
    monkeypatch.setattr("sys.argv", ["run_pipeline.py", ".env", "--skip-dataset"])

    with patch("run_pipeline.run_full_pipeline") as mock_run:
        mock_run.return_value = True
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs.get("skip_dataset_override") is True


def test_run_full_pipeline_cli_default_no_skip(monkeypatch: pytest.MonkeyPatch):
    """Verify CLI default without --skip-dataset passes skip_dataset_override=None."""
    monkeypatch.setattr("sys.argv", ["run_pipeline.py", ".env"])

    with patch("run_pipeline.run_full_pipeline") as mock_run:
        mock_run.return_value = True
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs.get("skip_dataset_override") is None
