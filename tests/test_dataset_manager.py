"""Unit tests for DatasetManager discovery, validation, and metadata management."""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import pytest

from src.config import AppConfig
from src.dataset_manager import DatasetManager


@pytest.fixture(autouse=True)
def setup_default_env(monkeypatch: pytest.MonkeyPatch):
    """Ensure baseline environment variables are loaded for all dataset manager tests."""
    AppConfig.from_env()


def test_dataset_manager_find_and_validate_standard(tmp_path: Path):
    """Verify DatasetManager locates and validates buy/sell CSVs and metadata."""
    config = AppConfig.from_env()

    terminal_data = tmp_path / "terminal"
    common_data = tmp_path / "common"
    files_dir = terminal_data / "MQL5" / "Files"
    files_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy datasets
    buy_csv = files_dir / f"{config.symbol}_{config.clean_timeframe}_buy.csv"
    sell_csv = files_dir / f"{config.symbol}_{config.clean_timeframe}_sell.csv"
    meta_json = files_dir / f"{config.symbol}_{config.clean_timeframe}_metadata.json"

    df = pd.DataFrame({"f0": [1.0, 2.0], "label": [1, 0]})
    df.to_csv(buy_csv, index=False)
    df.to_csv(sell_csv, index=False)

    meta_content = {"symbol": config.symbol, "timeframe": config.timeframe}
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta_content, f)

    manager = DatasetManager(config, tmp_path, terminal_data, common_data)
    found_buy, found_sell, found_meta = manager.find_and_validate_datasets()

    assert found_buy == buy_csv
    assert found_sell == sell_csv
    assert found_meta == meta_json

    meta = manager.load_metadata(found_meta, 1, ["f0"], {"roc_auc": 0.8}, {"roc_auc": 0.85})
    assert meta["num_features"] == 1
    assert "metrics" in meta
    assert meta["metrics"]["buy"]["roc_auc"] == 0.8
    assert meta["metrics"]["sell"]["roc_auc"] == 0.85
    assert "timestamp" in meta


def test_dataset_manager_search_in_tester_agent_directories(tmp_path: Path):
    """Verify DatasetManager resolves and finds datasets inside MT5 Tester agent directories."""
    config = AppConfig.from_env()

    terminal_data = tmp_path / "terminal"
    common_data = tmp_path / "common"
    agent_files = terminal_data / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files"
    agent_files.mkdir(parents=True, exist_ok=True)

    buy_csv = agent_files / f"{config.symbol}_{config.clean_timeframe}_buy.csv"
    sell_csv = agent_files / f"{config.symbol}_{config.clean_timeframe}_sell.csv"

    df = pd.DataFrame({"f0": [1.0, 2.0], "label": [1, 0]})
    df.to_csv(buy_csv, index=False)
    df.to_csv(sell_csv, index=False)

    manager = DatasetManager(config, tmp_path, terminal_data, common_data)
    found_buy, found_sell, found_meta = manager.find_and_validate_datasets()

    assert found_buy == buy_csv
    assert found_sell == sell_csv
    assert found_meta is None


def test_dataset_manager_missing_files_raises_runtime_error(tmp_path: Path):
    """Verify DatasetManager raises RuntimeError if dataset files are missing."""
    config = AppConfig.from_env()
    manager = DatasetManager(config, tmp_path, tmp_path / "term", tmp_path / "comm")

    with pytest.raises(RuntimeError, match="required datasets"):
        manager.find_and_validate_datasets()


def test_dataset_manager_missing_label_column_raises_value_error(tmp_path: Path):
    """Verify DatasetManager raises ValueError if dataset CSV does not have 'label' column."""
    config = AppConfig.from_env()

    files_dir = tmp_path / "MQL5" / "Files"
    files_dir.mkdir(parents=True, exist_ok=True)

    buy_csv = files_dir / f"{config.symbol}_{config.clean_timeframe}_buy.csv"
    sell_csv = files_dir / f"{config.symbol}_{config.clean_timeframe}_sell.csv"

    # CSV without label column
    df = pd.DataFrame({"f0": [1.0, 2.0], "f1": [3.0, 4.0]})
    df.to_csv(buy_csv, index=False)
    df.to_csv(sell_csv, index=False)

    manager = DatasetManager(config, tmp_path, tmp_path, tmp_path)
    with pytest.raises(ValueError, match="must contain 'label' column"):
        manager.find_and_validate_datasets()


def test_dataset_manager_load_metadata_fallback(tmp_path: Path):
    """Verify metadata generation when metadata json is absent or None."""
    config = AppConfig.from_env()
    manager = DatasetManager(config, tmp_path, tmp_path, tmp_path)

    meta = manager.load_metadata(
        meta_path=None,
        num_features=105,
        feature_names=["f0", "f1"],
        metrics_buy={"accuracy": 0.75},
        metrics_sell={"accuracy": 0.78},
    )

    assert meta["symbol"] == config.symbol
    assert meta["timeframe"] == config.timeframe
    assert meta["num_features"] == 105
    assert meta["feature_names"] == ["f0", "f1"]
    assert meta["metrics"]["buy"]["accuracy"] == 0.75
    assert meta["metrics"]["sell"]["accuracy"] == 0.78


def test_has_existing_datasets_all_scenarios(tmp_path: Path):
    """Verify has_existing_datasets correctly validates presence and non-empty sizes of buy and sell CSVs."""
    config = AppConfig.from_env()
    term_files = tmp_path / "MQL5" / "Files"
    comm_files = tmp_path / "Common" / "Files"
    term_files.mkdir(parents=True, exist_ok=True)
    comm_files.mkdir(parents=True, exist_ok=True)

    manager = DatasetManager(config, tmp_path, tmp_path, tmp_path / "Common")

    # 1. No files present
    assert manager.has_existing_datasets() is False

    # 2. Only BUY present and non-empty
    buy_csv = term_files / f"{config.symbol}_{config.clean_timeframe}_buy.csv"
    buy_csv.write_text("f0,label\n1.0,1\n", encoding="utf-8")
    assert manager.has_existing_datasets() is False

    # 3. Both BUY and SELL present and non-empty
    sell_csv = comm_files / f"{config.symbol}_{config.clean_timeframe}_sell.csv"
    sell_csv.write_text("f0,label\n1.0,0\n", encoding="utf-8")
    assert manager.has_existing_datasets() is True

    # 4. SELL exists but has 0 bytes
    sell_csv.write_text("", encoding="utf-8")
    assert manager.has_existing_datasets() is False

    # 5. BUY exists but has 0 bytes
    buy_csv.write_text("", encoding="utf-8")
    sell_csv.write_text("f0,label\n1.0,0\n", encoding="utf-8")
    assert manager.has_existing_datasets() is False


def test_dataset_manager_search_in_appdata_metaquotes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify DatasetManager resolves and finds datasets inside %APPDATA%/MetaQuotes/Tester directories."""
    appdata_dir = tmp_path / "AppData" / "Roaming"
    metaquotes_tester = appdata_dir / "MetaQuotes" / "Tester" / "Common" / "Agent-01" / "MQL5" / "Files"
    metaquotes_tester.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("APPDATA", str(appdata_dir))

    config = AppConfig.from_env()
    buy_csv = metaquotes_tester / f"{config.symbol}_{config.clean_timeframe}_buy.csv"
    sell_csv = metaquotes_tester / f"{config.symbol}_{config.clean_timeframe}_sell.csv"

    df = pd.DataFrame({"f0": [1.0, 2.0], "label": [1, 0]})
    df.to_csv(buy_csv, index=False)
    df.to_csv(sell_csv, index=False)

    manager = DatasetManager(config, tmp_path / "ws", tmp_path / "term", tmp_path / "comm")
    found_buy, found_sell, _ = manager.find_and_validate_datasets()

    assert found_buy == buy_csv
    assert found_sell == sell_csv
    assert manager.has_existing_datasets() is True


def test_dataset_manager_dump_tester_logs_utf16_and_utf8(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Verify _dump_tester_logs prints lines from latest tester log in UTF-16 and UTF-8."""
    config = AppConfig.from_env()
    term_dir = tmp_path / "terminal"
    tester_logs = term_dir / "Tester" / "logs"
    tester_logs.mkdir(parents=True, exist_ok=True)

    # Write a UTF-16 encoded log file
    log_file = tester_logs / "20260902.log"
    log_file.write_text("Line 1: Tester started\nLine 2: Tester ended\n", encoding="utf-16")

    manager = DatasetManager(config, tmp_path, term_dir, tmp_path / "comm")
    manager._dump_tester_logs(max_lines=10)

    captured = capsys.readouterr().out
    assert "Tester started" in captured
    assert "Tester ended" in captured
