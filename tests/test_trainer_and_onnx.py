"""Integration tests for DualXGBoostTrainer and ONNXExporter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import AppConfig
from src.onnx_exporter import ONNXExporter
from src.trainer import DualXGBoostTrainer


@pytest.fixture(autouse=True)
def setup_default_env(monkeypatch: pytest.MonkeyPatch):
    """Ensure baseline environment variables are loaded for all trainer and onnx tests."""
    AppConfig.from_env()


def test_dual_xgboost_trainer_and_onnx_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test full training on synthetic dataset, early stopping, and ONNX export for BUY and SELL."""
    monkeypatch.setenv("OPTUNA_TRIALS", "2")
    monkeypatch.setenv("XGB_ROUNDS", "10")
    monkeypatch.setenv("XGB_EARLY_STOPPING_ROUNDS", "5")
    monkeypatch.setenv("VALIDATION_PERCENTAGE", "0.20")

    config = AppConfig.from_env()

    # Generate synthetic CSV dataset
    num_features = 20
    num_samples = 120
    rng = np.random.default_rng(42)
    x = rng.standard_normal((num_samples, num_features), dtype=np.float32)
    y_buy = (x[:, 0] + x[:, 1] > 0).astype(int)
    y_sell = (x[:, 0] - x[:, 1] > 0).astype(int)

    cols = [f"feat_{i}" for i in range(num_features)]

    df_buy = pd.DataFrame(x, columns=cols)
    df_buy["label"] = y_buy
    buy_csv = tmp_path / "EURUSD_H1_buy.csv"
    df_buy.to_csv(buy_csv, index=False)

    df_sell = pd.DataFrame(x, columns=cols)
    df_sell["label"] = y_sell
    sell_csv = tmp_path / "EURUSD_H1_sell.csv"
    df_sell.to_csv(sell_csv, index=False)

    trainer = DualXGBoostTrainer(config)

    # 1. Train BUY model
    clf_buy, metrics_buy, feat_names_buy = trainer.train(buy_csv, "buy")
    assert len(feat_names_buy) == num_features
    assert metrics_buy["direction"] == "BUY"
    assert "roc_auc" in metrics_buy
    assert "accuracy" in metrics_buy
    assert "log_loss" in metrics_buy
    assert "best_iteration" in metrics_buy

    # 2. Train SELL model
    clf_sell, metrics_sell, feat_names_sell = trainer.train(sell_csv, "sell")
    assert len(feat_names_sell) == num_features
    assert metrics_sell["direction"] == "SELL"

    # 3. Export to pure ONNX
    exporter = ONNXExporter(config, terminal_data_path=tmp_path / "term", common_path=tmp_path / "comm")
    buy_onnx = exporter.export_and_validate(clf_buy, len(feat_names_buy), "buy")
    sell_onnx = exporter.export_and_validate(clf_sell, len(feat_names_sell), "sell")

    assert buy_onnx.exists()
    assert sell_onnx.exists()
    assert buy_onnx.name == f"{config.symbol}_{config.clean_timeframe}_model_buy.onnx"
    assert sell_onnx.name == f"{config.symbol}_{config.clean_timeframe}_model_sell.onnx"

    # 4. Deploy models and metadata
    metadata = {
        "symbol": config.symbol,
        "timeframe": config.clean_timeframe,
        "num_features": num_features,
        "metrics": {"buy": metrics_buy, "sell": metrics_sell},
    }
    exporter.deploy(buy_onnx, sell_onnx, metadata)

    term_models = tmp_path / "term" / "MQL5" / "Files" / "Models"
    comm_models = tmp_path / "comm" / "Files" / "Models"

    assert (term_models / f"{config.symbol}_{config.clean_timeframe}_model_buy.onnx").exists()
    assert (term_models / f"{config.symbol}_{config.clean_timeframe}_model_sell.onnx").exists()
    assert (comm_models / f"{config.symbol}_{config.clean_timeframe}_model_buy.onnx").exists()
    assert (comm_models / f"{config.symbol}_{config.clean_timeframe}_model_sell.onnx").exists()

    meta_deployed = term_models / f"{config.symbol}_{config.clean_timeframe}_metadata.json"
    assert meta_deployed.exists()
    with open(meta_deployed, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    assert meta_data["symbol"] == config.symbol
    assert meta_data["metrics"]["buy"]["direction"] == "BUY"


def test_trainer_insufficient_samples_raises(tmp_path: Path):
    """Verify trainer raises ValueError when dataset has fewer than 10 samples."""
    config = AppConfig.from_env()
    trainer = DualXGBoostTrainer(config)

    df_tiny = pd.DataFrame({"f0": [1.0, 2.0, 3.0], "label": [0, 1, 0]})
    tiny_csv = tmp_path / "EURUSD_H1_tiny.csv"
    df_tiny.to_csv(tiny_csv, index=False)

    with pytest.raises(ValueError, match="too few samples"):
        trainer.train(tiny_csv, "buy")


def test_trainer_missing_label_raises(tmp_path: Path):
    """Verify trainer raises ValueError when CSV lacks a 'label' column."""
    config = AppConfig.from_env()
    trainer = DualXGBoostTrainer(config)

    df_nolabel = pd.DataFrame({"f0": np.random.randn(20), "f1": np.random.randn(20)})
    nolabel_csv = tmp_path / "EURUSD_H1_nolabel.csv"
    df_nolabel.to_csv(nolabel_csv, index=False)

    with pytest.raises(ValueError, match="missing 'label' column"):
        trainer.train(nolabel_csv, "buy")


def test_trainer_timestamp_and_elapsed_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    """Verify trainer prints start timestamp, end timestamp, and elapsed duration formatted as HH:MM:SS."""
    monkeypatch.setenv("OPTUNA_TRIALS", "1")
    monkeypatch.setenv("XGB_ROUNDS", "5")
    monkeypatch.setenv("XGB_EARLY_STOPPING_ROUNDS", "2")
    monkeypatch.setenv("VALIDATION_PERCENTAGE", "0.20")

    config = AppConfig.from_env()
    trainer = DualXGBoostTrainer(config)

    num_samples = 25
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "f0": rng.standard_normal(num_samples),
        "f1": rng.standard_normal(num_samples),
        "label": rng.integers(0, 2, size=num_samples),
    })
    csv_path = tmp_path / "EURUSD_M15_buy.csv"
    df.to_csv(csv_path, index=False)

    trainer.train(csv_path, "buy")
    captured = capsys.readouterr().out

    import re
    assert "[*] Optimizing and Training XGBoost BUY Model on 'EURUSD_M15_buy.csv'..." in captured
    assert re.search(r"\[\*\] Training started at: \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]", captured)
    assert re.search(
        r"\[\*\] Training completed at: \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \(Elapsed: \d{2}:\d{2}:\d{2}\)",
        captured,
    )
