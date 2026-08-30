"""Unit tests for Triple Barrier labeling, GARCH features schema, and position control parameters."""

from __future__ import annotations

from pathlib import Path
import tempfile
import pytest

from src.config import AppConfig
from src.preset_generator import PresetGenerator
from src.mt5_client import MT5Client


@pytest.fixture(autouse=True)
def setup_default_env(monkeypatch: pytest.MonkeyPatch):
    """Ensure baseline environment variables are loaded."""
    AppConfig.from_env()


def test_triple_barrier_config_defaults(monkeypatch: pytest.MonkeyPatch):
    """Verify that new Triple Barrier and GARCH parameters load properly or fallback safely."""
    monkeypatch.setenv("USE_GARCH_FEATURES", "1")
    monkeypatch.setenv("LABEL_HORIZON_BARS", "15")
    monkeypatch.setenv("LABEL_MIN_POINTS", "200")
    monkeypatch.setenv("LABEL_MAX_ADVERSE_POINTS", "120")

    cfg = AppConfig.from_env(load_env_file=False)
    assert cfg.use_garch_features is True
    assert cfg.label_horizon_bars == 15
    assert cfg.label_min_points == 200
    assert cfg.label_max_adverse_points == 120


def test_triple_barrier_fallbacks(monkeypatch: pytest.MonkeyPatch):
    """Verify default fallbacks when optional keys are omitted."""
    monkeypatch.delenv("USE_GARCH_FEATURES", raising=False)
    monkeypatch.delenv("LABEL_HORIZON_BARS", raising=False)
    monkeypatch.delenv("LABEL_MIN_POINTS", raising=False)
    monkeypatch.delenv("LABEL_MAX_ADVERSE_POINTS", raising=False)

    cfg = AppConfig.from_env(load_env_file=False)
    assert cfg.use_garch_features is True
    assert cfg.label_horizon_bars == 12
    assert cfg.label_min_points == 150
    assert cfg.label_max_adverse_points == 150


def test_dmatrix_and_live_preset_content_triple_barrier():
    """Verify that generated presets contain feature parity and clean LiveONNX settings."""
    cfg = AppConfig.from_env()

    with tempfile.TemporaryDirectory() as tmp_dir:
        target_dir = Path(tmp_dir)
        generator = PresetGenerator(cfg, target_dir, target_dir)

        live_content = generator.build_live_preset_content()
        assert "InpMaxPositions=" not in live_content
        assert "InpHoldingBarsTimeout=" not in live_content
        assert "InpEnableSRSnapping=1" in live_content
        assert "InpSRLookbackBars=12" in live_content
        assert "InpSRPivotStrength=2" in live_content
        assert "InpSROffsetPoints=30" in live_content
        assert "InpSRZoneSelection=0" in live_content
        assert "InpEnableRiskFilter=1" in live_content
        assert "InpEnableDynamicLotSizing=0" in live_content
        assert "InpMaxLotSize=0.05" in live_content
        assert "InpMarginSafetyMultiplier=1.5" in live_content
        assert "InpMaxRiskRewardRatio=1.5" in live_content
        assert "InpMaxTradeRiskPct=3.0" in live_content
        assert "InpEnableCalendarFilter=1" in live_content
        assert "InpCalendarTrailingPoints=" not in live_content
        assert f"InpRiskGarchHorizon={cfg.garch_horizon}" in live_content
        assert "InpEnableNewsFilter=1" in live_content
        assert "InpKTP=1.5" in live_content
        assert "InpKSL=1.5" in live_content
        assert "InpTradeMonday=1" in live_content
        assert f"InpUseGarchFeatures={1 if cfg.use_garch_features else 0}" in live_content

        dmatrix_content = generator.build_dmatrix_preset_content()
        assert f"InpUseGarchFeatures={1 if cfg.use_garch_features else 0}" in dmatrix_content
        assert f"InpLabelHorizonBars={cfg.label_horizon_bars}" in dmatrix_content
        assert f"InpLabelMinPoints={cfg.label_min_points}" in dmatrix_content
        assert f"InpLabelMaxAdversePoints={cfg.label_max_adverse_points}" in dmatrix_content
        assert "InpTradeMonday=1" in dmatrix_content


def test_mt5_client_tester_config_includes_triple_barrier(tmp_path: Path):
    """Verify that MT5Client generates .ini with Triple Barrier and GARCH inputs."""
    cfg = AppConfig.from_env()
    client = MT5Client(cfg, tmp_path)
    client.terminal_data_path = tmp_path

    ini_file = client.generate_tester_ini()
    assert ini_file.exists()
    content = ini_file.read_text(encoding="ascii")

    assert f"InpUseGarchFeatures={'true' if cfg.use_garch_features else 'false'}" in content
    assert f"InpLabelHorizonBars={cfg.label_horizon_bars}" in content
    assert f"InpLabelMinPoints={cfg.label_min_points}" in content
    assert f"InpLabelMaxAdversePoints={cfg.label_max_adverse_points}" in content
    assert "InpTradeMonday=true" in content
