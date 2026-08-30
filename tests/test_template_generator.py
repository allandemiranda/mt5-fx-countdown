"""Tests for MT5 Chart Template (.tpl) Generator."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from src.config import AppConfig
from src.template_generator import TemplateGenerator


def test_template_generator_candlestick_and_colors(monkeypatch: pytest.MonkeyPatch):
    """Verify template has candlestick mode (mode=1), green bull candles, and red bear candles."""
    monkeypatch.setenv("SYMBOL", "EURUSD")
    monkeypatch.setenv("TIMEFRAME", "H1")
    monkeypatch.setenv("USE_BANDS", "1")
    monkeypatch.setenv("USE_FAST_MA", "1")
    monkeypatch.setenv("USE_SLOW_MA", "1")
    monkeypatch.setenv("USE_MACD", "1")
    monkeypatch.setenv("USE_RSI", "1")
    monkeypatch.setenv("USE_STOCHASTIC", "0")
    monkeypatch.setenv("USE_ATR", "1")
    monkeypatch.setenv("USE_ADX", "1")

    config = AppConfig.from_env()

    with tempfile.TemporaryDirectory() as tmp_dir:
        target_dir = Path(tmp_dir)
        generator = TemplateGenerator(config, target_dir, target_dir)
        tpl_path = generator.generate_all([target_dir])

        assert tpl_path.exists()
        assert tpl_path.name == "EURUSD_H1.tpl"

        content = tpl_path.read_text(encoding="ascii")

        # Check Candlestick mode and Colors
        assert "mode=1" in content, "Chart must be configured in Candlestick mode (mode=1)"
        assert "bullcandle_color=65280" in content, "Bullish candle body must be green (65280)"
        assert "bearcandle_color=255" in content, "Bearish candle body must be red (255)"
        assert "barup_color=65280" in content, "Bar up outline must be green (65280)"
        assert "bardown_color=255" in content, "Bar down outline must be red (255)"
        assert "background_color=0" in content, "Background color must be black"

        # Check Indicators
        assert "name=Bollinger Bands" in content
        assert f"period={config.bands_period}" in content
        assert "name=Moving Average" in content
        assert f"period={config.fast_ma_period}" in content
        assert f"period={config.slow_ma_period}" in content
        assert "name=MACD" in content
        assert f"fast_ema={config.macd_fast}" in content
        assert "name=Relative Strength Index" in content
        assert f"period={config.rsi_period}" in content
        assert "name=Average True Range" in content
        assert f"period={config.atr_period}" in content
        assert "name=Average Directional Movement Index" in content
        assert f"period={config.adx_period}" in content

        assert "grid=0" in content, "Background grid must be disabled (grid=0)"
        assert "color=13749760" in content, "Bollinger Bands must use custom turquoise color"
        assert "name=Stochastic Oscillator" not in content

        # Verify replacement of pre-existing template file
        generator.generate_all([target_dir])
        assert tpl_path.exists()


def test_template_generator_toggle_exclusion(monkeypatch: pytest.MonkeyPatch):
    """Verify disabled indicators are excluded from the chart template."""
    monkeypatch.setenv("SYMBOL", "GBPUSD")
    monkeypatch.setenv("TIMEFRAME", "M15")
    monkeypatch.setenv("USE_BANDS", "0")
    monkeypatch.setenv("USE_FAST_MA", "0")
    monkeypatch.setenv("USE_SLOW_MA", "0")
    monkeypatch.setenv("USE_MACD", "0")
    monkeypatch.setenv("USE_RSI", "0")
    monkeypatch.setenv("USE_STOCHASTIC", "1")
    monkeypatch.setenv("USE_ATR", "0")
    monkeypatch.setenv("USE_ADX", "0")

    config = AppConfig.from_env()

    with tempfile.TemporaryDirectory() as tmp_dir:
        target_dir = Path(tmp_dir)
        generator = TemplateGenerator(config, target_dir, target_dir)
        tpl_path = generator.generate_all([target_dir])

        assert tpl_path.name == "GBPUSD_M15.tpl"
        content = tpl_path.read_text(encoding="ascii")

        assert "name=Bollinger Bands" not in content
        assert "name=MACD" not in content
        assert "name=Relative Strength Index" not in content
        assert "name=Average True Range" not in content
        assert "name=Average Directional Movement Index" not in content
        assert "name=Stochastic Oscillator" in content
        assert f"kperiod={config.stoch_k}" in content
