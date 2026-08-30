"""Unit tests for feature schema consistency, dimension parity, and .set preset generation."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.config import AppConfig
from src.preset_generator import PresetGenerator


@pytest.fixture(autouse=True)
def setup_default_env(monkeypatch: pytest.MonkeyPatch):
    """Ensure baseline environment variables are loaded for all feature schema tests."""
    AppConfig.from_env()


def test_feature_vector_dimension_parity_all_combinations(monkeypatch: pytest.MonkeyPatch):
    """Verify that Python feature dimension calculation strictly matches MQL5 FeatureExtractor formula."""
    # Full default configuration with lookback=4 without GARCH:
    # 3(ADX) + 1(ATR) + 2(Bands) + 2(MACD) + 1(FastMA) + 1(SlowMA) + 1(RSI) + 2(Stoch)
    # + 4(Candle) + 1(Week) + 1(Day) + 1(OpenMarkets) + 1(Spread) = 21 base features.
    # Total = 21 * (4 + 1) = 105.
    monkeypatch.setenv("FEATURE_LOOKBACK", "4")
    monkeypatch.setenv("USE_ADX", "1")
    monkeypatch.setenv("USE_ATR", "1")
    monkeypatch.setenv("USE_BANDS", "1")
    monkeypatch.setenv("USE_MACD", "1")
    monkeypatch.setenv("USE_FAST_MA", "1")
    monkeypatch.setenv("USE_SLOW_MA", "1")
    monkeypatch.setenv("USE_RSI", "1")
    monkeypatch.setenv("USE_STOCHASTIC", "1")
    monkeypatch.setenv("USE_CANDLESTICK", "1")
    monkeypatch.setenv("USE_TIMESTAMP_WEEK", "1")
    monkeypatch.setenv("USE_TIMESTAMP_DAY", "1")
    monkeypatch.setenv("USE_OPEN_MARKETS", "1")
    monkeypatch.setenv("USE_SPREAD", "1")
    monkeypatch.setenv("USE_GARCH_FEATURES", "0")

    config = AppConfig.from_env()
    assert config.base_feature_count == 21
    assert config.active_feature_count == 105

    # With GARCH features enabled: 21 + 5 = 26 base features, 26 * 5 = 130
    monkeypatch.setenv("USE_GARCH_FEATURES", "1")
    config_garch = AppConfig.from_env()
    assert config_garch.base_feature_count == 26
    assert config_garch.active_feature_count == 130

    # Lookback variation: L=0 -> 26, L=10 -> 286
    monkeypatch.setenv("FEATURE_LOOKBACK", "0")
    assert AppConfig.from_env().active_feature_count == 26

    monkeypatch.setenv("FEATURE_LOOKBACK", "10")
    assert AppConfig.from_env().active_feature_count == 286

    # Custom indicator subset (GARCH disabled)
    monkeypatch.setenv("USE_GARCH_FEATURES", "0")
    monkeypatch.setenv("USE_ADX", "1")         # 3
    monkeypatch.setenv("USE_ATR", "1")         # 1
    monkeypatch.setenv("USE_BANDS", "0")       # 0
    monkeypatch.setenv("USE_MACD", "0")        # 0
    monkeypatch.setenv("USE_FAST_MA", "0")     # 0
    monkeypatch.setenv("USE_SLOW_MA", "0")     # 0
    monkeypatch.setenv("USE_RSI", "1")         # 1
    monkeypatch.setenv("USE_STOCHASTIC", "0")  # 0
    monkeypatch.setenv("USE_CANDLESTICK", "1")  # 4
    monkeypatch.setenv("USE_TIMESTAMP_WEEK", "0")
    monkeypatch.setenv("USE_TIMESTAMP_DAY", "0")
    monkeypatch.setenv("USE_OPEN_MARKETS", "0")
    monkeypatch.setenv("USE_SPREAD", "0")
    monkeypatch.setenv("FEATURE_LOOKBACK", "3")

    custom_config = AppConfig.from_env()
    # Base features: 3 + 1 + 1 + 4 = 9. Total = 9 * (3 + 1) = 36
    assert custom_config.base_feature_count == 9
    assert custom_config.active_feature_count == 36


def test_preset_generation_and_all_keys(monkeypatch: pytest.MonkeyPatch):
    """Verify that MT5 Expert Preset (.set) files contain all required input keys with proper formatting."""
    monkeypatch.setenv("SYMBOL", "EURUSD")
    monkeypatch.setenv("TIMEFRAME", "H1")
    monkeypatch.setenv("FEATURE_LOOKBACK", "4")
    monkeypatch.setenv("GARCH_HORIZON", "8")
    monkeypatch.setenv("GARCH_ALPHA", "0.08")
    monkeypatch.setenv("GARCH_BETA", "0.90")

    config = AppConfig.from_env()

    with tempfile.TemporaryDirectory() as tmp_dir:
        target_dir = Path(tmp_dir)
        generator = PresetGenerator(config, target_dir, target_dir)
        live_set = generator.generate_all([target_dir])

        assert live_set.exists(), f"Missing LiveONNX preset: {live_set}"
        assert live_set.name == "LiveONNX-EA_EURUSD_H1.set"

        # Verify strictly scoped naming and absence of generic aliases
        live_alias = target_dir / "LiveONNX-EA.set"
        dmatrix_alias = target_dir / "DMatrix-EA.set"
        assert not live_alias.exists(), "Generic alias should not be created"
        assert not dmatrix_alias.exists(), "Generic alias should not be created"

        # Read contents
        live_content = live_set.read_text(encoding="utf-8")

        # 1. Check all required parameters in LiveONNX preset
        required_live_keys = [
            f"InpMagicNumber={config.magic_number}",
            "InpTradeDirection=0",
            "InpMinimalLevelAcceptedBuy=0.50",
            "InpMinimalLevelAcceptedSell=0.50",
            "InpLotSize=0.01",
            f"InpFeatureLookback={config.feature_lookback}",
            "InpUseADX=1",
            "InpUseATR=1",
            "InpUseBands=1",
            "InpUseMACD=1",
            "InpUseFastMA=1",
            "InpUseSlowMA=1",
            "InpUseRSI=1",
            f"InpUseStochastic={1 if config.use_stochastic else 0}",
            "InpUseCandlestick=1",
            "InpUseTimestampWeek=1",
            "InpUseTimestampDay=1",
            "InpUseOpenMarkets=1",
            "InpUseSpread=1",
            f"InpUseGarchFeatures={1 if config.use_garch_features else 0}",
            f"InpGarchHorizon={config.garch_horizon}",
            f"InpPriceSize={config.price_size}",
            f"InpGarchAlpha={config.garch_alpha}",
            f"InpGarchBeta={config.garch_beta}",
            "InpEnableSRSnapping=1",
            "InpSRLookbackBars=12",
            "InpSRPivotStrength=2",
            "InpSROffsetPoints=30",
            "InpSRZoneSelection=0",
            "InpEnableRiskFilter=1",
            "InpEnableDynamicLotSizing=0",
            "InpMaxLotSize=0.05",
            "InpMarginSafetyMultiplier=1.5",
            "InpMaxRiskRewardRatio=1.5",
            "InpMaxTradeRiskPct=3.0",
            "InpEnableCalendarFilter=1",
            "InpEnableNewsFilter=1",
            f"InpRiskGarchHorizon={config.garch_horizon}",
            "InpKTP=1.5",
            "InpKSL=1.5",
            f"InpTradeMonday={1 if config.trade_monday else 0}",
            f"InpMondayStartTime={config.trade_monday_start}",
            f"InpMondayEndTime={config.trade_monday_end}",
            f"InpTradeFriday={1 if config.trade_friday else 0}",
            f"InpFridayStartTime={config.trade_friday_start}",
            f"InpFridayEndTime={config.trade_friday_end}",
            "InpModelBuyPath=Models/EURUSD_H1_model_buy.onnx",
            "InpModelSellPath=Models/EURUSD_H1_model_sell.onnx",
            f"InpADXPeriod={config.adx_period}",
            f"InpATRPeriod={config.atr_period}",
            f"InpBandsPeriod={config.bands_period}",
            f"InpBandsShift={config.bands_shift}",
            f"InpBandsDev={config.bands_dev}",
            f"InpBandsAppliedPrice={config.bands_applied_price}",
            f"InpMACDFastPeriod={config.macd_fast}",
            f"InpMACDSlowPeriod={config.macd_slow}",
            f"InpMACDSignalPeriod={config.macd_signal}",
            f"InpMACDAppliedPrice={config.macd_applied_price}",
            f"InpFastMAPeriod={config.fast_ma_period}",
            f"InpFastMAShift={config.fast_ma_shift}",
            f"InpFastMAMethod={config.fast_ma_method}",
            f"InpFastMAAppliedPrice={config.fast_ma_applied_price}",
            f"InpSlowMAPeriod={config.slow_ma_period}",
            f"InpSlowMAShift={config.slow_ma_shift}",
            f"InpSlowMAMethod={config.slow_ma_method}",
            f"InpSlowMAAppliedPrice={config.slow_ma_applied_price}",
            f"InpRSIPeriod={config.rsi_period}",
            f"InpRSIAppliedPrice={config.rsi_applied_price}",
            f"InpStochK={config.stoch_k}",
            f"InpStochD={config.stoch_d}",
            f"InpStochSlowing={config.stoch_slowing}",
            f"InpStochMethod={config.stoch_method}",
            f"InpStochPriceField={config.stoch_price_field}",
            "InpIgnoreAudit=0",
        ]

        for key in required_live_keys:
            assert key in live_content, f"LiveONNX preset missing required key: {key}"

        # Assert LiveONNX does NOT contain pandemic blackout keys
        assert "InpAvoidPandemicTime" not in live_content
        assert "InpPandemicStartTime" not in live_content
        assert "InpPandemicEndTime" not in live_content

        # Assert DMatrix preset DOES contain pandemic blackout keys
        dmat_content = generator.build_dmatrix_preset_content()
        assert f"InpAvoidPandemicTime={'1' if config.avoid_pandemictime else '0'}" in dmat_content
        assert f"InpPandemicStartTime={config.pandemic_start_date}" in dmat_content
        assert f"InpPandemicEndTime={config.pandemic_end_date}" in dmat_content


def test_time_series_chronological_split():
    """Verify that dataset splits strictly follow chronological order without data leakage."""
    np.random.seed(42)
    n_samples = 200
    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="1h")
    values = np.random.randn(n_samples)

    df = pd.DataFrame({"timestamp": timestamps, "feature": values, "label": (values > 0).astype(int)})

    val_percentage = 0.20
    val_size = int(len(df) * val_percentage)
    train_size = len(df) - val_size

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    # Zero overlap & chronological continuity
    assert train_df["timestamp"].max() < val_df["timestamp"].min()
    assert len(train_df) == 160
    assert len(val_df) == 40
    assert len(train_df) + len(val_df) == n_samples
    assert set(train_df.index).isdisjoint(set(val_df.index))


def test_chronological_split_edge_cases():
    """Verify validation split behavior on small and boundary dataset sizes."""
    # Small dataset of 20 samples with 20% validation
    n_samples = 20
    df = pd.DataFrame({
        "f0": np.linspace(0, 19, n_samples),
        "label": np.random.randint(0, 2, n_samples),
    })

    val_size = int(len(df) * 0.20)
    if val_size < 10:
        val_size = max(5, int(len(df) * 0.2))
    train_size = len(df) - val_size

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    assert len(train_df) == 15
    assert len(val_df) == 5
    assert train_df["f0"].max() < val_df["f0"].min()
