"""Unit tests for src/config.py strictly typed configuration dataclass."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from src.config import AppConfig, DirectionalXGBConfig


@pytest.fixture(autouse=True)
def setup_default_env(monkeypatch: pytest.MonkeyPatch):
    """Ensure baseline environment variables are loaded for all config tests."""
    AppConfig.from_env()


def test_app_config_load_and_types():
    """Verify AppConfig loads properly with correct types from environment."""
    config = AppConfig.from_env()

    # Paths
    assert isinstance(config.mt5_path, Path)
    assert isinstance(config.metaeditor_path, Path)

    # Backtest params
    assert isinstance(config.symbol, str)
    assert isinstance(config.timeframe, str)
    assert isinstance(config.magic_number, int)
    assert isinstance(config.from_date, str)
    assert isinstance(config.to_date, str)
    assert isinstance(config.shutdown_terminal, int)
    assert isinstance(config.skip_dataset_generation, bool)
    assert isinstance(config.avoid_pandemictime, bool)
    assert isinstance(config.pandemic_start_date, str)
    assert isinstance(config.pandemic_end_date, str)

    # GARCH & ML parameters
    assert isinstance(config.feature_lookback, int)
    assert isinstance(config.garch_horizon, int)
    assert isinstance(config.garch_alpha, float)
    assert isinstance(config.garch_beta, float)
    assert isinstance(config.xgb_max_depth, int)
    assert isinstance(config.xgb_eta, float)
    assert isinstance(config.xgb_early_stopping_rounds, int)
    assert config.xgb_early_stopping_rounds > 0
    assert isinstance(config.validation_percentage, float)

    # Hyperparameters & Toggles
    assert isinstance(config.optuna_trials, int)
    assert isinstance(config.use_adx, bool)
    assert isinstance(config.use_candlestick, bool)
    assert isinstance(config.use_garch_features, bool)
    assert isinstance(config.label_horizon_bars, int)
    assert isinstance(config.label_min_points, int)
    assert isinstance(config.label_max_adverse_points, int)
    assert isinstance(config.trade_monday, bool)
    assert isinstance(config.trade_monday_start, str)
    assert isinstance(config.trade_monday_end, str)
    assert isinstance(config.trade_tuesday, bool)
    assert isinstance(config.trade_wednesday, bool)
    assert isinstance(config.trade_thursday, bool)
    assert isinstance(config.trade_friday, bool)
    assert config.clean_timeframe == config.timeframe.replace("PERIOD_", "")
    assert config.active_feature_count > 0

    # Directional Evaluation & Sensitivity Grid
    assert isinstance(config.eval_classification_threshold, float)
    assert 0.0 < config.eval_classification_threshold < 1.0
    assert isinstance(config.optuna_objective_metric, str)
    assert config.optuna_objective_metric in {"logloss", "roc_auc", "precision", "f1"}
    assert isinstance(config.eval_enable_threshold_grid, bool)
    assert isinstance(config.eval_threshold_min, float)
    assert isinstance(config.eval_threshold_max, float)
    assert isinstance(config.eval_threshold_step, float)
    assert 0.0 <= config.eval_threshold_min < config.eval_threshold_max <= 1.0
    assert config.eval_threshold_step > 0.0


def test_clean_timeframe_variations(monkeypatch: pytest.MonkeyPatch):
    """Verify clean_timeframe strips 'PERIOD_' prefix if present or retains clean string."""
    monkeypatch.setenv("TIMEFRAME", "PERIOD_M15")
    cfg1 = AppConfig.from_env(load_env_file=False)
    assert cfg1.clean_timeframe == "M15"

    monkeypatch.setenv("TIMEFRAME", "PERIOD_H4")
    cfg2 = AppConfig.from_env(load_env_file=False)
    assert cfg2.clean_timeframe == "H4"

    monkeypatch.setenv("TIMEFRAME", "D1")
    cfg3 = AppConfig.from_env(load_env_file=False)
    assert cfg3.clean_timeframe == "D1"


def test_active_and_base_feature_count_permutations(monkeypatch: pytest.MonkeyPatch):
    """Verify base_feature_count and active_feature_count calculations across all toggles."""
    # 1. All features disabled
    toggles = [
        "USE_ADX", "USE_ATR", "USE_BANDS", "USE_MACD", "USE_FAST_MA", "USE_SLOW_MA",
        "USE_RSI", "USE_STOCHASTIC", "USE_CANDLESTICK", "USE_TIMESTAMP_WEEK",
        "USE_TIMESTAMP_DAY", "USE_OPEN_MARKETS", "USE_SPREAD", "USE_GARCH_FEATURES",
    ]
    for toggle in toggles:
        monkeypatch.setenv(toggle, "0")
    monkeypatch.setenv("FEATURE_LOOKBACK", "0")

    cfg_none = AppConfig.from_env(load_env_file=False)
    assert cfg_none.base_feature_count == 0
    assert cfg_none.active_feature_count == 0

    # 2. Enable each feature group individually with lookback=0
    expected_weights = {
        "USE_ADX": 3,
        "USE_ATR": 1,
        "USE_BANDS": 2,
        "USE_MACD": 2,
        "USE_FAST_MA": 1,
        "USE_SLOW_MA": 1,
        "USE_RSI": 1,
        "USE_STOCHASTIC": 2,
        "USE_CANDLESTICK": 4,
        "USE_TIMESTAMP_WEEK": 1,
        "USE_TIMESTAMP_DAY": 1,
        "USE_OPEN_MARKETS": 1,
        "USE_SPREAD": 1,
        "USE_GARCH_FEATURES": 5,
    }

    for toggle_key, weight in expected_weights.items():
        monkeypatch.setenv(toggle_key, "1")
        cfg_single = AppConfig.from_env(load_env_file=False)
        assert cfg_single.base_feature_count == weight, f"Failed for {toggle_key}"
        assert cfg_single.active_feature_count == weight * (0 + 1)
        monkeypatch.setenv(toggle_key, "0")

    # 3. All features enabled with lookback=4
    for toggle in toggles:
        monkeypatch.setenv(toggle, "1")
    monkeypatch.setenv("FEATURE_LOOKBACK", "4")

    cfg_all = AppConfig.from_env(load_env_file=False)
    assert cfg_all.base_feature_count == 26  # 21 indicators + 5 GARCH features
    assert cfg_all.active_feature_count == 26 * (4 + 1)  # 130


@pytest.mark.parametrize("missing_key", [
    "SYMBOL",
    "TIMEFRAME",
    "FROM_DATE",
    "TO_DATE",
    "MT5_PATH",
    "METAEDITOR_PATH",
    "FEATURE_LOOKBACK",
    "GARCH_ALPHA",
    "XGB_MAX_DEPTH",
    "USE_ADX",
    "ADX_PERIOD",
    "SKIP_DATASET_GENERATION",
])
def test_missing_mandatory_keys_raise_error(missing_key: str, monkeypatch: pytest.MonkeyPatch):
    """Verify that AppConfig strictly rejects missing mandatory parameters without hidden defaults."""
    monkeypatch.delenv(missing_key, raising=False)
    with pytest.raises(ValueError, match=f"Mandatory configuration parameter '{missing_key}' is missing or empty."):
        AppConfig.from_env(load_env_file=False)


@pytest.mark.parametrize("empty_value", ["", "   ", "\t"])
def test_empty_string_keys_raise_error(empty_value: str, monkeypatch: pytest.MonkeyPatch):
    """Verify that whitespace-only or empty strings for mandatory parameters raise ValueError."""
    monkeypatch.setenv("SYMBOL", empty_value)
    with pytest.raises(ValueError, match="Mandatory configuration parameter 'SYMBOL' is missing or empty."):
        AppConfig.from_env(load_env_file=False)


def test_app_config_invalid_int(monkeypatch: pytest.MonkeyPatch):
    """Verify AppConfig raises ValueError for non-integer int fields."""
    monkeypatch.setenv("FEATURE_LOOKBACK", "not_a_number")
    with pytest.raises(ValueError, match="must be a valid integer"):
        AppConfig.from_env(load_env_file=False)

    monkeypatch.setenv("FEATURE_LOOKBACK", "100.55")
    with pytest.raises(ValueError, match="must be a valid integer"):
        AppConfig.from_env(load_env_file=False)


def test_app_config_invalid_float(monkeypatch: pytest.MonkeyPatch):
    """Verify AppConfig raises ValueError for non-float float fields."""
    monkeypatch.setenv("GARCH_ALPHA", "invalid_float")
    with pytest.raises(ValueError, match="must be a valid float"):
        AppConfig.from_env(load_env_file=False)


@pytest.mark.parametrize("invalid_bool", ["maybe", "2", "none", "enabled", "on"])
def test_app_config_invalid_bool(invalid_bool: str, monkeypatch: pytest.MonkeyPatch):
    """Verify AppConfig raises ValueError for invalid boolean fields."""
    monkeypatch.setenv("USE_ADX", invalid_bool)
    with pytest.raises(ValueError, match="must be a boolean"):
        AppConfig.from_env(load_env_file=False)


@pytest.mark.parametrize("truthy_val", ["1", "true", "TRUE", "True", "yes", "YES"])
def test_app_config_valid_bool_truthy(truthy_val: str, monkeypatch: pytest.MonkeyPatch):
    """Verify boolean parsing handles all standard truthy formats."""
    monkeypatch.setenv("USE_ADX", truthy_val)
    cfg = AppConfig.from_env(load_env_file=False)
    assert cfg.use_adx is True


@pytest.mark.parametrize("falsy_val", ["0", "false", "FALSE", "False", "no", "NO"])
def test_app_config_valid_bool_falsy(falsy_val: str, monkeypatch: pytest.MonkeyPatch):
    """Verify boolean parsing handles all standard falsy formats."""
    monkeypatch.setenv("USE_ADX", falsy_val)
    cfg = AppConfig.from_env(load_env_file=False)
    assert cfg.use_adx is False


def test_app_config_immutability():
    """Verify AppConfig is a frozen dataclass preventing accidental runtime modification."""
    config = AppConfig.from_env()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.symbol = "EURUSD"  # type: ignore


def test_optional_mt5_paths(monkeypatch: pytest.MonkeyPatch):
    """Verify MT5_DATA_PATH and MT5_COMMON_PATH parse to Path or None when empty."""
    monkeypatch.setenv("MT5_DATA_PATH", "C:/Custom/MT5/Data")
    monkeypatch.setenv("MT5_COMMON_PATH", "C:/Custom/MT5/Common")
    cfg = AppConfig.from_env(load_env_file=False)
    assert cfg.mt5_data_path == Path("C:/Custom/MT5/Data")
    assert cfg.mt5_common_path == Path("C:/Custom/MT5/Common")

    monkeypatch.setenv("MT5_DATA_PATH", "")
    monkeypatch.setenv("MT5_COMMON_PATH", "   ")
    cfg2 = AppConfig.from_env(load_env_file=False)
    assert cfg2.mt5_data_path is None
    assert cfg2.mt5_common_path is None


def test_eval_metrics_and_grid_validation(monkeypatch: pytest.MonkeyPatch):
    """Verify bounds checking and validation logic for evaluation and grid parameters."""
    # 1. Invalid EVAL_CLASSIFICATION_THRESHOLD (<= 0 or >= 1)
    monkeypatch.setenv("EVAL_CLASSIFICATION_THRESHOLD", "0.0")
    with pytest.raises(ValueError, match=r"must be between 0\.0 and 1\.0"):
        AppConfig.from_env(load_env_file=False)

    monkeypatch.setenv("EVAL_CLASSIFICATION_THRESHOLD", "1.0")
    with pytest.raises(ValueError, match=r"must be between 0\.0 and 1\.0"):
        AppConfig.from_env(load_env_file=False)

    # 2. Invalid OPTUNA_OBJECTIVE_METRIC
    monkeypatch.setenv("EVAL_CLASSIFICATION_THRESHOLD", "0.50")
    monkeypatch.setenv("OPTUNA_OBJECTIVE_METRIC", "invalid_metric")
    with pytest.raises(ValueError, match="OPTUNA_OBJECTIVE_METRIC.*must be one of"):
        AppConfig.from_env(load_env_file=False)

    # 3. Invalid EVAL_THRESHOLD_MIN / MAX bounds (min >= max)
    monkeypatch.setenv("OPTUNA_OBJECTIVE_METRIC", "logloss")
    monkeypatch.setenv("EVAL_THRESHOLD_MIN", "0.80")
    monkeypatch.setenv("EVAL_THRESHOLD_MAX", "0.60")
    with pytest.raises(ValueError, match="must be strictly less than 'EVAL_THRESHOLD_MAX'"):
        AppConfig.from_env(load_env_file=False)

    # 4. Invalid EVAL_THRESHOLD_STEP (<= 0)
    monkeypatch.setenv("EVAL_THRESHOLD_MIN", "0.40")
    monkeypatch.setenv("EVAL_THRESHOLD_MAX", "0.70")
    monkeypatch.setenv("EVAL_THRESHOLD_STEP", "0.0")
    with pytest.raises(ValueError, match="must be strictly positive"):
        AppConfig.from_env(load_env_file=False)


def test_directional_xgb_config_and_fallback(monkeypatch: pytest.MonkeyPatch):
    """Verify get_directional_config returns transparent fallbacks or directional overrides."""
    # 1. Base config without directional overrides (must fallback to global settings)
    all_directional_keys = [
        "XGB_BUY_MAX_DEPTH", "XGB_BUY_ETA", "XGB_BUY_SUBSAMPLE", "XGB_BUY_COLSAMPLE_BYTREE",
        "XGB_BUY_MIN_CHILD_WEIGHT", "XGB_BUY_LAMBDA", "XGB_BUY_ALPHA", "XGB_BUY_ROUNDS",
        "XGB_BUY_EARLY_STOPPING_ROUNDS", "OPTUNA_BUY_TRIALS", "OPTUNA_BUY_OBJECTIVE_METRIC",
        "EVAL_BUY_CLASSIFICATION_THRESHOLD",
        "XGB_SELL_MAX_DEPTH", "XGB_SELL_ETA", "XGB_SELL_SUBSAMPLE", "XGB_SELL_COLSAMPLE_BYTREE",
        "XGB_SELL_MIN_CHILD_WEIGHT", "XGB_SELL_LAMBDA", "XGB_SELL_ALPHA", "XGB_SELL_ROUNDS",
        "XGB_SELL_EARLY_STOPPING_ROUNDS", "OPTUNA_SELL_TRIALS", "OPTUNA_SELL_OBJECTIVE_METRIC",
        "EVAL_SELL_CLASSIFICATION_THRESHOLD",
    ]
    for k in all_directional_keys:
        monkeypatch.delenv(k, raising=False)

    config = AppConfig.from_env(load_env_file=False)

    buy_cfg = config.get_directional_config("buy")
    sell_cfg = config.get_directional_config("SELL")

    assert isinstance(buy_cfg, DirectionalXGBConfig)
    assert isinstance(sell_cfg, DirectionalXGBConfig)

    # Validate fallbacks match global AppConfig
    assert buy_cfg.max_depth == config.xgb_max_depth
    assert buy_cfg.eta == config.xgb_eta
    assert buy_cfg.subsample == config.xgb_subsample
    assert buy_cfg.colsample_bytree == config.xgb_colsample_bytree
    assert buy_cfg.min_child_weight == config.xgb_min_child_weight
    assert buy_cfg.reg_lambda == config.xgb_lambda
    assert buy_cfg.reg_alpha == config.xgb_alpha
    assert buy_cfg.rounds == config.xgb_rounds
    assert buy_cfg.early_stopping_rounds == config.xgb_early_stopping_rounds
    assert buy_cfg.optuna_trials == config.optuna_trials
    assert buy_cfg.optuna_objective_metric == config.optuna_objective_metric
    assert buy_cfg.classification_threshold == config.eval_classification_threshold

    assert sell_cfg.max_depth == config.xgb_max_depth
    assert sell_cfg.classification_threshold == config.eval_classification_threshold

    # 2. Overrides for BUY only
    monkeypatch.setenv("XGB_BUY_MAX_DEPTH", "6")
    monkeypatch.setenv("XGB_BUY_ETA", "0.03")
    monkeypatch.setenv("XGB_BUY_ALPHA", "0.1")
    monkeypatch.setenv("OPTUNA_BUY_TRIALS", "45")
    monkeypatch.setenv("OPTUNA_BUY_OBJECTIVE_METRIC", "precision")
    monkeypatch.setenv("EVAL_BUY_CLASSIFICATION_THRESHOLD", "0.48")

    cfg_overridden = AppConfig.from_env(load_env_file=False)
    buy_overridden = cfg_overridden.get_directional_config("buy")
    sell_overridden = cfg_overridden.get_directional_config("sell")

    # BUY should have overridden values
    assert buy_overridden.max_depth == 6
    assert buy_overridden.eta == 0.03
    assert buy_overridden.reg_alpha == 0.1
    assert buy_overridden.optuna_trials == 45
    assert buy_overridden.optuna_objective_metric == "precision"
    assert buy_overridden.classification_threshold == 0.48
    # BUY unchanged parameters should still fallback to global
    assert buy_overridden.subsample == cfg_overridden.xgb_subsample

    # SELL should remain unchanged (falling back to global)
    assert sell_overridden.max_depth == cfg_overridden.xgb_max_depth
    assert sell_overridden.eta == cfg_overridden.xgb_eta
    assert sell_overridden.classification_threshold == cfg_overridden.eval_classification_threshold

    # 3. Invalid direction error
    with pytest.raises(ValueError, match="Invalid direction 'hold'"):
        config.get_directional_config("hold")


def test_env_and_example_parity_and_no_liveonnx_only_keys():
    """Verify .env and .env.example contain all AppConfig parameters and exclude LiveONNX-only parameters."""
    env_path = Path(".env")
    example_path = Path(".env.example")
    assert env_path.exists(), "Missing .env file"
    assert example_path.exists(), "Missing .env.example file"

    def parse_env_keys(path: Path) -> set[str]:
        keys = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key = line.split("=", 1)[0].strip()
                keys.add(key)
        return keys

    env_keys = parse_env_keys(env_path)
    example_keys = parse_env_keys(example_path)

    # 1. Verify critical pipeline keys in AppConfig are present in both files
    critical_pipeline_keys = {
        "MT5_PATH", "METAEDITOR_PATH", "SYMBOL", "TIMEFRAME", "MAGIC_NUMBER", "FROM_DATE", "TO_DATE",
        "SHUTDOWN_TERMINAL", "BACKTEST_TIMEOUT", "SKIP_DATASET_GENERATION",
        "AVOID_PANDEMICTIME", "PANDEMIC_START_DATE", "PANDEMIC_END_DATE",
        "FEATURE_LOOKBACK", "GARCH_HORIZON", "PRICE_SIZE", "GARCH_ALPHA", "GARCH_BETA",
        "LABEL_HORIZON_BARS", "LABEL_MIN_POINTS", "LABEL_MAX_ADVERSE_POINTS",
        "TRADE_MONDAY", "TRADE_MONDAY_START", "TRADE_MONDAY_END",
        "TRADE_TUESDAY", "TRADE_TUESDAY_START", "TRADE_TUESDAY_END",
        "TRADE_WEDNESDAY", "TRADE_WEDNESDAY_START", "TRADE_WEDNESDAY_END",
        "TRADE_THURSDAY", "TRADE_THURSDAY_START", "TRADE_THURSDAY_END",
        "TRADE_FRIDAY", "TRADE_FRIDAY_START", "TRADE_FRIDAY_END",
        "XGB_MAX_DEPTH", "XGB_ETA", "XGB_SUBSAMPLE", "XGB_COLSAMPLE_BYTREE",
        "XGB_MIN_CHILD_WEIGHT", "XGB_LAMBDA", "XGB_ALPHA", "XGB_ROUNDS",
        "XGB_EARLY_STOPPING_ROUNDS", "VALIDATION_PERCENTAGE", "OPTUNA_TRIALS",
        "EVAL_CLASSIFICATION_THRESHOLD", "OPTUNA_OBJECTIVE_METRIC",
        "EVAL_ENABLE_THRESHOLD_GRID", "EVAL_THRESHOLD_MIN", "EVAL_THRESHOLD_MAX", "EVAL_THRESHOLD_STEP",
    }
    missing_in_env = critical_pipeline_keys - env_keys
    missing_in_example = critical_pipeline_keys - example_keys
    assert not missing_in_env, f".env is missing critical pipeline keys: {missing_in_env}"
    assert not missing_in_example, f".env.example is missing critical pipeline keys: {missing_in_example}"

    # 2. Verify LiveONNX-only parameters are NOT present in .env or .env.example
    liveonnx_only_keys = {
        "INP_ENABLE_SR_SNAPPING", "ENABLE_SR_SNAPPING",
        "INP_SR_LOOKBACK_BARS", "SR_LOOKBACK_BARS",
        "INP_SR_PIVOT_STRENGTH", "SR_PIVOT_STRENGTH",
        "INP_SR_OFFSET_POINTS", "SR_OFFSET_POINTS",
        "INP_SR_ZONE_SELECTION", "SR_ZONE_SELECTION",
        "INP_ENABLE_RISK_FILTER", "ENABLE_RISK_FILTER",
        "INP_ENABLE_DYNAMIC_LOT_SIZING", "ENABLE_DYNAMIC_LOT_SIZING",
        "INP_MAX_LOT_SIZE", "MAX_LOT_SIZE",
        "INP_MARGIN_SAFETY_MULTIPLIER", "MARGIN_SAFETY_MULTIPLIER",
        "INP_MIN_MARGIN_LEVEL", "MIN_MARGIN_LEVEL",
        "INP_MAX_RISK_REWARD_RATIO", "MAX_RISK_REWARD_RATIO",
        "INP_MAX_TRADE_RISK_PCT", "MAX_TRADE_RISK_PCT",
        "INP_ENABLE_CALENDAR_FILTER", "ENABLE_CALENDAR_FILTER",
        "INP_CALENDAR_TRAILING_POINTS", "CALENDAR_TRAILING_POINTS",
        "INP_ENABLE_NEWS_FILTER", "ENABLE_NEWS_FILTER",
        "INP_USE_SUPPORT_RESISTANCE", "USE_SUPPORT_RESISTANCE",
        "INP_SR_BARS", "SR_BARS", "INP_BUFFER_POINTS", "BUFFER_POINTS",
        "INP_TARGET_EXACT_SR", "TARGET_EXACT_SR",
        "INP_TRADE_DIRECTION", "TRADE_DIRECTION",
        "INP_MINIMAL_LEVEL_ACCEPTED_BUY", "MINIMAL_LEVEL_ACCEPTED_BUY",
        "INP_MINIMAL_LEVEL_ACCEPTED_SELL", "MINIMAL_LEVEL_ACCEPTED_SELL",
        "INP_LOT_SIZE", "LOT_SIZE", "INP_MAGIC_NUMBER",
        "INP_IGNORE_AUDIT", "IGNORE_AUDIT",
        "INP_CONSECUTIVE_MODE", "CONSECUTIVE_MODE",
        "INP_MAX_CONSECUTIVE_ORDERS", "MAX_CONSECUTIVE_ORDERS",
        "INP_HURDLE_PROFIT_PCT", "HURDLE_PROFIT_PCT",
        "INP_PROFIT_LOCK_PCT", "PROFIT_LOCK_PCT",
        "INP_ANTI_CHOP_MIN_DISPLACEMENT", "ANTI_CHOP_MIN_DISPLACEMENT",
        "INP_SAFETY_OFFSET_POINTS", "SAFETY_OFFSET_POINTS",
        "INP_ENABLE_SWAP_AMORTIZATION", "ENABLE_SWAP_AMORTIZATION",
        "INP_CONSECUTIVE_SLOT_FILTER", "CONSECUTIVE_SLOT_FILTER",
        "INP_IGNORE_CONFLICTING_SIGNALS", "IGNORE_CONFLICTING_SIGNALS",
        "INP_ENABLE_OPPOSING_REGIME_FILTER", "ENABLE_OPPOSING_REGIME_FILTER",
        "INP_OPPOSING_STREAK_THRESHOLD", "OPPOSING_STREAK_THRESHOLD",
        "INP_OPPOSING_ACTION", "OPPOSING_ACTION",
        "INP_OPPOSING_TRAILING_POINTS", "OPPOSING_TRAILING_POINTS",
        "INP_OPPOSING_RECALCULATE_RATIO", "OPPOSING_RECALCULATE_RATIO",
    }
    leaked_in_env = env_keys & liveonnx_only_keys
    leaked_in_example = example_keys & liveonnx_only_keys
    assert not leaked_in_env, f".env contains LiveONNX-only keys: {leaked_in_env}"
    assert not leaked_in_example, f".env.example contains LiveONNX-only keys: {leaked_in_example}"


def test_preset_generator_threshold_propagation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Verify PAR-01: PresetGenerator propagates thresholds from AppConfig and env."""
    from src.preset_generator import PresetGenerator

    # 1. Default config fallback to 0.50 (when directional thresholds are None)
    monkeypatch.delenv("INP_MINIMAL_LEVEL_ACCEPTED_BUY", raising=False)
    monkeypatch.delenv("INP_MINIMAL_LEVEL_ACCEPTED_SELL", raising=False)
    monkeypatch.delenv("EVAL_BUY_CLASSIFICATION_THRESHOLD", raising=False)
    monkeypatch.delenv("EVAL_SELL_CLASSIFICATION_THRESHOLD", raising=False)
    monkeypatch.delenv("EVAL_CLASSIFICATION_THRESHOLD", raising=False)
    config = AppConfig.from_env(load_env_file=False)
    gen = PresetGenerator(config, tmp_path, tmp_path)
    content = gen.build_live_preset_content()
    assert "InpMinimalLevelAcceptedBuy=0.50" in content
    assert "InpMinimalLevelAcceptedSell=0.50" in content

    # 2. Global EVAL_CLASSIFICATION_THRESHOLD propagation
    monkeypatch.setenv("EVAL_CLASSIFICATION_THRESHOLD", "0.55")
    config_global = AppConfig.from_env(load_env_file=False)
    gen_global = PresetGenerator(config_global, tmp_path, tmp_path)
    content_global = gen_global.build_live_preset_content()
    assert "InpMinimalLevelAcceptedBuy=0.55" in content_global
    assert "InpMinimalLevelAcceptedSell=0.55" in content_global

    # 3. Directional overrides propagation
    monkeypatch.setenv("EVAL_BUY_CLASSIFICATION_THRESHOLD", "0.48")
    monkeypatch.setenv("EVAL_SELL_CLASSIFICATION_THRESHOLD", "0.52")
    config_dir = AppConfig.from_env(load_env_file=False)
    gen_dir = PresetGenerator(config_dir, tmp_path, tmp_path)
    content_dir = gen_dir.build_live_preset_content()
    assert "InpMinimalLevelAcceptedBuy=0.48" in content_dir
    assert "InpMinimalLevelAcceptedSell=0.52" in content_dir

    # 4. Environment variable precedence
    monkeypatch.setenv("INP_MINIMAL_LEVEL_ACCEPTED_BUY", "0.61")
    monkeypatch.setenv("INP_MINIMAL_LEVEL_ACCEPTED_SELL", "0.62")
    content_env = gen_dir.build_live_preset_content()
    assert "InpMinimalLevelAcceptedBuy=0.61" in content_env
    assert "InpMinimalLevelAcceptedSell=0.62" in content_env

    # 5. Fallback to 0.50 when all thresholds in config are None
    monkeypatch.delenv("INP_MINIMAL_LEVEL_ACCEPTED_BUY", raising=False)
    monkeypatch.delenv("INP_MINIMAL_LEVEL_ACCEPTED_SELL", raising=False)
    config_none = dataclasses.replace(
        config,
        eval_buy_classification_threshold=None,
        eval_sell_classification_threshold=None,
        eval_classification_threshold=None,
    )
    gen_none = PresetGenerator(config_none, tmp_path, tmp_path)
    content_none = gen_none.build_live_preset_content()
    assert "InpMinimalLevelAcceptedBuy=0.50" in content_none
    assert "InpMinimalLevelAcceptedSell=0.50" in content_none
