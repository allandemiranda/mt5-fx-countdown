"""Strictly typed configuration module for MetaTrader 5 MLOps Pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv


def _get_required_env(key: str, env_dict: Dict[str, str] | None = None) -> str:
    """Retrieve a mandatory environment variable or raise ValueError."""
    val = env_dict.get(key) if env_dict is not None else os.getenv(key)
    if val is None or val.strip() == "":
        raise ValueError(f"Mandatory configuration parameter '{key}' is missing or empty.")
    return val.strip()


def _get_required_int(key: str, env_dict: Dict[str, str] | None = None) -> int:
    """Retrieve and parse a mandatory integer environment variable."""
    val_str = _get_required_env(key, env_dict)
    try:
        return int(val_str)
    except ValueError as exc:
        raise ValueError(f"Configuration parameter '{key}' must be a valid integer, got '{val_str}'.") from exc


def _get_required_float(key: str, env_dict: Dict[str, str] | None = None) -> float:
    """Retrieve and parse a mandatory float environment variable."""
    val_str = _get_required_env(key, env_dict)
    try:
        return float(val_str)
    except ValueError as exc:
        raise ValueError(f"Configuration parameter '{key}' must be a valid float, got '{val_str}'.") from exc


def _get_required_bool(key: str, env_dict: Dict[str, str] | None = None) -> bool:
    """Retrieve and parse a mandatory boolean/flag environment variable (1/0, true/false)."""
    val_str = _get_required_env(key, env_dict).lower()
    if val_str in ("1", "true", "yes"):
        return True
    if val_str in ("0", "false", "no"):
        return False
    raise ValueError(f"Configuration parameter '{key}' must be a boolean (1/0, true/false), got '{val_str}'.")


def _get_optional_int(key: str, default: int, env_dict: Dict[str, str] | None = None) -> int:
    """Retrieve and parse an optional integer environment variable with fallback."""
    val = env_dict.get(key) if env_dict is not None else os.getenv(key)
    if val is None or val.strip() == "":
        return default
    val_str = val.strip()
    try:
        return int(val_str)
    except ValueError as exc:
        raise ValueError(f"Configuration parameter '{key}' must be a valid integer, got '{val_str}'.") from exc


def _get_optional_float(key: str, default: float, env_dict: Dict[str, str] | None = None) -> float:
    """Retrieve and parse an optional float environment variable with fallback."""
    val = env_dict.get(key) if env_dict is not None else os.getenv(key)
    if val is None or val.strip() == "":
        return default
    val_str = val.strip()
    try:
        return float(val_str)
    except ValueError as exc:
        raise ValueError(f"Configuration parameter '{key}' must be a valid float, got '{val_str}'.") from exc


def _get_optional_bool(key: str, default: bool, env_dict: Dict[str, str] | None = None) -> bool:
    """Retrieve and parse an optional boolean environment variable with fallback."""
    val = env_dict.get(key) if env_dict is not None else os.getenv(key)
    if val is None or val.strip() == "":
        return default
    val_str = val.strip().lower()
    if val_str in ("1", "true", "yes"):
        return True
    if val_str in ("0", "false", "no"):
        return False
    raise ValueError(f"Configuration parameter '{key}' must be a boolean (1/0, true/false), got '{val_str}'.")


def _get_optional_env(key: str, default: str, env_dict: Dict[str, str] | None = None) -> str:
    """Retrieve an optional string environment variable with fallback."""
    val = env_dict.get(key) if env_dict is not None else os.getenv(key)
    if val is None or val.strip() == "":
        return default
    return val.strip()


def _get_optional_nullable_int(key: str, env_dict: Dict[str, str] | None = None) -> int | None:
    """Retrieve and parse an optional nullable integer environment variable."""
    val = env_dict.get(key) if env_dict is not None else os.getenv(key)
    if val is None or val.strip() == "":
        return None
    val_str = val.strip()
    try:
        return int(val_str)
    except ValueError as exc:
        raise ValueError(f"Configuration parameter '{key}' must be a valid integer, got '{val_str}'.") from exc


def _get_optional_nullable_float(key: str, env_dict: Dict[str, str] | None = None) -> float | None:
    """Retrieve and parse an optional nullable float environment variable."""
    val = env_dict.get(key) if env_dict is not None else os.getenv(key)
    if val is None or val.strip() == "":
        return None
    val_str = val.strip()
    try:
        return float(val_str)
    except ValueError as exc:
        raise ValueError(f"Configuration parameter '{key}' must be a valid float, got '{val_str}'.") from exc


def _get_optional_nullable_str(key: str, env_dict: Dict[str, str] | None = None) -> str | None:
    """Retrieve an optional nullable string environment variable."""
    val = env_dict.get(key) if env_dict is not None else os.getenv(key)
    if val is None or val.strip() == "":
        return None
    return val.strip()


@dataclass(frozen=True)
class DirectionalXGBConfig:
    """XGBoost and evaluation configuration for a specific trade direction (BUY or SELL)."""

    max_depth: int
    eta: float
    subsample: float
    colsample_bytree: float
    min_child_weight: float
    reg_lambda: float
    reg_alpha: float
    rounds: int
    early_stopping_rounds: int
    optuna_trials: int
    optuna_objective_metric: str
    classification_threshold: float



@dataclass(frozen=True)
class AppConfig:
    """Strictly typed configuration dataclass for the MLOps pipeline."""

    # 1. MT5 & MetaEditor Paths
    mt5_path: Path
    metaeditor_path: Path
    mt5_data_path: Path | None
    mt5_common_path: Path | None

    # 2. Backtest & Tester Parameters
    symbol: str
    timeframe: str
    magic_number: int
    from_date: str
    to_date: str
    shutdown_terminal: int
    backtest_timeout: int
    watchdog_poll_interval: int
    skip_dataset_generation: bool
    avoid_pandemictime: bool
    pandemic_start_date: str
    pandemic_end_date: str

    # 3. Feature Extraction & GARCH Settings
    feature_lookback: int
    garch_horizon: int
    price_size: int
    garch_alpha: float
    garch_beta: float
    use_garch_features: bool
    label_horizon_bars: int
    label_min_points: int
    label_max_adverse_points: int

    # Daily Schedule Settings (MT5 Server Time)
    trade_monday: bool
    trade_monday_start: str
    trade_monday_end: str
    trade_tuesday: bool
    trade_tuesday_start: str
    trade_tuesday_end: str
    trade_wednesday: bool
    trade_wednesday_start: str
    trade_wednesday_end: str
    trade_thursday: bool
    trade_thursday_start: str
    trade_thursday_end: str
    trade_friday: bool
    trade_friday_start: str
    trade_friday_end: str

    # 4. XGBoost ML Hyperparameters
    xgb_max_depth: int
    xgb_eta: float
    xgb_subsample: float
    xgb_colsample_bytree: float
    xgb_min_child_weight: float
    xgb_lambda: float
    xgb_alpha: float
    xgb_rounds: int
    xgb_early_stopping_rounds: int
    validation_percentage: float

    # 5. Hyperparameter Optimization
    optuna_trials: int

    # 6. ML Directional Evaluation & Threshold Sensitivity Grid
    eval_classification_threshold: float
    optuna_objective_metric: str
    eval_enable_threshold_grid: bool
    eval_threshold_min: float
    eval_threshold_max: float
    eval_threshold_step: float

    # 7. Feature Toggles
    use_adx: bool
    use_atr: bool
    use_bands: bool
    use_macd: bool
    use_fast_ma: bool
    use_slow_ma: bool
    use_rsi: bool
    use_stochastic: bool
    use_candlestick: bool
    use_timestamp_week: bool
    use_timestamp_day: bool
    use_open_markets: bool
    use_spread: bool

    # 7. Indicator Parameters
    adx_period: int
    atr_period: int
    bands_period: int
    bands_shift: int
    bands_dev: float
    bands_applied_price: int
    macd_fast: int
    macd_slow: int
    macd_signal: int
    macd_applied_price: int
    fast_ma_period: int
    fast_ma_shift: int
    fast_ma_method: int
    fast_ma_applied_price: int
    slow_ma_period: int
    slow_ma_shift: int
    slow_ma_method: int
    slow_ma_applied_price: int
    rsi_period: int
    rsi_applied_price: int
    stoch_k: int
    stoch_d: int
    stoch_slowing: int
    stoch_method: int
    stoch_price_field: int

    # 8. Directional XGBoost & Optuna Overrides (Optional with fallback to global settings)
    xgb_buy_max_depth: int | None = None
    xgb_buy_eta: float | None = None
    xgb_buy_subsample: float | None = None
    xgb_buy_colsample_bytree: float | None = None
    xgb_buy_min_child_weight: float | None = None
    xgb_buy_lambda: float | None = None
    xgb_buy_alpha: float | None = None
    xgb_buy_rounds: int | None = None
    xgb_buy_early_stopping_rounds: int | None = None
    optuna_buy_trials: int | None = None
    optuna_buy_objective_metric: str | None = None
    eval_buy_classification_threshold: float | None = None

    xgb_sell_max_depth: int | None = None
    xgb_sell_eta: float | None = None
    xgb_sell_subsample: float | None = None
    xgb_sell_colsample_bytree: float | None = None
    xgb_sell_min_child_weight: float | None = None
    xgb_sell_lambda: float | None = None
    xgb_sell_alpha: float | None = None
    xgb_sell_rounds: int | None = None
    xgb_sell_early_stopping_rounds: int | None = None
    optuna_sell_trials: int | None = None
    optuna_sell_objective_metric: str | None = None
    eval_sell_classification_threshold: float | None = None

    def get_directional_config(self, direction: str) -> DirectionalXGBConfig:
        """Resolve directional XGBoost and evaluation configuration with transparent fallback to global settings."""
        dir_clean = direction.lower().strip()
        if dir_clean == "buy":
            return DirectionalXGBConfig(
                max_depth=self.xgb_buy_max_depth if self.xgb_buy_max_depth is not None else self.xgb_max_depth,
                eta=self.xgb_buy_eta if self.xgb_buy_eta is not None else self.xgb_eta,
                subsample=self.xgb_buy_subsample if self.xgb_buy_subsample is not None else self.xgb_subsample,
                colsample_bytree=(
                    self.xgb_buy_colsample_bytree
                    if self.xgb_buy_colsample_bytree is not None
                    else self.xgb_colsample_bytree
                ),
                min_child_weight=(
                    self.xgb_buy_min_child_weight
                    if self.xgb_buy_min_child_weight is not None
                    else self.xgb_min_child_weight
                ),
                reg_lambda=self.xgb_buy_lambda if self.xgb_buy_lambda is not None else self.xgb_lambda,
                reg_alpha=self.xgb_buy_alpha if self.xgb_buy_alpha is not None else self.xgb_alpha,
                rounds=self.xgb_buy_rounds if self.xgb_buy_rounds is not None else self.xgb_rounds,
                early_stopping_rounds=(
                    self.xgb_buy_early_stopping_rounds
                    if self.xgb_buy_early_stopping_rounds is not None
                    else self.xgb_early_stopping_rounds
                ),
                optuna_trials=self.optuna_buy_trials if self.optuna_buy_trials is not None else self.optuna_trials,
                optuna_objective_metric=(
                    self.optuna_buy_objective_metric
                    if self.optuna_buy_objective_metric is not None
                    else self.optuna_objective_metric
                ),
                classification_threshold=(
                    self.eval_buy_classification_threshold
                    if self.eval_buy_classification_threshold is not None
                    else self.eval_classification_threshold
                ),
            )
        elif dir_clean == "sell":
            return DirectionalXGBConfig(
                max_depth=self.xgb_sell_max_depth if self.xgb_sell_max_depth is not None else self.xgb_max_depth,
                eta=self.xgb_sell_eta if self.xgb_sell_eta is not None else self.xgb_eta,
                subsample=self.xgb_sell_subsample if self.xgb_sell_subsample is not None else self.xgb_subsample,
                colsample_bytree=(
                    self.xgb_sell_colsample_bytree
                    if self.xgb_sell_colsample_bytree is not None
                    else self.xgb_colsample_bytree
                ),
                min_child_weight=(
                    self.xgb_sell_min_child_weight
                    if self.xgb_sell_min_child_weight is not None
                    else self.xgb_min_child_weight
                ),
                reg_lambda=self.xgb_sell_lambda if self.xgb_sell_lambda is not None else self.xgb_lambda,
                reg_alpha=self.xgb_sell_alpha if self.xgb_sell_alpha is not None else self.xgb_alpha,
                rounds=self.xgb_sell_rounds if self.xgb_sell_rounds is not None else self.xgb_rounds,
                early_stopping_rounds=(
                    self.xgb_sell_early_stopping_rounds
                    if self.xgb_sell_early_stopping_rounds is not None
                    else self.xgb_early_stopping_rounds
                ),
                optuna_trials=self.optuna_sell_trials if self.optuna_sell_trials is not None else self.optuna_trials,
                optuna_objective_metric=(
                    self.optuna_sell_objective_metric
                    if self.optuna_sell_objective_metric is not None
                    else self.optuna_objective_metric
                ),
                classification_threshold=(
                    self.eval_sell_classification_threshold
                    if self.eval_sell_classification_threshold is not None
                    else self.eval_classification_threshold
                ),
            )
        else:
            raise ValueError(f"Invalid direction '{direction}'. Expected 'buy' or 'sell'.")

    @property
    def clean_timeframe(self) -> str:
        """Return standardized timeframe string without 'PERIOD_' prefix."""
        return self.timeframe.replace("PERIOD_", "")

    @property
    def base_feature_count(self) -> int:
        """Calculate the number of active base indicator features per bar."""
        count = 0
        if self.use_adx:
            count += 3
        if self.use_atr:
            count += 1
        if self.use_bands:
            count += 2
        if self.use_macd:
            count += 2
        if self.use_fast_ma:
            count += 1
        if self.use_slow_ma:
            count += 1
        if self.use_rsi:
            count += 1
        if self.use_stochastic:
            count += 2
        if self.use_candlestick:
            count += 4
        if self.use_timestamp_week:
            count += 1
        if self.use_timestamp_day:
            count += 1
        if self.use_open_markets:
            count += 1
        if self.use_spread:
            count += 1
        if self.use_garch_features:
            count += 5
        return count

    @property
    def active_feature_count(self) -> int:
        """Calculate total feature vector dimensions across lookback horizon."""
        return self.base_feature_count * (self.feature_lookback + 1)

    @classmethod
    def from_env(cls, env_path: str | Path | None = ".env", load_env_file: bool = True) -> AppConfig:
        """Load configuration from environment, optionally loading from .env without overriding pre-set vars."""
        if load_env_file:
            if env_path is not None and Path(env_path).exists():
                load_dotenv(dotenv_path=env_path, override=False)
            elif env_path is None or Path(".env").exists():
                load_dotenv(dotenv_path=".env", override=False)

        # Explicit MT5 Data & Common paths (optional in .env, discovered dynamically if missing)
        raw_data_path = os.getenv("MT5_DATA_PATH", "").strip()
        data_path = Path(raw_data_path) if raw_data_path else None

        raw_common_path = os.getenv("MT5_COMMON_PATH", "").strip()
        common_path = Path(raw_common_path) if raw_common_path else None

        # ML Directional Evaluation & Threshold Sensitivity Grid Parameters
        eval_classification_threshold = _get_optional_float("EVAL_CLASSIFICATION_THRESHOLD", 0.50)
        if not (0.0 < eval_classification_threshold < 1.0):
            raise ValueError(
                f"Configuration parameter 'EVAL_CLASSIFICATION_THRESHOLD' must be between 0.0 and 1.0, "
                f"got {eval_classification_threshold}."
            )

        optuna_objective_metric = _get_optional_env("OPTUNA_OBJECTIVE_METRIC", "logloss").lower()
        if optuna_objective_metric not in ("logloss", "roc_auc", "precision", "f1"):
            raise ValueError(
                f"Configuration parameter 'OPTUNA_OBJECTIVE_METRIC' must be one of "
                f"['logloss', 'roc_auc', 'precision', 'f1'], got '{optuna_objective_metric}'."
            )

        eval_enable_threshold_grid = _get_optional_bool("EVAL_ENABLE_THRESHOLD_GRID", True)
        eval_threshold_min = _get_optional_float("EVAL_THRESHOLD_MIN", 0.40)
        eval_threshold_max = _get_optional_float("EVAL_THRESHOLD_MAX", 0.70)
        eval_threshold_step = _get_optional_float("EVAL_THRESHOLD_STEP", 0.02)

        if not (0.0 <= eval_threshold_min < eval_threshold_max <= 1.0):
            raise ValueError(
                f"'EVAL_THRESHOLD_MIN' ({eval_threshold_min}) must be strictly less than "
                f"'EVAL_THRESHOLD_MAX' ({eval_threshold_max}) within [0.0, 1.0]."
            )
        if eval_threshold_step <= 0.0 or eval_threshold_step > (eval_threshold_max - eval_threshold_min):
            raise ValueError(
                f"'EVAL_THRESHOLD_STEP' ({eval_threshold_step}) must be strictly positive and <= range span."
            )

        # Build dataclass with strict typed validation
        cfg = cls(
            # 1. MT5 Paths
            mt5_path=Path(_get_required_env("MT5_PATH")),
            metaeditor_path=Path(_get_required_env("METAEDITOR_PATH")),
            mt5_data_path=data_path,
            mt5_common_path=common_path,

            # 2. Backtest Parameters
            symbol=_get_required_env("SYMBOL"),
            timeframe=_get_required_env("TIMEFRAME"),
            magic_number=_get_optional_int("MAGIC_NUMBER", 222100),
            from_date=_get_required_env("FROM_DATE"),
            to_date=_get_required_env("TO_DATE"),
            shutdown_terminal=_get_required_int("SHUTDOWN_TERMINAL"),
            backtest_timeout=_get_required_int("BACKTEST_TIMEOUT"),
            watchdog_poll_interval=_get_required_int("WATCHDOG_POLL_INTERVAL"),
            skip_dataset_generation=_get_required_bool("SKIP_DATASET_GENERATION"),
            avoid_pandemictime=_get_optional_bool("AVOID_PANDEMICTIME", False),
            pandemic_start_date=_get_optional_env("PANDEMIC_START_DATE", "2020.01.01 00:00:00"),
            pandemic_end_date=_get_optional_env("PANDEMIC_END_DATE", "2021.06.01 00:00:00"),

            # 3. Feature & GARCH Settings
            feature_lookback=_get_required_int("FEATURE_LOOKBACK"),
            garch_horizon=_get_required_int("GARCH_HORIZON"),
            price_size=_get_required_int("PRICE_SIZE"),
            garch_alpha=_get_required_float("GARCH_ALPHA"),
            garch_beta=_get_required_float("GARCH_BETA"),
            use_garch_features=_get_optional_bool("USE_GARCH_FEATURES", True),
            label_horizon_bars=_get_optional_int("LABEL_HORIZON_BARS", 12),
            label_min_points=_get_optional_int("LABEL_MIN_POINTS", 150),
            label_max_adverse_points=_get_optional_int("LABEL_MAX_ADVERSE_POINTS", 150),

            # Daily Schedule Settings (MT5 Server Time)
            trade_monday=_get_optional_bool("TRADE_MONDAY", True),
            trade_monday_start=_get_optional_env("TRADE_MONDAY_START", "11:00:00"),
            trade_monday_end=_get_optional_env("TRADE_MONDAY_END", "18:00:00"),
            trade_tuesday=_get_optional_bool("TRADE_TUESDAY", True),
            trade_tuesday_start=_get_optional_env("TRADE_TUESDAY_START", "10:00:00"),
            trade_tuesday_end=_get_optional_env("TRADE_TUESDAY_END", "18:00:00"),
            trade_wednesday=_get_optional_bool("TRADE_WEDNESDAY", True),
            trade_wednesday_start=_get_optional_env("TRADE_WEDNESDAY_START", "10:00:00"),
            trade_wednesday_end=_get_optional_env("TRADE_WEDNESDAY_END", "18:00:00"),
            trade_thursday=_get_optional_bool("TRADE_THURSDAY", True),
            trade_thursday_start=_get_optional_env("TRADE_THURSDAY_START", "10:00:00"),
            trade_thursday_end=_get_optional_env("TRADE_THURSDAY_END", "18:00:00"),
            trade_friday=_get_optional_bool("TRADE_FRIDAY", True),
            trade_friday_start=_get_optional_env("TRADE_FRIDAY_START", "10:00:00"),
            trade_friday_end=_get_optional_env("TRADE_FRIDAY_END", "16:00:00"),

            # 4. XGBoost Hyperparameters
            xgb_max_depth=_get_required_int("XGB_MAX_DEPTH"),
            xgb_eta=_get_required_float("XGB_ETA"),
            xgb_subsample=_get_required_float("XGB_SUBSAMPLE"),
            xgb_colsample_bytree=_get_required_float("XGB_COLSAMPLE_BYTREE"),
            xgb_min_child_weight=_get_required_float("XGB_MIN_CHILD_WEIGHT"),
            xgb_lambda=_get_required_float("XGB_LAMBDA"),
            xgb_alpha=_get_required_float("XGB_ALPHA"),
            xgb_rounds=_get_required_int("XGB_ROUNDS"),
            xgb_early_stopping_rounds=_get_required_int("XGB_EARLY_STOPPING_ROUNDS"),
            validation_percentage=_get_required_float("VALIDATION_PERCENTAGE"),

            # 5. Hyperparameter Optimization
            optuna_trials=_get_required_int("OPTUNA_TRIALS"),

            # 6. ML Directional Evaluation & Threshold Sensitivity Grid
            eval_classification_threshold=eval_classification_threshold,
            optuna_objective_metric=optuna_objective_metric,
            eval_enable_threshold_grid=eval_enable_threshold_grid,
            eval_threshold_min=eval_threshold_min,
            eval_threshold_max=eval_threshold_max,
            eval_threshold_step=eval_threshold_step,

            # 7. Feature Toggles
            use_adx=_get_required_bool("USE_ADX"),
            use_atr=_get_required_bool("USE_ATR"),
            use_bands=_get_required_bool("USE_BANDS"),
            use_macd=_get_required_bool("USE_MACD"),
            use_fast_ma=_get_required_bool("USE_FAST_MA"),
            use_slow_ma=_get_required_bool("USE_SLOW_MA"),
            use_rsi=_get_required_bool("USE_RSI"),
            use_stochastic=_get_required_bool("USE_STOCHASTIC"),
            use_candlestick=_get_required_bool("USE_CANDLESTICK"),
            use_timestamp_week=_get_required_bool("USE_TIMESTAMP_WEEK"),
            use_timestamp_day=_get_required_bool("USE_TIMESTAMP_DAY"),
            use_open_markets=_get_required_bool("USE_OPEN_MARKETS"),
            use_spread=_get_required_bool("USE_SPREAD"),

            # 7. Indicator Parameters
            adx_period=_get_required_int("ADX_PERIOD"),
            atr_period=_get_required_int("ATR_PERIOD"),
            bands_period=_get_required_int("BANDS_PERIOD"),
            bands_shift=_get_required_int("BANDS_SHIFT"),
            bands_dev=_get_required_float("BANDS_DEV"),
            bands_applied_price=_get_required_int("BANDS_APPLIED_PRICE"),
            macd_fast=_get_required_int("MACD_FAST"),
            macd_slow=_get_required_int("MACD_SLOW"),
            macd_signal=_get_required_int("MACD_SIGNAL"),
            macd_applied_price=_get_required_int("MACD_APPLIED_PRICE"),
            fast_ma_period=_get_required_int("FAST_MA_PERIOD"),
            fast_ma_shift=_get_required_int("FAST_MA_SHIFT"),
            fast_ma_method=_get_required_int("FAST_MA_METHOD"),
            fast_ma_applied_price=_get_required_int("FAST_MA_APPLIED_PRICE"),
            slow_ma_period=_get_required_int("SLOW_MA_PERIOD"),
            slow_ma_shift=_get_required_int("SLOW_MA_SHIFT"),
            slow_ma_method=_get_required_int("SLOW_MA_METHOD"),
            slow_ma_applied_price=_get_required_int("SLOW_MA_APPLIED_PRICE"),
            rsi_period=_get_required_int("RSI_PERIOD"),
            rsi_applied_price=_get_required_int("RSI_APPLIED_PRICE"),
            stoch_k=_get_required_int("STOCH_K"),
            stoch_d=_get_required_int("STOCH_D"),
            stoch_slowing=_get_required_int("STOCH_SLOWING"),
            stoch_method=_get_required_int("STOCH_METHOD"),
            stoch_price_field=_get_required_int("STOCH_PRICE_FIELD"),

            # 8. Directional XGBoost & Optuna Overrides (Optional with fallback to global settings)
            xgb_buy_max_depth=_get_optional_nullable_int("XGB_BUY_MAX_DEPTH"),
            xgb_buy_eta=_get_optional_nullable_float("XGB_BUY_ETA"),
            xgb_buy_subsample=_get_optional_nullable_float("XGB_BUY_SUBSAMPLE"),
            xgb_buy_colsample_bytree=_get_optional_nullable_float("XGB_BUY_COLSAMPLE_BYTREE"),
            xgb_buy_min_child_weight=_get_optional_nullable_float("XGB_BUY_MIN_CHILD_WEIGHT"),
            xgb_buy_lambda=_get_optional_nullable_float("XGB_BUY_LAMBDA"),
            xgb_buy_alpha=_get_optional_nullable_float("XGB_BUY_ALPHA"),
            xgb_buy_rounds=_get_optional_nullable_int("XGB_BUY_ROUNDS"),
            xgb_buy_early_stopping_rounds=_get_optional_nullable_int("XGB_BUY_EARLY_STOPPING_ROUNDS"),
            optuna_buy_trials=_get_optional_nullable_int("OPTUNA_BUY_TRIALS"),
            optuna_buy_objective_metric=_get_optional_nullable_str("OPTUNA_BUY_OBJECTIVE_METRIC"),
            eval_buy_classification_threshold=_get_optional_nullable_float("EVAL_BUY_CLASSIFICATION_THRESHOLD"),

            xgb_sell_max_depth=_get_optional_nullable_int("XGB_SELL_MAX_DEPTH"),
            xgb_sell_eta=_get_optional_nullable_float("XGB_SELL_ETA"),
            xgb_sell_subsample=_get_optional_nullable_float("XGB_SELL_SUBSAMPLE"),
            xgb_sell_colsample_bytree=_get_optional_nullable_float("XGB_SELL_COLSAMPLE_BYTREE"),
            xgb_sell_min_child_weight=_get_optional_nullable_float("XGB_SELL_MIN_CHILD_WEIGHT"),
            xgb_sell_lambda=_get_optional_nullable_float("XGB_SELL_LAMBDA"),
            xgb_sell_alpha=_get_optional_nullable_float("XGB_SELL_ALPHA"),
            xgb_sell_rounds=_get_optional_nullable_int("XGB_SELL_ROUNDS"),
            xgb_sell_early_stopping_rounds=_get_optional_nullable_int("XGB_SELL_EARLY_STOPPING_ROUNDS"),
            optuna_sell_trials=_get_optional_nullable_int("OPTUNA_SELL_TRIALS"),
            optuna_sell_objective_metric=_get_optional_nullable_str("OPTUNA_SELL_OBJECTIVE_METRIC"),
            eval_sell_classification_threshold=_get_optional_nullable_float("EVAL_SELL_CLASSIFICATION_THRESHOLD"),
        )

        if not (0.0 < cfg.garch_alpha + cfg.garch_beta < 1.0):
            raise ValueError(
                f"GARCH covariance stationarity violated: GARCH_ALPHA ({cfg.garch_alpha}) + "
                f"GARCH_BETA ({cfg.garch_beta}) = {cfg.garch_alpha + cfg.garch_beta:.4f} >= 1.0. "
                f"Persistence sum must be strictly less than 1.0."
            )

        return cfg
