"""Mathematical verification of GARCH(1,1) log-returns, variance recursion, and multi-step aggregation."""

from __future__ import annotations

import math
import numpy as np


def python_garch_sigma_agg(
    prices: np.ndarray,
    price_size: int,
    horizon: int,
    alpha: float = 0.08,
    beta: float = 0.90,
) -> float:
    """Reference Python implementation of GARCH(1,1) multi-step aggregated volatility
    strictly matching MQL5 GarchEngine.mqh.
    """
    assert len(prices) >= price_size + 1

    # Extract historical window (chronological: oldest to newest)
    p_window = prices[-(price_size + 1):]

    # 1. Log returns: r_t = ln(Close_t / Close_{t-1})
    returns = np.diff(np.log(p_window))
    n = len(returns)
    assert n == price_size

    # Mean return: mu = (1/N) * sum(r_i)
    mean_ret = np.mean(returns)

    # 2. Sample variance: s^2 = (1/(N-1)) * sum( (r_i - mu)^2 )
    sample_var = np.var(returns, ddof=1)
    if sample_var <= 0.0:
        sample_var = 1e-6

    # 3. Stationarity and Omega
    persistence = alpha + beta
    if persistence >= 1.0:
        alpha = 0.05
        beta = 0.92
        persistence = alpha + beta

    omega = sample_var * (1.0 - persistence)
    if omega <= 0.0:
        omega = 1e-8
    long_run_var = omega / (1.0 - persistence)

    # 4. Conditional variance recursion: sigma_t^2 = omega + alpha * (r_{t-1} - mu)^2 + beta * sigma_{t-1}^2
    current_sigma2 = sample_var
    for r in returns:
        shock = r - mean_ret
        current_sigma2 = omega + alpha * (shock ** 2) + beta * current_sigma2

    # 5. Multi-step analytical variance forecast over horizon H
    # E[sigma_{t+h}^2] = V_L + (alpha + beta)^h * (sigma_t^2 - V_L)
    sum_forecast_var = 0.0
    persistence_power = persistence
    for h in range(1, horizon + 1):
        f_var = long_run_var + persistence_power * (current_sigma2 - long_run_var)
        if f_var < 1e-8:
            f_var = 1e-8
        sum_forecast_var += f_var
        persistence_power *= persistence

    # 6. Aggregated standard deviation: sigma_agg = sqrt(sum_forecast_var)
    sigma_agg = np.sqrt(sum_forecast_var)
    if sigma_agg <= 0.0:
        sigma_agg = 1e-4
    return float(sigma_agg)


def test_garch_step_by_step_analytical_parity():
    """Verify each analytical formula step (mean, sample var, omega, recursion, forecast sum)."""
    # Deterministic price series: 5 prices -> 4 returns
    prices = np.array([100.0, 102.0, 101.0, 103.0, 102.5], dtype=np.float64)
    price_size = 4
    horizon = 2
    alpha = 0.10
    beta = 0.80

    # Step 1: Log returns
    expected_returns = np.log(prices[1:] / prices[:-1])
    assert len(expected_returns) == 4

    # Step 2: Mean & Sample Var
    mean_ret = float(np.mean(expected_returns))
    sample_var = float(np.sum((expected_returns - mean_ret) ** 2) / (price_size - 1))

    # Step 3: Persistence & Omega
    persistence = alpha + beta  # 0.90
    omega = sample_var * (1.0 - persistence)
    long_run_var = omega / (1.0 - persistence)
    assert np.isclose(long_run_var, sample_var)

    # Step 4: Recursion
    curr_sig2 = sample_var
    for r in expected_returns:
        shock2 = (r - mean_ret) ** 2
        curr_sig2 = omega + alpha * shock2 + beta * curr_sig2

    # Step 5: Multi-step forecast
    step1_var = long_run_var + persistence * (curr_sig2 - long_run_var)
    step2_var = long_run_var + (persistence ** 2) * (curr_sig2 - long_run_var)
    expected_sigma_agg = math.sqrt(step1_var + step2_var)

    actual_sigma_agg = python_garch_sigma_agg(prices, price_size, horizon, alpha, beta)
    assert np.isclose(actual_sigma_agg, expected_sigma_agg, atol=1e-8)


def test_garch_realistic_simulation_and_stop_levels():
    """Verify GARCH volatility calculation on simulated FX walk and dynamic TP/SL point mapping."""
    np.random.seed(42)
    initial_price = 1.1000
    steps = 400
    daily_returns = np.random.normal(0.0001, 0.005, size=steps)
    prices = initial_price * np.exp(np.cumsum(daily_returns))

    price_size = 200
    horizon = 5
    alpha = 0.08
    beta = 0.90

    sigma_agg = python_garch_sigma_agg(prices, price_size, horizon, alpha, beta)

    assert sigma_agg > 0.0
    assert 0.001 < sigma_agg < 0.10  # Plausible FX multi-bar volatility

    # Stop level mapping
    k_tp = 2.0
    k_sl = 1.5
    point = 0.00001
    current_price = prices[-1]

    price_risk = current_price * sigma_agg
    risk_points = price_risk / point

    tp_points = k_tp * risk_points
    sl_points = k_sl * risk_points

    assert tp_points > sl_points
    assert tp_points > 10.0  # Above typical broker minimum stops level
    assert sl_points > 10.0


def test_garch_horizon_monotonicity():
    """Verify aggregated volatility strictly increases with forecast horizon H."""
    np.random.seed(123)
    prices = 1.2500 * np.exp(np.cumsum(np.random.normal(0, 0.004, size=300)))
    price_size = 150

    v_h1 = python_garch_sigma_agg(prices, price_size, horizon=1)
    v_h3 = python_garch_sigma_agg(prices, price_size, horizon=3)
    v_h8 = python_garch_sigma_agg(prices, price_size, horizon=8)
    v_h15 = python_garch_sigma_agg(prices, price_size, horizon=15)

    assert v_h1 < v_h3 < v_h8 < v_h15


def test_garch_stationarity_fallback():
    """Verify that parameters violating stationarity (alpha + beta >= 1.0) are gracefully clamped."""
    np.random.seed(42)
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, size=100)))

    # Violating stationarity: 0.6 + 0.6 = 1.2
    sigma_agg = python_garch_sigma_agg(prices, price_size=50, horizon=5, alpha=0.60, beta=0.60)
    assert not np.isnan(sigma_agg)
    assert not np.isinf(sigma_agg)
    assert sigma_agg > 0.0


def test_garch_flat_prices_boundary():
    """Verify numerical stability when prices are perfectly flat (zero variance edge case)."""
    prices = np.full(100, 1.1000)
    sigma_agg = python_garch_sigma_agg(prices, price_size=50, horizon=5, alpha=0.08, beta=0.90)
    assert not np.isnan(sigma_agg)
    assert sigma_agg > 0.0  # Gracefully uses lower bound safety clamps


def test_garch_dynamic_feature_metrics():
    """Verify calculation of vol_ratio and vol_trend econometric features."""
    np.random.seed(42)
    prices = 1.1000 * np.exp(np.cumsum(np.random.normal(0, 0.005, size=200)))
    price_size = 100
    horizon = 8
    alpha = 0.05
    beta = 0.92

    p_window = prices[-(price_size + 1):]
    returns = np.diff(np.log(p_window))
    mean_ret = np.mean(returns)
    sample_var = np.var(returns, ddof=1)
    persistence = alpha + beta
    omega = sample_var * (1.0 - persistence)
    long_run_var = omega / (1.0 - persistence)

    current_sigma2 = sample_var
    for r in returns:
        shock = r - mean_ret
        current_sigma2 = omega + alpha * (shock ** 2) + beta * current_sigma2

    sigma_cond = math.sqrt(current_sigma2)
    vol_ratio = sigma_cond / math.sqrt(sample_var)

    sum_f_var = 0.0
    p_pow = persistence
    for h in range(1, horizon + 1):
        f_var = max(1e-8, long_run_var + p_pow * (current_sigma2 - long_run_var))
        sum_f_var += f_var
        p_pow *= persistence

    sigma_agg = math.sqrt(sum_f_var)
    horizon_factor = math.sqrt(horizon)
    vol_trend = sigma_agg / (horizon_factor * sigma_cond)

    assert vol_ratio > 0.0
    assert vol_trend > 0.0
    assert not np.isnan(vol_ratio)
    assert not np.isnan(vol_trend)
