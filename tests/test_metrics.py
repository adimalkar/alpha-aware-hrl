"""
Unit contract for src/utils/metrics.

These tests exist to pin the two bug classes found in the audit:
  M1 - an annual risk-free rate subtracted from per-tick returns, which
       swamped the mean and inverted the sign of every Sharpe.
  M2 - max drawdown multiplied by 100 in the metric and again at the call
       site, producing values like 6619% that are definitionally impossible.
"""

import numpy as np
import pytest

from src.utils.metrics import (
    MetricUnitError,
    compute_all_metrics,
    compute_cvar,
    compute_max_drawdown,
    compute_pnl,
    compute_sharpe,
    compute_var,
    compute_win_rate_pct,
)


def test_sharpe_rejects_frequency_mismatch():
    """M1: a daily rf against tick returns must raise, not silently flip sign."""
    rng = np.random.default_rng(0)
    tick_returns = rng.normal(1e-5, 1e-5, size=5000)  # sd ~ 1e-5, daily rf ~ 7.9e-5
    with pytest.raises(MetricUnitError, match="does not match the sampling frequency"):
        compute_sharpe(tick_returns, risk_free_rate=0.02, periods_per_year=252)


def test_sharpe_sign_agrees_with_mean_return():
    """A strictly profitable series cannot have a negative Sharpe."""
    rng = np.random.default_rng(1)
    returns = rng.normal(2e-5, 1e-5, size=5000)
    assert returns.mean() > 0
    assert compute_sharpe(returns) > 0


def test_sharpe_per_period_ignores_risk_free():
    """periods_per_year=None means no annualisation and no rf conversion."""
    rng = np.random.default_rng(2)
    r = rng.normal(1e-4, 1e-3, size=1000)
    assert compute_sharpe(r, risk_free_rate=0.02) == compute_sharpe(r, risk_free_rate=0.99)


def test_sharpe_annualisation_scales_by_sqrt():
    rng = np.random.default_rng(3)
    r = rng.normal(1e-4, 1e-3, size=2000)
    base = compute_sharpe(r, risk_free_rate=0.0, periods_per_year=None)
    ann = compute_sharpe(r, risk_free_rate=0.0, periods_per_year=252)
    assert ann == pytest.approx(base * np.sqrt(252), rel=1e-9)


def test_drawdown_is_bounded_and_already_percent():
    """M2: the return value is percent; squaring the scale must be impossible."""
    equity = np.array([100.0, 120.0, 60.0, 90.0])
    dd_pct, peak, trough = compute_max_drawdown(equity)
    assert dd_pct == pytest.approx(50.0)  # 120 -> 60
    assert 0.0 <= dd_pct <= 100.0
    assert (peak, trough) == (1, 2)


def test_drawdown_rejects_nonpositive_equity():
    with pytest.raises(MetricUnitError, match="non-positive"):
        compute_max_drawdown(np.array([100.0, 50.0, 0.0]))


def test_drawdown_flat_curve_is_zero():
    assert compute_max_drawdown(np.full(10, 100.0))[0] == 0.0


def test_pnl_rejects_misaligned_inputs():
    with pytest.raises(MetricUnitError, match="must align"):
        compute_pnl(np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5]))


def test_pnl_rejects_nonpositive_prices():
    with pytest.raises(MetricUnitError, match="non-positive"):
        compute_pnl(np.array([1.0, 0.0, 3.0]), np.zeros(3))


def test_pnl_flat_position_is_flat_equity():
    prices = np.array([100.0, 101.0, 99.0, 103.0])
    eq, ret = compute_pnl(prices, np.zeros(4), initial_capital=1000.0)
    assert np.allclose(eq, 1000.0)
    assert ret == pytest.approx(0.0)


def test_pnl_full_long_tracks_price():
    prices = np.array([100.0, 110.0])
    eq, ret = compute_pnl(prices, np.ones(2), initial_capital=1000.0)
    assert ret == pytest.approx(10.0)


def test_returns_validator_rejects_non_finite():
    with pytest.raises(MetricUnitError, match="non-finite"):
        compute_sharpe(np.array([0.01, np.nan, 0.02]))


def test_returns_validator_warns_on_percent_scale():
    with pytest.warns(RuntimeWarning, match="fractions"):
        compute_sharpe(np.array([5.0, -3.0, 20.0, 11.0]))


def test_var_cvar_are_positive_losses_and_ordered():
    rng = np.random.default_rng(4)
    r = rng.normal(0, 0.01, size=20000)
    var, cvar = compute_var(r), compute_cvar(r)
    assert var > 0 and cvar > 0
    assert cvar >= var  # expected shortfall is at least the quantile


def test_win_rate_bounds():
    assert compute_win_rate_pct(np.array([1.0, -1.0, 1.0, 1.0])) == pytest.approx(75.0)
    assert compute_win_rate_pct(np.array([])) == 0.0


def test_all_metrics_refuses_empty_evaluation():
    """An evaluation that collected nothing is an error, not a zero-return result."""
    with pytest.raises(MetricUnitError, match="nothing to measure"):
        compute_all_metrics(np.array([100.0]), np.array([0.0]))


def test_all_metrics_reports_annualisation_flag():
    prices = 100.0 * np.cumprod(1 + np.random.default_rng(5).normal(0, 1e-3, 500))
    m = compute_all_metrics(prices, np.ones(500))
    assert m["annualised"] is False
    assert "calmar_ratio" not in m
    assert 0.0 <= m["max_drawdown_pct"] <= 100.0
    assert m["n_periods"] == 499


def test_flat_rate_separates_zero_from_loss():
    """
    win_rate near zero is ambiguous on sparse data: it can mean 'mostly flat'
    rather than 'mostly losing'. The two must be distinguishable.
    """
    from src.utils.metrics import compute_flat_rate_pct

    r = np.array([0.0, 0.0, 0.0, 0.0, 0.01, -0.01])
    assert compute_flat_rate_pct(r) == pytest.approx(100 * 4 / 6)
    assert compute_win_rate_pct(r) == pytest.approx(100 * 1 / 6)


def test_rates_sum_to_one_hundred():
    rng = np.random.default_rng(7)
    r = np.where(rng.random(1000) < 0.5, 0.0, rng.normal(0, 1e-3, 1000))
    from src.utils.metrics import compute_flat_rate_pct

    prices = 100.0 * np.cumprod(1 + rng.normal(0, 1e-3, 500))
    m = compute_all_metrics(prices, np.ones(500))
    total = m["win_rate_pct"] + m["flat_rate_pct"] + m["loss_rate_pct"]
    assert total == pytest.approx(100.0)
