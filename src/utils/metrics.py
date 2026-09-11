"""
Evaluation Metrics for Trading Performance

Unit conventions (enforced, not assumed):
  * `returns` are ALWAYS simple per-period returns (fractions, not percent).
  * `periods_per_year` MUST match the sampling frequency of `returns`.
    Pass None to report a non-annualised, per-period ratio. For tick- or
    event-sampled data the annualisation factor is not well defined and
    None is the honest choice -- annualising tick returns by sqrt(N) is how
    a Sharpe of 365 gets produced.
  * Every function returning a percentage has `_pct` in its name and returns
    a number already multiplied by 100. Callers must NOT re-scale.
"""

import warnings
import numpy as np
from typing import List, Tuple, Optional


class MetricUnitError(ValueError):
    """Raised when inputs are inconsistent with the documented unit convention."""


def _as_returns(returns: np.ndarray) -> np.ndarray:
    """Validate and coerce a return series to a clean 1-D float array."""
    r = np.asarray(returns, dtype=np.float64).ravel()
    if r.size == 0:
        return r
    if not np.all(np.isfinite(r)):
        n_bad = int((~np.isfinite(r)).sum())
        raise MetricUnitError(
            f"Return series contains {n_bad} non-finite values. This usually means "
            "the equity curve touched zero upstream -- fix the source, do not mask it."
        )
    if np.max(np.abs(r)) > 10.0:
        warnings.warn(
            f"Return series has |max| = {np.max(np.abs(r)):.2f}. Returns must be "
            "fractions (0.01 = 1%), not percentages. Check the caller.",
            RuntimeWarning,
            stacklevel=3,
        )
    return r


def _rf_per_period(risk_free_rate: float, periods_per_year: Optional[int]) -> float:
    """
    Convert an ANNUAL risk-free rate to a per-period rate by compounding.

    Returns 0.0 when `periods_per_year` is None: without a stated sampling
    frequency an annual rate cannot be converted, and subtracting an
    unconverted annual rate from per-tick returns is the classic sign-flip bug.
    """
    if periods_per_year is None:
        return 0.0
    if periods_per_year <= 0:
        raise MetricUnitError(f"periods_per_year must be positive, got {periods_per_year}")
    return (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0


def compute_pnl(
    prices: np.ndarray,
    positions: np.ndarray,
    initial_capital: float = 100000.0,
) -> Tuple[np.ndarray, float]:
    """
    Compute an equity curve from a price path and a position series.

    Args:
        prices: Asset prices over time, length T. Must be strictly positive.
        positions: Portfolio weight at each step, length T. `positions[t]` is
            the weight held over the interval (t, t+1].
        initial_capital: Starting capital.

    Returns:
        equity_curve: Portfolio value over time, length T.
        total_return_pct: Final return, in percent.
    """
    prices = np.asarray(prices, dtype=np.float64).ravel()
    positions = np.asarray(positions, dtype=np.float64).ravel()

    if prices.size < 2:
        return np.array([initial_capital], dtype=np.float64), 0.0
    if np.any(prices <= 0):
        raise MetricUnitError(
            "Price path contains non-positive values; simple returns are undefined."
        )
    if positions.size != prices.size:
        raise MetricUnitError(
            f"positions (len {positions.size}) and prices (len {prices.size}) must align."
        )

    price_returns = np.diff(prices) / prices[:-1]
    position_returns = positions[:-1] * price_returns

    # A weight of |w| > 1 is leverage; compounding can drive equity negative.
    growth = 1.0 + position_returns
    if np.any(growth <= 0):
        first = int(np.argmax(growth <= 0))
        warnings.warn(
            f"Equity curve wiped out at step {first} (growth factor <= 0). "
            "Truncating there; the run is a blow-up, not a result.",
            RuntimeWarning,
            stacklevel=2,
        )
        growth = growth[:first]

    equity_curve = initial_capital * np.cumprod(growth)
    equity_curve = np.insert(equity_curve, 0, initial_capital)

    total_return_pct = (equity_curve[-1] - initial_capital) / initial_capital * 100.0
    return equity_curve, total_return_pct


def compute_sharpe(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: Optional[int] = None,
) -> float:
    """
    Sharpe ratio.

    With `periods_per_year=None` (default) this is the per-period Sharpe and
    the risk-free rate is ignored -- the honest reading for tick data. Pass an
    integer only when `returns` really are sampled at that frequency.

    Raises MetricUnitError if the risk-free rate dominates the return series,
    which means the annual rate and the return frequency disagree.
    """
    r = _as_returns(returns)
    if r.size < 2:
        return 0.0

    sd = float(np.std(r, ddof=1))
    if sd == 0.0 or not np.isfinite(sd):
        return 0.0

    rf = _rf_per_period(risk_free_rate, periods_per_year)

    # Units guard: the whole point of the original bug. A daily rf subtracted
    # from tick returns swamps the mean and flips the sign of the ratio.
    if rf != 0.0 and abs(rf) > 0.5 * sd:
        raise MetricUnitError(
            f"Risk-free rate per period ({rf:.3e}) is large relative to the "
            f"standard deviation of returns ({sd:.3e}). periods_per_year="
            f"{periods_per_year} does not match the sampling frequency of these "
            "returns; the resulting Sharpe would be dominated by the rf term. "
            "Pass periods_per_year=None for a per-period Sharpe, or supply the "
            "correct frequency."
        )

    excess = r - rf
    ratio = float(np.mean(excess) / sd)
    if periods_per_year is not None:
        ratio *= np.sqrt(periods_per_year)
    return ratio


def compute_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Maximum drawdown.

    Returns:
        max_dd_pct: Maximum drawdown IN PERCENT, guaranteed in [0, 100].
                    Do not multiply by 100 again at the call site.
        peak_idx: Index of the peak preceding the max drawdown.
        trough_idx: Index of the trough.
    """
    eq = np.asarray(equity_curve, dtype=np.float64).ravel()
    if eq.size < 2:
        return 0.0, 0, 0
    if np.any(eq <= 0):
        raise MetricUnitError(
            "Equity curve contains non-positive values; drawdown is undefined. "
            "The run blew up -- report that, do not compute a ratio."
        )

    running_max = np.maximum.accumulate(eq)
    drawdown = (running_max - eq) / running_max

    max_dd = float(np.max(drawdown))
    trough_idx = int(np.argmax(drawdown))
    peak_idx = int(np.argmax(eq[: trough_idx + 1]))

    max_dd_pct = max_dd * 100.0
    # Unlevered drawdown is definitionally in [0, 100]. Anything else is a bug.
    if not (0.0 <= max_dd_pct <= 100.0):
        raise MetricUnitError(
            f"Max drawdown computed as {max_dd_pct:.2f}%, which is outside [0, 100]. "
            "This indicates a double-scaling bug or a non-monotone equity input."
        )
    return max_dd_pct, peak_idx, trough_idx


def compute_sortino(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: Optional[int] = None,
) -> float:
    """Sortino ratio. Same unit contract as `compute_sharpe`."""
    r = _as_returns(returns)
    if r.size < 2:
        return 0.0

    rf = _rf_per_period(risk_free_rate, periods_per_year)
    excess = r - rf
    downside = excess[excess < 0]
    if downside.size == 0:
        return float("inf")

    downside_std = float(np.std(downside, ddof=1)) if downside.size > 1 else float(np.std(downside))
    if downside_std == 0.0:
        return float("inf")

    ratio = float(np.mean(excess) / downside_std)
    if periods_per_year is not None:
        ratio *= np.sqrt(periods_per_year)
    return ratio


def compute_calmar(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    periods_per_year: Optional[int] = None,
) -> float:
    """
    Calmar ratio: annualised return over max drawdown.

    Requires `periods_per_year` -- an annualised numerator is the definition of
    this ratio, so there is no meaningful per-period variant.
    """
    if periods_per_year is None:
        raise MetricUnitError(
            "compute_calmar requires periods_per_year: the ratio is defined on an "
            "annualised return and cannot be formed from per-period returns alone."
        )
    r = _as_returns(returns)
    if r.size == 0:
        return 0.0

    annual_return = float(np.mean(r) * periods_per_year)
    max_dd_pct, _, _ = compute_max_drawdown(equity_curve)
    if max_dd_pct == 0.0:
        return float("inf")
    return annual_return / (max_dd_pct / 100.0)


def compute_win_rate_pct(returns: np.ndarray) -> float:
    """Percentage of periods with a positive return."""
    r = _as_returns(returns)
    if r.size == 0:
        return 0.0
    return float(np.sum(r > 0) / r.size * 100.0)


def compute_profit_factor(returns: np.ndarray) -> float:
    """Ratio of gross profit to gross loss."""
    r = _as_returns(returns)
    if r.size == 0:
        return 0.0
    gains = float(r[r > 0].sum())
    losses = float(np.abs(r[r < 0].sum()))
    if losses == 0.0:
        return float("inf")
    return gains / losses


def compute_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Value at Risk, returned as a POSITIVE number representing a loss fraction.
    """
    r = _as_returns(returns)
    if r.size == 0:
        return 0.0
    if not (0.0 < confidence < 1.0):
        raise MetricUnitError(f"confidence must be in (0, 1), got {confidence}")
    return float(-np.percentile(r, (1.0 - confidence) * 100.0))


def compute_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Conditional VaR / Expected Shortfall, as a POSITIVE loss fraction.
    """
    r = _as_returns(returns)
    if r.size == 0:
        return 0.0
    var = compute_var(r, confidence)
    tail = r[r <= -var]
    if tail.size == 0:
        return var
    return float(-np.mean(tail))


def compute_all_metrics(
    prices: np.ndarray,
    positions: np.ndarray,
    initial_capital: float = 100000.0,
    periods_per_year: Optional[int] = None,
) -> dict:
    """
    Compute the full metric set from a price path and position series.

    `periods_per_year` is threaded through to the annualised ratios; leave it
    None for tick data. Calmar is omitted when it is None, because that ratio
    has no per-period form.
    """
    equity_curve, total_return_pct = compute_pnl(prices, positions, initial_capital)
    if equity_curve.size < 2:
        raise MetricUnitError(
            "Equity curve has fewer than 2 points; there is nothing to measure. "
            "This is an empty evaluation, not a zero-return result."
        )

    returns = np.diff(equity_curve) / equity_curve[:-1]

    metrics = {
        "total_return_pct": total_return_pct,
        "sharpe_ratio": compute_sharpe(returns, periods_per_year=periods_per_year),
        "sortino_ratio": compute_sortino(returns, periods_per_year=periods_per_year),
        "max_drawdown_pct": compute_max_drawdown(equity_curve)[0],
        "win_rate_pct": compute_win_rate_pct(returns),
        "profit_factor": compute_profit_factor(returns),
        "var_95": compute_var(returns, 0.95),
        "cvar_95": compute_cvar(returns, 0.95),
        "final_equity": float(equity_curve[-1]),
        "n_periods": int(returns.size),
        "annualised": periods_per_year is not None,
    }
    if periods_per_year is not None:
        metrics["calmar_ratio"] = compute_calmar(returns, equity_curve, periods_per_year)
    return metrics


def format_metrics_table(metrics: dict) -> str:
    """Format metrics as a table for printing/logging."""
    lines = ["=" * 46, "Trading Performance Metrics", "=" * 46]
    for key, value in metrics.items():
        if isinstance(value, bool):
            lines.append(f"{key:28s}: {str(value):>14}")
        elif isinstance(value, float):
            lines.append(f"{key:28s}: {value:>14.4f}")
        else:
            lines.append(f"{key:28s}: {value:>14}")
    lines.append("=" * 46)
    return "\n".join(lines)
