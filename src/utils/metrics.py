"""
Evaluation Metrics for Trading Performance

Includes standard financial metrics required for ICML submission.
"""

import numpy as np
from typing import List, Tuple, Optional


def compute_pnl(
    prices: np.ndarray,
    positions: np.ndarray,
    initial_capital: float = 100000.0,
) -> Tuple[np.ndarray, float]:
    """
    Compute Profit and Loss (PnL) from trading positions.
    
    Args:
        prices: Asset prices over time
        positions: Position sizes at each timestep
        initial_capital: Starting capital
        
    Returns:
        equity_curve: Portfolio value over time
        total_return: Final return percentage
    """
    # Calculate returns
    price_returns = np.diff(prices) / prices[:-1]
    
    # Position returns (shifted by 1 since position at t affects return at t+1)
    position_returns = positions[:-1] * price_returns
    
    # Cumulative returns
    equity_curve = initial_capital * np.cumprod(1 + position_returns)
    equity_curve = np.insert(equity_curve, 0, initial_capital)
    
    total_return = (equity_curve[-1] - initial_capital) / initial_capital * 100
    
    return equity_curve, total_return


def compute_sharpe(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized Sharpe Ratio.
    
    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    # Convert annual risk-free to per-period
    rf_per_period = risk_free_rate / periods_per_year
    
    # Excess returns
    excess_returns = returns - rf_per_period
    
    # Annualized Sharpe
    sharpe = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(returns)
    
    return sharpe


def compute_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Compute Maximum Drawdown.
    
    Args:
        equity_curve: Portfolio value over time
        
    Returns:
        max_dd: Maximum drawdown percentage
        peak_idx: Index of peak before max drawdown
        trough_idx: Index of trough during max drawdown
    """
    # Running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Drawdown at each point
    drawdown = (running_max - equity_curve) / running_max
    
    # Maximum drawdown
    max_dd = np.max(drawdown)
    trough_idx = np.argmax(drawdown)
    peak_idx = np.argmax(equity_curve[:trough_idx + 1])
    
    return max_dd * 100, peak_idx, trough_idx


def compute_sortino(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """
    Compute Sortino Ratio (downside risk-adjusted return).
    
    Unlike Sharpe, Sortino only penalizes downside volatility.
    """
    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period
    
    # Downside returns only
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf  # No downside volatility
    
    downside_std = np.std(downside_returns)
    
    if downside_std == 0:
        return np.inf
    
    sortino = np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_std
    
    return sortino


def compute_calmar(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    Compute Calmar Ratio (return / max drawdown).
    """
    annual_return = np.mean(returns) * periods_per_year
    max_dd, _, _ = compute_max_drawdown(equity_curve)
    
    if max_dd == 0:
        return np.inf
    
    return annual_return / (max_dd / 100)


def compute_win_rate(returns: np.ndarray) -> float:
    """Compute percentage of profitable trades."""
    if len(returns) == 0:
        return 0.0
    
    return np.sum(returns > 0) / len(returns) * 100


def compute_profit_factor(returns: np.ndarray) -> float:
    """Compute ratio of gross profit to gross loss."""
    gains = returns[returns > 0].sum()
    losses = np.abs(returns[returns < 0].sum())
    
    if losses == 0:
        return np.inf
    
    return gains / losses


def compute_var(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Compute Value at Risk (VaR).
    
    VaR at confidence level alpha is the loss that is exceeded
    only (1-alpha)% of the time.
    
    Args:
        returns: Array of period returns
        confidence: Confidence level (e.g. 0.95 for 95% VaR)
        
    Returns:
        VaR as a positive number representing potential loss
    """
    if len(returns) == 0:
        return 0.0
    return -np.percentile(returns, (1 - confidence) * 100)


def compute_cvar(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Compute Conditional Value at Risk (CVaR / Expected Shortfall).
    
    CVaR is the expected loss given that the loss exceeds VaR.
    This captures tail risk better than VaR alone.
    
    Args:
        returns: Array of period returns
        confidence: Confidence level
        
    Returns:
        CVaR as a positive number representing expected tail loss
    """
    if len(returns) == 0:
        return 0.0
    var = compute_var(returns, confidence)
    tail_returns = returns[returns <= -var]
    if len(tail_returns) == 0:
        return var
    return -np.mean(tail_returns)


def compute_all_metrics(
    prices: np.ndarray,
    positions: np.ndarray,
    initial_capital: float = 100000.0,
) -> dict:
    """
    Compute all trading metrics.
    
    Returns a dictionary with all metrics for easy logging/reporting.
    """
    equity_curve, total_return = compute_pnl(prices, positions, initial_capital)
    
    # Period returns from equity curve
    returns = np.diff(equity_curve) / equity_curve[:-1]
    
    return {
        "total_return_pct": total_return,
        "sharpe_ratio": compute_sharpe(returns),
        "sortino_ratio": compute_sortino(returns),
        "max_drawdown_pct": compute_max_drawdown(equity_curve)[0],
        "calmar_ratio": compute_calmar(returns, equity_curve),
        "win_rate_pct": compute_win_rate(returns),
        "profit_factor": compute_profit_factor(returns),
        "var_95": compute_var(returns, 0.95),
        "cvar_95": compute_cvar(returns, 0.95),
        "final_equity": equity_curve[-1],
    }


def format_metrics_table(metrics: dict) -> str:
    """Format metrics as a table for printing/logging."""
    lines = [
        "=" * 40,
        "Trading Performance Metrics",
        "=" * 40,
    ]
    
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key:25s}: {value:>12.4f}")
        else:
            lines.append(f"{key:25s}: {value:>12}")
    
    lines.append("=" * 40)
    
    return "\n".join(lines)
