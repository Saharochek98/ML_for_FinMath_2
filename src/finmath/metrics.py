
"""Financial performance metrics."""

import numpy as np
import pandas as pd

PERIODS_PER_YEAR_4H = 365 * 6

def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR_4H,
) -> float:
    """Calculate annualized volatility of periodic returns."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    clean = returns.dropna()

    if len(clean) < 2:
        return float("nan")

    return float(
        clean.std(ddof=1) * np.sqrt(periods_per_year)
    )


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR_4H,
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate annualized Sharpe ratio.

    Args:
        returns: Periodic simple returns, expressed as fractions.
        periods_per_year: Number of return observations per year.
        risk_free_rate: Annualized risk-free rate, expressed as
            a fraction.

    Returns:
        Annualized Sharpe ratio, or NaN when it cannot be computed.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    clean = returns.dropna()

    if len(clean) < 2:
        return float("nan")

    periodic_rf = (1 + risk_free_rate) ** (
        1 / periods_per_year
    ) - 1

    excess_returns = clean - periodic_rf
    volatility = excess_returns.std(ddof=1)

    if np.isclose(volatility, 0.0):
        return float("nan")

    return float(
        excess_returns.mean()
        / volatility
        * np.sqrt(periods_per_year)
    )


def max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from periodic simple returns.

    Returns:
        A non-negative drawdown magnitude between 0 and 1
        for a long-only equity curve that remains positive.
    """
    clean = returns.dropna()

    if clean.empty:
        return float("nan")

    if (clean < -1).any():
        raise ValueError(
            "Simple returns cannot be below -100%"
        )

    equity = (1 + clean).cumprod()

    # Include initial equity to capture first-period losses.
    equity = pd.concat([
        pd.Series([1.0]),
        equity.reset_index(drop=True),
    ], ignore_index=True)

    running_peak = equity.cummax()
    drawdown = 1 - equity / running_peak

    return float(drawdown.max())
