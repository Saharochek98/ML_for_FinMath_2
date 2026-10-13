
import numpy as np
import pandas as pd
import pytest

from finmath.metrics import (
    PERIODS_PER_YEAR_4H,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
)


def test_annualized_volatility():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02])

    expected = returns.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR_4H)

    assert annualized_volatility(returns) == pytest.approx(
        expected
    )


def test_sharpe_ratio():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03])

    expected = (
        returns.mean()
        / returns.std(ddof=1)
        * np.sqrt(PERIODS_PER_YEAR_4H)
    )

    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_zero_volatility_returns_nan():
    returns = pd.Series([0.01, 0.01, 0.01])

    assert np.isnan(sharpe_ratio(returns))


def test_max_drawdown():
    returns = pd.Series([0.10, -0.20, 0.05])

    # Equity: 1.0 -> 1.1 -> 0.88 -> 0.924
    # Largest decline: 20% from 1.1 to 0.88.
    assert max_drawdown(returns) == pytest.approx(0.20)


def test_first_period_loss_counts_as_drawdown():
    returns = pd.Series([-0.10, 0.05])

    assert max_drawdown(returns) == pytest.approx(0.10)


def test_empty_returns():
    returns = pd.Series(dtype=float)

    assert np.isnan(sharpe_ratio(returns))
    assert np.isnan(annualized_volatility(returns))
    assert np.isnan(max_drawdown(returns))


def test_invalid_annualization_factor():
    returns = pd.Series([0.01, 0.02])

    with pytest.raises(ValueError):
        sharpe_ratio(returns, periods_per_year=0)


def test_four_hour_annualization():
    returns = pd.Series([0.01, -0.02, 0.015, 0.005])

    expected = (
        returns.mean()
        / returns.std(ddof=1)
        * np.sqrt(2190)
    )

    assert sharpe_ratio(returns) == pytest.approx(expected)
