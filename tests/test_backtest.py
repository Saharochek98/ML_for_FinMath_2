
import pandas as pd
import pytest

from finmath.backtest import portfolio_returns


def test_portfolio_returns_use_previous_weights():
    dates = pd.date_range(
        "2025-01-01", periods=3, freq="4h"
    )

    weights = pd.DataFrame(
        {
            "BTC": [1.0, 0.0, 1.0],
            "ETH": [0.0, 1.0, 0.0],
        },
        index=dates,
    )

    returns = pd.DataFrame(
        {
            "BTC": [0.10, 0.02, 0.03],
            "ETH": [0.01, 0.04, -0.01],
        },
        index=dates,
    )

    result = portfolio_returns(weights, returns)

    # t=1: previous BTC weight = 1 -> 0.02
    # t=2: previous ETH weight = 1 -> -0.01
    assert result.tolist() == pytest.approx(
        [0.02, -0.01]
    )


def test_first_period_is_excluded():
    dates = pd.date_range(
        "2025-01-01", periods=2, freq="4h"
    )

    weights = pd.DataFrame(
        {"BTC": [1.0, 1.0]}, index=dates
    )
    returns = pd.DataFrame(
        {"BTC": [0.50, 0.01]}, index=dates
    )

    result = portfolio_returns(weights, returns)

    assert len(result) == 1
    assert result.iloc[0] == pytest.approx(0.01)


def test_mismatched_assets_raise_error():
    weights = pd.DataFrame({"BTC": [1.0, 1.0]})
    returns = pd.DataFrame({"ETH": [0.01, 0.02]})

    with pytest.raises(ValueError):
        portfolio_returns(weights, returns)


def test_mismatched_dates_raise_error():
    weights = pd.DataFrame(
        {"BTC": [1.0, 1.0]},
        index=pd.date_range("2025-01-01", periods=2),
    )
    returns = pd.DataFrame(
        {"BTC": [0.01, 0.02]},
        index=pd.date_range("2025-02-01", periods=2),
    )

    with pytest.raises(ValueError):
        portfolio_returns(weights, returns)


def test_unsorted_index_raises_error():
    dates = pd.to_datetime(
        ["2025-01-02", "2025-01-01"]
    )

    weights = pd.DataFrame(
        {"BTC": [1.0, 1.0]}, index=dates
    )
    returns = pd.DataFrame(
        {"BTC": [0.01, 0.02]}, index=dates
    )

    with pytest.raises(ValueError):
        portfolio_returns(weights, returns)
