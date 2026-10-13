
"""Portfolio backtesting utilities."""

import numpy as np
import pandas as pd


def portfolio_returns(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    lag: int = 1,
) -> pd.Series:
    """Calculate portfolio returns without look-ahead bias.

    Assumes:
        - weights[t] are determined at the end of period t;
        - asset_returns[t] are returns realized during period t;
        - rows share the same timestamps;
        - weights and returns use identical asset columns;
        - weights represent portfolio exposure per unit of capital.

    The default lag=1 applies weights[t-1] to returns[t].
    """
    if lag < 1:
        raise ValueError("lag must be at least 1")

    if not weights.index.equals(asset_returns.index):
        raise ValueError("Weights and returns must have the same index")

    if not weights.columns.equals(asset_returns.columns):
        raise ValueError("Weights and returns must have the same columns")

    if not weights.index.is_unique or not weights.index.is_monotonic_increasing:
        raise ValueError("Index must be unique and sorted")

    if weights.isna().any().any() or asset_returns.isna().any().any():
        raise ValueError("Inputs must not contain missing values")

    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("Weights must be finite")

    if not np.isfinite(asset_returns.to_numpy()).all():
        raise ValueError("Returns must be finite")

    lagged_weights = weights.shift(lag)

    # The first lag periods have no previously known weights.
    valid = lagged_weights.notna().all(axis=1)

    result = (
        lagged_weights.loc[valid] * asset_returns.loc[valid]
    ).sum(axis=1)

    result.name = "portfolio_return"
    return result
