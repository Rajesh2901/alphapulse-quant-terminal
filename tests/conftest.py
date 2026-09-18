"""
AlphaPulse Test Fixtures & Mock Generators
Provides reproducible market dataframes, return series, and mock pipelines.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_ohlcv_data() -> pd.DataFrame:
    """Generates 120 bars of synthetic geometric Brownian motion OHLCV data."""
    np.random.seed(42)
    n_bars = 120
    dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n_bars)

    returns = np.random.normal(0.0005, 0.015, n_bars)
    close_prices = 100.0 * np.exp(np.cumsum(returns))

    high_spread = np.abs(np.random.normal(0, 0.5, n_bars))
    low_spread = np.abs(np.random.normal(0, 0.5, n_bars))

    highs = close_prices + high_spread
    lows = close_prices - low_spread
    opens = np.roll(close_prices, 1)
    opens[0] = 100.0
    volumes = np.random.lognormal(mean=14.0, sigma=0.4, size=n_bars)

    df = pd.DataFrame(
        {
            "Open": np.round(opens, 2),
            "High": np.round(highs, 2),
            "Low": np.round(lows, 2),
            "Close": np.round(close_prices, 2),
            "Volume": np.round(volumes, 0),
        },
        index=dates,
    )
    return df


@pytest.fixture
def mock_returns(mock_ohlcv_data: pd.DataFrame) -> pd.Series:
    """Provides daily simple returns series."""
    return mock_ohlcv_data["Close"].pct_change().dropna()
