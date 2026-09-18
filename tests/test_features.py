"""
Unit Tests for FeatureEngine
Validates indicator accuracy, missing values handling, and ML tensor dimensions.
"""

import pandas as pd

from features import FeatureEngine


def test_compute_all_features(mock_ohlcv_data: pd.DataFrame):
    df_feat = FeatureEngine.compute_all_features(mock_ohlcv_data)

    # Core indicator columns
    expected_cols = [
        "Log_Return",
        "Return",
        "EMA_9",
        "EMA_21",
        "SMA_50",
        "BB_Upper",
        "BB_Lower",
        "BB_PctB",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "ATR",
        "Realized_Vol_20",
        "OBV",
    ]
    for col in expected_cols:
        assert col in df_feat.columns, f"Missing feature column: {col}"

    # RSI must be bounded between 0 and 100
    valid_rsi = df_feat["RSI"].dropna()
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    # Bollinger Bands relationship
    valid_bb = df_feat.dropna(subset=["BB_Upper", "BB_Lower", "BB_Mid"])
    assert (valid_bb["BB_Upper"] >= valid_bb["BB_Mid"]).all()
    assert (valid_bb["BB_Mid"] >= valid_bb["BB_Lower"]).all()


def test_build_ml_feature_matrix(mock_ohlcv_data: pd.DataFrame):
    X, y, feature_cols = FeatureEngine.build_ml_feature_matrix(
        mock_ohlcv_data, n_lags=4, target_horizon=1
    )

    assert len(X) == len(y)
    assert len(X) > 20
    assert "Lag_Return_1" in feature_cols
    assert "Lag_Return_4" in feature_cols
    assert "Target_Return" not in X.columns
    assert not X.isnull().values.any(), "ML feature matrix contains unexpected NaNs"
