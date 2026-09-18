"""
AlphaPulse Quantitative Terminal - Feature Engineering & Technical Analysis
Calculates statistical rolling metrics, momentum oscillators, volatility bands, and ML feature tensors.
"""

import numpy as np
import pandas as pd


class FeatureEngine:
    """
    Institutional feature engineering pipeline transforming raw OHLCV time-series
    into quantitative signal matrices, econometric stationary series, and ML training frames.
    """

    @staticmethod
    def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        # 1. Log Returns & Simple Returns
        data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))
        data["Return"] = data["Close"].pct_change()

        # 2. Moving Averages
        data["EMA_9"] = data["Close"].ewm(span=9, adjust=False).mean()
        data["EMA_21"] = data["Close"].ewm(span=21, adjust=False).mean()
        data["SMA_50"] = data["Close"].rolling(window=50, min_periods=10).mean()
        data["SMA_200"] = data["Close"].rolling(window=200, min_periods=30).mean()

        # 3. Bollinger Bands (20, 2 std)
        bb_window = 20
        data["BB_Mid"] = data["Close"].rolling(window=bb_window, min_periods=5).mean()
        data["BB_Std"] = data["Close"].rolling(window=bb_window, min_periods=5).std()
        data["BB_Upper"] = data["BB_Mid"] + (2.0 * data["BB_Std"])
        data["BB_Lower"] = data["BB_Mid"] - (2.0 * data["BB_Std"])
        data["BB_Bandwidth"] = (data["BB_Upper"] - data["BB_Lower"]) / (data["BB_Mid"] + 1e-6)
        data["BB_PctB"] = (data["Close"] - data["BB_Lower"]) / (
            data["BB_Upper"] - data["BB_Lower"] + 1e-6
        )

        # 4. Relative Strength Index (RSI 14)
        delta = data["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        data["RSI"] = 100.0 - (100.0 / (1.0 + rs))

        # 5. Moving Average Convergence Divergence (MACD 12, 26, 9)
        ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = ema_12 - ema_26
        data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
        data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

        # 6. Average True Range (ATR 14)
        high_low = data["High"] - data["Low"]
        high_close_prev = (data["High"] - data["Close"].shift(1)).abs()
        low_close_prev = (data["Low"] - data["Close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        data["ATR"] = true_range.ewm(span=14, adjust=False).mean()
        data["ATR_Pct"] = data["ATR"] / (data["Close"] + 1e-6)

        # 7. Realized Volatility (Rolling 20-bar annualized)
        data["Realized_Vol_20"] = data["Log_Return"].rolling(
            window=20, min_periods=5
        ).std() * np.sqrt(252)

        # 8. On-Balance Volume (OBV)
        obv_direction = np.where(
            data["Close"] > data["Close"].shift(1),
            1,
            np.where(data["Close"] < data["Close"].shift(1), -1, 0),
        )
        data["OBV"] = (obv_direction * data["Volume"]).cumsum()

        # 9. Volume Moving Average Ratio
        data["Vol_SMA_20"] = data["Volume"].rolling(window=20, min_periods=5).mean()
        data["Vol_Ratio"] = data["Volume"] / (data["Vol_SMA_20"] + 1e-6)

        return data

    @staticmethod
    def build_ml_feature_matrix(
        df: pd.DataFrame, n_lags: int = 5, target_horizon: int = 1
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        features_df = FeatureEngine.compute_all_features(df).copy()

        feature_cols: list[str] = []
        for i in range(1, n_lags + 1):
            col_name = f"Lag_Return_{i}"
            features_df[col_name] = features_df["Log_Return"].shift(i)
            feature_cols.append(col_name)

        core_features = [
            "RSI",
            "MACD_Hist",
            "BB_PctB",
            "BB_Bandwidth",
            "ATR_Pct",
            "Realized_Vol_20",
            "Vol_Ratio",
        ]

        features_df["Dist_EMA_21"] = (features_df["Close"] - features_df["EMA_21"]) / (
            features_df["EMA_21"] + 1e-6
        )
        core_features.append("Dist_EMA_21")

        for col in core_features:
            if col in features_df.columns:
                feature_cols.append(col)

        future_close = features_df["Close"].shift(-target_horizon)
        features_df["Target_Return"] = np.log(future_close / features_df["Close"])

        clean_ml_df = features_df.dropna(subset=feature_cols + ["Target_Return"])

        X = clean_ml_df[feature_cols]
        y = clean_ml_df["Target_Return"]

        return X, y, feature_cols
