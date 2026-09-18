"""
AlphaPulse Quantitative Terminal - Predictive Analytics & Consensus Signal Engine
Econometric ARIMA modeling, Machine Learning Ensemble forecasting, and Multi-Factor quantitative signals.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from features import FeatureEngine

warnings.filterwarnings("ignore")


class ARIMAForecaster:
    """
    Institutional econometric time-series forecaster with automated order parameterization,
    stationarity validation (ADF), and analytical confidence interval fan charts.
    """

    @staticmethod
    def fit_and_forecast(
        series: pd.Series, horizon: int = 10, order: tuple[int, int, int] = (2, 1, 2)
    ) -> dict[str, Any]:
        close_prices = series.dropna().astype(float)
        if len(close_prices) < 30:
            raise ValueError(
                "Insufficient data points for econometric estimation (minimum 30 required)."
            )

        log_prices = np.log(close_prices)

        # 1. Augmented Dickey-Fuller Stationarity Test
        log_diff = log_prices.diff().dropna()
        adf_res = adfuller(log_diff)
        adf_pvalue = float(adf_res[1])
        is_stationary = adf_pvalue < 0.05

        # 2. Fit ARIMA model on 1D array to guarantee index-agnostic forecasting
        p, d, q = order
        model = ARIMA(log_prices.values, order=(p, d, q))
        try:
            fitted_model = model.fit()
        except Exception:
            model = ARIMA(log_prices.values, order=(1, 1, 1))
            fitted_model = model.fit()

        # 3. Generate out-of-sample forecast
        forecast_res = fitted_model.get_forecast(steps=horizon)
        pred_log_mean = np.asarray(forecast_res.predicted_mean)

        conf_int_95 = np.asarray(forecast_res.conf_int(alpha=0.05))
        conf_int_80 = np.asarray(forecast_res.conf_int(alpha=0.20))

        pred_prices = np.exp(pred_log_mean)
        lower_95 = np.exp(conf_int_95[:, 0])
        upper_95 = np.exp(conf_int_95[:, 1])
        lower_80 = np.exp(conf_int_80[:, 0])
        upper_80 = np.exp(conf_int_80[:, 1])

        last_date = series.index[-1]
        freq = pd.infer_freq(series.index) or "B"
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

        pred_series = pd.Series(pred_prices, index=future_dates)
        lower_95_series = pd.Series(lower_95, index=future_dates)
        upper_95_series = pd.Series(upper_95, index=future_dates)
        lower_80_series = pd.Series(lower_80, index=future_dates)
        upper_80_series = pd.Series(upper_80, index=future_dates)

        # 4. Residual Diagnostics
        residuals = fitted_model.resid
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        lb_pvalue = float(lb_test["lb_pvalue"].iloc[0])

        current_price = float(close_prices.iloc[-1])
        final_forecast = float(pred_series.iloc[-1])
        expected_change_pct = ((final_forecast - current_price) / current_price) * 100.0

        return {
            "pred_series": pred_series,
            "lower_95": lower_95_series,
            "upper_95": upper_95_series,
            "lower_80": lower_80_series,
            "upper_80": upper_80_series,
            "expected_change_pct": round(expected_change_pct, 2),
            "aic": round(float(fitted_model.aic), 2),
            "bic": round(float(fitted_model.bic), 2),
            "adf_pvalue": round(adf_pvalue, 4),
            "is_stationary": is_stationary,
            "ljung_box_pvalue": round(lb_pvalue, 4),
            "residuals": residuals,
        }


class MLEnsembleForecaster:
    """
    Ensemble machine learning predictor utilizing Gradient Boosting and Random Forest
    regressors trained on technical features, volatility, and autoregressive lags.
    """

    def __init__(self, n_estimators: int = 120, max_depth: int = 4):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        self.gb = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.04, random_state=42
        )
        self.feature_names: list[str] = []
        self.feature_importances_: dict[str, float] = {}

    def train_and_predict(
        self, df: pd.DataFrame, horizon: int = 10, n_lags: int = 5
    ) -> dict[str, Any]:
        X, y, feature_cols = FeatureEngine.build_ml_feature_matrix(
            df, n_lags=n_lags, target_horizon=1
        )
        self.feature_names = feature_cols

        if len(X) < 40:
            raise ValueError("Insufficient sample length for machine learning training.")

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self.rf.fit(X_train, y_train)
        self.gb.fit(X_train, y_train)

        pred_test_rf = self.rf.predict(X_test)
        pred_test_gb = self.gb.predict(X_test)
        pred_test = 0.5 * (pred_test_rf + pred_test_gb)

        mse = mean_squared_error(y_test, pred_test)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, pred_test))

        actual_direction = np.sign(y_test)
        pred_direction = np.sign(pred_test)
        directional_acc = float(np.mean(actual_direction == pred_direction) * 100.0)

        self.rf.fit(X, y)
        self.gb.fit(X, y)

        importances = 0.5 * (self.rf.feature_importances_ + self.gb.feature_importances_)
        feat_imp_df = pd.DataFrame(
            {"Feature": feature_cols, "Importance": importances}
        ).sort_values(by="Importance", ascending=False)

        current_features = X.iloc[-1:].copy()
        current_price = float(df["Close"].iloc[-1])
        predicted_prices = []
        last_price = current_price

        resid_std = float(np.std(y - (0.5 * (self.rf.predict(X) + self.gb.predict(X)))))

        for _step in range(horizon):
            pred_ret_rf = float(self.rf.predict(current_features)[0])
            pred_ret_gb = float(self.gb.predict(current_features)[0])
            step_return = 0.5 * (pred_ret_rf + pred_ret_gb)

            next_price = last_price * np.exp(step_return)
            predicted_prices.append(next_price)
            last_price = next_price

            if "Lag_Return_1" in current_features.columns:
                for lag in range(n_lags, 1, -1):
                    current_features[f"Lag_Return_{lag}"] = current_features[
                        f"Lag_Return_{lag - 1}"
                    ].values
                current_features["Lag_Return_1"] = step_return

        last_date = df.index[-1]
        freq = pd.infer_freq(df.index) or "B"
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
        ml_pred_series = pd.Series(predicted_prices, index=future_dates)

        total_pred_return = np.log(predicted_prices[-1] / current_price)
        z_score = total_pred_return / (resid_std * np.sqrt(horizon) + 1e-6)
        prob_bullish = float(norm.cdf(z_score) * 100.0)

        return {
            "pred_series": ml_pred_series,
            "prob_bullish": round(prob_bullish, 1),
            "prob_bearish": round(100.0 - prob_bullish, 1),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "directional_accuracy": round(directional_acc, 1),
            "feature_importance": feat_imp_df,
            "expected_change_pct": round(
                ((predicted_prices[-1] - current_price) / current_price) * 100.0, 2
            ),
        }


class ConsensusSignalEngine:
    """
    Synthesizes Trend, Momentum, Mean Reversion, Econometric ARIMA, and ML Ensemble
    signals into an institutional rating with risk-reward parameters.
    """

    @staticmethod
    def generate_consensus_signal(
        df_features: pd.DataFrame, arima_results: dict[str, Any], ml_results: dict[str, Any]
    ) -> dict[str, Any]:
        latest = df_features.iloc[-1]
        score = 0.0

        # 1. Trend Factor (Weight: 25%)
        close = latest["Close"]
        ema_9 = latest.get("EMA_9", close)
        ema_21 = latest.get("EMA_21", close)
        sma_50 = latest.get("SMA_50", close)

        trend_subscore = 0.0
        if ema_9 > ema_21:
            trend_subscore += 15.0
        else:
            trend_subscore -= 15.0

        if close > sma_50:
            trend_subscore += 10.0
        else:
            trend_subscore -= 10.0

        score += trend_subscore

        # 2. Momentum Factor (Weight: 20%)
        rsi = latest.get("RSI", 50.0)
        macd_hist = latest.get("MACD_Hist", 0.0)

        momentum_subscore = 0.0
        if rsi < 30.0:
            momentum_subscore += 10.0
        elif rsi > 70.0:
            momentum_subscore -= 10.0
        elif rsi > 50.0:
            momentum_subscore += 5.0
        else:
            momentum_subscore -= 5.0

        if macd_hist > 0:
            momentum_subscore += 10.0
        else:
            momentum_subscore -= 10.0

        score += momentum_subscore

        # 3. Mean Reversion Factor (Weight: 15%)
        pct_b = latest.get("BB_PctB", 0.5)
        reversion_subscore = 0.0
        if pct_b < 0.1:
            reversion_subscore += 15.0
        elif pct_b > 0.9:
            reversion_subscore -= 15.0
        score += reversion_subscore

        # 4. Econometric ARIMA Forecast (Weight: 20%)
        arima_chg = arima_results.get("expected_change_pct", 0.0)
        arima_subscore = max(-20.0, min(20.0, arima_chg * 4.0))
        score += arima_subscore

        # 5. ML Ensemble Forecast (Weight: 20%)
        ml_prob = ml_results.get("prob_bullish", 50.0)
        ml_subscore = (ml_prob - 50.0) * 0.40
        score += ml_subscore

        score = max(-100.0, min(100.0, score))

        if score >= 45.0:
            action = "STRONG BUY"
            color = "#00e676"
        elif score >= 15.0:
            action = "BUY"
            color = "#4ade80"
        elif score <= -45.0:
            action = "STRONG SELL"
            color = "#ff1744"
        elif score <= -15.0:
            action = "SELL"
            color = "#f87171"
        else:
            action = "NEUTRAL"
            color = "#ffb300"

        confidence_pct = min(100.0, abs(score) + 20.0)

        atr = latest.get("ATR", close * 0.02)
        if "BUY" in action:
            stop_loss = round(close - (1.75 * atr), 2)
            take_profit = round(close + (3.0 * atr), 2)
        elif "SELL" in action:
            stop_loss = round(close + (1.75 * atr), 2)
            take_profit = round(close - (3.0 * atr), 2)
        else:
            stop_loss = round(close - (2.0 * atr), 2)
            take_profit = round(close + (2.0 * atr), 2)

        risk_amount = abs(close - stop_loss)
        reward_amount = abs(take_profit - close)
        rr_ratio = round(reward_amount / (risk_amount + 1e-6), 2)

        return {
            "action": action,
            "score": round(score, 1),
            "color": color,
            "confidence_pct": round(confidence_pct, 1),
            "current_price": round(close, 2),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": rr_ratio,
            "trend_subscore": round(trend_subscore, 1),
            "momentum_subscore": round(momentum_subscore, 1),
            "reversion_subscore": round(reversion_subscore, 1),
            "arima_subscore": round(arima_subscore, 1),
            "ml_subscore": round(ml_subscore, 1),
        }
