"""
Unit Tests for Predictive Modeling Engines
Validates ARIMA forecasting, ML ensemble projections, and Consensus signal generator.
"""

import pandas as pd

from features import FeatureEngine
from models import ARIMAForecaster, ConsensusSignalEngine, MLEnsembleForecaster


def test_arima_forecaster(mock_ohlcv_data: pd.DataFrame):
    horizon = 8
    order = (1, 1, 1)
    res = ARIMAForecaster.fit_and_forecast(mock_ohlcv_data["Close"], horizon=horizon, order=order)

    assert "pred_series" in res
    assert len(res["pred_series"]) == horizon
    assert len(res["lower_95"]) == horizon
    assert len(res["upper_95"]) == horizon
    assert (res["upper_95"] >= res["lower_95"]).all()
    assert isinstance(res["aic"], float)
    assert 0.0 <= res["adf_pvalue"] <= 1.0


def test_ml_ensemble_forecaster(mock_ohlcv_data: pd.DataFrame):
    horizon = 6
    forecaster = MLEnsembleForecaster(n_estimators=30, max_depth=3)
    res = forecaster.train_and_predict(mock_ohlcv_data, horizon=horizon, n_lags=3)

    assert "pred_series" in res
    assert len(res["pred_series"]) == horizon
    assert 0.0 <= res["prob_bullish"] <= 100.0
    assert 0.0 <= res["prob_bearish"] <= 100.0
    assert round(res["prob_bullish"] + res["prob_bearish"], 1) == 100.0
    assert res["rmse"] >= 0.0
    assert not res["feature_importance"].empty


def test_consensus_signal_engine(mock_ohlcv_data: pd.DataFrame):
    df_feat = FeatureEngine.compute_all_features(mock_ohlcv_data)
    arima_res = ARIMAForecaster.fit_and_forecast(df_feat["Close"], horizon=5, order=(1, 1, 1))
    ml_res = MLEnsembleForecaster(n_estimators=20, max_depth=3).train_and_predict(
        mock_ohlcv_data, horizon=5, n_lags=3
    )

    signal = ConsensusSignalEngine.generate_consensus_signal(df_feat, arima_res, ml_res)
    assert signal["action"] in ["STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL"]
    assert -100.0 <= signal["score"] <= 100.0
    assert signal["confidence_pct"] >= 0.0
    assert signal["stop_loss"] > 0
    assert signal["take_profit"] > 0
    assert signal["risk_reward_ratio"] > 0
