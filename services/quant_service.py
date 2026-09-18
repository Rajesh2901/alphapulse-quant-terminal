"""
AlphaPulse Enterprise Service Layer - Quant Orchestrator
Decoupled domain service orchestrating data ingestion, feature generation,
and concurrent model inference across multi-threaded workers.
"""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from data_ingestion import MarketDataPipeline
from exceptions import (
    InsufficientDataError,
    InvalidSymbolError,
    ModelConvergenceError,
)
from features import FeatureEngine
from models import ARIMAForecaster, ConsensusSignalEngine, MLEnsembleForecaster
from risk_engine import QuantitativeRiskEngine
from schemas.analytics import (
    ARIMAForecastOutput,
    ConsensusSignalOutput,
    FullAnalysisResult,
    MLForecastOutput,
    RiskAuditReport,
)

logger = logging.getLogger("AlphaPulse.QuantService")


class QuantService:
    """
    Enterprise quantitative service implementing the Facade pattern.
    Manages end-to-end data preparation, parallel modeling, and signal aggregation.
    """

    SYMBOL_REGEX = re.compile(r"^[A-Za-z0-9\-\=\.\^]{1,15}$")

    def __init__(self, data_pipeline: MarketDataPipeline | None = None):
        self.pipeline = data_pipeline or MarketDataPipeline(cache_ttl_seconds=10)

    @classmethod
    def validate_symbol(cls, symbol: str) -> str:
        """Sanitizes and enforces strict character limits for financial tickers."""
        cleaned = symbol.strip().upper()
        if not cleaned or not cls.SYMBOL_REGEX.match(cleaned):
            raise InvalidSymbolError(
                f"Invalid symbol format: '{symbol}'",
                details="Ticker must be 1-15 alphanumeric characters (with standard separators -, =, ., ^)",
            )
        return cleaned

    def execute_analysis(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        forecast_horizon: int = 10,
        arima_order: tuple[int, int, int] = (2, 1, 2),
        n_lags: int = 5,
        n_estimators: int = 100,
    ) -> FullAnalysisResult:
        """
        Executes full quantitative analysis pipeline with parallel execution of models.
        """
        start_time = time.perf_counter()
        valid_symbol = self.validate_symbol(symbol)

        # 1. Ingest Data & Live Quote
        df_raw = self.pipeline.fetch_historical_data(valid_symbol, period=period, interval=interval)
        live_quote = self.pipeline.fetch_live_quote(valid_symbol)

        if df_raw.empty or len(df_raw) < 25:
            raise InsufficientDataError(
                f"Insufficient historical data for {valid_symbol}",
                details=f"Retrieved {len(df_raw)} bars, minimum 25 bars required.",
            )

        # 2. Compute Indicators
        df_features = FeatureEngine.compute_all_features(df_raw)
        recent_vol = float(df_features["Log_Return"].tail(20).std())
        order_book = self.pipeline.generate_order_book_depth(
            live_quote["price"], volatility=max(0.01, recent_vol)
        )

        # 3. Parallel Execution of Independent Quantitative Models
        # Concurrently runs ARIMA, ML Ensemble, and Risk Engine across thread pool
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_arima = executor.submit(
                self._run_arima, df_features["Close"], forecast_horizon, arima_order
            )
            future_ml = executor.submit(
                self._run_ml, df_raw, forecast_horizon, n_lags, n_estimators
            )
            future_risk = executor.submit(self._run_risk, df_features["Return"])

            arima_data = future_arima.result()
            ml_data = future_ml.result()
            risk_data = future_risk.result()

        # 4. Synthesize Consensus Signal
        raw_signal = ConsensusSignalEngine.generate_consensus_signal(
            df_features, arima_data, ml_data
        )

        # 5. Pack into Strongly-Typed Pydantic Domain Objects
        arima_output = ARIMAForecastOutput(
            pred_series=arima_data["pred_series"],
            lower_95=arima_data["lower_95"],
            upper_95=arima_data["upper_95"],
            lower_80=arima_data["lower_80"],
            upper_80=arima_data["upper_80"],
            expected_change_pct=arima_data["expected_change_pct"],
            aic=arima_data["aic"],
            bic=arima_data["bic"],
            adf_pvalue=arima_data["adf_pvalue"],
            is_stationary=arima_data["is_stationary"],
            ljung_box_pvalue=arima_data["ljung_box_pvalue"],
        )

        ml_output = MLForecastOutput(
            pred_series=ml_data["pred_series"],
            prob_bullish=ml_data["prob_bullish"],
            prob_bearish=ml_data["prob_bearish"],
            rmse=ml_data["rmse"],
            mae=ml_data["mae"],
            directional_accuracy=ml_data["directional_accuracy"],
            feature_importance=ml_data["feature_importance"],
            expected_change_pct=ml_data["expected_change_pct"],
        )

        signal_output = ConsensusSignalOutput(**raw_signal)
        risk_output = RiskAuditReport(**risk_data)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return FullAnalysisResult(
            symbol=valid_symbol,
            quote=live_quote,
            order_book=order_book,
            features_df=df_features,
            arima=arima_output,
            ml=ml_output,
            signal=signal_output,
            risk=risk_output,
            execution_time_ms=round(elapsed_ms, 1),
        )

    def _run_arima(self, series: pd.Series, horizon: int, order: tuple[int, int, int]) -> dict:
        try:
            return ARIMAForecaster.fit_and_forecast(series, horizon=horizon, order=order)
        except Exception as e:
            logger.error(f"ARIMA optimization failure: {e}")
            raise ModelConvergenceError("ARIMA model estimation failed", details=str(e)) from e

    def _run_ml(self, df: pd.DataFrame, horizon: int, n_lags: int, n_estimators: int) -> dict:
        try:
            forecaster = MLEnsembleForecaster(n_estimators=n_estimators)
            return forecaster.train_and_predict(df, horizon=horizon, n_lags=n_lags)
        except Exception as e:
            logger.error(f"ML ensemble inference failure: {e}")
            raise ModelConvergenceError(
                "Machine learning ensemble execution failed", details=str(e)
            ) from e

    def _run_risk(self, returns: pd.Series) -> dict:
        return QuantitativeRiskEngine.calculate_risk_metrics(returns)
