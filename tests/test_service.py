"""
Integration Tests for QuantService
Validates input validation, error handling, parallel execution, and Pydantic outputs.
"""

import pytest

from exceptions import InvalidSymbolError
from schemas.analytics import FullAnalysisResult
from services.quant_service import QuantService


def test_symbol_validation():
    assert QuantService.validate_symbol("nvda") == "NVDA"
    assert QuantService.validate_symbol("BTC-USD") == "BTC-USD"
    assert QuantService.validate_symbol("EURUSD=X") == "EURUSD=X"

    # Malformed / Injection attempts
    with pytest.raises(InvalidSymbolError):
        QuantService.validate_symbol("")

    with pytest.raises(InvalidSymbolError):
        QuantService.validate_symbol("DROP TABLE;--")

    with pytest.raises(InvalidSymbolError):
        QuantService.validate_symbol("A" * 20)


def test_service_execution(mock_ohlcv_data):
    class MockPipeline:
        def fetch_historical_data(self, symbol, period, interval):
            return mock_ohlcv_data

        def fetch_live_quote(self, symbol):
            return {
                "symbol": symbol,
                "price": 105.50,
                "prev_close": 104.00,
                "change": 1.50,
                "pct_change": 1.44,
                "timestamp": 1726000000.0,
                "status": "SIMULATED",
            }

        def generate_order_book_depth(self, price, volatility):
            return {
                "spread": 0.05,
                "spread_bps": 4.8,
                "imbalance": 0.12,
                "best_bid": 105.47,
                "best_ask": 105.52,
                "bids": [],
                "asks": [],
            }

    service = QuantService(data_pipeline=MockPipeline())
    result = service.execute_analysis(
        symbol="SPY",
        period="1y",
        interval="1d",
        forecast_horizon=5,
        arima_order=(1, 1, 1),
        n_lags=3,
        n_estimators=25,
    )

    assert isinstance(result, FullAnalysisResult)
    assert result.symbol == "SPY"
    assert result.execution_time_ms > 0
    assert result.signal.action in ["STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL"]
    assert result.risk.sharpe_ratio is not None
    assert len(result.arima.pred_series) == 5
    assert len(result.ml.pred_series) == 5
