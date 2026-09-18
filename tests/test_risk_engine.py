"""
Unit Tests for QuantitativeRiskEngine
Validates Value-at-Risk, Cornish-Fisher expansion, and drawdown attribution.
"""

import pandas as pd

from risk_engine import QuantitativeRiskEngine


def test_calculate_risk_metrics(mock_returns: pd.Series):
    metrics = QuantitativeRiskEngine.calculate_risk_metrics(mock_returns, risk_free_rate=0.045)

    assert "ann_return_pct" in metrics
    assert "ann_volatility_pct" in metrics
    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert "hist_var_95_pct" in metrics
    assert "cf_var_95_pct" in metrics
    assert "cvar_95_pct" in metrics
    assert "max_drawdown_pct" in metrics
    assert "drawdown_series" in metrics

    # Volatility and VaR must be positive quantities in percentage
    assert metrics["ann_volatility_pct"] > 0
    assert metrics["hist_var_95_pct"] > 0
    assert metrics["cvar_95_pct"] >= metrics["hist_var_95_pct"]  # Expected shortfall >= VaR
    assert metrics["max_drawdown_pct"] <= 0  # Drawdowns are <= 0%
    assert 0.0 <= metrics["win_rate_pct"] <= 100.0
