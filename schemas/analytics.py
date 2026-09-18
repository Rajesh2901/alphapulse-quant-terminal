"""
AlphaPulse Domain Schemas - Analytics & Predictive Inference
Strongly-typed Pydantic domain models for forecasting, risk attribution, and signals.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ARIMAForecastOutput(BaseModel):
    """Econometric time-series forecast results."""

    model_config = {"arbitrary_types_allowed": True}

    pred_series: Any = Field(..., description="Projected price series (pd.Series)")
    lower_95: Any = Field(..., description="Lower 95% confidence boundary (pd.Series)")
    upper_95: Any = Field(..., description="Upper 95% confidence boundary (pd.Series)")
    lower_80: Any = Field(..., description="Lower 80% confidence boundary (pd.Series)")
    upper_80: Any = Field(..., description="Upper 80% confidence boundary (pd.Series)")
    expected_change_pct: float = Field(..., description="Expected percentage drift over horizon")
    aic: float = Field(..., description="Akaike Information Criterion")
    bic: float = Field(..., description="Bayesian Information Criterion")
    adf_pvalue: float = Field(..., description="Augmented Dickey-Fuller stationarity test p-value")
    is_stationary: bool = Field(
        ..., description="True if differenced series is stationary (p < 0.05)"
    )
    ljung_box_pvalue: float = Field(..., description="Ljung-Box residual white-noise test p-value")


class FeatureImportanceItem(BaseModel):
    """Feature ranking item from machine learning model."""

    feature: str
    importance: float


class MLForecastOutput(BaseModel):
    """Ensemble machine learning forecast results."""

    model_config = {"arbitrary_types_allowed": True}

    pred_series: Any = Field(..., description="Multi-step reconstructed price path (pd.Series)")
    prob_bullish: float = Field(
        ..., ge=0, le=100, description="Calibrated probability of upward movement (%)"
    )
    prob_bearish: float = Field(
        ..., ge=0, le=100, description="Calibrated probability of downward movement (%)"
    )
    rmse: float = Field(..., ge=0, description="Out-of-sample Root Mean Squared Error")
    mae: float = Field(..., ge=0, description="Out-of-sample Mean Absolute Error")
    directional_accuracy: float = Field(
        ..., ge=0, le=100, description="Out-of-sample directional accuracy (%)"
    )
    feature_importance: Any = Field(..., description="Feature importance table (pd.DataFrame)")
    expected_change_pct: float = Field(
        ..., description="Expected cumulative return over prediction horizon"
    )


class ConsensusSignalOutput(BaseModel):
    """Multi-factor quantitative consensus signal."""

    action: Literal["STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL"]
    score: float = Field(
        ..., ge=-100.0, le=100.0, description="Composite quantitative score (-100 to +100)"
    )
    color: str = Field(..., description="UI hex color code corresponding to action")
    confidence_pct: float = Field(..., ge=0, le=100, description="Model conviction percentage")
    current_price: float = Field(..., gt=0)
    stop_loss: float = Field(..., gt=0, description="ATR-anchored risk boundary")
    take_profit: float = Field(..., gt=0, description="Target profit boundary")
    risk_reward_ratio: float = Field(..., ge=0, description="Reward-to-risk ratio")
    trend_subscore: float
    momentum_subscore: float
    reversion_subscore: float
    arima_subscore: float
    ml_subscore: float


class RiskAuditReport(BaseModel):
    """Comprehensive portfolio and risk attribution audit."""

    model_config = {"arbitrary_types_allowed": True}

    ann_return_pct: float
    ann_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    hist_var_95_pct: float
    hist_var_99_pct: float
    param_var_95_pct: float
    param_var_99_pct: float
    cf_var_95_pct: float
    cf_var_99_pct: float
    cvar_95_pct: float
    cvar_99_pct: float
    skewness: float
    kurtosis: float
    win_rate_pct: float
    profit_factor: float
    drawdown_series: Any = Field(..., description="Underwater drawdown time series (pd.Series)")


class FullAnalysisResult(BaseModel):
    """Unified payload returned by the QuantService orchestration layer."""

    model_config = {"arbitrary_types_allowed": True}

    symbol: str
    quote: Any
    order_book: Any
    features_df: Any
    arima: ARIMAForecastOutput
    ml: MLForecastOutput
    signal: ConsensusSignalOutput
    risk: RiskAuditReport
    execution_time_ms: float
