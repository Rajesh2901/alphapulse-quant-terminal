"""
AlphaPulse Quantitative Terminal - Quantitative Risk & Portfolio Engine
Value-at-Risk (VaR), Expected Shortfall (CVaR), Cornish-Fisher expansion,
drawdown analytics, and risk-adjusted performance attribution.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew


class QuantitativeRiskEngine:
    """
    Institutional risk analytics engine computing parametric, non-parametric,
    and extreme-value risk metrics across portfolio returns.
    """

    @staticmethod
    def calculate_risk_metrics(
        returns_series: pd.Series, risk_free_rate: float = 0.045, trading_days: int = 252
    ) -> dict[str, Any]:
        r = returns_series.dropna().values
        if len(r) < 10:
            raise ValueError("Insufficient return observations for risk calculation.")

        # 1. Statistical Moments
        mean_daily = float(np.mean(r))
        std_daily = float(np.std(r, ddof=1))
        skew_val = float(skew(r))
        kurt_val = float(kurtosis(r))

        # 2. Annualized Performance Metrics
        ann_return = float((1.0 + mean_daily) ** trading_days - 1.0)
        ann_volatility = float(std_daily * np.sqrt(trading_days))

        excess_return = ann_return - risk_free_rate
        sharpe_ratio = float(excess_return / (ann_volatility + 1e-6))

        downside_returns = r[r < 0]
        downside_std_daily = (
            float(np.std(downside_returns, ddof=1)) if len(downside_returns) > 1 else std_daily
        )
        ann_downside_vol = float(downside_std_daily * np.sqrt(trading_days))
        sortino_ratio = float(excess_return / (ann_downside_vol + 1e-6))

        # 3. Maximum Drawdown (MDD)
        cum_wealth = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(cum_wealth)
        drawdown_series = (cum_wealth - peak) / peak
        max_drawdown = float(np.min(drawdown_series))
        calmar_ratio = float(ann_return / (abs(max_drawdown) + 1e-6))

        # 4. Value-at-Risk (VaR) Analytics (Daily)
        hist_var_95 = float(-np.percentile(r, 5))
        hist_var_99 = float(-np.percentile(r, 1))

        z_95 = norm.ppf(0.95)
        z_99 = norm.ppf(0.99)
        param_var_95 = float(-(mean_daily - z_95 * std_daily))
        param_var_99 = float(-(mean_daily - z_99 * std_daily))

        def cornish_fisher_z(z_crit: float, s: float, k: float) -> float:
            return (
                z_crit
                + (z_crit**2 - 1.0) * s / 6.0
                + (z_crit**3 - 3.0 * z_crit) * k / 24.0
                - (2.0 * z_crit**3 - 5.0 * z_crit) * (s**2) / 36.0
            )

        z_cf_95 = cornish_fisher_z(z_95, skew_val, kurt_val)
        z_cf_99 = cornish_fisher_z(z_99, skew_val, kurt_val)
        cf_var_95 = float(-(mean_daily - z_cf_95 * std_daily))
        cf_var_99 = float(-(mean_daily - z_cf_99 * std_daily))

        # 5. Conditional Value at Risk (CVaR / Expected Shortfall)
        tail_losses_95 = r[r <= -hist_var_95]
        cvar_95 = float(-np.mean(tail_losses_95)) if len(tail_losses_95) > 0 else hist_var_95

        tail_losses_99 = r[r <= -hist_var_99]
        cvar_99 = float(-np.mean(tail_losses_99)) if len(tail_losses_99) > 0 else hist_var_99

        win_rate = float(np.mean(r > 0) * 100.0)
        gains = r[r > 0]
        losses = np.abs(r[r < 0])
        profit_factor = float(np.sum(gains) / (np.sum(losses) + 1e-6))

        dd_series = pd.Series(drawdown_series, index=returns_series.dropna().index)

        return {
            "ann_return_pct": round(ann_return * 100.0, 2),
            "ann_volatility_pct": round(ann_volatility * 100.0, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "max_drawdown_pct": round(max_drawdown * 100.0, 2),
            "calmar_ratio": round(calmar_ratio, 2),
            "hist_var_95_pct": round(hist_var_95 * 100.0, 2),
            "hist_var_99_pct": round(hist_var_99 * 100.0, 2),
            "param_var_95_pct": round(param_var_95 * 100.0, 2),
            "param_var_99_pct": round(param_var_99 * 100.0, 2),
            "cf_var_95_pct": round(cf_var_95 * 100.0, 2),
            "cf_var_99_pct": round(cf_var_99 * 100.0, 2),
            "cvar_95_pct": round(cvar_95 * 100.0, 2),
            "cvar_99_pct": round(cvar_99 * 100.0, 2),
            "skewness": round(skew_val, 2),
            "kurtosis": round(kurt_val, 2),
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "drawdown_series": dd_series,
        }

    @staticmethod
    def compute_correlation_matrix(asset_returns_dict: dict[str, pd.Series]) -> pd.DataFrame:
        combined_df = pd.DataFrame(asset_returns_dict).dropna()
        return combined_df.corr().round(2)
