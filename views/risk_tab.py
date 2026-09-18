"""
AlphaPulse Presentation Views - Quantitative Risk Tab
Renders parametric/empirical VaR, Expected Shortfall, drawdowns, and comprehensive audits.
"""

import pandas as pd
import streamlit as st

from schemas.analytics import FullAnalysisResult
from ui_components import create_drawdown_chart, create_risk_distribution_chart


def render_risk_workspace(analysis: FullAnalysisResult):
    """Renders Tab 2: Quantitative Risk & Portfolio."""
    risk = analysis.risk
    st.markdown("### 🛡️ Institutional Risk & Tail Analysis")

    rk1, rk2, rk3, rk4, rk5 = st.columns(5)
    rk1.metric("Annualized Return", f"{risk.ann_return_pct:.2f}%")
    rk2.metric("Sortino Ratio", f"{risk.sortino_ratio:.2f}")
    rk3.metric("Calmar Ratio", f"{risk.calmar_ratio:.2f}")
    rk4.metric("Cornish-Fisher VaR (95%)", f"-{risk.cf_var_95_pct:.2f}%")
    rk5.metric("Max Peak-to-Trough DD", f"{risk.max_drawdown_pct:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    col_dist, col_dd = st.columns(2)

    risk_dict = {
        "hist_var_95_pct": risk.hist_var_95_pct,
        "cvar_95_pct": risk.cvar_95_pct,
    }

    with col_dist:
        fig_dist = create_risk_distribution_chart(analysis.features_df["Return"], risk_dict)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_dd:
        fig_dd = create_drawdown_chart(risk.drawdown_series)
        st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("#### 📋 Comprehensive Risk Audit")
    audit_data = {
        "Risk Metric": [
            "Annualized Realized Volatility",
            "Annualized Sharpe Ratio (Rf=4.5%)",
            "Downside Sortino Ratio",
            "Maximum Historical Drawdown",
            "Calmar Quotient (Ann. Return / MDD)",
            "Historical VaR (95% confidence, 1-day)",
            "Historical VaR (99% confidence, 1-day)",
            "Cornish-Fisher Adjusted VaR (95%)",
            "Expected Shortfall / CVaR (95%)",
            "Distribution Skewness",
            "Distribution Excess Kurtosis (Fat-Tail Indicator)",
            "Historical Trade Win Rate",
        ],
        "Value": [
            f"{risk.ann_volatility_pct:.2f}%",
            f"{risk.sharpe_ratio:.2f}",
            f"{risk.sortino_ratio:.2f}",
            f"{risk.max_drawdown_pct:.2f}%",
            f"{risk.calmar_ratio:.2f}",
            f"-{risk.hist_var_95_pct:.2f}%",
            f"-{risk.hist_var_99_pct:.2f}%",
            f"-{risk.cf_var_95_pct:.2f}%",
            f"-{risk.cvar_95_pct:.2f}%",
            f"{risk.skewness:.2f}",
            f"{risk.kurtosis:.2f}",
            f"{risk.win_rate_pct:.1f}%",
        ],
        "Institutional Benchmark / Guidance": [
            "Equities baseline 15-25%",
            "> 1.0 is acceptable; > 2.0 is elite",
            "> 1.5 indicates asymmetric upside capture",
            "Acceptable threshold depends on risk mandate (<20% target)",
            "> 1.0 implies returns outpace peak drawdowns",
            "Daily capital at risk with 95% certainty",
            "Extreme daily shock boundary",
            "Higher than normal VaR if distribution has fat left tails",
            "Expected tail loss during structural market failure",
            "Negative skewness implies sharp sudden crashes",
            "> 3.0 indicates significant tail risk leptokurtosis",
            "Percentage of positive closing sessions",
        ],
    }
    st.table(pd.DataFrame(audit_data))
