"""
AlphaPulse Presentation Views - Telemetry Header & KPI Cards
Renders the top navigation bar, feed status, microstructural metrics, and executive KPIs.
"""

import streamlit as st

from schemas.analytics import FullAnalysisResult


def render_telemetry_header(analysis: FullAnalysisResult):
    """Renders the top telemetry navigation ribbon with latency and status."""
    quote = analysis.quote
    order_book = analysis.order_book

    st.markdown(
        f"""
        <div class="quant-header">
            <div style="display:flex; align-items:center; flex-wrap:wrap; gap:12px;">
                <div class="quant-title">⚡ ALPHAPULSE</div>
                <div class="quant-badge-group">
                    <span class="quant-tag">ENTERPRISE DESK</span>
                    <span class="quant-tag" style="background:rgba(0,230,118,0.1); color:#00e676; border-color:rgba(0,230,118,0.3);">FEED: {quote.get("status", "LIVE")}</span>
                    <span class="quant-tag" style="background:rgba(59,130,246,0.1); color:#38bdf8; border-color:rgba(59,130,246,0.3);">AI: GEMINI ONLINE</span>
                </div>
            </div>
            <div style="display:flex; align-items:center; flex-wrap:wrap; gap:18px; font-family:'JetBrains Mono', monospace; font-size:0.82rem;">
                <span style="color:#8fa0b5;">TICKER: <strong style="color:#f8fafc;">{analysis.symbol}</strong></span>
                <span style="color:#8fa0b5;">SPREAD: <strong style="color:#00f2fe;">{order_book.get("spread_bps", 0.0):.1f} bps</strong></span>
                <span style="color:#8fa0b5;">LATENCY: <strong style="color:#00e676;">{analysis.execution_time_ms:.0f} ms</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(analysis: FullAnalysisResult, forecast_horizon: int):
    """Renders the 5 primary executive telemetry metric cards."""
    quote = analysis.quote
    signal = analysis.signal
    arima = analysis.arima
    ml = analysis.ml
    risk = analysis.risk

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        chg_color = "text-bullish" if quote["pct_change"] >= 0 else "text-bearish"
        sign = "+" if quote["pct_change"] >= 0 else ""
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">{analysis.symbol} Last Price</div>
                <div class="metric-value">${quote["price"]:.2f}</div>
                <div class="metric-sub {chg_color}">{sign}{quote["change"]:.2f} ({sign}{quote["pct_change"]:.2f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Consensus Rating</div>
                <div class="metric-value" style="color:{signal.color};">{signal.action}</div>
                <div class="metric-sub text-cyan">Score: {signal.score:+.1f} | Conf: {signal.confidence_pct:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Predictive Drift ({forecast_horizon}b)</div>
                <div class="metric-value text-cyan">{arima.expected_change_pct:+.2f}%</div>
                <div class="metric-sub" style="color:#8fa0b5;">ML Prob: {ml.prob_bullish:.0f}% Bullish</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Annualized Volatility</div>
                <div class="metric-value">{risk.ann_volatility_pct:.1f}%</div>
                <div class="metric-sub text-neutral">Sharpe (Rf=4.5%): {risk.sharpe_ratio:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c5:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Daily VaR (95%)</div>
                <div class="metric-value text-bearish">-{risk.hist_var_95_pct:.2f}%</div>
                <div class="metric-sub" style="color:#8fa0b5;">CVaR 95%: -{risk.cvar_95_pct:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
