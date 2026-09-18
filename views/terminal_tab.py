"""
AlphaPulse Presentation Views - Live Terminal Tab
Renders multi-factor candlestick charts, predictive confidence fans, signal cards, and order books.
"""

import streamlit as st

from schemas.analytics import FullAnalysisResult
from ui_components import (
    create_candlestick_chart,
    create_forecast_cone_chart,
    create_order_book_chart,
    create_oscillators_chart,
)


def render_terminal_workspace(analysis: FullAnalysisResult):
    """Renders Tab 1: Live Terminal & Predictive Analytics."""
    col_chart, col_signals = st.columns([0.68, 0.32])

    with col_chart:
        st.markdown("#### 📊 Multi-Factor Price Action & Volume Profile")
        fig_candle = create_candlestick_chart(analysis.features_df, analysis.symbol)
        st.plotly_chart(fig_candle, use_container_width=True, config={"displayModeBar": False})

        st.markdown("#### 🔮 Predictive Price Fan & Analytical Confidence Cones")
        # Adapt Pydantic models to chart inputs
        arima_dict = {
            "pred_series": analysis.arima.pred_series,
            "upper_95": analysis.arima.upper_95,
            "lower_95": analysis.arima.lower_95,
            "upper_80": analysis.arima.upper_80,
            "lower_80": analysis.arima.lower_80,
            "expected_change_pct": analysis.arima.expected_change_pct,
        }
        ml_dict = {
            "pred_series": analysis.ml.pred_series,
            "expected_change_pct": analysis.ml.expected_change_pct,
        }
        fig_forecast = create_forecast_cone_chart(
            analysis.features_df, arima_dict, ml_dict, analysis.symbol
        )
        st.plotly_chart(fig_forecast, use_container_width=True, config={"displayModeBar": False})

    with col_signals:
        signal = analysis.signal
        st.markdown("#### ⚡ Algorithmic Consensus Signal")
        st.markdown(
            f"""
            <div class="signal-box">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="signal-badge" style="background:{signal.color}22; color:{signal.color}; border:1px solid {signal.color};">
                        {signal.action}
                    </span>
                    <span style="font-family:monospace; font-size:1.1rem; color:#f8fafc;">
                        Score: <strong>{signal.score:+.1f}</strong>
                    </span>
                </div>
                <div style="margin-top:14px; display:grid; grid-template-columns: 1fr 1fr; gap:10px; font-family:monospace; font-size:0.85rem;">
                    <div>Stop-Loss: <strong style="color:#ff1744;">${signal.stop_loss:.2f}</strong></div>
                    <div>Take-Profit: <strong style="color:#00e676;">${signal.take_profit:.2f}</strong></div>
                    <div>R:R Ratio: <strong style="color:#00f2fe;">{signal.risk_reward_ratio}:1</strong></div>
                    <div>Confidence: <strong style="color:#f8fafc;">{signal.confidence_pct:.0f}%</strong></div>
                </div>
                <hr style="border-color:#1c2738; margin:12px 0;">
                <div style="font-size:0.75rem; color:#8fa0b5; font-family:monospace;">
                    FACTOR ATTRIBUTION BREAKDOWN:<br>
                    • Trend Following: {signal.trend_subscore:+.1f}<br>
                    • Momentum (RSI/MACD): {signal.momentum_subscore:+.1f}<br>
                    • Mean Reversion: {signal.reversion_subscore:+.1f}<br>
                    • ARIMA Projected Drift: {signal.arima_subscore:+.1f}<br>
                    • ML Directional Bias: {signal.ml_subscore:+.1f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 🪜 Order Flow & Market Depth")
        fig_ob = create_order_book_chart(analysis.order_book)
        st.plotly_chart(fig_ob, use_container_width=True, config={"displayModeBar": False})

        st.markdown("#### 🌊 Momentum Oscillators")
        fig_osc = create_oscillators_chart(analysis.features_df)
        st.plotly_chart(fig_osc, use_container_width=True, config={"displayModeBar": False})
