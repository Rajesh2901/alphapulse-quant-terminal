"""
AlphaPulse Presentation Views - AI Intelligence Desk Tab
Renders Google Gemini institutional research briefs and scenario probability matrix.
"""

import streamlit as st

from ai_analyst import AIAssistantAnalyst
from schemas.analytics import FullAnalysisResult


def render_ai_workspace(
    analysis: FullAnalysisResult,
    ai_analyst: AIAssistantAnalyst,
    asset_name: str,
):
    """Renders Tab 3: Google Gemini Systematic Research Desk."""
    st.markdown("### 🧠 Google Gemini Systematic Research Desk")
    st.caption(
        "Autonomous synthesis of quantitative telemetries, predictive divergence, and tail risk regimes"
    )

    ai_col1, ai_col2 = st.columns([0.7, 0.3])
    with ai_col2:
        re_gen_btn = st.button("⚡ Re-Generate Institutional Memo", use_container_width=True)

    cache_key = f"ai_memo_{analysis.symbol}"
    if cache_key not in st.session_state or re_gen_btn:
        with st.spinner("Generating institutional memo via Google Gemini (gemini-3.6-flash)..."):
            # Map Pydantic structures to analyst expectations
            memo_text = ai_analyst.generate_institutional_brief(
                symbol=analysis.symbol,
                asset_name=asset_name,
                latest_price=analysis.quote["price"],
                price_change_pct=analysis.quote["pct_change"],
                technical_data={
                    "RSI": float(analysis.features_df["RSI"].iloc[-1]),
                    "MACD": float(analysis.features_df["MACD"].iloc[-1]),
                },
                signal_data=analysis.signal.model_dump(),
                arima_data={
                    "expected_change_pct": analysis.arima.expected_change_pct,
                    "aic": analysis.arima.aic,
                    "adf_pvalue": analysis.arima.adf_pvalue,
                },
                ml_data=analysis.ml.model_dump(),
                risk_data=analysis.risk.model_dump(),
                order_book_data=analysis.order_book,
            )
            st.session_state[cache_key] = memo_text
    else:
        memo_text = st.session_state[cache_key]

    st.markdown(
        f"""
        <div style="background:#0c121e; border:1px solid #1c2738; border-radius:8px; padding:22px; line-height:1.6;">
            {memo_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🎯 Scenario Probability Matrix")
    sc_cols = st.columns(3)
    signal = analysis.signal
    ml = analysis.ml
    df_features = analysis.features_df

    with sc_cols[0]:
        st.markdown(
            f"""
            <div style="background:#0b1915; border:1px solid #00e67644; border-radius:8px; padding:16px;">
                <div style="color:#00e676; font-weight:700;">🟢 BULL CASE ({ml.prob_bullish:.0f}%)</div>
                <div style="font-size:0.85rem; margin-top:8px; color:#cbd5e1;">
                    Catalyst: Sustained breakout beyond ${signal.take_profit:.2f}. Volatility compression enables aggressive momentum expansion.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sc_cols[1]:
        st.markdown(
            f"""
            <div style="background:#1c1912; border:1px solid #ffb30044; border-radius:8px; padding:16px;">
                <div style="color:#ffb300; font-weight:700;">🟡 BASE CASE (Neutral Drift)</div>
                <div style="font-size:0.85rem; margin-top:8px; color:#cbd5e1;">
                    Mean-reverting consolidation within Bollinger range [${df_features["BB_Lower"].iloc[-1]:.2f} - ${df_features["BB_Upper"].iloc[-1]:.2f}].
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sc_cols[2]:
        st.markdown(
            f"""
            <div style="background:#1f1013; border:1px solid #ff174444; border-radius:8px; padding:16px;">
                <div style="color:#ff1744; font-weight:700;">🔴 BEAR CASE ({ml.prob_bearish:.0f}%)</div>
                <div style="font-size:0.85rem; margin-top:8px; color:#cbd5e1;">
                    Breach of ${signal.stop_loss:.2f} stop boundary triggering stop cascade and liquidity vacuum.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
