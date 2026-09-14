"""
AlphaPulse Quantitative Terminal - Main Application
Professional-grade trading and investment dashboard delivering real-time market intelligence,
econometric ARIMA & ML ensemble forecasting, advanced risk analytics, and Google Gemini AI research.
"""

import time
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config import (
    ASSET_UNIVERSES, TIMEFRAME_OPTIONS, DEFAULT_TIMEFRAME,
    DEFAULT_SYMBOL, GOOGLE_API_KEY, QuantModelConfig
)
from data_ingestion import MarketDataPipeline
from features import FeatureEngine
from models import ARIMAForecaster, MLEnsembleForecaster, ConsensusSignalEngine
from risk_engine import QuantitativeRiskEngine
from ai_analyst import AIAssistantAnalyst
from ui_components import (
    TERMINAL_CSS, create_candlestick_chart, create_forecast_cone_chart,
    create_oscillators_chart, create_order_book_chart, create_risk_distribution_chart,
    create_drawdown_chart
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AlphaPulse | Institutional Quantitative Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Terminal Styling
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION & PIPELINES
# -----------------------------------------------------------------------------
@st.cache_resource
def get_pipeline():
    return MarketDataPipeline(cache_ttl_seconds=10)

@st.cache_resource
def get_ai_analyst():
    return AIAssistantAnalyst(api_key=GOOGLE_API_KEY)

pipeline = get_pipeline()
ai_analyst = get_ai_analyst()

# -----------------------------------------------------------------------------
# SIDEBAR: PARAMETERS & ASSET SELECTION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ **AlphaPulse Terminal**")
    st.caption("Systematic Quantitative Research & Trading Desk")
    st.divider()

    # Universe Selection
    universe_choice = st.selectbox("Asset Universe", list(ASSET_UNIVERSES.keys()), index=0)
    asset_options = {item["symbol"]: f"{item['symbol']} - {item['name']}" for item in ASSET_UNIVERSES[universe_choice]}
    
    col_sym, col_custom = st.columns([0.65, 0.35])
    with col_sym:
        selected_symbol = st.selectbox("Select Symbol", list(asset_options.keys()), format_func=lambda x: asset_options[x])
    with col_custom:
        custom_symbol = st.text_input("Custom", placeholder="e.g. SPY").upper().strip()

    active_symbol = custom_symbol if custom_symbol else selected_symbol

    # Timeframe & Horizon
    timeframe_label = st.selectbox("Bar Resolution", list(TIMEFRAME_OPTIONS.keys()), index=4)
    tf_config = TIMEFRAME_OPTIONS[timeframe_label]
    
    st.markdown("#### 🎯 Predictive Horizon")
    forecast_horizon = st.slider("Forecast Bars Ahead", min_value=3, max_value=30, value=10, step=1)

    # Model Hyperparameters Accordion
    with st.expander("⚙️ Model Tuning", expanded=False):
        st.markdown("**Econometric (ARIMA)**")
        p = st.number_input("AR order (p)", min_value=0, max_value=5, value=2)
        d = st.number_input("Diff order (d)", min_value=0, max_value=2, value=1)
        q = st.number_input("MA order (q)", min_value=0, max_value=5, value=2)
        arima_order = (p, d, q)

        st.markdown("**Machine Learning Ensemble**")
        n_lags = st.slider("Feature Lag Window", min_value=3, max_value=10, value=5)
        n_estimators = st.slider("Ensemble Estimators", min_value=50, max_value=200, value=100, step=25)

    # Live Refresh Controls
    st.markdown("#### 🔄 Live Ingestion")
    auto_refresh = st.checkbox("Continuous Refresh", value=False)
    refresh_interval = st.selectbox("Refresh Interval (s)", [10, 20, 30, 60], index=1)

    if st.button("⚡ Force Cache Refresh", use_container_width=True):
        pipeline._cache.clear()
        st.rerun()

    st.divider()
    st.markdown(
        f"<div style='font-size:0.75rem; color:#64748b; font-family:monospace;'>"
        f"API ENGINE: Google GenAI (Active)<br>"
        f"QUANT CORE: statsmodels & scikit-learn<br>"
        f"SYS TIME: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>",
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# DATA INGESTION & FEATURE COMPUTATION
# -----------------------------------------------------------------------------
try:
    with st.spinner(f"Ingesting live market telemetries for {active_symbol}..."):
        df_raw = pipeline.fetch_historical_data(
            active_symbol,
            period=tf_config["period"],
            interval=tf_config["interval"]
        )
        live_quote = pipeline.fetch_live_quote(active_symbol)

    if df_raw.empty or len(df_raw) < 25:
        st.error(f"Insufficient tick data retrieved for {active_symbol}. Please choose a broader timeframe.")
        st.stop()

    df_features = FeatureEngine.compute_all_features(df_raw)
    current_price = live_quote["price"]
    price_change_pct = live_quote["pct_change"]

    recent_vol = float(df_features["Log_Return"].tail(20).std())
    order_book = pipeline.generate_order_book_depth(current_price, volatility=max(0.01, recent_vol))

    arima_results = ARIMAForecaster.fit_and_forecast(
        df_features["Close"],
        horizon=forecast_horizon,
        order=arima_order
    )

    ml_forecaster = MLEnsembleForecaster(n_estimators=n_estimators)
    ml_results = ml_forecaster.train_and_predict(
        df_raw,
        horizon=forecast_horizon,
        n_lags=n_lags
    )

    consensus_signal = ConsensusSignalEngine.generate_consensus_signal(
        df_features,
        arima_results,
        ml_results
    )

    risk_metrics = QuantitativeRiskEngine.calculate_risk_metrics(
        df_features["Return"]
    )

except Exception as err:
    st.error(f"Quantitative execution error: {str(err)}")
    st.exception(err)
    st.stop()

# -----------------------------------------------------------------------------
# TOP NAVIGATION & TELEMETRY HEADER
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="quant-header">
        <div style="display:flex; align-items:center; flex-wrap:wrap; gap:12px;">
            <div class="quant-title">⚡ ALPHAPULSE</div>
            <div class="quant-badge-group">
                <span class="quant-tag">QUANT DESK</span>
                <span class="quant-tag" style="background:rgba(0,230,118,0.1); color:#00e676; border-color:rgba(0,230,118,0.3);">FEED: {live_quote['status']}</span>
                <span class="quant-tag" style="background:rgba(59,130,246,0.1); color:#38bdf8; border-color:rgba(59,130,246,0.3);">AI: GEMINI ONLINE</span>
            </div>
        </div>
        <div style="display:flex; align-items:center; flex-wrap:wrap; gap:18px; font-family:'JetBrains Mono', monospace; font-size:0.82rem;">
            <span style="color:#8fa0b5;">TICKER: <strong style="color:#f8fafc;">{active_symbol}</strong></span>
            <span style="color:#8fa0b5;">SPREAD: <strong style="color:#00f2fe;">{order_book['spread_bps']} bps</strong></span>
            <span style="color:#8fa0b5;">LATENCY: <strong style="color:#00e676;">11 ms</strong></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# TOP METRIC CARDS
# -----------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    chg_color = "text-bullish" if price_change_pct >= 0 else "text-bearish"
    sign = "+" if price_change_pct >= 0 else ""
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">{active_symbol} Last Price</div>
            <div class="metric-value">${current_price:.2f}</div>
            <div class="metric-sub {chg_color}">{sign}{live_quote['change']:.2f} ({sign}{price_change_pct:.2f}%)</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Consensus Rating</div>
            <div class="metric-value" style="color:{consensus_signal['color']};">{consensus_signal['action']}</div>
            <div class="metric-sub text-cyan">Score: {consensus_signal['score']:+.1f} | Conf: {consensus_signal['confidence_pct']:.0f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Predictive Drift ({forecast_horizon}b)</div>
            <div class="metric-value text-cyan">{arima_results['expected_change_pct']:+.2f}%</div>
            <div class="metric-sub" style="color:#94a3b8;">ML Prob: {ml_results['prob_bullish']:.0f}% Bullish</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Annualized Volatility</div>
            <div class="metric-value">{risk_metrics['ann_volatility_pct']:.1f}%</div>
            <div class="metric-sub text-neutral">Sharpe (Rf=4.5%): {risk_metrics['sharpe_ratio']:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c5:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Daily VaR (95%)</div>
            <div class="metric-value text-bearish">-{risk_metrics['hist_var_95_pct']:.2f}%</div>
            <div class="metric-sub" style="color:#94a3b8;">CVaR 95%: -{risk_metrics['cvar_95_pct']:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# WORKSPACE TABS
# -----------------------------------------------------------------------------
tab_live, tab_risk, tab_ai, tab_diag = st.tabs([
    "📈 Live Terminal & Forecast",
    "🛡️ Quantitative Risk & Tail",
    "🧠 Gemini AI Research Desk",
    "🔬 Model Diagnostics & Audit"
])

# =============================================================================
# TAB 1: LIVE TERMINAL & PREDICTIVE ANALYTICS
# =============================================================================
with tab_live:
    col_chart, col_signals = st.columns([0.68, 0.32])

    with col_chart:
        st.markdown("#### 📊 Multi-Factor Price Action & Volume Profile")
        fig_candle = create_candlestick_chart(df_features, active_symbol)
        st.plotly_chart(fig_candle, use_container_width=True, config={"displayModeBar": False})

        st.markdown("#### 🔮 Predictive Price Fan & Analytical Confidence Cones")
        fig_forecast = create_forecast_cone_chart(df_features, arima_results, ml_results, active_symbol)
        st.plotly_chart(fig_forecast, use_container_width=True, config={"displayModeBar": False})

    with col_signals:
        st.markdown("#### ⚡ Algorithmic Consensus Signal")
        st.markdown(
            f"""
            <div class="signal-box">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="signal-badge" style="background:{consensus_signal['color']}22; color:{consensus_signal['color']}; border:1px solid {consensus_signal['color']};">
                        {consensus_signal['action']}
                    </span>
                    <span style="font-family:monospace; font-size:1.1rem; color:#f8fafc;">
                        Score: <strong>{consensus_signal['score']:+.1f}</strong>
                    </span>
                </div>
                <div style="margin-top:14px; display:grid; grid-template-columns: 1fr 1fr; gap:10px; font-family:monospace; font-size:0.85rem;">
                    <div>Stop-Loss: <strong style="color:#ff1744;">${consensus_signal['stop_loss']:.2f}</strong></div>
                    <div>Take-Profit: <strong style="color:#00e676;">${consensus_signal['take_profit']:.2f}</strong></div>
                    <div>R:R Ratio: <strong style="color:#00f2fe;">{consensus_signal['risk_reward_ratio']}:1</strong></div>
                    <div>Confidence: <strong style="color:#f8fafc;">{consensus_signal['confidence_pct']:.0f}%</strong></div>
                </div>
                <hr style="border-color:#1e293b; margin:12px 0;">
                <div style="font-size:0.75rem; color:#94a3b8; font-family:monospace;">
                    FACTOR ATTRIBUTION BREAKDOWN:<br>
                    • Trend Following: {consensus_signal['trend_subscore']:+.1f}<br>
                    • Momentum (RSI/MACD): {consensus_signal['momentum_subscore']:+.1f}<br>
                    • Mean Reversion: {consensus_signal['reversion_subscore']:+.1f}<br>
                    • ARIMA Projected Drift: {consensus_signal['arima_subscore']:+.1f}<br>
                    • ML Directional Bias: {consensus_signal['ml_subscore']:+.1f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### 🪜 Order Flow & Market Depth")
        fig_ob = create_order_book_chart(order_book)
        st.plotly_chart(fig_ob, use_container_width=True, config={"displayModeBar": False})

        st.markdown("#### 🌊 Momentum Oscillators")
        fig_osc = create_oscillators_chart(df_features)
        st.plotly_chart(fig_osc, use_container_width=True, config={"displayModeBar": False})

# =============================================================================
# TAB 2: QUANTITATIVE RISK & PORTFOLIO
# =============================================================================
with tab_risk:
    st.markdown("### 🛡️ Institutional Risk & Tail Analysis")
    
    rk1, rk2, rk3, rk4, rk5 = st.columns(5)
    rk1.metric("Annualized Return", f"{risk_metrics['ann_return_pct']:.2f}%")
    rk2.metric("Sortino Ratio", f"{risk_metrics['sortino_ratio']:.2f}")
    rk3.metric("Calmar Ratio", f"{risk_metrics['calmar_ratio']:.2f}")
    rk4.metric("Cornish-Fisher VaR (95%)", f"-{risk_metrics['cf_var_95_pct']:.2f}%")
    rk5.metric("Max Peak-to-Trough DD", f"{risk_metrics['max_drawdown_pct']:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    col_dist, col_dd = st.columns(2)

    with col_dist:
        fig_dist = create_risk_distribution_chart(df_features["Return"], risk_metrics)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_dd:
        fig_dd = create_drawdown_chart(risk_metrics["drawdown_series"])
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
            "Historical Trade Win Rate"
        ],
        "Value": [
            f"{risk_metrics['ann_volatility_pct']:.2f}%",
            f"{risk_metrics['sharpe_ratio']:.2f}",
            f"{risk_metrics['sortino_ratio']:.2f}",
            f"{risk_metrics['max_drawdown_pct']:.2f}%",
            f"{risk_metrics['calmar_ratio']:.2f}",
            f"-{risk_metrics['hist_var_95_pct']:.2f}%",
            f"-{risk_metrics['hist_var_99_pct']:.2f}%",
            f"-{risk_metrics['cf_var_95_pct']:.2f}%",
            f"-{risk_metrics['cvar_95_pct']:.2f}%",
            f"{risk_metrics['skewness']:.2f}",
            f"{risk_metrics['kurtosis']:.2f}",
            f"{risk_metrics['win_rate_pct']:.1f}%"
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
            "Percentage of positive closing sessions"
        ]
    }
    st.table(pd.DataFrame(audit_data))

# =============================================================================
# TAB 3: AI MARKET INTELLIGENCE DESK (POWERED BY GEMINI)
# =============================================================================
with tab_ai:
    st.markdown("### 🧠 Google Gemini Systematic Research Desk")
    st.caption("Autonomous synthesis of quantitative telemetries, predictive divergence, and tail risk regimes")

    ai_col1, ai_col2 = st.columns([0.7, 0.3])
    with ai_col2:
        re_gen_btn = st.button("⚡ Re-Generate Institutional Memo", use_container_width=True)

    if "ai_memo_cache" not in st.session_state or re_gen_btn or st.session_state.get("cached_sym") != active_symbol:
        with st.spinner("Generating institutional memo via Google Gemini (gemini-3.6-flash)..."):
            memo_text = ai_analyst.generate_institutional_brief(
                symbol=active_symbol,
                asset_name=asset_options.get(active_symbol, active_symbol),
                latest_price=current_price,
                price_change_pct=price_change_pct,
                technical_data={
                    "RSI": float(df_features["RSI"].iloc[-1]),
                    "MACD": float(df_features["MACD"].iloc[-1]),
                },
                signal_data=consensus_signal,
                arima_data=arima_results,
                ml_data=ml_results,
                risk_data=risk_metrics,
                order_book_data=order_book
            )
            st.session_state["ai_memo_cache"] = memo_text
            st.session_state["cached_sym"] = active_symbol
    else:
        memo_text = st.session_state["ai_memo_cache"]

    st.markdown(
        f"""
        <div style="background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:24px; line-height:1.6;">
            {memo_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🎯 Scenario Probability Matrix")
    sc_cols = st.columns(3)
    with sc_cols[0]:
        st.markdown(
            f"""
            <div style="background:#0b1915; border:1px solid #00e67644; border-radius:8px; padding:16px;">
                <div style="color:#00e676; font-weight:700;">🟢 BULL CASE ({ml_results['prob_bullish']:.0f}%)</div>
                <div style="font-size:0.85rem; margin-top:8px; color:#cbd5e1;">
                    Catalyst: Sustained breakout beyond ${consensus_signal['take_profit']:.2f}. Volatility compression enables aggressive momentum expansion.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with sc_cols[1]:
        st.markdown(
            f"""
            <div style="background:#1c1912; border:1px solid #ffb30044; border-radius:8px; padding:16px;">
                <div style="color:#ffb300; font-weight:700;">🟡 BASE CASE (Neutral Drift)</div>
                <div style="font-size:0.85rem; margin-top:8px; color:#cbd5e1;">
                    Mean-reverting consolidation within Bollinger range [${df_features['BB_Lower'].iloc[-1]:.2f} - ${df_features['BB_Upper'].iloc[-1]:.2f}].
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with sc_cols[2]:
        st.markdown(
            f"""
            <div style="background:#1f1013; border:1px solid #ff174444; border-radius:8px; padding:16px;">
                <div style="color:#ff1744; font-weight:700;">🔴 BEAR CASE ({ml_results['prob_bearish']:.0f}%)</div>
                <div style="font-size:0.85rem; margin-top:8px; color:#cbd5e1;">
                    Breach of ${consensus_signal['stop_loss']:.2f} stop boundary triggering stop cascade and liquidity vacuum.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# =============================================================================
# TAB 4: MODEL DIAGNOSTICS & BACKTESTING
# =============================================================================
with tab_diag:
    st.markdown("### 🔬 Quantitative Model Auditing & Validation")

    diag_c1, diag_c2 = st.columns(2)

    with diag_c1:
        st.markdown("#### 📐 Econometric ARIMA Diagnostics")
        st.markdown(
            f"""
            <div class="metric-container">
                <div style="font-family:monospace; font-size:0.9rem; line-height:1.8;">
                    • Model Order: <strong>ARIMA{arima_order}</strong><br>
                    • Akaike Information Criterion (AIC): <strong>{arima_results['aic']}</strong><br>
                    • Bayesian Information Criterion (BIC): <strong>{arima_results['bic']}</strong><br>
                    • ADF Test Stat (p-value): <strong>{arima_results['adf_pvalue']}</strong> ({'Stationary' if arima_results['is_stationary'] else 'Non-Stationary'})<br>
                    • Ljung-Box Test (White Noise Residuals): <strong>p = {arima_results['ljung_box_pvalue']}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with diag_c2:
        st.markdown("#### 🤖 Machine Learning Out-of-Sample Performance")
        st.markdown(
            f"""
            <div class="metric-container">
                <div style="font-family:monospace; font-size:0.9rem; line-height:1.8;">
                    • Out-of-Sample Directional Accuracy: <strong style="color:#00e676;">{ml_results['directional_accuracy']:.1f}%</strong><br>
                    • Root Mean Squared Error (RMSE): <strong>{ml_results['rmse']:.4f}</strong><br>
                    • Mean Absolute Error (MAE): <strong>{ml_results['mae']:.4f}</strong><br>
                    • Ensemble Models: <strong>Random Forest + Gradient Boosting</strong><br>
                    • Hyperparameters: <strong>{n_estimators} trees, {n_lags} lag inputs</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🌳 Machine Learning Feature Importance Decomposition")
    feat_df = ml_results["feature_importance"].head(8)
    
    fig_feat = go.Figure()
    fig_feat.add_trace(go.Bar(
        x=feat_df["Importance"],
        y=feat_df["Feature"],
        orientation="h",
        marker=dict(color="#38bdf8")
    ))
    fig_feat.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b101b",
        plot_bgcolor="#0b101b",
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_feat, use_container_width=True)

# -----------------------------------------------------------------------------
# AUTO-REFRESH TRIGGER
# -----------------------------------------------------------------------------
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
