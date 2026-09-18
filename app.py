"""
AlphaPulse Quantitative Terminal - Main Application Controller
Professional-grade trading and investment dashboard delivering real-time market intelligence,
econometric ARIMA & ML ensemble forecasting, advanced risk analytics, and Google Gemini AI research.
"""

import datetime
import time

import streamlit as st

from ai_analyst import AIAssistantAnalyst
from config import (
    ASSET_UNIVERSES,
    GOOGLE_API_KEY,
    TIMEFRAME_OPTIONS,
)
from exceptions import AlphaPulseBaseException, InvalidSymbolError
from services.quant_service import QuantService
from ui_components import TERMINAL_CSS
from views import (
    render_ai_workspace,
    render_diagnostics_workspace,
    render_kpi_cards,
    render_risk_workspace,
    render_telemetry_header,
    render_terminal_workspace,
)

# -----------------------------------------------------------------------------
# APPLICATION CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AlphaPulse | Institutional Quantitative Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(TERMINAL_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SERVICE INITIALIZATION (CACHED SINGLETONS)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_quant_service() -> QuantService:
    return QuantService()


@st.cache_resource
def get_ai_analyst() -> AIAssistantAnalyst:
    return AIAssistantAnalyst(api_key=GOOGLE_API_KEY)


quant_service = get_quant_service()
ai_analyst = get_ai_analyst()


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & PARAMETERS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ **AlphaPulse Terminal**")
    st.caption("Systematic Quantitative Research & Trading Desk")
    st.divider()

    # Universe Selection
    universe_choice = st.selectbox("Asset Universe", list(ASSET_UNIVERSES.keys()), index=0)
    asset_options = {
        item["symbol"]: f"{item['symbol']} - {item['name']}"
        for item in ASSET_UNIVERSES[universe_choice]
    }

    col_sym, col_custom = st.columns([0.65, 0.35])
    with col_sym:
        selected_symbol = st.selectbox(
            "Select Symbol", list(asset_options.keys()), format_func=lambda x: asset_options[x]
        )
    with col_custom:
        custom_symbol = st.text_input("Custom", placeholder="e.g. SPY").upper().strip()

    raw_symbol = custom_symbol if custom_symbol else selected_symbol

    # Validate ticker input
    try:
        active_symbol = QuantService.validate_symbol(raw_symbol)
    except InvalidSymbolError as e:
        st.error(str(e))
        st.stop()

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
        arima_order = (int(p), int(d), int(q))

        st.markdown("**Machine Learning Ensemble**")
        n_lags = st.slider("Feature Lag Window", min_value=3, max_value=10, value=5)
        n_estimators = st.slider(
            "Ensemble Estimators", min_value=50, max_value=200, value=100, step=25
        )

    # Live Refresh Controls
    st.markdown("#### 🔄 Live Ingestion")
    auto_refresh = st.checkbox("Continuous Refresh", value=False)
    refresh_interval = st.selectbox("Refresh Interval (s)", [10, 20, 30, 60], index=1)

    if st.button("⚡ Force Cache Refresh", use_container_width=True):
        quant_service.pipeline._cache.clear()
        st.rerun()

    st.divider()
    st.markdown(
        f"<div style='font-size:0.75rem; color:#64748b; font-family:monospace;'>"
        f"API ENGINE: Google GenAI (Active)<br>"
        f"QUANT CORE: statsmodels & scikit-learn<br>"
        f"SYS TIME: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# DOMAIN ORCHESTRATION & EXECUTION
# -----------------------------------------------------------------------------
try:
    with st.spinner(f"Executing parallel quantitative pipelines for {active_symbol}..."):
        analysis = quant_service.execute_analysis(
            symbol=active_symbol,
            period=tf_config["period"],
            interval=tf_config["interval"],
            forecast_horizon=forecast_horizon,
            arima_order=arima_order,
            n_lags=n_lags,
            n_estimators=n_estimators,
        )
except AlphaPulseBaseException as domain_err:
    st.error(f"Domain Execution Error: {domain_err}")
    st.stop()
except Exception as sys_err:
    st.error(f"System Error: {sys_err}")
    st.stop()


# -----------------------------------------------------------------------------
# PRESENTATION LAYER
# -----------------------------------------------------------------------------
# 1. Telemetry Header Ribbon
render_telemetry_header(analysis)

# 2. Executive KPI Cards
render_kpi_cards(analysis, forecast_horizon)

# 3. Workspace Tabs
tab_live, tab_risk, tab_ai, tab_diag = st.tabs(
    [
        "📈 Live Terminal & Forecast",
        "🛡️ Quantitative Risk & Tail",
        "🧠 Gemini AI Research Desk",
        "🔬 Model Diagnostics & Audit",
    ]
)

with tab_live:
    render_terminal_workspace(analysis)

with tab_risk:
    render_risk_workspace(analysis)

with tab_ai:
    asset_name = asset_options.get(active_symbol, active_symbol)
    render_ai_workspace(analysis, ai_analyst, asset_name)

with tab_diag:
    render_diagnostics_workspace(analysis, arima_order, n_estimators, n_lags)


# -----------------------------------------------------------------------------
# REACTIVE REFRESH CONTROLLER
# -----------------------------------------------------------------------------
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
