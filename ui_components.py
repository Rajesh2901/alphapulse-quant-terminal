"""
AlphaPulse Quantitative Terminal - UI Components & Data Visualization Suite
Institutional dark-theme styling, interactive Plotly financial charts, and telemetry cards.
"""

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TERMINAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Hide Streamlit default overhead navigation to maintain native terminal look */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    #MainMenu, footer {
        visibility: hidden !important;
    }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 98% !important;
    }

    .stApp {
        background-color: #070a10;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Responsive Quant Header */
    .quant-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        padding: 12px 20px;
        background: linear-gradient(180deg, #0e1524 0%, #0a0f1a 100%);
        border: 1px solid #1c2738;
        border-radius: 8px;
        margin-bottom: 16px;
    }

    .quant-title {
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .quant-badge-group {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px;
    }

    .quant-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        padding: 3px 8px;
        border-radius: 4px;
        background: rgba(0, 242, 254, 0.08);
        color: #00f2fe;
        border: 1px solid rgba(0, 242, 254, 0.25);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        white-space: nowrap;
    }

    /* Metric Cards */
    .metric-container {
        background: #0d131f;
        border: 1px solid #1a2333;
        border-radius: 8px;
        padding: 12px 16px;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-container:hover {
        border-color: #38bdf8;
        box-shadow: 0 4px 18px rgba(0, 242, 254, 0.06);
    }
    .metric-label {
        font-size: 0.7rem;
        color: #8fa0b5;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.45rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.02em;
    }
    .metric-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        margin-top: 4px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .text-bullish { color: #00e676; }
    .text-bearish { color: #ff1744; }
    .text-neutral { color: #ffb300; }
    .text-cyan { color: #00f2fe; }

    /* Signal Card */
    .signal-box {
        background: linear-gradient(135deg, #0c1422 0%, #131e31 100%);
        border: 1px solid #1f2d42;
        border-radius: 8px;
        padding: 16px;
        position: relative;
        overflow: hidden;
    }
    .signal-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.05rem;
        font-weight: 800;
        padding: 5px 12px;
        border-radius: 5px;
        letter-spacing: 0.04em;
    }

    /* Tabs Styling & Overflow Fix */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid #1c2738;
        width: 100%;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        padding: 8px 14px;
        border-radius: 6px 6px 0 0;
        background-color: transparent;
        color: #8fa0b5;
        white-space: nowrap;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0f1828 !important;
        color: #00f2fe !important;
        border-bottom: 2px solid #00f2fe !important;
    }
    /* Style tab pagination buttons */
    .stTabs button[data-baseweb="tab-highlight"] {
        background-color: #00f2fe !important;
    }
    .stTabs button[aria-label="Previous"], .stTabs button[aria-label="Next"] {
        background-color: #0f1828 !important;
        color: #00f2fe !important;
        border: 1px solid #1c2738 !important;
        border-radius: 4px;
    }

    /* Sidebar Dark Theming & Inputs */
    [data-testid="stSidebar"] {
        background-color: #070b12;
        border-right: 1px solid #141b27;
    }
    div[data-baseweb="select"] > div {
        background-color: #0c121e !important;
        border-color: #1a2333 !important;
        color: #f8fafc !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="input"] > div {
        background-color: #0c121e !important;
        border-color: #1a2333 !important;
        color: #f8fafc !important;
        border-radius: 6px !important;
    }
    input {
        color: #f8fafc !important;
    }
</style>
"""


def create_candlestick_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25]
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=f"{symbol} Price",
            increasing_line_color="#00e676",
            decreasing_line_color="#ff1744",
            increasing_fillcolor="#00e676",
            decreasing_fillcolor="#ff1744",
        ),
        row=1,
        col=1,
    )

    if "EMA_9" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["EMA_9"], line=dict(color="#00f2fe", width=1.2), name="EMA 9"
            ),
            row=1,
            col=1,
        )
    if "EMA_21" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["EMA_21"], line=dict(color="#ffb300", width=1.2), name="EMA 21"
            ),
            row=1,
            col=1,
        )
    if "SMA_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_50"],
                line=dict(color="#38bdf8", width=1.2, dash="dash"),
                name="SMA 50",
            ),
            row=1,
            col=1,
        )

    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Upper"],
                line=dict(color="rgba(255, 255, 255, 0.15)", width=1),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Lower"],
                line=dict(color="rgba(255, 255, 255, 0.15)", width=1),
                fill="tonexty",
                fillcolor="rgba(56, 189, 248, 0.04)",
                name="Bollinger Bands (20,2)",
            ),
            row=1,
            col=1,
        )

    colors = [
        "#00e676" if c >= o else "#ff1744"
        for c, o in zip(df["Close"], df["Open"], strict=False)
    ]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], marker_color=colors, opacity=0.6, name="Volume"),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b101b",
        plot_bgcolor="#0b101b",
        margin=dict(l=10, r=10, t=20, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=520,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#182234")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#182234")

    return fig


def create_forecast_cone_chart(
    historical_df: pd.DataFrame,
    arima_res: dict[str, Any],
    ml_res: dict[str, Any],
    symbol: str,
    lookback_bars: int = 40,
) -> go.Figure:
    fig = go.Figure()
    hist_subset = historical_df.iloc[-lookback_bars:]

    fig.add_trace(
        go.Scatter(
            x=hist_subset.index,
            y=hist_subset["Close"],
            mode="lines",
            name="Historical Close",
            line=dict(color="#cbd5e1", width=2),
        )
    )

    last_dt = hist_subset.index[-1]
    last_price = hist_subset["Close"].iloc[-1]

    cone_x = [last_dt] + list(arima_res["upper_95"].index)
    upper_95 = [last_price] + list(arima_res["upper_95"].values)
    lower_95 = [last_price] + list(arima_res["lower_95"].values)

    fig.add_trace(
        go.Scatter(
            x=cone_x + cone_x[::-1],
            y=upper_95 + lower_95[::-1],
            fill="toself",
            fillcolor="rgba(0, 242, 254, 0.08)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% Confidence Band",
            hoverinfo="skip",
        )
    )

    upper_80_vals = [last_price] + list(arima_res.get("upper_80", arima_res["upper_95"]).values)
    lower_80_vals = [last_price] + list(arima_res.get("lower_80", arima_res["lower_95"]).values)

    fig.add_trace(
        go.Scatter(
            x=cone_x + cone_x[::-1],
            y=upper_80_vals + lower_80_vals[::-1],
            fill="toself",
            fillcolor="rgba(0, 242, 254, 0.16)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% Confidence Band",
            hoverinfo="skip",
        )
    )

    arima_x = [last_dt] + list(arima_res["pred_series"].index)
    arima_y = [last_price] + list(arima_res["pred_series"].values)
    fig.add_trace(
        go.Scatter(
            x=arima_x,
            y=arima_y,
            mode="lines+markers",
            name=f"ARIMA Forecast ({arima_res.get('expected_change_pct', 0):+.2f}%)",
            line=dict(color="#00f2fe", width=2.5, dash="dot"),
            marker=dict(size=4),
        )
    )

    if "pred_series" in ml_res:
        ml_x = [last_dt] + list(ml_res["pred_series"].index)
        ml_y = [last_price] + list(ml_res["pred_series"].values)
        fig.add_trace(
            go.Scatter(
                x=ml_x,
                y=ml_y,
                mode="lines+markers",
                name=f"ML Ensemble ({ml_res.get('expected_change_pct', 0):+.2f}%)",
                line=dict(color="#00e676", width=2.5, dash="dash"),
                marker=dict(size=4),
            )
        )

    fig.add_hline(
        y=last_price,
        line_dash="dash",
        line_color="#64748b",
        annotation_text=f"Anchor: ${last_price:.2f}",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b101b",
        plot_bgcolor="#0b101b",
        margin=dict(l=10, r=10, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=450,
        title=dict(
            text=f"{symbol} Predictive Trajectory & Confidence Cones",
            font=dict(size=14, color="#94a3b8"),
        ),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#182234")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#182234")

    return fig


def create_oscillators_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("MACD (12, 26, 9)", "RSI (14)"),
    )

    if "MACD" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["MACD"], line=dict(color="#00f2fe", width=1.5), name="MACD"
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD_Signal"],
                line=dict(color="#ffb300", width=1.5),
                name="Signal",
            ),
            row=1,
            col=1,
        )
        hist_colors = ["#00e676" if h >= 0 else "#ff1744" for h in df["MACD_Hist"]]
        fig.add_trace(
            go.Bar(x=df.index, y=df["MACD_Hist"], marker_color=hist_colors, name="Hist"),
            row=1,
            col=1,
        )

    if "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI"], line=dict(color="#a855f7", width=1.8), name="RSI"),
            row=2,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#ff1744", row=2, col=1, opacity=0.7)
        fig.add_hline(y=30, line_dash="dash", line_color="#00e676", row=2, col=1, opacity=0.7)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b101b",
        plot_bgcolor="#0b101b",
        margin=dict(l=10, r=10, t=30, b=20),
        height=380,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#182234")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#182234")
    return fig


def create_order_book_chart(order_book: dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    bids = order_book["bids"].sort_values(by="Price", ascending=False)
    asks = order_book["asks"].sort_values(by="Price", ascending=True)

    fig.add_trace(
        go.Scatter(
            x=bids["Price"],
            y=bids["Total"],
            fill="tozeroy",
            fillcolor="rgba(0, 230, 118, 0.25)",
            line=dict(color="#00e676", width=2),
            name="Cumulative Bids (Buy Depth)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=asks["Price"],
            y=asks["Total"],
            fill="tozeroy",
            fillcolor="rgba(255, 23, 68, 0.25)",
            line=dict(color="#ff1744", width=2),
            name="Cumulative Asks (Sell Depth)",
        )
    )

    fig.add_vline(x=order_book["best_bid"], line_dash="dot", line_color="#00e676", opacity=0.5)
    fig.add_vline(x=order_book["best_ask"], line_dash="dot", line_color="#ff1744", opacity=0.5)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b101b",
        plot_bgcolor="#0b101b",
        margin=dict(l=10, r=10, t=30, b=20),
        height=320,
        title=dict(
            text=f"L2 Market Depth Ladder | Spread: {order_book['spread_bps']:.1f} bps | Imbalance: {order_book['imbalance']:+.2%}",
            font=dict(size=13, color="#94a3b8"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#182234")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#182234")
    return fig


def create_risk_distribution_chart(
    returns_series: pd.Series, risk_stats: dict[str, Any]
) -> go.Figure:
    r = returns_series.dropna() * 100.0
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(x=r, nbinsx=50, name="Daily Return Dist", marker_color="#38bdf8", opacity=0.65)
    )

    var_95 = risk_stats.get("hist_var_95_pct", 2.0)
    fig.add_vline(
        x=-var_95,
        line_color="#ffb300",
        line_width=2,
        line_dash="dash",
        annotation_text=f"VaR 95%: -{var_95:.2f}%",
        annotation_position="top left",
    )

    cvar_95 = risk_stats.get("cvar_95_pct", 3.0)
    fig.add_vline(
        x=-cvar_95,
        line_color="#ff1744",
        line_width=2,
        line_dash="dot",
        annotation_text=f"CVaR 95%: -{cvar_95:.2f}%",
        annotation_position="bottom left",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b101b",
        plot_bgcolor="#0b101b",
        margin=dict(l=10, r=10, t=30, b=20),
        height=320,
        title=dict(
            text="Empirical Returns Distribution & Tail Risk Cutoffs",
            font=dict(size=13, color="#94a3b8"),
        ),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#182234", title="Daily Return (%)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#182234", title="Frequency")
    return fig


def create_drawdown_chart(dd_series: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd_series.index,
            y=dd_series * 100.0,
            fill="tozeroy",
            fillcolor="rgba(255, 23, 68, 0.2)",
            line=dict(color="#ff1744", width=1.5),
            name="Drawdown %",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b101b",
        plot_bgcolor="#0b101b",
        margin=dict(l=10, r=10, t=30, b=20),
        height=300,
        title=dict(
            text="Historical Underwater Drawdown Curve (%)", font=dict(size=13, color="#94a3b8")
        ),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#182234")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#182234")
    return fig
