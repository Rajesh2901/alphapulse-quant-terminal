"""
AlphaPulse Quantitative Terminal - Configuration Module
Centralized settings for market universes, model parameters, API credentials, and terminal styling.
"""

import os
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# API CREDENTIALS
# -----------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GOOGLE_API_KEY:
    try:
        import streamlit as st

        GOOGLE_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        GOOGLE_API_KEY = ""

# -----------------------------------------------------------------------------
# ASSET UNIVERSES
# -----------------------------------------------------------------------------
ASSET_UNIVERSES: dict[str, list[dict[str, str]]] = {
    "Equities & Tech Titans": [
        {"symbol": "NVDA", "name": "NVIDIA Corporation", "category": "Semiconductors / AI"},
        {"symbol": "AAPL", "name": "Apple Inc.", "category": "Consumer Electronics"},
        {"symbol": "MSFT", "name": "Microsoft Corporation", "category": "Enterprise Cloud / AI"},
        {"symbol": "TSLA", "name": "Tesla, Inc.", "category": "EV / Autonomous"},
        {"symbol": "GOOGL", "name": "Alphabet Inc.", "category": "Digital Advertising / AI"},
        {"symbol": "AMZN", "name": "Amazon.com Inc.", "category": "E-Commerce / Cloud"},
        {"symbol": "META", "name": "Meta Platforms Inc.", "category": "Social / AI"},
    ],
    "Indices & ETFs": [
        {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "category": "US Large Cap"},
        {"symbol": "QQQ", "name": "Invesco QQQ Trust (Nasdaq 100)", "category": "Tech Index"},
        {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "category": "Small Cap"},
        {"symbol": "TLT", "name": "iShares 20+ Year Treasury Bond ETF", "category": "Fixed Income"},
        {"symbol": "GLD", "name": "SPDR Gold Shares", "category": "Precious Metals"},
    ],
    "Cryptocurrency Assets": [
        {"symbol": "BTC-USD", "name": "Bitcoin USD", "category": "Crypto / Digital Gold"},
        {"symbol": "ETH-USD", "name": "Ethereum USD", "category": "Smart Contract Platform"},
        {"symbol": "SOL-USD", "name": "Solana USD", "category": "Layer 1 High-Throughput"},
        {"symbol": "BNB-USD", "name": "BNB USD", "category": "Exchange Token"},
    ],
    "FX & Commodities": [
        {"symbol": "EURUSD=X", "name": "EUR/USD Cross Rate", "category": "Foreign Exchange"},
        {"symbol": "GBPUSD=X", "name": "GBP/USD Cross Rate", "category": "Foreign Exchange"},
        {"symbol": "USDJPY=X", "name": "USD/JPY Cross Rate", "category": "Foreign Exchange"},
        {"symbol": "GC=F", "name": "Gold Continuous Contract", "category": "Commodities"},
        {"symbol": "CL=F", "name": "Crude Oil Continuous Contract", "category": "Energy"},
    ],
}

# -----------------------------------------------------------------------------
# TIMEFRAME & INTERVAL CONFIGURATIONS
# -----------------------------------------------------------------------------
TIMEFRAME_OPTIONS = {
    "1 Minute (High-Freq)": {"interval": "1m", "period": "5d"},
    "5 Minutes (Intraday)": {"interval": "5m", "period": "1mo"},
    "15 Minutes (Short-Term)": {"interval": "15m", "period": "1mo"},
    "1 Hour (Swing/Tactical)": {"interval": "1h", "period": "3mo"},
    "1 Day (Positional/Macro)": {"interval": "1d", "period": "1y"},
}

DEFAULT_TIMEFRAME = "1 Day (Positional/Macro)"
DEFAULT_SYMBOL = "NVDA"


# -----------------------------------------------------------------------------
# MODEL PARAMETERS
# -----------------------------------------------------------------------------
@dataclass
class QuantModelConfig:
    # Econometric (ARIMA) settings
    arima_order: tuple = (2, 1, 2)
    auto_arima: bool = True
    max_p: int = 4
    max_q: int = 4

    # ML Ensemble settings
    n_estimators: int = 120
    learning_rate: float = 0.05
    max_depth: int = 4
    lags: int = 5
    forecast_horizon: int = 10

    # Quantitative Risk Parameters
    risk_free_rate: float = 0.045
    var_confidence_95: float = 0.95
    var_confidence_99: float = 0.99
    trading_days_per_year: int = 252


DEFAULT_MODEL_CONFIG = QuantModelConfig()

# -----------------------------------------------------------------------------
# TERMINAL UI PALETTE & DESIGN SYSTEM
# -----------------------------------------------------------------------------
UI_THEME = {
    "bg_color": "#0a0e17",
    "surface_color": "#111827",
    "surface_elevated": "#1a2234",
    "border_color": "#2d3748",
    "text_primary": "#f8fafc",
    "text_muted": "#94a3b8",
    "accent_cyan": "#00f2fe",
    "accent_blue": "#3b82f6",
    "bullish_green": "#00e676",
    "bearish_red": "#ff1744",
    "neutral_amber": "#ffb300",
    "font_mono": "'JetBrains Mono', 'Fira Code', 'Courier New', monospace",
    "font_sans": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
}
