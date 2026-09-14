# AlphaPulse: Institutional Quantitative Trading & Investment Terminal

AlphaPulse is a professional-grade quantitative research and trading terminal engineered in Python. It combines real-time market data ingestion, dual-engine predictive modeling (econometric ARIMA + non-linear Machine Learning ensembles), institutional risk analytics (VaR, CVaR, Cornish-Fisher, Sharpe, Drawdown), and generative AI market commentary powered by **Google Gemini**.

Designed for hedge fund analysts, systematic portfolio managers, and quantitative researchers, the application provides sub-second analytics, Level-2 (L2) order book visualization, and automated quantitative research memorandums.

---

## Architecture Overview

```
quant_terminal/
├── config.py                 # System configurations, asset universes, model hyperparameters, API credentials
├── data_ingestion.py         # Resilient live pipelines, caching, auto-retry, and synthetic market microstructure
├── features.py               # 27 quantitative factors (EMA, MACD, RSI, Bollinger Bands, ATR, Realized Volatility)
├── models.py                 # Econometric ARIMA models, ML Ensemble predictors (RF + GBDT), Consensus Signal Engine
├── risk_engine.py            # VaR (Historical & Parametric), Cornish-Fisher expansion, CVaR, Sharpe, Drawdowns
├── ai_analyst.py             # Google Gemini (gemini-3.6-flash) integration for autonomous institutional research
├── ui_components.py          # Bloomberg-style dark terminal theme & Plotly interactive financial charts
├── app.py                    # Multi-workspace Streamlit terminal interface
├── test_pipeline.py          # End-to-end automated verification test suite
├── requirements.txt          # Python production dependencies
└── run_terminal.bat          # One-click Windows batch launcher
```

---

## Quantitative & Predictive Methodology

### 1. Econometric Time-Series Modeling (ARIMA)
- **Stationarity Verification**: Augmented Dickey-Fuller (ADF) hypothesis test applied to first-differenced log prices:
  $$\Delta \ln(P_t) = \alpha + \beta t + \gamma \ln(P_{t-1}) + \sum_{i=1}^p \delta_i \Delta \ln(P_{t-i}) + \epsilon_t$$
- **ARIMA$(p, d, q)$ Formulation**: Models the stationary return innovations with autoregressive and moving average polynomials:
  $$\phi(B)(1 - B)^d \ln(P_t) = \theta(B)\epsilon_t$$
- **Analytical Confidence Cones**: Forward projections include $80\%$ and $95\%$ analytical error bands mapped back into geometric price space:
  $$\hat{P}_{t+h} = \exp\left(\hat{y}_{t+h} \pm z_{\alpha/2} \sigma_h\right)$$

### 2. Machine Learning Ensemble Forecasting
- **Algorithms**: Ensemble blending Random Forest Regressors and Gradient Boosted Decision Trees (GBDT).
- **Feature Tensor**:
  - Autoregressive return lags: $r_{t-1}, r_{t-2}, \dots, r_{t-5}$
  - Momentum: RSI-14, MACD Histogram
  - Volatility: Bollinger Bandwidth, Bollinger $\%B$, ATR percentage
  - Microstructure: Volume-to-20SMA ratio, Price distance from 21-EMA
- **Directional Probability**: Estimates cumulative horizon return probability using calibrated residual standard error:
  $$P(\text{Bullish}) = \Phi\left(\frac{\hat{R}_{t+h}}{\sigma_{\text{resid}} \sqrt{h}}\right)$$

### 3. Consensus Signal Engine
Synthesizes orthogonal signal dimensions into a normalized Quantitative Score $[-100, +100]$:
- **Trend Following** (25%): 9-EMA / 21-EMA alignment and 50-SMA baseline
- **Momentum** (20%): RSI bounds ($<30$ oversold, $>70$ overbought) and MACD histogram sign
- **Mean Reversion** (15%): Bollinger Band penetration ($\%B < 0.1$ or $> 0.9$)
- **ARIMA Drift** (20%): Econometric expected path
- **ML Ensemble** (20%): Non-linear probability estimation

**Rating Categorization**:
- Score $\ge +45$: `STRONG BUY`
- $+15 \le$ Score $< +45$: `BUY`
- $-15 <$ Score $< +15$: `NEUTRAL`
- $-45 <$ Score $\le -15$: `SELL`
- Score $\le -45$: `STRONG SELL`

Execution boundaries (Stop-Loss and Take-Profit) are dynamically calibrated to Average True Range (ATR) maintaining a minimum 1:2.0 risk-reward ratio.

### 4. Quantitative Risk Analytics
- **Historical Value-at-Risk (VaR)**: Empirical $5\text{th}$ and $1\text{st}$ percentiles of historical return distribution.
- **Parametric Gaussian VaR**:
  $$\text{VaR}_\alpha = -(\mu - z_\alpha \sigma)$$
- **Cornish-Fisher VaR Expansion**: Accounts for higher-order statistical moments (skewness $S$, excess kurtosis $K$):
  $$z_{\text{cf}} = z_\alpha + \frac{z_\alpha^2 - 1}{6}S + \frac{z_\alpha^3 - 3z_\alpha}{24}K - \frac{2z_\alpha^3 - 5z_\alpha}{36}S^2$$
- **Conditional Value-at-Risk (CVaR / Expected Shortfall)**:
  $$\text{CVaR}_\alpha = -\mathbb{E}[R \mid R \le -\text{VaR}_\alpha]$$
- **Performance Attribution**: Annualized Sharpe Ratio ($R_f = 4.5\%$), Downside Sortino Ratio, Calmar Ratio, and Maximum Drawdown recovery dynamics.

### 5. Google Gemini AI Market Intelligence
Powered by `gemini-3.6-flash`, the terminal automatically converts mathematical telemetry, forecast paths, L2 order book imbalances, and VaR risk parameters into an institutional research memo covering:
1. Executive Quantitative Synthesis
2. Predictive Model Alignment & Divergence
3. Microstructure & Order Flow Intelligence
4. Tail-Risk & Volatility Regimes
5. Tactical Execution Matrix

---

## Quickstart & Installation

### Prerequisites
- Python 3.11+ installed.
- Network access to market data endpoints.

### Setup Virtual Environment
```powershell
cd C:\Users\rajes\.gemini\antigravity\scratch\quant_terminal
uv venv .venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### Run Automated Test Suite
```powershell
python test_pipeline.py
```

### Launch the Terminal
Simply double-click `run_terminal.bat` or run:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Workspace Navigation

1. **📈 Live Terminal & Predictive Analytics**:
   - Candlesticks with 9-EMA, 21-EMA, 50-SMA, and Bollinger Bands.
   - Predictive Fan Chart displaying ARIMA & ML forecast trajectories with $80\%$ and $95\%$ confidence cones.
   - Algorithmic Consensus Signal box with factor attribution.
   - Simulated Level-2 (L2) market depth ladder showing cumulative bids vs asks.
   - MACD and RSI secondary oscillators.

2. **🛡️ Quantitative Risk & Portfolio**:
   - VaR and CVaR statistical audit table.
   - Empirical return distribution with tail risk cutoff boundaries.
   - Historical underwater drawdown curve.

3. **🧠 AI Market Intelligence Desk**:
   - Real-time institutional research memorandum generated by Google Gemini.
   - Bull, Base, and Bear scenario probability matrix.

4. **🔬 Model Diagnostics & Backtesting**:
   - Econometric parameter validation (AIC, BIC, ADF p-value, Ljung-Box test).
   - Machine Learning out-of-sample directional accuracy, RMSE, and MAE.
   - Feature importance rankings (Random Forest + Gradient Boosting).
