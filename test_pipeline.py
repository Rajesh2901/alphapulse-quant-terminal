"""
AlphaPulse Quantitative Terminal - Automated Pipeline Verification
Tests data ingestion, feature transformations, econometric modeling, ML ensemble,
risk engine calculations, and Google Gemini AI intelligence.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("AlphaPulse.Test")

def run_tests():
    logger.info("=== [1/6] TESTING DATA INGESTION & ORDER BOOK ===")
    from data_ingestion import MarketDataPipeline
    pipeline = MarketDataPipeline(cache_ttl_seconds=5)
    
    df = pipeline.fetch_historical_data("NVDA", period="3mo", interval="1d")
    assert not df.empty, "Historical dataframe is empty!"
    assert len(df) >= 20, f"Insufficient rows in df: {len(df)}"
    logger.info(f"Data Ingestion OK: Ingested {len(df)} rows for NVDA. Latest Close: ${df['Close'].iloc[-1]:.2f}")

    live_quote = pipeline.fetch_live_quote("NVDA")
    assert "price" in live_quote, "Live quote missing price!"
    logger.info(f"Live Quote OK: ${live_quote['price']} ({live_quote['pct_change']:+.2f}%)")

    order_book = pipeline.generate_order_book_depth(live_quote["price"], volatility=0.02)
    assert "bids" in order_book and "asks" in order_book, "Order book generation failed!"
    logger.info(f"L2 Order Book OK: Spread={order_book['spread_bps']} bps, Imbalance={order_book['imbalance']:+.2%}")

    logger.info("=== [2/6] TESTING FEATURE ENGINEERING ===")
    from features import FeatureEngine
    df_features = FeatureEngine.compute_all_features(df)
    for col in ["EMA_9", "EMA_21", "RSI", "MACD", "MACD_Hist", "BB_Upper", "BB_Lower", "ATR", "Realized_Vol_20"]:
        assert col in df_features.columns, f"Missing technical indicator: {col}"
    logger.info(f"Features OK: Generated {len(df_features.columns)} feature columns. Latest RSI: {df_features['RSI'].iloc[-1]:.1f}")

    logger.info("=== [3/6] TESTING ECONOMETRIC ARIMA MODELING ===")
    from models import ARIMAForecaster
    arima_res = ARIMAForecaster.fit_and_forecast(df_features["Close"], horizon=7, order=(1, 1, 1))
    assert "pred_series" in arima_res, "ARIMA missing pred_series!"
    assert len(arima_res["pred_series"]) == 7, "ARIMA horizon mismatch!"
    assert "lower_95" in arima_res and "upper_95" in arima_res, "ARIMA missing confidence intervals!"
    logger.info(f"ARIMA OK: Horizon=7, Expected Change={arima_res['expected_change_pct']:+.2f}%, AIC={arima_res['aic']}")

    logger.info("=== [4/6] TESTING MACHINE LEARNING ENSEMBLE FORECASTER ===")
    from models import MLEnsembleForecaster
    ml_forecaster = MLEnsembleForecaster(n_estimators=50, max_depth=3)
    ml_res = ml_forecaster.train_and_predict(df, horizon=7, n_lags=4)
    assert "pred_series" in ml_res, "ML forecaster missing pred_series!"
    assert len(ml_res["pred_series"]) == 7, "ML forecast horizon mismatch!"
    logger.info(f"ML Ensemble OK: Prob Bullish={ml_res['prob_bullish']}%, Directional Accuracy={ml_res['directional_accuracy']}%")

    logger.info("=== [5/6] TESTING CONSENSUS SIGNAL ENGINE & RISK METRICS ===")
    from models import ConsensusSignalEngine
    from risk_engine import QuantitativeRiskEngine
    
    signal = ConsensusSignalEngine.generate_consensus_signal(df_features, arima_res, ml_res)
    assert "action" in signal and "score" in signal, "Signal generation failed!"
    logger.info(f"Consensus Signal OK: {signal['action']} (Score: {signal['score']:+.1f}, SL: ${signal['stop_loss']}, TP: ${signal['take_profit']})")

    risk_stats = QuantitativeRiskEngine.calculate_risk_metrics(df_features["Return"])
    assert "hist_var_95_pct" in risk_stats and "sharpe_ratio" in risk_stats, "Risk calculation failed!"
    logger.info(f"Risk Engine OK: Sharpe={risk_stats['sharpe_ratio']}, Ann Vol={risk_stats['ann_volatility_pct']}%, VaR 95%={risk_stats['hist_var_95_pct']}%")

    logger.info("=== [6/6] TESTING GOOGLE GEMINI AI MARKET INTELLIGENCE ===")
    from ai_analyst import AIAssistantAnalyst
    from config import GOOGLE_API_KEY
    
    ai_agent = AIAssistantAnalyst(api_key=GOOGLE_API_KEY)
    memo = ai_agent.generate_institutional_brief(
        symbol="NVDA",
        asset_name="NVIDIA Corporation",
        latest_price=live_quote["price"],
        price_change_pct=live_quote["pct_change"],
        technical_data={"RSI": float(df_features["RSI"].iloc[-1]), "MACD": float(df_features["MACD"].iloc[-1])},
        signal_data=signal,
        arima_data=arima_res,
        ml_data=ml_res,
        risk_data=risk_stats,
        order_book_data=order_book
    )
    assert len(memo) > 100, "AI memo too short or failed!"
    logger.info(f"AI Analyst OK: Generated {len(memo)} chars institutional memo.")

    logger.info("ALL 6 SUITES PASSED CLEANLY!")

if __name__ == "__main__":
    run_tests()
