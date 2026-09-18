"""
AlphaPulse Presentation Views - Diagnostics Tab
Renders econometric ARIMA parameters, residual tests, ML holdout validation, and feature rankings.
"""

import plotly.graph_objects as go
import streamlit as st

from schemas.analytics import FullAnalysisResult


def render_diagnostics_workspace(
    analysis: FullAnalysisResult,
    arima_order: tuple,
    n_estimators: int,
    n_lags: int,
):
    """Renders Tab 4: Model Diagnostics & Audit."""
    st.markdown("### 🔬 Quantitative Model Auditing & Validation")

    diag_c1, diag_c2 = st.columns(2)
    arima = analysis.arima
    ml = analysis.ml

    with diag_c1:
        st.markdown("#### 📐 Econometric ARIMA Diagnostics")
        st.markdown(
            f"""
            <div class="metric-container">
                <div style="font-family:monospace; font-size:0.9rem; line-height:1.8;">
                    • Model Order: <strong>ARIMA{arima_order}</strong><br>
                    • Akaike Information Criterion (AIC): <strong>{arima.aic:.2f}</strong><br>
                    • Bayesian Information Criterion (BIC): <strong>{arima.bic:.2f}</strong><br>
                    • ADF Test Stat (p-value): <strong>{arima.adf_pvalue:.4f}</strong> ({"Stationary" if arima.is_stationary else "Non-Stationary"})<br>
                    • Ljung-Box Test (White Noise Residuals): <strong>p = {arima.ljung_box_pvalue:.4f}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with diag_c2:
        st.markdown("#### 🤖 Machine Learning Out-of-Sample Performance")
        st.markdown(
            f"""
            <div class="metric-container">
                <div style="font-family:monospace; font-size:0.9rem; line-height:1.8;">
                    • Out-of-Sample Directional Accuracy: <strong style="color:#00e676;">{ml.directional_accuracy:.1f}%</strong><br>
                    • Root Mean Squared Error (RMSE): <strong>{ml.rmse:.4f}</strong><br>
                    • Mean Absolute Error (MAE): <strong>{ml.mae:.4f}</strong><br>
                    • Ensemble Models: <strong>Random Forest + Gradient Boosting</strong><br>
                    • Hyperparameters: <strong>{n_estimators} trees, {n_lags} lag inputs</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🌳 Machine Learning Feature Importance Decomposition")
    feat_df = ml.feature_importance.head(8)

    fig_feat = go.Figure()
    fig_feat.add_trace(
        go.Bar(
            x=feat_df["Importance"],
            y=feat_df["Feature"],
            orientation="h",
            marker=dict(color="#38bdf8"),
        )
    )
    fig_feat.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b101b",
        plot_bgcolor="#0b101b",
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_feat, use_container_width=True)
