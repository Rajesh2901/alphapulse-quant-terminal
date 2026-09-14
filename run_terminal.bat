@echo off
title AlphaPulse Quantitative Terminal
echo =====================================================================
echo           ALPHAPULSE QUANTITATIVE TRADING & RESEARCH DESK
echo =====================================================================
echo.
echo Initializing Python virtual environment...
cd /d "%~dp0"
call .venv\Scripts\activate.bat

echo Launching Streamlit Terminal Server...
echo Terminal will be available at: http://localhost:8501
echo.
streamlit run app.py --server.port 8501 --server.headless false

pause
