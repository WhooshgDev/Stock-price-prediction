@echo off
cd /d D:\Stock-price-prediction
echo [TFT Training]
python demo/tft/model.py
echo.
echo [TFT Evaluation]
python demo/tft/eval.py
echo.
echo [Launching Demo App]
start streamlit run demo/app.py
echo.
echo Done!
pause
