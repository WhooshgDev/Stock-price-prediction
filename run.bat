@echo off
cd /d D:\Stock-price-prediction
echo [Training]
python scripts/tft.py
echo.
echo [Evaluation]
python scripts/eval.py
echo.
echo Done!
pause
