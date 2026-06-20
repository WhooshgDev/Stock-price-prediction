@echo off
cd /d D:\Stock-price-prediction
echo [Training]
python tft.py
echo.
echo [Evaluation]
python eval.py
echo.
echo Done!
pause
