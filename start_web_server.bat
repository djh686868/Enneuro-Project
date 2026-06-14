@echo off
set PYTHONNOUSERSITE=1
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set PATH=%CUDA_PATH%\bin;%PATH%
set ENNEURO_PYTHON=C:\Users\Administrator\.conda\envs\EnNeuro\python.exe
cd /d "%~dp0code"
"%ENNEURO_PYTHON%" -m uvicorn web_server.main:app --host 0.0.0.0 --port 8000 --reload
pause
