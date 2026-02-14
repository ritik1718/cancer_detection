@echo off
setlocal

:: Define the path to the Python executable in the virtual environment
set PYTHON_EXE=venv\Scripts\python.exe

echo ========================================================
echo Starting Sequential Training for All 4 Experiments
echo ========================================================
echo.

:: 1. ViT Base Experiment
echo [1/4] Running ViT Base Experiment...
cd vit-base-experiment
..\%PYTHON_EXE% src/train.py
if %ERRORLEVEL% NEQ 0 (
    echo Error running ViT Base Experiment!
    pause
    exit /b %ERRORLEVEL%
)
cd ..
echo [1/4] ViT Base Experiment Completed.
echo.

:: 2. Swin Transformer Experiment (Large)
echo [2/4] Running Swin Transformer (Large) Experiment...
cd swin-transformer-experiment
..\%PYTHON_EXE% src/train.py
if %ERRORLEVEL% NEQ 0 (
    echo Error running Swin Transformer Experiment!
    pause
    exit /b %ERRORLEVEL%
)
cd ..
echo [2/4] Swin Transformer Experiment Completed.
echo.

:: 3. DenseNet Experiment
echo [3/4] Running DenseNet Experiment...
cd densenet-experiment
..\%PYTHON_EXE% src/train.py
if %ERRORLEVEL% NEQ 0 (
    echo Error running DenseNet Experiment!
    pause
    exit /b %ERRORLEVEL%
)
cd ..
echo [3/4] DenseNet Experiment Completed.
echo.

:: 4. Original Source Experiment (Swin Base)
echo [4/4] Running Original Swin Base Experiment...
:: This one is in the root src, so we run it from root
%PYTHON_EXE% src/train.py
if %ERRORLEVEL% NEQ 0 (
    echo Error running Original Swin Base Experiment!
    pause
    exit /b %ERRORLEVEL%
)
echo [4/4] Original Swin Base Experiment Completed.
echo.

echo ========================================================
echo All Experiments Completed Successfully!
echo ========================================================
pause
