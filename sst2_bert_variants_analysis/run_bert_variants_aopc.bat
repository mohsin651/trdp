@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "PYTHON_EXE=%PROJECT_ROOT%\sst2_quantitative_analysis\.venv\Scripts\python.exe"
set "DATA_FILE=%PROJECT_ROOT%\sst2_quantitative_analysis\data\SST-2\sst2_train_10k_seed42.tsv"

if not exist "%PYTHON_EXE%" (
    echo Expected Python venv was not found:
    echo   %PYTHON_EXE%
    exit /b 1
)

if not exist "%DATA_FILE%" (
    echo Expected 10k SST-2 data file was not found:
    echo   %DATA_FILE%
    exit /b 1
)

"%PYTHON_EXE%" "%SCRIPT_DIR%bert_variants_analysis.py" ^
    --data-file "%DATA_FILE%" ^
    --run-name bert_variants_sa_gw_aopc_10k ^
    --grad-weighted-max-examples 0 ^
    --aopc ^
    --aopc-max-examples 0 ^
    %*
