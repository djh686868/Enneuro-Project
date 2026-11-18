@echo off
setlocal
rem ============================================================
rem  批量编译 cuda/cu/*.cu  ->  cuda/dll/*.dll
rem  用法: 双击运行, 或在终端执行 code\cuda\build.bat
rem ============================================================
set "SCRIPT_DIR=%~dp0"
set "CU_DIR=%SCRIPT_DIR%cu"
set "OUT_DIR=%SCRIPT_DIR%dll"

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

for %%f in ("%CU_DIR%\*.cu") do (
    echo [nvcc] %%~nxf -^> %%~nf.dll
    nvcc -shared -o "%OUT_DIR%\%%~nf.dll" "%%~ff" -Xcompiler "/EHsc /W3 /nologo /O2 /FS /utf-8"
    if errorlevel 1 (
        echo [FAILED] %%~nxf
        exit /b 1
    )
)

echo.
echo All CUDA operators compiled to "%OUT_DIR%".
endlocal
