@echo off
REM Build spermC.exe.
REM Works with cl.exe when the MSVC environment is already initialized.

setlocal

where cl.exe >nul 2>nul
if not errorlevel 1 goto :build_cl

where gcc.exe >nul 2>nul
if not errorlevel 1 goto :build_gcc

if exist "%MSVCDir%\bin\vcvars32.bat" call "%MSVCDir%\bin\vcvars32.bat"
where cl.exe >nul 2>nul
if not errorlevel 1 goto :build_cl

echo Build failed. Install Visual C++ or MinGW GCC and make sure it is on PATH.
goto :done

:build_cl
cl /nologo /O2 /Ob2 /MT /W3 /TC /D_CRT_SECURE_NO_WARNINGS /Fe:spermC.exe gguf_infer.c
if errorlevel 1 goto :error
goto :success

:build_gcc
gcc -O3 -march=native -mtune=native -funroll-loops -std=c99 -Wall -Wextra -Wno-unused-parameter -o spermC.exe gguf_infer.c -lm
if errorlevel 1 goto :error

:success
echo Build successful: spermC.exe
goto :done

:error
echo Build failed.

:done
endlocal
