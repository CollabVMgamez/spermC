@echo off
REM Build gguf_infer.exe for Windows 2000 Beta 1
REM Requires Visual C++ 6.0 (cl.exe)

if exist "%MSVCDir%\bin\vcvars32.bat" call "%MSVCDir%\bin\vcvars32.bat"

cl /Ox /Ob2 /G6 /MT /W3 /Fe:gguf_infer.exe gguf_infer.c
if errorlevel 1 goto :error

echo Build successful: gguf_infer.exe
goto :done

:error
echo Build failed. Make sure cl.exe (Visual C++ 6.0) is in PATH.

:done
