@echo off
REM Build Atlas for Windows 32-bit.
REM Requires the 32-bit FreeBASIC toolchain (fbc.exe) in PATH.
REM -mt : multi-threaded runtime (REQUIRED -- training uses threads).
REM -march=native is omitted for portability across 32-bit CPUs.
REM Note: the default hyper-parameters build a ~2.7M-param model; on a
REM small 32-bit machine you may want to lower NEMB/NLAYER/BLOCK/NTHREAD
REM at the top of atlas.bas.
cd /d "%~dp0"
Z:\home\ronen\freebasic\fb_programming\FreeBASIC-1.10.1-winlibs-gcc-9.3.0\fbc32.exe -gen gcc -O 3 -mt -Wc -funroll-loops atlas.bas -x atlas.exe
if errorlevel 1 (
    echo build failed
    exit /b 1
)
echo built atlas.exe
echo   atlas.exe train [steps]   train from data\corpus.txt -^> model.bin
echo   atlas.exe chat            chat with the trained model
