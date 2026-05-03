@echo off
rem Mirrors J:\candle-src\build-cuda.bat — the recipe Matt actually uses for
rem his candle fork. Uses BuildTools' vcvars64 (not Community), unsets
rem RUSTC_WRAPPER, sets CUDA_COMPUTE_CAP=86 for the 3070 Ti, and overrides
rem ~/.cargo/config.toml's lld-link with MSVC link.exe.

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set RUSTC_WRAPPER=
set CUDA_COMPUTE_CAP=86
set CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=link.exe
cd /d "%~dp0candle"
rem Aether's default Rust toolchain is GNU, but candle-kernels' .o files
rem use MSVC stack-cookie / SEH (__security_check_cookie, __GSHandlerCheck)
rem which mingw ld can't resolve. Pin the bench to the MSVC toolchain
rem (`+stable-msvc`) and explicitly target windows-msvc.
cargo +stable-x86_64-pc-windows-msvc build --release --target x86_64-pc-windows-msvc
if errorlevel 1 exit /b %errorlevel%
target\x86_64-pc-windows-msvc\release\candle_matmul_bench.exe
