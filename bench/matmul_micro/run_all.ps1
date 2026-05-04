# 3-way matmul bench: Aether vs Candle vs PyTorch — CPU + GPU.
# Same sizes, same iters, same warm-up discipline, same hardware.

$ErrorActionPreference = 'Continue'
$repo = "J:\aether"
$out  = Join-Path $repo "scratch\bench_3way.txt"
"" | Out-File -Encoding utf8 $out

function Header($s) {
    $line = "==== $s ===="
    Write-Host $line
    $line | Out-File -Encoding utf8 -Append $out
}

# 1) Aether
Header "Aether (own runtime, cuBLAS via cudarc)"
Push-Location $repo
cargo build -p aether_rt --features cuda 2>&1 | Select-Object -Last 1 | Out-File -Encoding utf8 -Append $out
cargo build -p aetherc 2>&1 | Select-Object -Last 1 | Out-File -Encoding utf8 -Append $out
& "$repo\target\debug\aetherc.exe" "$repo\scratch\bench_batch.aether" --emit=aether-bin -o "$repo\scratch\bench_batch_3way.exe" 2>&1 | Out-File -Encoding utf8 -Append $out
$aether_out = & "$repo\scratch\bench_batch_3way.exe" 2>&1
$aether_out | Out-File -Encoding utf8 -Append $out
$aether_out | Where-Object { $_ -match '^batch' } | ForEach-Object { Write-Host $_ }
Pop-Location

# 2) Candle (local fork at J:/candle-src)
Header "Candle (J:/candle-src local fork)"
Push-Location "$repo\bench\matmul_micro"
$candle_out = cmd /c .\run_candle.bat 2>&1
$candle_out | Out-File -Encoding utf8 -Append $out
$candle_out | Where-Object { $_ -match '^candle-' } | ForEach-Object { Write-Host $_ }
Pop-Location

# 3) PyTorch
Header "PyTorch"
$torch_out = python "$repo\bench\matmul_micro\pytorch\bench.py" 2>&1
$torch_out | Out-File -Encoding utf8 -Append $out
$torch_out | Where-Object { $_ -match '^pytorch-' } | ForEach-Object { Write-Host $_ }

Write-Host ""
Write-Host "Full log: $out"
