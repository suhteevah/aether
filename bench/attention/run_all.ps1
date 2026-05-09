# bench/attention/run_all.ps1 — placeholder for the SDPA bench. Runs the
# existing cuda_attention witness and times it; bench-runner subagent
# extends with PyTorch SDPA cross-comparison once parity tables are in.

$ErrorActionPreference = 'Continue'
$repo = "J:\aether"
$out  = Join-Path $repo "scratch\bench_attention.txt"
"" | Out-File -Encoding utf8 $out

Push-Location $repo

"==== Aether SDPA (causal attention) ====" | Tee-Object -Append $out
& "$repo\target\debug\aetherc.exe" "$repo\tests\runtime\cuda_attention.aether" --emit=aether-bin -o "$repo\scratch\attention.exe" 2>&1 | Tee-Object -Append $out
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& "$repo\scratch\attention.exe"
$sw.Stop()
"attention wall: {0:N2} ms" -f $sw.Elapsed.TotalMilliseconds | Tee-Object -Append $out

Pop-Location
