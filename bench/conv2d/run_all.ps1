# bench/conv2d/run_all.ps1 — placeholder for the ResNet-style conv2d bench.
# Today this just builds the runtime + reports that conv2d is on the
# (op, dtype, device) coverage matrix at TBD position. The bench-runner
# subagent will fill in real measurements once `aether_op_conv2d_*` is
# wired (P7.3).

$ErrorActionPreference = 'Continue'
$repo = "J:\aether"
$out  = Join-Path $repo "scratch\bench_conv2d.txt"
"" | Out-File -Encoding utf8 $out

Push-Location $repo

"==== Aether conv2d (placeholder) ====" | Tee-Object -Append $out
cargo build -p aether_rt 2>&1 | Select-Object -Last 1 | Out-File -Encoding utf8 -Append $out
"conv2d harness ready; awaits aether_op_conv2d_* wiring (P7.3)" | Tee-Object -Append $out

Pop-Location
