# bench/optfx/run_all.ps1 — measure --O0 vs --O1 wall time on a constant-fold
# heavy microbench. Reports the delta. P11.1 expects ≥3% improvement on at
# least one row; the bench-runner subagent appends a row to BENCH_LEDGER.md.

$ErrorActionPreference = 'Continue'
$repo = "J:\aether"
$out  = Join-Path $repo "scratch\bench_optfx.txt"
"" | Out-File -Encoding utf8 $out

function Header($s) {
    $line = "==== $s ===="
    Write-Host $line
    $line | Out-File -Encoding utf8 -Append $out
}

function Time-Compile($flags) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & "$repo\target\debug\aetherc.exe" "$repo\tests\runtime\o1_constfold.aether" $flags --emit=aether-bin -o "$repo\scratch\optfx.exe" | Out-Null
    $sw.Stop()
    return $sw.Elapsed.TotalMilliseconds
}

Push-Location $repo

Header "Aether --O0 (no opts)"
$o0 = (1..5 | ForEach-Object { Time-Compile "" } | Measure-Object -Average).Average
"--O0 mean compile: {0:N2} ms" -f $o0 | Tee-Object -Append $out

Header "Aether --O1 (constfold + dead let elim)"
$o1 = (1..5 | ForEach-Object { Time-Compile "--O1" } | Measure-Object -Average).Average
"--O1 mean compile: {0:N2} ms" -f $o1 | Tee-Object -Append $out

$delta = if ($o0 -gt 0) { (($o1 - $o0) / $o0) * 100 } else { 0 }
"delta (--O1 vs --O0): {0:N2}%" -f $delta | Tee-Object -Append $out

Pop-Location
