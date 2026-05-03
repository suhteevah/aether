# Kill stale rustc / cargo / nvcc / link.exe processes that pile up when an
# IDE checker and a CLI cargo run simultaneously on the same workspace.
# Safe to run any time — anything in flight will get re-spawned by the next
# build trigger. See .cargo/config.toml for the long-term fix (per-tool
# target dirs via `cargo ide-check` / `cargo ide-clippy`).

param([switch]$WhatIf)

$names = @('rustc', 'cargo', 'nvcc', 'link', 'lld-link', 'cl')
$killed = 0
$skipped = 0
foreach ($name in $names) {
    $procs = Get-Process -Name $name -ErrorAction SilentlyContinue
    foreach ($p in $procs) {
        $age_sec = (Get-Date) - $p.StartTime
        if ($age_sec.TotalSeconds -lt 5) {
            $skipped++
            Write-Host ("skip  {0,-10} pid={1,-6}  (just started)" -f $p.ProcessName, $p.Id)
            continue
        }
        if ($WhatIf) {
            Write-Host ("would-kill {0,-10} pid={1,-6}  rss={2,7:N0} MB" -f $p.ProcessName, $p.Id, ($p.WorkingSet64 / 1MB))
        } else {
            try {
                Stop-Process -Id $p.Id -Force -ErrorAction Stop
                Write-Host ("killed     {0,-10} pid={1,-6}  rss={2,7:N0} MB" -f $p.ProcessName, $p.Id, ($p.WorkingSet64 / 1MB))
                $killed++
            } catch {
                Write-Host ("FAIL       {0,-10} pid={1,-6}  $_" -f $p.ProcessName, $p.Id) -ForegroundColor Red
            }
        }
    }
}
Write-Host ""
Write-Host "killed: $killed  skipped: $skipped"
