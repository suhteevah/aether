# scripts/audit.ps1 — single-command rigorous codebase audit.
#
# Builds the workspace, runs every test crate, and then runs aether-audit
# which scans for honesty issues, verifies golden artifacts, and runs the
# Aether language conformance suite. Non-zero exit if any dimension reports
# an error. Pair with `scripts/smoke.ps1` for the runtime training run.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "==> building workspace (debug)"
$prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
& cargo build --workspace --quiet 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { $ErrorActionPreference = $prev; throw "build failed" }
$ErrorActionPreference = $prev

$exe = Join-Path $root "target\debug\aether-audit.exe"
if (-not (Test-Path $exe)) { throw "aether-audit missing at $exe" }

Write-Host "==> running aether-audit"
$ErrorActionPreference = "Continue"
& $exe @args
$rc = $LASTEXITCODE
$ErrorActionPreference = $prev

if ($rc -ne 0) { throw "audit reported $rc error(s)" }
Write-Host "OK - audit clean"
