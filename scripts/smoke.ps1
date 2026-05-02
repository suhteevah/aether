$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "==> cargo build --workspace"
cargo build --workspace --quiet
if ($LASTEXITCODE -ne 0) { throw "build failed" }

Write-Host "==> cargo test --workspace"
cargo test --workspace --quiet
if ($LASTEXITCODE -ne 0) { throw "tests failed" }

$exe = Join-Path $root "target\debug\aetherc.exe"

Write-Host "==> hello end-to-end"
& $exe examples\00_hello.aether -o hello.exe | Out-Null
if (-not (Test-Path hello.exe)) { throw "hello.exe not produced" }
$out = & .\hello.exe
if ($out -notmatch "Hello from Aether") { throw "hello output wrong: $out" }

Write-Host "==> train_mlp MIR has all_reduce + cross_entropy partial"
& $exe examples\02_train_mlp.aether --emit=mir -o train.mir | Out-Null
$mir = Get-Content train.mir -Raw
if ($mir -notmatch "all_reduce grads world_size=8") { throw "missing all_reduce" }
if ($mir -notmatch "softmax")                       { throw "missing cross_entropy partial" }
if ($mir -notmatch "tape_reverse")                  { throw "missing tape_reverse" }

Write-Host "==> train_mlp LLVM IR has tape alloca + intrinsics"
& $exe examples\02_train_mlp.aether --emit=llvm-ir -o train.ll | Out-Null
$ll = Get-Content train.ll -Raw
if ($ll -notmatch "alloca \[1024 x")                            { throw "missing tape alloca" }
if ($ll -notmatch "@aether_autodiff_reverse")                   { throw "missing autodiff_reverse" }
if ($ll -notmatch "@aether_dist_all_reduce\(i8\* null, i32 8")  { throw "missing all_reduce intrinsic" }

Write-Host "==> matmul + serve_llama parse cleanly"
& $exe examples\01_matmul.aether     --emit=mir -o mm.mir    | Out-Null
& $exe examples\03_serve_llama.aether --emit=mir -o serve.mir | Out-Null

Write-Host "==> aether_lm parses + has real symbolic partials for matmul"
& $exe examples\aether_lm.aether --emit=mir -o lm.mir | Out-Null
$lm = Get-Content lm.mir -Raw
if ($lm -notmatch 'grad\[\d+\] @ v\[\d+\]\.T') { throw "missing matmul partial in causal_attention" }
if ($lm -notmatch 'softmax\(v\[\d+\]\) - onehot') { throw "missing cross_entropy partial" }
if ($lm -notmatch 'tape_reverse')                { throw "missing tape_reverse" }

Write-Host "==> aether_lm LLVM IR has aether_autodiff_partial calls"
& $exe examples\aether_lm.aether --emit=llvm-ir -o lm.ll | Out-Null
$ll = Get-Content lm.ll -Raw
if ($ll -notmatch '@aether_autodiff_partial\(') { throw "missing autodiff_partial intrinsic" }

Write-Host "==> --check on broken file emits AE0002 JSON"
"fn broken( {" | Out-File -Encoding ascii _broken.aether
$prev = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $exe _broken.aether --check --json-errors 2>_err.txt | Out-Null
$ErrorActionPreference = $prev
$err = Get-Content _err.txt -Raw
Remove-Item _broken.aether,_err.txt -Force
if ("$err" -notmatch 'AE0002')        { throw "expected AE0002 code in JSON output: $err" }
if ("$err" -notmatch '"line":\s*\d+') { throw "expected line in JSON output: $err" }

Write-Host "==> stdlib + AetherLM model parse"
foreach ($f in @("stdlib\ops.aether","stdlib\optim.aether","stdlib\nn.aether","examples\aether_lm.aether")) {
    & $exe $f --check | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "$f failed --check" }
}

Write-Host "==> AetherLM MIR has tape + symbolic partials + extern markers"
& $exe examples\aether_lm.aether --emit=mir -o lm.mir | Out-Null
$lm = Get-Content lm.mir -Raw
if ($lm -notmatch 'tape_reverse')          { throw "missing tape_reverse" }
if ($lm -notmatch 'softmax')               { throw "missing softmax partial" }

Write-Host "==> AetherLM LLVM IR contains autodiff_partial calls"
& $exe examples\aether_lm.aether --emit=llvm-ir -o lm.ll | Out-Null
$ll = Get-Content lm.ll -Raw
if ($ll -notmatch '@aether_autodiff_partial\(') { throw "missing @aether_autodiff_partial" }

Write-Host "==> aether_asm: encoder + COFF + parser tests"
$prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
& cargo test -p aether_asm --quiet 2>$null | Out-Null
$rc = $LASTEXITCODE
$ErrorActionPreference = $prev
if ($rc -ne 0) { throw "aether_asm tests failed" }

Write-Host "==> Aether-only compile chain: aetherc -> aether-asm -> link -> exe"
$prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
& $exe examples\00_hello.aether --emit=aether-bin -o hello_aether.exe 2>$null | Out-Null
$rc = $LASTEXITCODE
$ErrorActionPreference = $prev
if ($rc -ne 0) { throw "aether-bin chain failed (rc=$rc)" }
$out = & .\hello_aether.exe
if ("$out" -notmatch "Hello from Aether") { throw "aether-bin output wrong: $out" }
Remove-Item hello_aether.exe,hello_aether.obj,hello_aether.s -ErrorAction SilentlyContinue

Write-Host "==> Real training run on CPU through libaether_rt (no framework)"
$prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
& cargo build --release -p trainer 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { $ErrorActionPreference = $prev; throw "release build failed" }
& .\target\release\aether-train.exe --config nano --steps 40 --batch 8 --seq 32 --lr 3e-3 --log-every 10 1>_train_out.txt 2>_train_err.txt
$ErrorActionPreference = $prev
$tlog = (Get-Content _train_err.txt -Raw) + (Get-Content _train_out.txt -Raw)
Remove-Item _train_out.txt,_train_err.txt -ErrorAction SilentlyContinue
$m0 = [regex]::Match($tlog, 'step=\s*0 loss=([0-9.]+)')
$mN = [regex]::Match($tlog, 'step=\s*39 loss=([0-9.]+)')
if (-not $m0.Success) { throw "no step=0 loss in: $tlog" }
if (-not $mN.Success) { throw "no step=39 loss in: $tlog" }
$loss0 = [float]$m0.Groups[1].Value
$lossN = [float]$mN.Groups[1].Value
if ($lossN -ge $loss0) { throw "loss did not decrease: $loss0 -> $lossN" }
if (-not (Test-Path checkpoints/aether_lm.weights)) { throw "no checkpoint produced" }
Write-Host ("    loss {0:F3} -> {1:F3}  (drop {2:F3})" -f $loss0, $lossN, ($loss0 - $lossN))

Write-Host "==> Inference round-trip"
$prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
& .\target\release\aether-infer.exe --ckpt checkpoints\aether_lm --prompt "the quick" --max-new 16 --temperature 0.5 --top-k 8 --seed 1 1>_infer.txt 2>$null
$ErrorActionPreference = $prev
$ilog = Get-Content _infer.txt -Raw
Remove-Item _infer.txt -ErrorAction SilentlyContinue
if ($ilog.Length -lt 8) { throw "infer produced no output: $ilog" }

Write-Host "OK - Phase 0/0.5 smoke passed (Aether-only, no framework, real training run)."
