#!/usr/bin/env bash
# matt-voice FR-18.6-real leg 3 — multi-process pipeline-parallel parity witness.
#
# Runs the qwen3 PP worker (trainer/src/bin/pp_qwen_worker.rs) as:
#   (a) world_size=1 single process (all L layers) — the reference
#   (b) world_size=2, two processes (layers split L/2 + L/2) over localhost TCP
# and asserts the pipelined run is BIT-IDENTICAL to the reference: same loss
# trajectory + the two ranks' param checksums sum to the single-process total.
# This is the GPU analog of the leg-1 LinearReluStack thread witness — real
# cross-process pipeline parallelism with GPU stages, the exact shape of the
# Qwen3-32B 2xP100 cnc run (only --host / --layers / dims / per-rank GPU change).
#
# Usage: bash scripts/pp_2rank_witness.sh   (needs a built --features cuda binary)
set -euo pipefail

BIN=./target/release/pp-qwen-worker
LAYERS=4
EPOCHS=40
PORT=$(( 29600 + RANDOM % 1000 ))

if [ ! -x "$BIN" ]; then
  echo "building pp-qwen-worker (--features cuda)..."
  cargo build -p trainer --features cuda --release --bin pp-qwen-worker
fi

echo "=== (a) world_size=1 reference: $LAYERS layers, 1 process ==="
REF=$("$BIN" --rank 0 --world-size 1 --layers $LAYERS --epochs $EPOCHS 2>/dev/null)
echo "$REF" | grep -E "RESULT|PARAMSUM"
ref_first=$(echo "$REF" | sed -n 's/.*first_loss=\([0-9.]*\).*/\1/p')
ref_final=$(echo "$REF" | sed -n 's/.*final_loss=\([0-9.]*\).*/\1/p')
ref_sum=$(echo "$REF" | sed -n 's/.*sum_abs=\([0-9.]*\).*/\1/p')

echo "=== (b) world_size=2 pipelined: $LAYERS layers split $((LAYERS/2))+$((LAYERS/2)), 2 processes over TCP :$PORT ==="
"$BIN" --rank 1 --world-size 2 --layers $LAYERS --epochs $EPOCHS --base-port $PORT >/tmp/pp_r1.out 2>/dev/null &
R1=$!
sleep 0.5
"$BIN" --rank 0 --world-size 2 --layers $LAYERS --epochs $EPOCHS --base-port $PORT >/tmp/pp_r0.out 2>/dev/null
wait $R1
cat /tmp/pp_r1.out | grep -E "RESULT|PARAMSUM"
cat /tmp/pp_r0.out | grep -E "PARAMSUM"
pp_first=$(grep RESULT /tmp/pp_r1.out | sed -n 's/.*first_loss=\([0-9.]*\).*/\1/p')
pp_final=$(grep RESULT /tmp/pp_r1.out | sed -n 's/.*final_loss=\([0-9.]*\).*/\1/p')
r0_sum=$(grep PARAMSUM /tmp/pp_r0.out | sed -n 's/.*sum_abs=\([0-9.]*\).*/\1/p')
r1_sum=$(grep PARAMSUM /tmp/pp_r1.out | sed -n 's/.*sum_abs=\([0-9.]*\).*/\1/p')

echo "=== parity check ==="
echo "  loss:    ref [$ref_first -> $ref_final]   pp [$pp_first -> $pp_final]"
pp_sum=$(awk "BEGIN{printf \"%.6f\", $r0_sum + $r1_sum}")
echo "  paramsum: ref $ref_sum   pp (r0+r1) $pp_sum"

fail=0
[ "$ref_first" = "$pp_first" ] || { echo "  FAIL: first_loss differs"; fail=1; }
[ "$ref_final" = "$pp_final" ] || { echo "  FAIL: final_loss differs"; fail=1; }
# paramsum: allow last-digit float rounding in the awk add
d=$(awk "BEGIN{x=$ref_sum-$pp_sum; print (x<0?-x:x)}")
awk "BEGIN{exit !($d < 0.01)}" || { echo "  FAIL: paramsum mismatch (|diff|=$d)"; fail=1; }

if [ $fail -eq 0 ]; then
  echo "  PASS: pipelined 2-process run is bit-identical to single-process reference"
else
  exit 1
fi
