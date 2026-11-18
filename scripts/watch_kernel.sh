#!/bin/bash
# usage: watch_kernel.sh <owner/slug> <cpu-slot-number|-> <outdir> [max_minutes]
# Every 5 min: `kaggle kernels status`. At a terminal state: fetch the SMALL outputs only (log/json/jsonl/csv/txt, never .npy/.onnx/.npz/.pt), release the CPU slot lock and exit.
# The slot is released on EVERY exit path (terminal state, watch timeout, INT/TERM, error) through the trap below, only if the lock is ours (slot.sh release checks the owner file).
# On a watch timeout the kernel may still be running on Kaggle (the kernels also stop themselves at their in-kernel wall cap): the script says so and exits non-zero.
K=$1; SLOT=$2; OUT=$3; MAXM=${4:-300}; END=$((SECONDS + MAXM*60)); S=""; HERE="$(cd "$(dirname "$0")" && pwd)"
rel() { if [ "$SLOT" != "-" ] && [ -n "$SLOT" ]; then "$HERE/slot.sh" release "$SLOT" "$K"; SLOT="-"; fi; }
trap rel EXIT
trap 'echo "interrupted"; exit 130' INT TERM
while [ $SECONDS -lt $END ]; do
  S=$(~/.local/bin/kaggle kernels status "$K" 2>&1 | tail -1)
  case "$S" in *COMPLETE*|*ERROR*|*CANCEL*)
    mkdir -p "$OUT"; ~/.local/bin/kaggle kernels output "$K" -p "$OUT" --file-pattern '\.(log|json|jsonl|csv|txt)$' -q 2>&1 | tail -3
    rel; echo "$(date -u +%H:%MZ) $K :: $S"; exit 0;;
  esac
  sleep 300
done
rel
echo "TIMEOUT watching $K after ${MAXM} min (last: $S): slot released; the kernel may still be running on Kaggle (its own wall cap will stop it): check 'kaggle kernels status $K' by hand" >&2
exit 4
