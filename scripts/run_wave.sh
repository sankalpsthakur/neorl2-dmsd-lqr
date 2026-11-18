#!/bin/bash
# usage: run_wave.sh <kernel_dir> <owner/slug> <outdir> [max_watch_minutes]
# Takes a free Kaggle CPU slot (waits if all four are taken; SLOT_MAX_WAIT_S=1200 gives up after 20 minutes), pushes the kernel (retrying while the account is at its session cap),
# then watches it (5-min status checks), fetches the SMALL outputs when it ends and releases the slot. Run it as ONE background command.
# The slot is released on every exit path. The FINAL kernel is refused unless the sha256 of kernels/final/script.py is in the COMMITTED TEST-LOG.md (Amendment 1).
cd "$(dirname "$0")/.." || exit 1
KD=$1; REF=$2; OUT=$3; MAXM=${4:-300}
if [ "$(basename "$KD")" = "final" ]; then
  FS=$(shasum -a 256 "$KD/script.py" | cut -d' ' -f1)
  if ! git show HEAD:TEST-LOG.md | grep -q "$FS"; then echo "REFUSED: sha256 of $KD/script.py ($FS) is not in the committed TEST-LOG.md" >&2; exit 5; fi
  if [ -n "$(git status --porcelain)" ]; then echo "REFUSED: working tree not clean" >&2; exit 5; fi
fi
N=$(scripts/slot.sh take "$REF") || exit 3
trap 'scripts/slot.sh release "$N" "$REF"' EXIT
echo "$(date -u +%H:%MZ) slot $N taken for $REF"
if scripts/push_when_free.sh "$KD" 30; then
  scripts/watch_kernel.sh "$REF" "$N" "$OUT" "$MAXM"
else
  echo "$(date -u +%H:%MZ) push failed; slot $N released"; exit 1
fi
