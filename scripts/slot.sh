#!/bin/bash
# Kaggle CPU slot helper (PLAN.md): slot.sh take "<kernel ref>"  -> prints the slot number taken (waits until one is free); owner file = "neorl2dmsd <kernel ref> <UTC start>"
#                                    slot.sh release <N> ["<kernel ref>"]   (removes the lock only if its owner file is ours (and, if given, names that kernel); never touches another track's lock)
LOCKS=${LOCKS_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/locks}
case "$1" in
  take)
    MAXW=${SLOT_MAX_WAIT_S:-0}; START=$SECONDS      # 0 = wait forever
    while true; do
      for n in 1 2 3 4; do
        if mkdir "$LOCKS/kaggle-cpu-$n" 2>/dev/null; then
          printf 'neorl2dmsd %s %s\n' "$2" "$(date -u +%Y-%m-%dT%H:%MZ)" > "$LOCKS/kaggle-cpu-$n/owner"; echo "$n"; exit 0
        fi
      done
      if [ "$MAXW" -gt 0 ] && [ $((SECONDS - START)) -ge "$MAXW" ]; then echo "no free slot after ${MAXW}s" >&2; exit 3; fi
      sleep 20
    done;;
  release)
    O="$LOCKS/kaggle-cpu-$2/owner"
    if [ -f "$O" ] && grep -q '^neorl2dmsd ' "$O" && { [ -z "$3" ] || grep -qF "$3" "$O"; }; then rm -rf "$LOCKS/kaggle-cpu-$2" && echo "released $2"
    else echo "slot $2 not ours (or already released): left alone"; fi;;
esac
