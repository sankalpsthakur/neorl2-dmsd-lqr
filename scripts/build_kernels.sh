#!/bin/bash
# Regenerates all kernel scripts and runs the static checks (nothing is pushed). The final kernel needs the committed frozen sha256:
#   scripts/build_kernels.sh <frozen_sha256>      (default: placeholder SET_AFTER_PREP, the final kernel then refuses to run)
# Wall caps (PREREGISTRATION section 6) are armed inside each kernel (CFG.wall_cap_s). The final kernel's n and seeds are hard-coded in its main.py.
# Prints the FULL sha256 of every script.py; the final one must be written into TEST-LOG and committed BEFORE the final push (run_wave.sh refuses otherwise).
cd "$(dirname "$0")/.." || exit 1
SHA=${1:-SET_AFTER_PREP}
python3 scripts/make_kernel.py kernels/dryrun --cfg '{"n": 100, "bc_epochs": 3, "wall_cap_s": 900}'
python3 scripts/make_kernel.py kernels/prep --cfg '{"wall_cap_s": 1800}'
python3 scripts/make_kernel.py kernels/g0 --cfg '{"n": 1000, "bc_epochs": 30, "wall_cap_s": 3600}'
python3 scripts/make_kernel.py kernels/final --cfg "{\"frozen_sha256\": \"$SHA\", \"wall_cap_s\": 3600}"
python3 scripts/check_kernels.py | grep -v '^PASS'
RC=${PIPESTATUS[0]}
for k in dryrun prep g0 final; do printf '%s  kernels/%s/script.py\n' "$(shasum -a 256 kernels/$k/script.py | cut -d' ' -f1)" "$k"; done
exit $RC
