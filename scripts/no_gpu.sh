#!/bin/bash
# This track has a billed-GPU-hour cap of 0.0 (PREREGISTRATION section 6). There is no GPU gate or launcher on purpose.
echo "REFUSED: neorl2-dmsd has a 0.0 billed GPU-hour cap; any GPU push violates the pre-registration and voids the claim." >&2
exit 1
