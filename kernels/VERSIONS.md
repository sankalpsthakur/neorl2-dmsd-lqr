# Kernel versions (script sha256 = sha256 of kernels/<dir>/script.py; regenerate with scripts/build_kernels.sh, which also runs scripts/check_kernels.py and prints the full hashes)

Regenerated 2026-11-18 16:2x UTC after PREREGISTRATION Amendment 1 (code commit 4a12818). All kernels are private, CPU only (`enable_gpu` false), named `bench-neorl2-dmsd-*`, and read no secret. There is **no GPU kernel by design** (billed-GPU-hour cap 0.0; `scripts/no_gpu.sh`). Each kernel arms an in-kernel wall cap (`CFG.wall_cap_s`).

| kernel | state | script sha256 (full) | notes |
|---|---|---|---|
| bench-neorl2-dmsd-dryrun | PUSHED 16:15Z v1, COMPLETE (STATUS.md) | 37a551e52eca9956ee08bc90197a6038509bb5db3502089a2d865f32fbc928ce | CPU, SYNTHETIC data, never a result. Installs the official package at commit b50d91d0 (5 files, hash-pinned), runs G0b, the BC path (3 epochs), audit/identification/selection/regime check and both scorers on seed sets 1, 2, 3, n = 100; its claim code path must return False. wall cap 900 s |
| bench-neorl2-dmsd-prep | PREPARED, not pushed | d90400d0025c0195d6062650c611c3e72b7f8cc8aa85cb451361b8fde1404c1b | CPU, internet. G0a, identification gate, validation-only selection, regime check (report-only), writes `frozen_config.json`. No `neorl2` import. wall cap 1,800 s |
| bench-neorl2-dmsd-g0 | PREPARED, not pushed | 4e3afb0a602a4dea2942e0325c2f904ccdaad60f9fe5f46aa337cc23e86c4d2f | CPU. G0b hard (replica vs official), G0c report-only (BC, 1000 episodes on seed set 7, 30 epochs; torch CPU). wall cap 3,600 s |
| bench-neorl2-dmsd-final | PREPARED with the placeholder `SET_AFTER_PREP` (refuses to run; hash 34f0bd8c37f6ff5c... is of the placeholder build and WILL CHANGE) | regenerate with `scripts/build_kernels.sh <sha256 of frozen_config.json>` after the freeze, then bind the new full hash in TEST-LOG | n = 1000 and seeds (7, 42, 210) hard-coded in its main.py; rows P before S, R, A; wall cap 3,600 s |

Static checks on generation (scripts/check_kernels.py): metadata private/CPU/names/sources, syntax, no undefined name, no secret/push/upload string in the script or the embedded package, embedded package equals `src/nd` and the protocol files, script body equals `main.py`, CFG parses, in-kernel wall cap armed, final hard-codes n and seeds.
Local flow tests (tests/test_kernel_flow.py, under `bin/guard.sh`): all four scripts executed with the replica standing in for the official package, no network, BC stubbed.
Not run anywhere before the dry run: the official package install on a Kaggle image, the parquet read, the BC training, the G0b comparison against the real `neorl2` code.
