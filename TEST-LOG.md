# TEST-LOG: every contact with the official simulator, announced before it runs

PREREGISTRATION sections 2, 4 and 5, as amended by Amendment 1 (2026-11-18 16:13 UTC). "Simulator contact" = any episode run in the official `gym.make("DMSD")` (the evaluation simulator). Episode seeds: evaluation sets `ep_seed(s, i) = 100,000,000 + 100,000 s + i` (i = 0..999); G0b `5,000,000 + i` (mode train) and `5,001,000 + i` (mode test); dry run seed sets 1, 2, 3. One evaluation per frozen configuration and seed set. Rows are filled whatever the outcome; failed and partial runs stay. Status of every row on 2026-11-18: **announced, not run** except D1-D5 (dry run, done 16:25Z, rehearsal on synthetic data; D6 mode-test row is in the same report) (the synthetic dry-run kernel is the only kernel pushed; no controller of ours exists; no simulator episode of rows G, P, S, R, A has been run).

| # | kernel | what is scored on the simulator | seeds / episodes | role | status |
|---|---|---|---|---|---|
| D1 | bench-neorl2-dmsd-dryrun | G0b policies zero, PD, bang-bang, official vs replica | 50 seeds each, mode train; 20 each, mode test | rehearsal (scripted policies) | announced |
| D2 | bench-neorl2-dmsd-dryrun | BC trained 3 epochs on SYNTHETIC data | 50 episodes of seed set 1 | rehearsal | announced |
| D3-D5 | bench-neorl2-dmsd-dryrun | controller identified from SYNTHETIC data | seed sets 1, 2, 3 x 100 episodes | rehearsal, never a result; the claim code path MUST return claim False (n and seeds wrong by design) | announced |
| D6 | bench-neorl2-dmsd-dryrun | the same synthetic controller in mode test | seed set 1 x 100 | rehearsal | announced |
| G1-G3 | bench-neorl2-dmsd-g0 | policies zero, PD, bang-bang; official vs replica (G0b) | 50 seeds `5,000,000 + i`, mode train | gate | announced |
| G4-G6 | bench-neorl2-dmsd-g0 | the same three policies in mode test (G0b) | 20 seeds `5,001,000 + i` | gate | announced |
| G7 | bench-neorl2-dmsd-g0 | BC (own MLP, 30 epochs, final epoch) on the official simulator (G0c) | seed set 7 x 1000 (the same episodes as P1; BC does not feed the controller) | REPORT-ONLY since Amendment 1 (band 51.64 to 70.09 recorded, not a stop) | announced |
| P1 | bench-neorl2-dmsd-final | frozen controller | seed set 7 x 1000, mode train | **primary** | announced |
| P2 | bench-neorl2-dmsd-final | frozen controller | seed set 42 x 1000, mode train | **primary** | announced |
| P3 | bench-neorl2-dmsd-final | frozen controller | seed set 210 x 1000, mode train | **primary** | announced |
| S1-S3 | bench-neorl2-dmsd-final | frozen controller, mode test (mass 30) | seed sets 7, 42, 210 x 1000 | secondary (distribution shift), no claim | announced; written after P1-P3 |
| R1-R3 | bench-neorl2-dmsd-final | oracle: true-parameter LQR with the frozen qp, qv | seed sets 7, 42, 210 x 1000, mode train | reference, no claim | announced; after P1-P3 |
| A1-A3 | bench-neorl2-dmsd-final | frozen design without target feed-forward (F = 0) | seed sets 7, 42, 210 x 1000, mode train | ablation, no claim | announced; after P1-P3 |

**Script binding (Amendment 1, A1.3).** The sha256 of `kernels/final/script.py` must be written here and committed BEFORE the final kernel is pushed; `scripts/run_wave.sh` refuses the final push otherwise. Line to fill at the freeze: `FINAL SCRIPT SHA256 (set by the lead at the freeze commit): 82672f9c8a4809d13a92b1e1a26806cea3d1463027a4239d01f76a0d1393b17c`. The final script hard-codes n = 1000 and seeds (7, 42, 210); the claim rule enforces both again. Other scripts as generated 2026-11-18 (commit 4a12818, final with the placeholder sha): dryrun 37a551e5, prep d90400d0, g0 4e3afb0a (hex prefixes; full values in kernels/VERSIONS.md).

Non-simulator checks (not rows): G0a (arithmetic on the train file), identification gate, validation-only selection, all in bench-neorl2-dmsd-prep before any row above except D1-D6.

Result columns to fill after each row: mean normalised score, per-episode sd, 95 % bootstrap interval, replica mean, maximum per-episode return difference between the replica and the official simulator, kernel version, UTC.

| # | mean | sd | 95 % interval | replica mean | max return diff | kernel version | UTC |
|---|---|---|---|---|---|---|---|
| D1 (zero, PD, bang; mode train 50 seeds, mode test 20) | n/a | n/a | n/a | official == replica | 0.0 | dryrun v1 | 2026-11-18T16:2xZ |
| D2 (BC 3 epochs, synthetic) | 62.87 (50 ep) | | | | | dryrun v1 | same |
| D3-D5 (synthetic controller, sets 1, 2, 3 x 100) | 97.98 / 98.20 / 98.37 | | | equal | 0.0 | dryrun v1 | same |
| (G, P, S, R, A rows: none yet) | | | | | | | |

FROZEN CONFIG SHA256 c3f031e99dbdfa12228f5f02add8745206754811b8dca6dc2e28e8b71dbec509 (Amendment 2, 2026-11-18 16:32 UTC; prep kernel v1; committed before g0 and final).

## Outcome rows (recorded 2026-11-18 16:45 UTC by the lead from kernel_outputs/final/testlog_rows.json; one scored run, nothing dropped or repeated)

| row | n | mode | mean | 95 % interval | max replica diff |
|---|---|---|---|---|---|
| P-7 | 1000 | train | 97.7770 | [97.5083, 98.0315] | 0.0 |
| P-42 | 1000 | train | 97.8260 | [97.5579, 98.0946] | 0.0 |
| P-210 | 1000 | train | 97.7383 | [97.4734, 98.0027] | 0.0 |
| S-7 | 1000 | test | 94.2609 | [93.8900, 94.6150] | 0.0 |
| S-42 | 1000 | test | 94.3245 | [93.9516, 94.6965] | 0.0 |
| S-210 | 1000 | test | 94.1889 | [93.8197, 94.5577] | 0.0 |
| R-7 | 1000 | train | 97.7770 | [97.5083, 98.0315] | 0.0 |
| R-42 | 1000 | train | 97.8260 | [97.5579, 98.0946] | 0.0 |
| R-210 | 1000 | train | 97.7383 | [97.4734, 98.0027] | 0.0 |
| A-7 | 1000 | train | 76.0853 | [75.5575, 76.5904] | 0.0 |
| A-42 | 1000 | train | 76.3376 | [75.8067, 76.8716] | 0.0 |
| A-210 | 1000 | train | 76.1500 | [75.6070, 76.7081] | 0.0 |

G rows: G0a pass (data score 56.6427), G0b pass (zero, PD, bang-bang: official equals replica, difference 0.0), G0c report-only (own BC 61.54, in band). Claim rule: claim = true, blocked_by = [].
