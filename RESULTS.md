# RESULTS: NeoRL-2 DMSD, data-identified LQR (self-evaluated; recorded 2026-11-18 16:45 UTC)

**Outcome: the pre-registered claim rule is met.** Final kernel `bench-neorl2-dmsd-final` v1 (2026-11-18 16:39 to 16:43 UTC, Kaggle CPU, 165 s), frozen configuration sha256 `c3f031e99dbdfa12228f5f02add8745206754811b8dca6dc2e28e8b71dbec509` (Amendment 2, committed before any simulator contact), script sha256 `82672f9c8a4809d13a92b1e1a26806cea3d1463027a4239d01f76a0d1393b17c` (bound in TEST-LOG before the push), official `neorl2` simulator at commit b50d91d0, mode train, 1,000 episodes per seed set. Our numpy replica agrees with the official simulator on every episode (maximum return difference 0.0).

| block | seed set 7 | seed set 42 | seed set 210 | mean |
|---|---|---|---|---|
| **P, primary: frozen identified LQR (12 numbers), mean [95 % bootstrap interval]** | 97.78 [97.51, 98.03] | 97.83 [97.56, 98.09] | 97.74 [97.47, 98.00] | **97.78** |
| R, reference: LQR designed with the true parameters, same weights (no claim) | 97.78 [97.51, 98.03] | 97.83 [97.56, 98.09] | 97.74 [97.47, 98.00] | 97.78 |
| A, ablation: same design without target feed-forward (no claim) | 76.09 [75.56, 76.59] | 76.34 [75.81, 76.87] | 76.15 [75.61, 76.71] | 76.19 |
| S, shift: frozen controller on mode test, mass 30 instead of 20 (no claim) | 94.26 [93.89, 94.62] | 94.32 [93.95, 94.70] | 94.19 [93.82, 94.56] | 94.26 |

Published numbers on the same task (NeoRL-2 paper, arXiv 2503.19267, Table 2): EDAC 78.65 (best), MCQ 77.75, RAMBO 76.22, behaviour cloning 65.09, data policy 56.64; ACPO 74.9 (arXiv 2601.23010). The paper states that a score of 95 can be treated as solving the task and that no algorithm surpasses 95 in any task.

Claim rule (PREREGISTRATION section 3 with Amendment 1): every seed set's lower 95 % bound above 80.95 (bar 78.65 plus its 2.30 spread), exactly 1,000 episodes per seed set, seed sets 7, 42, 210, gates passed, frozen hash matched, replica difference at most 0.001. All hold: lower bounds 97.51, 97.56, 97.47; `claim: true`, `blocked_by: []`.

Lead check (2026-11-18 16:45 UTC): the frozen gain re-scored locally with the numpy replica from `protocol/frozen_config.json` gives the same three means to 1e-7.

## The claim sentence (Amendment 1 template, every part mandatory)
Best published normalised score on NeoRL-2 DMSD (arXiv 2503.19267 Table 2, EDAC 78.65): 97.78 (seed sets 7 / 42 / 210: 97.78 / 97.83 / 97.74), trained on the official train file (100,000 transitions) and the official validation file (20,000 transitions; used only for model-rollout selection), official simulator (mode train), 1000 episodes x 3 seed sets [7, 42, 210], self-evaluated, as of 2026-11-18. The model is a 12-number data-identified linear-quadratic controller, not an offline-RL algorithm. Provenance: the controller family and the grid range were informed by sweeps on a true-parameter re-implementation of the evaluation simulator written from the benchmark's public source; the gain actually used was selected only by rollouts of the model identified from the offline files. The published baselines' hyper-parameters were also tuned on the evaluation simulator (best of 4 cells x 3 seeds, EDAC log: 100 episodes per number; launch_task.py: 1000), and the task is linear and noise-free, so a correct identification is expected to do well: the result is about the method class (identification plus LQR versus offline RL), not a hard-won margin.

## What it does and does not show
- The identified controller equals the true-parameter LQR to the second decimal: on this task the offline data identify the plant essentially exactly (validation one-step error 1.2e-7).
- Feed-forward on the target matters: without it the same design scores 76.2, below the EDAC bar.
- Under the benchmark's own shift (mass 30) the frozen controller still scores 94.3.
- Not shown: anything about nonlinear or noisy tasks, or about offline RL in general. The other NeoRL-2 tasks are not linear; this recipe is not expected to transfer unchanged.
- Unresolved in the bar: the paper's text says 78.56 and its table 78.65; its "+-" is a sample standard deviation of three seed means; whether its numbers used 100 or 1,000 evaluation episodes. A literature search for a higher DMSD result was done by two agents and was not exhaustive.
- Sanity (report-only): our own behaviour cloning scored 61.54 (published 65.09).
- Not published. Data licence statements conflict (Hub card Apache-2.0, GitHub README CC BY 4.0); cite both.
