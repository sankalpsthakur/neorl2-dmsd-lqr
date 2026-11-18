# neorl2-dmsd-lqr: a 12-number identified controller on NeoRL-2 DMSD

A linear model identified from the benchmark's offline data and a linear-quadratic gain with target feed-forward, scored on the official NeoRL-2 simulator. Self-evaluated, not an official ranking. No GPU was used.

## Result

- **Claim, in full (every part belongs to it):** best published normalised score on NeoRL-2 DMSD (arXiv 2503.19267 Table 2, EDAC 78.65): **97.78** (seed sets 7 / 42 / 210: 97.78 / 97.83 / 97.74), trained on the official train file (100,000 transitions) and the official validation file (20,000 transitions; used only for model-rollout selection), official simulator (mode train), 1000 episodes x 3 seed sets [7, 42, 210], self-evaluated, as of 2026-11-18. The model is a 12-number data-identified linear-quadratic controller, not an offline-RL algorithm. Provenance: the controller family and the grid range were informed by sweeps on a true-parameter re-implementation of the evaluation simulator written from the benchmark's public source; the gain actually used was selected only by rollouts of the model identified from the offline files. The published baselines' hyper-parameters were also tuned on the evaluation simulator (best of 4 cells x 3 seeds), and the task is linear and noise-free, so a correct identification is expected to do well: the result is about the method class (identification plus LQR versus offline RL), not a hard-won margin.

| block (1,000 episodes per seed set; mean [95 % bootstrap interval]) | seed set 7 | seed set 42 | seed set 210 | mean |
|---|---|---|---|---|
| **Primary: frozen identified LQR** | 97.78 [97.51, 98.03] | 97.83 [97.56, 98.09] | 97.74 [97.47, 98.00] | **97.78** |
| Reference: LQR designed with the true parameters, same weights (no claim) | 97.78 [97.51, 98.03] | 97.83 [97.56, 98.09] | 97.74 [97.47, 98.00] | 97.78 |
| Ablation: same design without target feed-forward (no claim) | 76.09 [75.56, 76.59] | 76.34 [75.81, 76.87] | 76.15 [75.61, 76.71] | 76.19 |
| Shift: frozen controller with mass 30 instead of 20 (no claim) | 94.26 [93.89, 94.62] | 94.32 [93.95, 94.70] | 94.19 [93.82, 94.56] | 94.26 |

Published scores on the same task: EDAC 78.65 (best), MCQ 77.75, RAMBO 76.22, behaviour cloning 65.09, data policy 56.64 (NeoRL-2 paper, Table 2); later papers report 74.9, 71.21 and 54.2 (`LITERATURE-CHECK.md`). The NeoRL-2 paper treats 95 as solving a task and says no algorithm passes 95 on any task.

## What this does and does not show

- **Expected, not surprising.** DMSD is a linear, noise-free double mass-spring-damper. The offline data identify it almost exactly (validation one-step error 1.2e-7), and the identified controller matches the true-parameter design to two decimals.
- **Feed-forward is the difference.** Without the target feed-forward term the same design scores 76.2, below the offline-RL bar.
- **Not shown:** anything about nonlinear or noisy tasks, or about offline RL in general. The other NeoRL-2 tasks are not linear.
- **Selection.** The baselines were tuned on the evaluation simulator; our controller family was informed by a replica of it. Only the choice among gains used the identified model alone.
- **Open points in the bar.** The paper's text says 78.56 and its table 78.65; its "+-" is a sample standard deviation of three seed means; it is not clear whether its numbers used 100 or 1,000 evaluation episodes. The literature check is not exhaustive.
- **One scored run.** The configuration (`protocol/frozen_config.json`, sha256 `c3f031e99dbdfa12...`) was committed before the simulator was contacted; the final script's hash was bound in the test log before the push; nothing was dropped or repeated (`PROVENANCE.md`, `TEST-LOG.md`). There is no external timestamp.

## Files

| path | what |
|---|---|
| `protocol/PREREGISTRATION.md` | the rule, Amendment 1 (honest provenance, completeness) and Amendment 2 (the freeze) |
| `protocol/frozen_config.json` | the identified A, B, the gain K (2 x 4) and feed-forward F (2 x 2): 12 numbers act on the plant |
| `src/nd/` | numpy replica of the simulator, identification, LQR design, scorer and claim rule |
| `tests/` | 34 tests, numpy only |
| `kernels/`, `scripts/` | the four Kaggle CPU kernels as run and their generators |
| `kernel_outputs/` | reports and per-episode scores of the prep, reproduction and final kernels |
| `BARS.md`, `RESULTS.md`, `TEST-LOG.md`, `LITERATURE-CHECK.md`, `PROVENANCE.md` | bars from primary sources, the result, every evaluation, the literature check, commit order |

## Reproduce

```bash
pip install numpy pytest
python -m pytest -q tests
python -c "import json,sys; sys.path.insert(0,'src'); from nd import env, model; p=model.policy_from_config(json.load(open('protocol/frozen_config.json'))); print([round(float(env.score_seed_set(p,s).mean()),2) for s in (7,42,210)])"
```

The second command re-scores the frozen gain with the replica and prints the three means above. To score on the official simulator, run `kernels/final` as a Kaggle CPU kernel (it fetches five files of github.com/polixir/NeoRL2 at commit b50d91d0 and checks their hashes).

## Licence and attribution

Code: Apache-2.0. Benchmark: NeoRL-2 by Gao, Tu, Qin, Sun, Chen and Yu (arXiv 2503.19267); no NeoRL-2 data or source file is redistributed here; see `NOTICE` for the two upstream licence statements. This is independent work, not affiliated with or endorsed by the NeoRL-2 authors.
