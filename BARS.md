# BARS: the published number on NeoRL-2 DMSD, verified from primary sources

Checked 2026-11-18 (UTC). "Not found" means searched and not found, not "does not exist". Nothing here is a result of ours. Verbatim copies of the source files read are in `protocol/sources/` (hashes in `protocol/sources.sha256`); they are Apache-2.0 files of the NeoRL2 repository, kept only to pin what was read.

## 1. The bar

| item | verified value | source and quote |
|---|---|---|
| benchmark | NeoRL-2 (Polixir / Nanjing University), task DMSD (double mass-spring-damper), offline control | arXiv 2503.19267 (one version, 2025-03-25), https://arxiv.org/abs/2503.19267 |
| bar | **EDAC 78.65** normalised score, "+- 2.30" | Table 2, DMSD row, read in https://arxiv.org/html/2503.19267v1: "DMSD 56.64 65.09 ± 1.64 70.24 ± 1.12 78.65 ± 2.30 77.75 ± 1.16 60.01 ± 0.79 68.23 ± 0.65 68.33 ± 0.41 76.22 ± 1.92 64.36 ± 0.78" (columns Data, BC, CQL, EDAC, MCQ, TD3BC, MOPO, COMBO, RAMBO, MOBILE) |
| inconsistency | the text says 78.56, the table says 78.65 | text: "the EDAC algorithm's performance increases to 78.56, representing an improvement of about 22 scores". The table value is the one the repository CSV reproduces (below); we use 78.65 |
| second and third | MCQ 77.75; RAMBO 76.22 | same Table 2 row |
| later paper | ACPO 74.9 +- 5.67 (below the bar) | arXiv 2601.23010 Table 3, https://arxiv.org/html/2601.23010v1: "DMSD 56.6 65.1 70.2 78.7 77.8 60.0 76.2 64.4 74.9 ± 5.67"; caption "Scores are reported as mean ± standard deviation" (ACPO's sd; the NeoRL-2 paper calls its +- "standard error") |
| Hub card | repeats 78.7 +- 2.3 for EDAC | https://huggingface.co/datasets/polixirai/NeoRL2 (README, DMSD row) |
| solved threshold | "A normalized score of 95 can be treated as having solved the task." | NeoRL-2 paper, footnote 2 of Sec. 5; same paragraph: "no algorithm surpasses 95 in any task" |
| model behind the bar | EDAC actor and critic MLPs, hidden 256, 2 layers, 10 or 50 critics, eta 1 or 5, 3000 epochs x 1000 steps | repository log `benchmark/results/DMSD-edac/log.csv` (copy in protocol/sources/edac_log.csv): columns hparams/hidden_layer_size 256, hparams/layer_num 2, hparams/epoch 3000, hparams/step_per_epoch 1000, grid_tune eta {1,5} x num_critics {10,50}. The paper itself states no size (not found in the text read) |
| data policy | 56.64 (raw return 494.4527) | `benchmark/task_score.csv` row DMSD: 56.642698068163774; `task_reward.csv`: 494.45271875 |

### How the bar was produced (read from the repository, not from the paper)
- Normalisation, `neorl2/envs/info.py`: `MIN_SCORES "DMSD": 30.60`, `MAX_SCORES "DMSD": 849.51`; `Env.get_normalized_score` (base.py): `(returns - min_score) / (max_score - min_score) * 100`.
- Arithmetic check we ran: (494.45271875 - 30.60) / 818.91 x 100 = 56.6427, (674.69 - 30.60) / 818.91 x 100 = 78.65, BC raw 563.63 gives 65.09. All three table entries reproduce.
- Data score definition, `benchmark/report_result.py`: `train_data_reward = np.sum(train_data["reward"]) / train_nums` with `train_nums = np.logical_or(done, truncated).sum()`; this is an arithmetic mean of stored trajectory returns over the training file, **not a simulator re-score** (our gate G0a is this arithmetic).
- Hyper-parameters were chosen on the evaluation simulator. Paper: "We run each configuration of the hyperparameters with 3 random seeds and choose the policy at the final training stage to conduct the online test. We report the results from the best hyperparameter across 3 seeds". Code: `launch_task.py` tunes `metric='Reward_Mean_Env', mode='max'` on the online return; `report_result.py` takes the `idxmax` of the per-configuration mean. We recomputed from `DMSD-edac/log.csv`: eta=1, critics=10 gives seed values 693.8, 656.2, 674.1, mean 674.69, sample std 18.81 (= 2.30 normalised); the other three cells give 77.62, 69.69, 69.23. The bar is therefore the best of 4 hyper-parameter cells (eta x num_critics), each run with 3 seeds (12 runs), picked on the test simulator, so it is optimistic.
- The "+- 2.30" is the **sample standard deviation of the three seed means** (pandas `.std()`, ddof 1: 18.81 / 818.91 x 100), although the paper labels it "standard error over three random seeds". The standard error of three seeds would be about 1.33.
- Evaluation episodes per number: **unresolved**. `launch_task.py` passes `number_of_runs=1000` to `callback.initialize`; `OnlineCallBackFunction.initialize` defaults to 100; the logged `hparams/eval_episodes` is 100 in all 12 EDAC rows. Which one produced Table 2 is not determinable from what was read. Our scorer uses 1000 per seed.
- Environment, `neorl2/__init__.py`: `register(id='DMSD', entry_point="neorl2.envs.dmsd:DoubleMassSpringDamperEnv", kwargs=dict(mode='train'), max_episode_steps=100)`. Mode `train` gives mass 20; mode `test` gives mass 30 and is not used by any benchmark script read (no use of `mode='test'` found in launch_task.py, report_result.py, launch_domain.py; the evaluation environment is built by `offlinerl.utils.env.get_env(task)` inside the OfflineRL package, which was **not read**, so that the baselines were evaluated in mode `train` is an inference). **Update 2026-11-18 (red-team report, not re-read by the builder):** on OfflineRL `master` (not the 2025 version used for the paper), `get_env("DMSD")` does `gym.make("DMSD")` (mode `train`) and `test_on_real_env` sums rewards over unseeded `env.reset()` episodes until `done or truncated`, mean over `number_of_runs` (default 100). That resolves, for `master` only, the mode and the metric (plain mean episode return); the 2025 version is unverified.
- Observation (6): position(2), velocity(2), target(2). Action (2) in [-1, 1], mapped to force `(a+1)/2*60-30` (so force = 30 a, bounds +-30). Reward `sum(-log(|position - target| + 0.01))` of `next_obs` (`dmsd_reward.py`, float32 arithmetic). dt 0.2, damping 4, stiffness 2, mass 20, linear and noise-free. `reset(seed=None)` draws a fresh `RandomState()`; **evaluation is not reproducible unless a seed is passed**.
- Dataset: 100,000 train and 20,000 val transitions (paper Table 4: "DMSD 6 2 False 100 100000 20000"); behaviour = PID controllers ("Rule-based behaviour policy ... DMSD" in the Hub card; a PID tune log `neorl2/envs/data/pid/DMSD-v0_tune_log.json` exists, not read).

## 2. Metric (the claim metric)
Normalised score = (mean episode return - 30.60) / (849.51 - 30.60) x 100, episode return = sum over 100 steps of the env reward, mean over evaluation episodes. Seeds 7, 42 and 210 are the benchmark's seeds; our controller is deterministic, so they only index three independent sets of 1000 evaluation episodes (see protocol/PREREGISTRATION.md).

## 3. Licence terms (quoted) and what may be published
| object | terms | source |
|---|---|---|
| dataset on the Hub | `license: apache-2.0` in the card header; `gated: false`; revision `d528956f2d4c3aaefe7314f5615307214841bbcc` | https://huggingface.co/api/datasets/polixirai/NeoRL2 (read 2026-11-18) |
| dataset per GitHub | "All datasets are licensed under the Creative Commons Attribution 4.0 License (CC BY), and code is licensed under the Apache 2.0 License." | https://github.com/polixir/NeoRL2 README. **Conflicts with the Hub card.** |
| code | the repository tree (read 2026-11-18, commit b50d91d0a65916a89342a4f6978188b17cb2d97a, committed 2025-10-27) has no LICENSE file at top level (entries: .gitignore, README.md, benchmark, neorl2, setup.py, test); the README badge and sentence say Apache 2.0 | GitHub tree API |
| the paper | CC BY-NC-ND 4.0 (paper text only, not data or code) | https://arxiv.org/abs/2503.19267 (licence link on the abstract page) |
Handling: both licences allow redistribution of results derived from the data with attribution, and a controller of 12 numbers contains no copy of the data. Cite both statements, attribute to Gao, Tu, Qin, Sun, Chen, Yu (NeoRL-2), and treat CC BY 4.0 as the stricter. Short quotes of the paper only (the paper is NC-ND). We do not redistribute the data or the repository code; copies in `protocol/sources/` are for hash pinning inside this working repository and must not be published.
**What may be published (by the lead, after verification):** the 12 numbers of the frozen controller and the identified A, B (about 36 numbers), our scorer and tests (our code), the pre-registration, TEST-LOG and result tables, and a card whose claim sentence follows PREREGISTRATION item 3. Nothing is uploaded by an agent.

## 4. Searched, not found, not verified
- A newer or higher DMSD result: none found by the skeptic (arXiv 2512.04341 and 2603.08118 report no DMSD number) nor by us; our arXiv API query returned no entries, so it adds no coverage. No official leaderboard found. The search is not exhaustive: re-run it before any dated claim.
- PIQL (arXiv 2501.08907) DMSD 54.2 +- 0.1 is quoted by the scout and was not re-read by us.
- The paper text says nothing on how the +- was computed beyond "standard errors computed from three random seeds"; the sd reading rests on `report_result.py` and the EDAC log (recomputed above).
- Whether `DMSD-v0_tune_log.json` describes the PID behaviour policy: not read.
- The HF parquet equals the official `DMSD-train.npz` (MD5 in info.py `88dfa39a...`, hosted at `http://datasets.polixir.site/neorl2`): not verified; the prep kernel tries to fetch the npz and compare, and records the outcome whatever it is.
