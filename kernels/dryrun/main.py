# DRY RUN (CPU, SYNTHETIC data only, never a result): the whole pipeline of prep, g0 and final on a synthetic dataset collected in the numpy replica, with the OFFICIAL package installed
# at the pinned commit (the official simulator needs no data). Its episode seed sets are 1, 2, 3 (not 7, 42, 210), so no evaluation episode of the real rows is touched. It exercises: hash-pinned install, G0b, the BC code path (3 epochs), audit/identification/selection, evaluation with both scorers, claim rule code.
import hashlib, json, os, time
import numpy as np
t0 = time.time()
from nd import official, bc, pipeline, synth, data, model, env, stats, kernel_util as ku
ku.arm_wall_cap(int(CFG["wall_cap_s"]))
OUT = "/kaggle/working"; os.makedirs(OUT, exist_ok=True)
inst = official.install(open("/tmp/pkg/nd/sources.sha256").read()); ver = official.versions(); print("official installed", ver, flush=True)
tr, _ = synth.make(300, seed=1); va, _ = synth.make(60, seed=2)
rep, frozen = pipeline.audit_and_fit(tr, va)
cfg, blob, sha = frozen
g0b = pipeline.g0b(official.returns); print("G0b", g0b["pass"], flush=True)
d = data.from_columns(tr)
policy, losses = bc.train_bc(d["obs"], d["action"], epochs=int(CFG["bc_epochs"]))
bc_orr, _ = official.returns(policy, [env.ep_seed(1, i) for i in range(50)])
cand = model.policy_from_config(cfg); N = int(CFG["n"])
sets = {s: pipeline.eval_set(official.returns, cand, s, n=N) for s in (1, 2, 3)}
S = pipeline.eval_set(official.returns, cand, 1, n=N, mode="test")
maxdiff = max(v["max_abs_return_diff"] for v in sets.values())
claim = stats.claim_rule({s: v["official"] for s, v in sets.items()}, True, True, maxdiff)   # seeds 1, 2, 3 and n=100: MUST come back claim False (fail closed)
out = dict(NOTE="SYNTHETIC DRY RUN: not a result", versions=ver, installed_sha256=inst, audit=rep, g0b=g0b, bc=dict(final_mse=float(losses[-1]), score_50_episodes=float(np.mean(env.normalize(bc_orr)))),
           cand={str(s): pipeline.summarize(v) for s, v in sets.items()}, mode_test=pipeline.summarize(S), claim_rule_code_path=claim, frozen_sha256=sha, seconds=round(time.time() - t0), utc=ku.utc())
ku.write_json(OUT + "/dryrun_report.json", out)
print(json.dumps({k: out[k] for k in ("g0b",)}, default=float)[:300]); print("DRYRUN DONE; max replica/official diff", maxdiff, "seconds", out["seconds"], flush=True)
if claim["claim"]:
    raise SystemExit("claim rule returned True for a dry-run seed set / n: the hard-coded completeness check is broken")
if not g0b["pass"] or maxdiff > 1e-3:
    raise SystemExit("dry run found a difference between the replica and the official package")
