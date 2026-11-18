# FINAL (CPU): scores the frozen controller on the official simulator, 3 seed sets x 1000 episodes, every episode also with the numpy replica. Rows P (primary) are written
# before the secondary rows S (mass 30), R (oracle) and A (no feed-forward). Aborts unless G0 passed (G0a, G0b) and the frozen configuration has the sha256 committed before the push (CFG["frozen_sha256"]).
# n = 1000 and seeds {7, 42, 210} are hard-coded here and again in nd/stats.claim_rule; the sha256 of THIS script is bound in TEST-LOG before the push (STATUS.md runbook).
import hashlib, json, os, time
import numpy as np
t0 = time.time()
from nd import official, pipeline, model, env, stats, kernel_util as ku
ku.arm_wall_cap(int(CFG["wall_cap_s"]))
OUT = "/kaggle/working"; os.makedirs(OUT, exist_ok=True)
prep = ku.find_dir("prep_report.json"); g0 = ku.find_dir("g0_report.json")
g0r = json.load(open(g0 + "/g0_report.json"))
if not g0r.get("pass_"):
    raise SystemExit("G0 did not pass: nothing is scored")
blob = open(prep + "/frozen_config.json").read(); sha = hashlib.sha256(blob.encode()).hexdigest()
want = str(CFG.get("frozen_sha256", ""))
if len(want) != 64 or want != sha:
    raise SystemExit(f"frozen configuration sha256 {sha} does not match the committed one {want!r}: regenerate this kernel with the committed sha (STATUS.md)")
frozen = json.loads(blob); frozen_ok = True
inst = official.install(open("/tmp/pkg/nd/sources.sha256").read()); ver = official.versions()
N = 1000; SEEDS = (7, 42, 210)   # HARD-CODED (Amendment 1): the claim rule in nd/stats.py requires exactly these; CFG cannot change them
cand = model.policy_from_config(frozen)
try:
    script_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()    # best effort: compare with the sha256 bound in TEST-LOG before the push
except Exception as e:
    script_sha = "unavailable: " + repr(e)[:80]
log = [dict(row="start", utc=ku.utc(), frozen_sha256=sha, script_sha256=script_sha, versions=ver, n=N)]


def run_rows(tag, policy, mode):
    sets = {}
    for s in SEEDS:
        t1 = time.time(); es = pipeline.eval_set(official.returns, policy, s, n=N, mode=mode); sets[s] = es
        np.savetxt(f"{OUT}/{tag}_scores_seed{s}.csv", np.stack([es["official"], es["replica"]], axis=1), delimiter=",", header="official_normalised,replica_normalised", comments="")
        log.append(dict(row=f"{tag}-{s}", seconds=round(time.time() - t1, 1), utc=ku.utc(), **pipeline.summarize(es)))
        print(tag, s, log[-1], flush=True)
        ku.write_json(OUT + "/testlog_rows.json", log)
    return sets


P = run_rows("P", cand, "train")
maxdiff = max(v["max_abs_return_diff"] for v in P.values())
claim = stats.claim_rule({s: v["official"] for s, v in P.items()}, True, frozen_ok, maxdiff)
ku.write_json(OUT + "/p_results.json", dict(claim_rule=claim, frozen_sha256=sha, g0=g0r, utc=ku.utc()))
print("PRIMARY", json.dumps(claim, default=float), flush=True)
S = run_rows("S", cand, "test")                                  # secondary: mass 30, not a claim
At, Bt = model.true_system(); orc, _ = model.design(At, Bt, frozen["qp"], frozen["qv"])
R = run_rows("R", orc, "train")                                  # reference: true-parameter design with the frozen weights
Aab = run_rows("A", model.policy_from_config(frozen, use_ff=False), "train")   # reference: no target feed-forward
summ = lambda sets: {str(s): pipeline.summarize(v) for s, v in sets.items()}
pred = dict(val_model_return_raw=[t["val_return"] for t in frozen["table"] if t["qp"] == frozen["qp"] and t["qv"] == frozen["qv"]][0])
pred["val_model_score_normalised"] = float(env.normalize(pred["val_model_return_raw"]))
pred["official_P_mean_normalised"] = {str(s): float(np.mean(v["official"])) for s, v in P.items()}   # ex-post regime check: identified-model prediction vs the simulator
res = dict(frozen_sha256=sha, script_sha256=script_sha, n=N, model_vs_simulator=pred, claim_rule=claim, P=summ(P), S_mass30=summ(S), R_oracle=summ(R), A_no_feedforward=summ(Aab), versions=ver, installed_sha256=inst, seconds=round(time.time() - t0), utc=ku.utc())
ku.write_json(OUT + "/results.json", res)
print("FINAL DONE; claim =", claim["claim"], flush=True)
