# G0 (CPU; G0a and G0b hard, G0c report-only per Amendment 1): the first and only part of the program that contacts the official simulator before the candidate. G0b = replica vs official package on identical seeds; G0c = BC scale anchor.
# Reads the prep output (kernel source). No secret.
import hashlib, json, os, time
import numpy as np
t0 = time.time()
from nd import official, bc, pipeline, data, env, kernel_util as ku
ku.arm_wall_cap(int(CFG["wall_cap_s"]))
OUT = "/kaggle/working"; os.makedirs(OUT, exist_ok=True)
prep = ku.find_dir("prep_report.json")
prep_rep = json.load(open(prep + "/prep_report.json"))
if not (prep_rep["g0a_pass"] and prep_rep["ident_pass"]):
    raise SystemExit("prep gates did not pass: nothing is scored")
inst = official.install(open("/tmp/pkg/nd/sources.sha256").read())
ver = official.versions(); print("installed official files at", official.COMMIT, ver, flush=True)
rep = dict(official_commit=official.COMMIT, installed_sha256=inst, versions=ver, g0a_pass=prep_rep["g0a_pass"], g0a_data_score=prep_rep["data_score"], g0a_data_raw=prep_rep["data_raw_return"])
rep["g0b"] = pipeline.g0b(official.returns); print("G0b", json.dumps(rep["g0b"], default=float), flush=True)
ku.write_json(OUT + "/g0_report.json", dict(rep, pass_=False, partial=True))      # the simulator contact is on record even if a later part fails
if rep["g0b"]["pass"]:
    tr = np.load(prep + "/data/train.npz"); d = data.from_columns({k: tr[k] for k in tr.files})
    policy, losses = bc.train_bc(d["obs"], d["action"], epochs=int(CFG["bc_epochs"]))
    seeds = [env.ep_seed(7, i) for i in range(int(CFG["n"]))]
    orr, _ = official.returns(policy, seeds)
    rep["g0c"] = dict(pipeline.bc_gate(float(np.mean(env.normalize(orr)))), n=len(seeds), final_train_mse=float(losses[-1]), epochs=int(CFG["bc_epochs"]), replica_score=float(np.mean(env.normalize(env.rollout(policy, seeds)))),
                      note="own torch MLP, not the OfflineRL package; sanity check of the scorer's scale")
    print("G0c", rep["g0c"], flush=True)
rep["pass_"] = bool(rep["g0a_pass"] and rep["g0b"]["pass"])   # G0c is report-only (Amendment 1); rep["seconds"] = round(time.time() - t0); rep["utc"] = ku.utc()
ku.write_json(OUT + "/g0_report.json", rep)
print("G0 PASS (G0a, G0b); G0c in band:", rep.get("g0c", {}).get("in_band"), flush=True) if rep["pass_"] else print("G0 FAILED: stop and report", flush=True)
if not rep["pass_"]:
    raise SystemExit("G0 failed")
