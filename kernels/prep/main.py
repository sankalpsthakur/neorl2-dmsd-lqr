# PREP (CPU, no simulator, no secret): downloads the two pinned Hub parquet files (public, not gated), verifies size and sha256, runs G0a, the identification gate and the
# validation-only selection (PREREGISTRATION sections 4 and 5), and writes the frozen configuration. This kernel never imports neorl2.
import hashlib, io, json, os, time
import numpy as np
t0 = time.time()
from nd import data, model, pipeline, kernel_util as ku
ku.arm_wall_cap(int(CFG["wall_cap_s"]))
OUT = "/kaggle/working"; os.makedirs(OUT + "/data", exist_ok=True)
MAN = json.loads(open("/tmp/pkg/nd/data_manifest.json").read())
REV = MAN["hub_revision"]; LIMIT = 50_000_000


def fetch(url, cap=LIMIT, timeout=120):
    import urllib.request
    b = urllib.request.urlopen(url, timeout=timeout).read(cap + 1)
    if len(b) > cap:
        raise SystemExit("download over the 50 MB cap: " + url)
    return b


def hub_parquet(rel):
    want = MAN["files"][rel]
    b = fetch(f"https://huggingface.co/datasets/polixirai/NeoRL2/resolve/{REV}/{rel}")
    got = dict(bytes=len(b), sha256=hashlib.sha256(b).hexdigest())
    if got["bytes"] != want["bytes"] or got["sha256"] != want["sha256"]:
        raise SystemExit(f"{rel}: size/sha256 differ from the pinned manifest: {got}")
    import pandas as pd
    df = pd.read_parquet(io.BytesIO(b))
    if list(df.columns) != MAN["columns"]:
        raise SystemExit(f"{rel}: unexpected columns {list(df.columns)}")
    return {c: df[c].to_numpy() for c in df.columns}, got


train, tr_info = hub_parquet("DMSD/train/DMSD-train.parquet")
val, va_info = hub_parquet("DMSD/val/DMSD-val.parquet")
np.savez_compressed(OUT + "/data/train.npz", **train); np.savez_compressed(OUT + "/data/val.npz", **val)
print("downloaded and verified", tr_info, va_info, flush=True)


def try_official_npz(name, md5, cols):
    """The official loader reads DMSD-*.npz from plain http; compare with the Hub parquet if the host still answers (recorded, never fatal)."""
    try:
        b = fetch("http://datasets.polixir.site/neorl2/DMSD/" + name, cap=40_000_000, timeout=30)
    except Exception as e:
        return dict(reachable=False, error=repr(e)[:200])
    res = dict(reachable=True, bytes=len(b), md5=hashlib.md5(b).hexdigest()); res["md5_ok"] = bool(res["md5"] == md5)
    try:
        z = np.load(io.BytesIO(b)); res["keys"] = sorted(z.files)
        for k, names in (("obs", [f"obs_{i}" for i in range(6)]), ("next_obs", [f"next_obs_{i}" for i in range(6)]), ("action", ["action_0", "action_1"])):
            ref = np.stack([cols[n] for n in names], axis=1)
            res["max_abs_diff_" + k] = float(np.max(np.abs(np.asarray(z[k], np.float64).reshape(len(ref), -1) - ref))) if z[k].shape[0] == len(ref) else "shape differs"
    except Exception as e:
        res["compare_error"] = repr(e)[:200]
    return res


npz = dict(train=try_official_npz("DMSD-train.npz", MAN["official_npz_md5_info_py"]["DMSD-train.npz"], train), val=try_official_npz("DMSD-val.npz", MAN["official_npz_md5_info_py"]["DMSD-val.npz"], val))
rep, frozen = pipeline.audit_and_fit(train, val, expect=pipeline.EXPECT_REAL, extra=dict(hub_revision=REV, train_sha256=tr_info["sha256"], val_sha256=va_info["sha256"]))
rep.update(hub_files=dict(train=tr_info, val=va_info), official_npz=npz, seconds=round(time.time() - t0), utc=ku.utc())
if frozen:
    cfg, blob, sha = frozen
    open(OUT + "/frozen_config.json", "w").write(blob); open(OUT + "/frozen_config.sha256", "w").write(sha + "\n")
    pol = model.policy_from_config(cfg); o = np.zeros((1, 6)); t1 = time.perf_counter()
    for _ in range(20000):
        pol(o)
    rep["edge"] = dict(controller_numbers=int(np.size(cfg["K"]) + np.size(cfg["F"])), identified_numbers=int(np.size(cfg["A"]) + np.size(cfg["B"])), float32_bytes=4 * int(np.size(cfg["K"]) + np.size(cfg["F"])),
                       numpy_latency_us_per_call_kaggle_cpu=round((time.perf_counter() - t1) / 20000 * 1e6, 1))
ku.write_json(OUT + "/prep_report.json", rep)
print(json.dumps({k: rep[k] for k in ("g0a_pass", "ident_pass", "data_score", "data_raw_return", "episodes_train", "episodes_val", "action_saturated_frac", "selected", "frozen_sha256") if k in rep}, default=float), flush=True)
if not (rep["g0a_pass"] and rep["ident_pass"]):
    raise SystemExit("GATE FAILED (G0a or identification): see prep_report.json; nothing frozen")
print("PREP DONE: commit frozen_config.json and its sha256 BEFORE pushing g0 and final", flush=True)
