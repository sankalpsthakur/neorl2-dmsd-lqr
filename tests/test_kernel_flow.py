"""Runs the four kernel scripts locally with stubs: the replica stands in for the official package, no network, no pandas, no torch (BC is stubbed), outputs in a tmp dir.
Checks the control flow, file names, gates and the sha256 binding between the prep output and the final kernel."""
import hashlib, json, pathlib, types
import numpy as np
import pytest
from nd import env, official, bc, pipeline, kernel_util, synth, model

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARMED = []


def stub_returns(policy, seeds, mode="train", record_first=0):
    if record_first:
        r, h = env.rollout(policy, seeds, mode, record=True)
        return r, h["obs"][:record_first]
    return env.rollout(policy, seeds, mode), None


def run_script(kernel, tmp, cfg, subs=(), extra_ns=None):
    src = (ROOT / "kernels" / kernel / "script.py").read_text()
    body_marker = "# ---------------------------------------------------------------- main.py\n"
    body = src.split(body_marker, 1)[1]
    for a, b in subs:
        assert a in body, a
        body = body.replace(a, b)
    body = body.replace('"/kaggle/working"', repr(str(tmp)))
    body = body.replace("/tmp/pkg/nd/data_manifest.json", str(ROOT / "protocol" / "data_manifest.json")).replace("/tmp/pkg/nd/sources.sha256", str(ROOT / "protocol" / "sources.sha256"))
    ns = dict(CFG=cfg, __name__="__main__"); ns.update(extra_ns or {})
    exec(compile(body, kernel, "exec"), ns)
    return ns


@pytest.fixture
def stubs(monkeypatch, tmp_path):
    monkeypatch.setattr(official, "install", lambda t, dest=None: {"stub": "x"})
    monkeypatch.setattr(official, "versions", lambda: {"stub": "1"})
    monkeypatch.setattr(official, "returns", stub_returns)
    monkeypatch.setattr(bc, "train_bc", lambda obs, act, epochs=3, **k: (lambda o: np.clip(2.0 * (np.atleast_2d(o)[:, 4:6] - np.atleast_2d(o)[:, :2]) - 1.0 * np.atleast_2d(o)[:, 2:4], -1, 1), [0.1]))
    monkeypatch.setattr(pipeline, "BC_BAND", (-1e9, 1e9))
    monkeypatch.setattr(kernel_util, "arm_wall_cap", lambda s: ARMED.append(s))        # never arm a real SIGALRM inside the pytest process
    ARMED.clear()
    return tmp_path


def test_dryrun(stubs):
    run_script("dryrun", stubs, dict(n=30, bc_epochs=1, wall_cap_s=900))
    rep = json.load(open(stubs / "dryrun_report.json"))
    assert rep["NOTE"].startswith("SYNTHETIC") and rep["g0b"]["pass"]
    assert rep["claim_rule_code_path"]["claim"] is False and rep["claim_rule_code_path"]["blocked_by"]    # dry-run seeds and n can never claim
    assert ARMED == [900]


def test_prep_g0_final_chain(stubs, monkeypatch):
    prep_dir = stubs / "prep"; g0_dir = stubs / "g0"; fin_dir = stubs / "final"
    subs = [('train, tr_info = hub_parquet("DMSD/train/DMSD-train.parquet")', 'train, tr_info = SYN_TR, dict(bytes=0, sha256="syn")'),
            ('val, va_info = hub_parquet("DMSD/val/DMSD-val.parquet")', 'val, va_info = SYN_VA, dict(bytes=0, sha256="syn")'),
            ('npz = dict(train=try_official_npz("DMSD-train.npz", MAN["official_npz_md5_info_py"]["DMSD-train.npz"], train), val=try_official_npz("DMSD-val.npz", MAN["official_npz_md5_info_py"]["DMSD-val.npz"], val))', 'npz = {}'),
            ("expect=pipeline.EXPECT_REAL", "expect=None")]
    tr, _ = synth.make(60, 1); va, _ = synth.make(20, 2)
    run_script("prep", prep_dir, dict(wall_cap_s=1800), subs, dict(SYN_TR=tr, SYN_VA=va))
    assert (prep_dir / "prep_report.json").exists() and (prep_dir / "data" / "train.npz").exists()
    sha = (prep_dir / "frozen_config.sha256").read_text().strip()
    assert sha == hashlib.sha256((prep_dir / "frozen_config.json").read_bytes()).hexdigest()
    rep = json.load(open(prep_dir / "prep_report.json")); assert rep["edge"]["controller_numbers"] == 12 and rep["edge"]["identified_numbers"] == 24
    monkeypatch.setattr(kernel_util, "find_dir", lambda marker, **k: str({"prep_report.json": prep_dir, "g0_report.json": g0_dir}[marker]))
    run_script("g0", g0_dir, dict(n=30, bc_epochs=1, wall_cap_s=3600))
    g0r = json.load(open(g0_dir / "g0_report.json")); assert g0r["pass_"] is True and g0r["g0c"]["report_only"]
    with pytest.raises(SystemExit, match="does not match"):                        # placeholder sha is refused
        run_script("final", fin_dir, dict(frozen_sha256="SET_AFTER_PREP", wall_cap_s=3600), SMALL_N)
    run_script("final", fin_dir, dict(frozen_sha256=sha, wall_cap_s=3600), SMALL_N)
    res = json.load(open(fin_dir / "results.json"))
    assert res["claim_rule"]["claim"] is False and res["n"] == 20                       # a kernel with n != 1000 cannot claim
    assert res["script_sha256"] and res["model_vs_simulator"]["val_model_score_normalised"] is not None
    assert res["frozen_sha256"] == sha and set(res["P"]) == {"7", "42", "210"} and res["claim_rule"]["replica_max_diff"] == 0.0
    rows = [r["row"] for r in json.load(open(fin_dir / "testlog_rows.json"))]
    assert rows == ["start"] + [f"{t}-{s}" for t in "PSRA" for s in (7, 42, 210)]    # P rows are written before S, R, A
    assert (fin_dir / "P_scores_seed7.csv").exists()


SMALL_N = [("N = 1000; SEEDS", "N = 20; SEEDS")]       # test-only text substitution in the script body, to run the flow fast; the rule must still refuse to claim


def test_g0c_out_of_band_does_not_fail_g0(stubs, monkeypatch):
    prep_dir = stubs / "p"; g0_dir = stubs / "g"; prep_dir.mkdir()
    tr, _ = synth.make(30, 1); np.savez(prep_dir / "train.npz", **tr); (prep_dir / "data").mkdir(); np.savez(prep_dir / "data" / "train.npz", **tr)
    (prep_dir / "prep_report.json").write_text(json.dumps(dict(g0a_pass=True, ident_pass=True, data_score=56.6, data_raw_return=494.4)))
    monkeypatch.setattr(kernel_util, "find_dir", lambda marker, **k: str(prep_dir))
    monkeypatch.setattr(pipeline, "BC_BAND", (1e6, 2e6))                       # BC score certainly outside the band
    run_script("g0", g0_dir, dict(n=20, bc_epochs=1, wall_cap_s=3600))
    r = json.load(open(g0_dir / "g0_report.json"))
    assert r["g0c"]["in_band"] is False and r["pass_"] is True


def test_final_script_hard_codes_n_and_seeds_and_has_no_cfg_n():
    src = (ROOT / "kernels" / "final" / "script.py").read_text()
    body = src.split("# ---------------------------------------------------------------- main.py\n", 1)[1]
    assert "N = 1000; SEEDS = (7, 42, 210)" in body and 'CFG["n"]' not in body and "CFG.get(\"n\"" not in body
    cfg = json.loads(eval([l for l in src.splitlines() if l.startswith("CFG = ")][0][len("CFG = json.loads("):-1]))
    assert "n" not in cfg and cfg["wall_cap_s"] == 3600


def test_arm_wall_cap_stops_the_script():
    import signal, time
    ku = kernel_util
    try:
        ku.arm_wall_cap(1)
        with pytest.raises(SystemExit, match="WALL CAP"):
            time.sleep(3)
    finally:
        signal.alarm(0)


def _sh(args, env_extra, cwd=ROOT):
    import subprocess, os
    e = dict(os.environ); e.update(env_extra)
    return subprocess.run(args, cwd=cwd, env=e, capture_output=True, text=True, timeout=30)


def test_slot_release_and_watch_timeout_leave_no_stale_lock(tmp_path):
    locks = tmp_path / "locks"; locks.mkdir(); env_ = dict(LOCKS_DIR=str(locks))
    r = _sh(["scripts/slot.sh", "take", "owner/bench-neorl2-dmsd-final"], env_); n = r.stdout.strip(); assert n == "1", r
    assert (locks / "kaggle-cpu-1" / "owner").read_text().startswith("neorl2dmsd ")
    # a lock of another track is never removed
    (locks / "kaggle-cpu-2").mkdir(); (locks / "kaggle-cpu-2" / "owner").write_text("gas k 2026-11-18T00:00Z\n")
    assert "left alone" in _sh(["scripts/slot.sh", "release", "2"], env_).stdout and (locks / "kaggle-cpu-2").exists()
    # the watcher times out (0 minutes: no kaggle call) -> lock released, kernel-may-still-run warning, non-zero exit
    w = _sh(["scripts/watch_kernel.sh", "owner/bench-neorl2-dmsd-final", n, str(tmp_path / "out"), "0"], env_)
    assert w.returncode == 4 and "TIMEOUT" in w.stderr and "may still be running" in w.stderr
    assert not (locks / "kaggle-cpu-1").exists(), "watcher left a stale lock on timeout"
    # SLOT_MAX_WAIT_S: four taken slots -> take gives up instead of waiting forever
    for i in (1, 2, 3, 4):
        (locks / f"kaggle-cpu-{i}").mkdir(exist_ok=True)
    assert _sh(["scripts/slot.sh", "take", "x"], dict(env_, SLOT_MAX_WAIT_S="1")).returncode == 3


def test_final_push_refused_without_sha_in_committed_testlog():
    r = _sh(["scripts/run_wave.sh", "kernels/final", "owner/bench-neorl2-dmsd-final", "/nonexistent-out", "0"], dict(LOCKS_DIR="/nonexistent-locks"))
    assert r.returncode == 5 and "REFUSED" in r.stderr


def test_final_refuses_when_g0_failed(stubs, monkeypatch):
    d = stubs / "x"; d.mkdir()
    (d / "g0_report.json").write_text(json.dumps(dict(pass_=False))); (d / "prep_report.json").write_text("{}")
    monkeypatch.setattr(kernel_util, "find_dir", lambda marker, **k: str(d))
    with pytest.raises(SystemExit, match="G0 did not pass"):
        run_script("final", stubs / "f", dict(frozen_sha256="0" * 64, wall_cap_s=3600))
