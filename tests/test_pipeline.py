import numpy as np
from nd import env, model, data, synth, pipeline, stats


def stub_official(policy, seeds, mode="train", record_first=0):
    """Replica standing in for the official package (the real G0b runs in Kaggle)."""
    if record_first:
        r, h = env.rollout(policy, seeds, mode, record=True)
        return r, h["obs"][:record_first]
    return env.rollout(policy, seeds, mode), None


def _cols(n_tr=60, n_va=20):
    tr, _ = synth.make(n_tr, 1); va, _ = synth.make(n_va, 2)
    return tr, va


def test_audit_and_fit_passes_on_synthetic_and_freezes():
    tr, va = _cols()
    rep, frozen = pipeline.audit_and_fit(tr, va)
    assert rep["g0a_pass"] and rep["ident_pass"] and frozen is not None
    assert rep["episodes_train"] == 60 and rep["episode_boundaries"] == "flags"
    assert rep["ident"]["audit_vs_physics"]["max_abs_A"] < 1e-5
    cfg, blob, sha = frozen
    assert sha == rep["frozen_sha256"] and len(cfg["table"]) == 35


def test_g0a_fails_on_real_expectations_for_synthetic_data():
    tr, va = _cols()
    rep, frozen = pipeline.audit_and_fit(tr, va, expect=pipeline.EXPECT_REAL)
    assert not rep["g0a_pass"] and frozen is None


def test_g0a_fails_when_action_out_of_range_or_reward_corrupt():
    tr, va = _cols()
    bad = dict(tr); bad["action_0"] = tr["action_0"] * 30
    assert not pipeline.audit_and_fit(bad, va)[0]["g0a_pass"]
    bad = dict(tr); bad["reward"] = tr["reward"] + 0.01
    assert not pipeline.audit_and_fit(bad, va)[0]["g0a_pass"]


def test_ident_gate_fails_on_nonlinear_data():
    tr, va = _cols()
    bad = dict(tr); bad["next_obs_2"] = tr["next_obs_2"] + 0.01 * np.sin(100 * tr["obs_0"])
    rep, frozen = pipeline.audit_and_fit(bad, va)
    assert not rep["ident_pass"] and frozen is None


def test_g0b_with_replica_stub_passes_and_detects_a_difference():
    r = pipeline.g0b(stub_official)
    assert r["pass"] and r["policies"]["pd"]["max_abs_obs_diff"] == 0.0

    def off(policy, seeds, mode="train", record_first=0):
        a, b = stub_official(policy, seeds, mode, record_first)
        return a + 0.01, b
    assert not pipeline.g0b(off)["pass"]


def test_bc_gate_band_is_report_only():
    assert pipeline.bc_gate(60.0)["in_band"] and not pipeline.bc_gate(40.0)["in_band"] and not pipeline.bc_gate(80.0)["in_band"]
    assert pipeline.bc_gate(40.0)["report_only"] and "pass_" not in pipeline.bc_gate(40.0)


def test_regime_check_reported_on_synthetic():
    tr, va = _cols()
    rep, frozen = pipeline.audit_and_fit(tr, va)
    rc = rep["regime_check"]
    assert rc["n_visited"] == 20 * 100 and rc["lev_train_max"] > 0 and rc["lev_ratio_max"] > 0 and len(rc["max_range_excess_per_coord"]) == 6
    assert np.isfinite(rep["selected_val_score_normalised"])
    X = np.random.RandomState(0).randn(500, 6)
    far = model.regime_check(X, 10 * X[:50])                    # far outside the training cloud: leverage ratio blows up, range excess positive
    assert far["lev_ratio_max"] > 50 and far["frac_visited_above_train_max"] > 0.5 and max(far["max_range_excess_per_coord"]) > 0
    near = model.regime_check(X, X[:50])
    assert near["lev_ratio_max"] <= 1.0 + 1e-9 and near["frac_visited_above_train_max"] == 0.0


def test_final_flow_with_stub():
    tr, va = _cols()
    rep, (cfg, blob, sha) = pipeline.audit_and_fit(tr, va)
    pol = model.policy_from_config(cfg)
    sets = {s: pipeline.eval_set(stub_official, pol, s, n=100) for s in (7, 42, 210)}
    assert all(v["max_abs_return_diff"] == 0.0 for v in sets.values())
    res = stats.claim_rule({s: v["official"] for s, v in sets.items()}, True, True, max(v["max_abs_return_diff"] for v in sets.values()))
    assert not res["claim"] and any("100 episodes" in b for b in res["blocked_by"])     # n = 100 can never report a claim (Amendment 1)
    noff = pipeline.eval_set(stub_official, model.policy_from_config(cfg, use_ff=False), 7, n=50)
    assert noff["official"].shape == (50,)
    s = pipeline.summarize(sets[7]); assert s["ci95"][0] < s["mean"] < s["ci95"][1]
