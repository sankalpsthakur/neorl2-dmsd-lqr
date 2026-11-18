"""Stage logic of the kernels, separated from I/O so that it is tested locally on synthetic data with the replica standing in for the official package
(tests/test_pipeline.py). The kernels (kernels/*/main.py) only fetch, call these functions and write files."""
import numpy as np
from . import env, model, data, stats

EXPECT_REAL = dict(rows_train=100000, rows_val=20000, episodes_train=1000, episodes_val=200, data_raw=494.4527, data_raw_tol=0.05)
BC_BAND = (56.64 - 5.0, 65.09 + 5.0)
G0B_TOL_RETURN = 1e-3
G0B_TOL_TRAJ = 1e-9


def audit_and_fit(train_cols, val_cols, expect=None, extra=None):
    """G0a, identification gate and validation-only selection. Never imports neorl2. Returns (report, frozen) where frozen = (cfg, blob, sha) or None when a gate fails."""
    d = data.from_columns(train_cols); v = data.from_columns(val_cols)
    rep = dict(rows_train=len(d["reward"]), rows_val=len(v["reward"]))
    sl, how = data.episode_slices(d); vsl, vhow = data.episode_slices(v)
    rep.update(episode_boundaries=how, episodes_train=len(sl), episodes_val=len(vsl), all_len_100=bool(all(b - a == 100 for a, b in sl + vsl)),
               continuity_train=data.continuity(d, sl), continuity_val=data.continuity(v, vsl))
    for k in ("obs", "action", "next_obs", "reward"):
        rep[f"finite_{k}"] = bool(np.all(np.isfinite(d[k])) and np.all(np.isfinite(v[k])))
    rep["action_max_abs"] = float(max(np.max(np.abs(d["action"])), np.max(np.abs(v["action"]))))
    rep["action_saturated_frac"] = [float(np.mean(np.abs(d["action"][:, j]) >= 1.0)) for j in range(2)]
    rep["target_range"] = [float(d["obs"][:, 4:6].min()), float(d["obs"][:, 4:6].max())]
    raw, k = data.data_score_raw(d)
    rep.update(data_raw_return=raw, data_trajectories=k, data_score=float(env.normalize(raw)))
    rep["reward_recompute_max_abs"] = float(np.max(np.abs(data.reward_from_next_obs(d["next_obs"]) - d["reward"])))
    g0a = (rep["all_len_100"] and rep["continuity_train"] == 0.0 and rep["action_max_abs"] <= 1.0 + 1e-6 and rep["reward_recompute_max_abs"] <= 1e-4
           and all(rep[f"finite_{k_}"] for k_ in ("obs", "action", "next_obs", "reward")))
    if expect:
        g0a = bool(g0a and rep["rows_train"] == expect["rows_train"] and rep["rows_val"] == expect["rows_val"] and rep["episodes_train"] == expect["episodes_train"]
                   and rep["episodes_val"] == expect["episodes_val"] and abs(raw - expect["data_raw"]) <= expect["data_raw_tol"])
        rep["g0a_expected"] = expect
    rep["g0a_pass"] = bool(g0a)
    # identification on train only; audit fits are reported, not used
    A, B, _, dg = model.fit_lti(d["obs"], d["action"], d["next_obs"])
    _, _, c, dgi = model.fit_lti(d["obs"], d["action"], d["next_obs"], intercept=True)
    At, Bt = model.true_system()
    rep["ident"] = dict(train=dg, intercept_fit=dict(max_abs_c=float(np.max(np.abs(c))), **dgi), val_one_step_max_abs=model.one_step_error(A, B, v["obs"], v["action"], v["next_obs"]),
                        audit_vs_physics=dict(max_abs_A=float(np.max(np.abs(A - At))), max_abs_B=float(np.max(np.abs(B - Bt)))), A=A.tolist(), B=B.tolist())
    rep["ident_pass"] = bool(rep["ident"]["val_one_step_max_abs"] <= 1e-3 and rep["ident"]["intercept_fit"]["max_abs_c"] <= 1e-4)
    frozen = None
    if rep["g0a_pass"] and rep["ident_pass"]:
        ve = data.val_episodes(v)
        best, table = model.select(A, B, ve)
        rep["selected"] = best
        frozen = model.freeze(A, B, best, table, extra=dict(n_val_episodes=int(ve[0].shape[0]), **(extra or {})))
        rep["frozen_sha256"] = frozen[2]
        pol_sel, _ = model.design(A, B, best["qp"], best["qv"])
        rep["regime_check"] = model.regime_check(np.concatenate([d["obs"][:, :4], d["action"]], axis=1), model.visited_regressors(A, B, pol_sel, *ve))
        rep["regime_check"]["note"] = "report-only; true one-step error in the LQR regime cannot be measured offline (no data under that policy); see PREREGISTRATION Amendment 1"
        rep["selected_val_score_normalised"] = float(env.normalize(best["val_return"]))
    return rep, frozen


# ---- G0b ---------------------------------------------------------------------------------------------------------------------------------------------------
def pol_zero(o):
    return np.zeros((len(o), 2))


def pol_pd(o):
    return np.clip(3.0 * (o[:, 4:6] - o[:, :2]) - 2.0 * o[:, 2:4], -1, 1)


def pol_bang(o):
    return np.sign(o[:, 4:6] - o[:, :2])


def g0b(official_returns, n=50, n_mode_test=20, seed0=5_000_000, n_traj=5):
    """official_returns(policy, seeds, mode, record_first) -> (returns, obs histories). Compares with the replica."""
    out = dict(policies={}, tol_return=G0B_TOL_RETURN, tol_traj=G0B_TOL_TRAJ)
    ok = True
    for name, pol in (("zero", pol_zero), ("pd", pol_pd), ("bang", pol_bang)):
        seeds = [seed0 + i for i in range(n)]
        orr, ohist = official_returns(pol, seeds, "train", n_traj if name == "pd" else 0)
        rr, rh = env.rollout(pol, seeds, "train", record=True)
        row = dict(n=n, max_abs_return_diff=float(np.max(np.abs(orr - rr))), official_mean=float(np.mean(orr)), replica_mean=float(np.mean(rr)))
        if name == "pd":
            row["max_abs_obs_diff"] = float(np.max(np.abs(ohist - rh["obs"][:n_traj])))
            ok &= row["max_abs_obs_diff"] <= G0B_TOL_TRAJ
        ok &= row["max_abs_return_diff"] <= G0B_TOL_RETURN
        out["policies"][name] = row
        seeds = [seed0 + 1_000 + i for i in range(n_mode_test)]
        orr, _ = official_returns(pol, seeds, "test", 0); rr = env.rollout(pol, seeds, "test")
        row_t = dict(n=n_mode_test, max_abs_return_diff=float(np.max(np.abs(orr - rr))))
        ok &= row_t["max_abs_return_diff"] <= G0B_TOL_RETURN
        out["policies"][name + "_mode_test"] = row_t
    out["pass"] = bool(ok)
    return out


def bc_gate(score_seed7):
    """G0c is REPORT-ONLY (Amendment 1): `in_band` is recorded, never a stop rule. The band is not derived from the paper (own BC, different implementation)."""
    return dict(score=float(score_seed7), band=list(BC_BAND), in_band=bool(BC_BAND[0] <= score_seed7 <= BC_BAND[1]), report_only=True)


# ---- evaluation rows ---------------------------------------------------------------------------------------------------------------------------------------
def eval_set(official_returns, policy, s, n=1000, mode="train"):
    """Scores every episode of the seed set s with BOTH the official simulator and the replica."""
    seeds = [env.ep_seed(s, i) for i in range(n)]
    orr, _ = official_returns(policy, seeds, mode, 0)
    rr = env.rollout(policy, seeds, mode)
    return dict(seed_set=int(s), mode=mode, n=n, official=env.normalize(orr), replica=env.normalize(rr), max_abs_return_diff=float(np.max(np.abs(orr - rr))))


def summarize(es):
    x = es["official"]
    lo, hi = stats.bootstrap_ci(x, 10000, seed=int(es["seed_set"]))
    return dict(seed_set=es["seed_set"], mode=es["mode"], n=es["n"], mean=float(x.mean()), sd=float(x.std(ddof=1)), ci95=[lo, hi], replica_mean=float(es["replica"].mean()), max_abs_return_diff=es["max_abs_return_diff"])
