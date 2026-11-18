import numpy as np
from nd import env, model, data, synth, stats


def _synth(n=60, seed=1):
    cols, _ = synth.make(n, seed)
    return data.from_columns(cols)


def test_synth_dataset_structure_and_data_score():
    d = _synth()
    sl, how = data.episode_slices(d)
    assert how == "flags" and len(sl) == 60 and all(b - a == 100 for a, b in sl)
    assert data.continuity(d, sl) == 0.0
    raw, k = data.data_score_raw(d); assert k == 60
    assert abs(raw - np.sum(d["reward"]) / 60) < 1e-9
    assert np.max(np.abs(data.reward_from_next_obs(d["next_obs"]) - d["reward"])) < 1e-4


def test_blocks_fallback_when_no_flags():
    d = _synth(5)
    d["truncated"][:] = 0
    sl, how = data.episode_slices(d)
    assert how == "blocks" and len(sl) == 5


def test_identification_recovers_true_system():
    d = _synth(60)
    A, B, c, dg = model.fit_lti(d["obs"], d["action"], d["next_obs"])
    At, Bt = model.true_system()
    assert np.max(np.abs(A - At)) < 1e-5 and np.max(np.abs(B - Bt)) < 1e-5 and dg["max_abs"] < 1e-5
    A2, B2, c2, _ = model.fit_lti(d["obs"], d["action"], d["next_obs"], intercept=True)
    assert np.max(np.abs(c2)) < 1e-5
    assert model.one_step_error(A, B, d["obs"], d["action"], d["next_obs"]) < 1e-5


def test_riccati_is_stabilising_and_matches_lqr_optimality():
    At, Bt = model.true_system()
    K = model.dare_gain(At, Bt, np.diag([10, 10, 1, 1.0]), np.eye(2))
    assert np.max(np.abs(np.linalg.eigvals(At - Bt @ K))) < 1
    # optimality: gain satisfies K = (R + B'PB)^-1 B'PA for the converged P; check against a perturbed gain on a quadratic cost
    def cost(K_, x0=np.array([1.0, -1, 0.2, 0.1])):
        x = x0.copy(); J = 0.0; Q = np.diag([10, 10, 1, 1.0])
        for _ in range(2000):
            u = -K_ @ x; J += x @ Q @ x + u @ u; x = At @ x + Bt @ u
        return J
    assert cost(K) <= cost(K * 1.05) + 1e-9 and cost(K) <= cost(K * 0.95) + 1e-9


def test_feedforward_holds_equilibrium():
    At, Bt = model.true_system()
    F = model.feedforward(At, Bt)
    for tgt in (np.array([1.0, -0.5]), np.array([-1.5, 1.5])):
        x = np.concatenate([tgt, [0, 0]])
        a = F @ tgt
        assert np.max(np.abs(At @ x + Bt @ a - x)) < 1e-9


def test_selection_is_validation_only_and_deterministic():
    d = _synth(80)
    A, B, _, _ = model.fit_lti(d["obs"][:6000], d["action"][:6000], d["next_obs"][:6000])
    vd = {k: v[6000:] for k, v in d.items()}
    ve = data.val_episodes(vd)
    b1, t1 = model.select(A, B, ve); b2, t2 = model.select(A, B, ve)
    assert b1 == b2 and len(t1) == 35
    cfg, blob, sha = model.freeze(A, B, b1, t1)
    assert len(sha) == 64 and np.array(cfg["K"]).shape == (2, 4) and np.array(cfg["F"]).shape == (2, 2)
    assert sha == model.freeze(A, B, b2, t2)[2]


def test_identified_controller_matches_oracle_on_replica_and_beats_bar():
    d = _synth(80)
    A, B, _, _ = model.fit_lti(d["obs"][:6000], d["action"][:6000], d["next_obs"][:6000])
    ve = data.val_episodes({k: v[6000:] for k, v in d.items()})
    best, _ = model.select(A, B, ve)
    pol, _ = model.design(A, B, best["qp"], best["qv"])
    At, Bt = model.true_system(); orc, _ = model.design(At, Bt, best["qp"], best["qv"])
    a = env.score_seed_set(pol, 7, 100); o = env.score_seed_set(orc, 7, 100)
    assert abs(a.mean() - o.mean()) < 1e-3          # identification is exact on noise-free data
    assert a.mean() > 78.65                          # synthetic replica only; not a result


def test_saturation_is_applied():
    pol = model.LQRPolicy(np.full((2, 4), 1e6), np.zeros((2, 2)))
    a = pol(np.array([[1.0, 1, 1, 1, 0, 0]]))
    assert np.all(np.abs(a) <= 1.0)


def test_scorer_is_deterministic_and_seed_sets_differ():
    pol = lambda o: np.zeros((len(o), 2))
    a = env.score_seed_set(pol, 7, 20); b = env.score_seed_set(pol, 7, 20); c = env.score_seed_set(pol, 42, 20)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert env.ep_seed(210, 999) == 100_000_000 + 21_000_000 + 999


def test_claim_rule_thresholds():
    rs = np.random.RandomState(0)
    good = {s: 95 + 5 * rs.randn(1000) for s in (7, 42, 210)}
    r = stats.claim_rule(good, True, True, 0.0)
    assert r["claim"] and all(v["lower_clears"] for v in r["seeds"].values())
    bad = dict(good); bad[42] = 80 + 5 * rs.randn(1000)            # mean above the bar, interval not above 80.95
    r = stats.claim_rule(bad, True, True, 0.0)
    assert not r["claim"] and r["above_bar_only"]
    assert not stats.claim_rule(good, False, True, 0.0)["claim"]     # G0 failed
    assert not stats.claim_rule(good, True, False, 0.0)["claim"]     # frozen sha mismatch
    assert not stats.claim_rule(good, True, True, 0.01)["claim"]     # replica disagreed
    short = {7: good[7][:900], 42: good[42], 210: good[210]}
    assert not stats.claim_rule(short, True, True, 0.0)["claim"]     # incomplete seed set
    assert abs(stats.THRESHOLD - 80.95) < 1e-12


def test_claim_rule_cannot_be_configured_away():
    """Amendment 1: n = 1000 and seeds {7, 42, 210} live inside the rule; anything else fails closed, even with scores far above the threshold."""
    rs = np.random.RandomState(1)
    ep = lambda n: 97 + 2 * rs.randn(n)
    assert stats.claim_rule({s: ep(1000) for s in (7, 42, 210)}, True, True, 0.0)["claim"]          # the one configuration that can pass
    r = stats.claim_rule({s: ep(100) for s in (7, 42, 210)}, True, True, 0.0)                         # n = 100
    assert not r["claim"] and not r["seeds"]["7"]["complete"] and r["blocked_by"]
    assert not stats.claim_rule({s: ep(20) for s in (7, 42, 210)}, True, True, 0.0)["claim"]          # the test-fixture size
    assert not stats.claim_rule({s: ep(1000) for s in (1, 2, 3)}, True, True, 0.0)["claim"]           # wrong seed set (the dry-run sets)
    assert not stats.claim_rule({s: ep(1000) for s in (7, 42, 211)}, True, True, 0.0)["claim"]        # one wrong seed
    assert not stats.claim_rule({s: ep(1000) for s in (7, 42)}, True, True, 0.0)["claim"]             # two seeds
    assert not stats.claim_rule({s: ep(1000) for s in (7, 42, 210, 5)}, True, True, 0.0)["claim"]     # an extra seed
    assert not stats.claim_rule({s: ep(1000) for s in ("a", "b", "c")}, True, True, 0.0)["claim"]
    import inspect
    assert list(inspect.signature(stats.claim_rule).parameters) == ["scores_by_seed", "g0_passed", "frozen_ok", "replica_max_diff"]   # no n_expected, no draws knob
    assert stats.N_REQUIRED == 1000 and stats.SEEDS_REQUIRED == {7, 42, 210} and stats.DRAWS_REQUIRED == 10000
    assert stats.claim_rule({"7": ep(1000), "42": ep(1000), "210": ep(1000)}, True, True, 0.0)["claim"]
