"""Bootstrap intervals and the pre-registered claim rule (PREREGISTRATION section 2 and Amendment 1).

Completeness is NOT configurable: the claim rule itself requires exactly 1000 episodes for each of the seed sets {7, 42, 210}, and 10,000 bootstrap draws.
Any other n, seed set or draw count fails closed (claim False, with the reasons listed in `blocked_by`)."""
import numpy as np

BAR = 78.65
BAR_SPREAD = 2.30
THRESHOLD = BAR + BAR_SPREAD   # 80.95
N_REQUIRED = 1000
SEEDS_REQUIRED = frozenset((7, 42, 210))
DRAWS_REQUIRED = 10000


def bootstrap_ci(x, draws=DRAWS_REQUIRED, seed=0, alpha=0.05, chunk=1000):
    x = np.asarray(x, np.float64)
    rs = np.random.RandomState(seed)
    m = np.empty(draws)
    for a in range(0, draws, chunk):
        b = min(draws, a + chunk)
        m[a:b] = x[rs.randint(0, len(x), size=(b - a, len(x)))].mean(axis=1)
    return float(np.percentile(m, 100 * alpha / 2)), float(np.percentile(m, 100 * (1 - alpha / 2)))


def _seed_key(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def claim_rule(scores_by_seed, g0_passed, frozen_ok, replica_max_diff):
    """scores_by_seed: {seed: per-episode normalised scores}. Returns dict with each condition, `blocked_by` (reasons) and the verdict `claim`.
    claim is True only if: seeds are exactly {7, 42, 210}; each has exactly 1000 episodes; each 10,000-draw 95 % percentile-bootstrap lower bound is above 80.95;
    G0 (G0a, G0b) passed; the frozen configuration sha matched; replica and official returns agree to 1e-3 on every episode."""
    out = dict(g0_passed=bool(g0_passed), frozen_ok=bool(frozen_ok), replica_max_diff=float(replica_max_diff), seeds={}, n_required=N_REQUIRED,
               seeds_required=sorted(SEEDS_REQUIRED), draws=DRAWS_REQUIRED)
    blocked = []
    keys = {k: _seed_key(k) for k in scores_by_seed}
    seed_set_ok = (len(keys) == len(SEEDS_REQUIRED) and set(keys.values()) == SEEDS_REQUIRED)
    if not seed_set_ok:
        blocked.append(f"seed set {sorted(str(k) for k in scores_by_seed)} is not {sorted(SEEDS_REQUIRED)}")
    lowers_ok = True
    for s, x in scores_by_seed.items():
        x = np.asarray(x, np.float64)
        lo, hi = bootstrap_ci(x, DRAWS_REQUIRED, seed=keys[s] if keys[s] is not None else 0)
        complete = bool(len(x) == N_REQUIRED)
        out["seeds"][str(s)] = dict(n=len(x), mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else None, ci95=[lo, hi], lower_clears=bool(lo > THRESHOLD), complete=complete)
        if not complete:
            blocked.append(f"seed {s}: {len(x)} episodes, {N_REQUIRED} required")
        if not lo > THRESHOLD:
            lowers_ok = False; blocked.append(f"seed {s}: lower bound {lo:.3f} not above {THRESHOLD}")
    means = [v["mean"] for v in out["seeds"].values()]
    out["mean_over_seeds"] = float(np.mean(means)) if means else None
    out["sd_over_seed_means"] = float(np.std(means, ddof=1)) if len(means) > 1 else None
    out["replica_ok"] = bool(replica_max_diff <= 1e-3)
    for flag, name in ((g0_passed, "G0 not passed"), (frozen_ok, "frozen configuration not verified"), (out["replica_ok"], "replica/official return difference above 1e-3")):
        if not flag:
            blocked.append(name)
    out["blocked_by"] = blocked
    out["claim"] = bool(not blocked and seed_set_ok and lowers_ok)
    out["above_bar_only"] = bool(out["mean_over_seeds"] is not None and out["mean_over_seeds"] > BAR and not out["claim"])
    return out
