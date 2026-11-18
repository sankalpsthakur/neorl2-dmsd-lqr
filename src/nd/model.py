"""System identification, LQR design and validation-only selection (numpy only). PREREGISTRATION section 4."""
import json, hashlib
import numpy as np

QP_GRID = (0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
QV_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


def fit_lti(obs, act, next_obs, intercept=False):
    """Unstructured least squares of next_obs[:4] on [obs[:4], action] (no structure). Returns A (4x4), B (4x2), c (4,) or None, diagnostics."""
    obs = np.asarray(obs, np.float64); act = np.asarray(act, np.float64); nxt = np.asarray(next_obs, np.float64)
    X = np.concatenate([obs[:, :4], act] + ([np.ones((len(obs), 1))] if intercept else []), axis=1)
    Y = nxt[:, :4]
    th, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ th
    diag = dict(rms=float(np.sqrt(np.mean(res ** 2))), max_abs=float(np.max(np.abs(res))), cond=float(np.linalg.cond(X)))
    A = th[:4].T.copy(); B = th[4:6].T.copy(); c = th[6].copy() if intercept else None
    return A, B, c, diag


def one_step_error(A, B, obs, act, next_obs):
    pred = np.asarray(obs, np.float64)[:, :4] @ A.T + np.asarray(act, np.float64) @ B.T
    return float(np.max(np.abs(pred - np.asarray(next_obs, np.float64)[:, :4])))


def dare_gain(A, B, Q, R, iters=100000, tol=1e-13):
    """Discrete LQR gain by iterating the Riccati recursion to convergence (K such that u = -K x)."""
    P = Q.copy()
    K = np.zeros((B.shape[1], A.shape[0]))
    for _ in range(iters):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        Pn = Q + A.T @ P @ (A - B @ K)
        if np.max(np.abs(Pn - P)) <= tol * max(1.0, np.max(np.abs(Pn))):
            P = Pn
            break
        P = Pn
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K


def feedforward(A, B):
    """F (2x2) with a_ff = F tgt holding the identified system at x = [tgt, 0]: B a = (I - A) [tgt, 0] in least squares."""
    return np.linalg.pinv(B) @ (np.eye(4) - A)[:, :2]


class LQRPolicy:
    """a = clip(F tgt - K (x - [tgt, 0]), -1, 1); batch interface (n, 6) -> (n, 2). 12 numbers: K (2x4), F (2x2)."""

    def __init__(self, K, F, use_ff=True):
        self.K = np.asarray(K, np.float64); self.F = np.asarray(F, np.float64) if use_ff else np.zeros((2, 2))

    def __call__(self, obs):
        obs = np.atleast_2d(obs)
        x = obs[:, :4]; tgt = obs[:, 4:6]
        e = x - np.concatenate([tgt, np.zeros_like(tgt)], axis=1)
        return np.clip(tgt @ self.F.T - e @ self.K.T, -1.0, 1.0)


def design(A, B, qp, qv, use_ff=True):
    K = dare_gain(A, B, np.diag([qp, qp, qv, qv]), np.eye(2))
    rho = float(np.max(np.abs(np.linalg.eigvals(A - B @ K))))
    return LQRPolicy(K, feedforward(A, B), use_ff), rho


def model_return(A, B, policy, x0, targets, next_targets):
    """Model rollout of val episodes. x0 (n,4); targets, next_targets (n,100,2). Reward float64. Returns (n,) returns."""
    x = np.array(x0, np.float64); ret = np.zeros(len(x))
    for t in range(targets.shape[1]):
        obs = np.concatenate([x[:, :2], x[:, 2:], targets[:, t]], axis=1)
        a = np.clip(policy(obs), -1.0, 1.0)
        x = x @ A.T + a @ B.T
        ret += np.sum(-np.log(np.abs(x[:, :2] - next_targets[:, t]) + 0.01), axis=1)
    return ret


def visited_regressors(A, B, policy, x0, targets, next_targets):
    """The (state, action) regressor rows z = [x(4), a(2)] that the closed loop visits in the identified model on the validation episodes (same rollout as `model_return`)."""
    x = np.array(x0, np.float64); zs = []
    for t in range(targets.shape[1]):
        obs = np.concatenate([x[:, :2], x[:, 2:], targets[:, t]], axis=1)
        a = np.clip(policy(obs), -1.0, 1.0)
        zs.append(np.concatenate([x, a], axis=1))
        x = x @ A.T + a @ B.T
    return np.concatenate(zs, axis=0)


def regime_check(X_train, Z_visited):
    """Offline check of the regime the controller visits (PREREGISTRATION Amendment 1). Cannot measure the true one-step error there (no data exist under the LQR policy);
    reports instead how far the visited regressors lie from the training regressors: leverage h(z) = z^T (X^T X)^-1 z (the factor by which least-squares noise on the
    parameters would be amplified at z), as a ratio of the visited maximum / 99th percentile to the training maximum / 99th percentile, plus the share of visited rows
    whose leverage exceeds the training maximum, and the per-coordinate range excess. Report-only."""
    X = np.asarray(X_train, np.float64); Z = np.asarray(Z_visited, np.float64)
    G = np.linalg.pinv(X.T @ X)
    h_tr = np.einsum("ij,jk,ik->i", X, G, X); h_vi = np.einsum("ij,jk,ik->i", Z, G, Z)
    lo, hi = X.min(axis=0), X.max(axis=0)
    excess = np.maximum(np.maximum(lo - Z, Z - hi), 0.0)
    return dict(n_visited=int(len(Z)), lev_train_max=float(h_tr.max()), lev_train_p99=float(np.percentile(h_tr, 99)), lev_visited_max=float(h_vi.max()), lev_visited_p99=float(np.percentile(h_vi, 99)),
                lev_ratio_max=float(h_vi.max() / h_tr.max()), lev_ratio_p99=float(np.percentile(h_vi, 99) / np.percentile(h_tr, 99)), frac_visited_above_train_max=float(np.mean(h_vi > h_tr.max())),
                max_range_excess_per_coord=excess.max(axis=0).tolist(), train_range_per_coord=[(hi - lo).tolist()][0])


def select(A, B, val_eps):
    """Validation-only selection over the frozen grid. val_eps = (x0, targets, next_targets). Returns (best dict, table list)."""
    table = []
    for qp in QP_GRID:
        for qv in QV_GRID:
            pol, rho = design(A, B, qp, qv)
            if not (rho < 1.0):
                table.append(dict(qp=qp, qv=qv, rho=rho, val_return=None, status="unstable")); continue
            r = model_return(A, B, pol, *val_eps)
            table.append(dict(qp=qp, qv=qv, rho=rho, val_return=float(np.mean(r)), status="ok"))
    ok = [t for t in table if t["status"] == "ok"]
    if not ok:
        raise RuntimeError("no stable candidate")
    best = max(ok, key=lambda t: (round(t["val_return"] / 1e-9) * 1e-9, -t["qp"], -t["qv"]))
    return best, table


def freeze(A, B, best, table, extra=None):
    pol, rho = design(A, B, best["qp"], best["qv"])
    cfg = dict(A=A.tolist(), B=B.tolist(), qp=best["qp"], qv=best["qv"], rho=rho, K=pol.K.tolist(), F=pol.F.tolist(),
               grid=dict(qp=list(QP_GRID), qv=list(QV_GRID)), table=table, **(extra or {}))
    blob = json.dumps(cfg, sort_keys=True, indent=1)
    return cfg, blob, hashlib.sha256(blob.encode()).hexdigest()


def policy_from_config(cfg, use_ff=True):
    return LQRPolicy(np.array(cfg["K"]), np.array(cfg["F"]), use_ff)


def true_system(mass=20.0, dt=0.2):
    """A, B of the true DMSD in action units (force = 30 a): used only for the oracle reference row and tests."""
    from .env import dynamics_from_task
    A, invm = dynamics_from_task([4.0, 4.0, 4.0, 2.0, 2.0, 2.0, mass, mass], dt)
    B = np.zeros((4, 2)); B[2, 0] = dt * invm[0] * 30.0; B[3, 1] = dt * invm[1] * 30.0
    return A, B
