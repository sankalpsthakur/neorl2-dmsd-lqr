"""Numpy replica of the NeoRL2 DMSD simulator (neorl2/envs/dmsd.py at commit b50d91d0, reward dmsd_reward.py) and the official metric.

It copies the official random-number order of `reset(seed)`: 8 task draws (damping x3, stiffness x3, mass x2), then the targets
(per dimension: randint(1, 3), then one uniform per segment), then the start state (2 positions, 2 velocities); the reward is evaluated
in float32 as in `get_reward` (numpy path). G0b checks it against the official package on identical seeds.
Policies take a batch of observations (n, 6) = [pos(2), vel(2), target(2)] and return actions (n, 2); they are clipped to [-1, 1] here as in the env.
"""
import numpy as np

MIN_SCORE, MAX_SCORE = 30.60, 849.51
DT = 0.2
T_MAX = 100
FORCE_LO, FORCE_HI = -30.0, 30.0
BASE_SEEDS = (7, 42, 210)
TRUE = dict(d=4.0, k=2.0, m_train=20.0, m_test=30.0)


def normalize(ret):
    return (np.asarray(ret, dtype=np.float64) - MIN_SCORE) / (MAX_SCORE - MIN_SCORE) * 100.0


def ep_seed(s, i):
    """Evaluation episode seed i of the seed set s (PREREGISTRATION item 2)."""
    return 100_000_000 + 100_000 * int(s) + int(i)


def dynamics_from_task(task, dt=DT):
    """task = [d1, d2, d3, k1, k2, k3, m1, m2] as in dmsd.py; returns A (4x4) in the env's explicit-Euler form and the 1/m vector."""
    d1, d2, d3, k1, k2, k3, m1, m2 = task
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [-dt * (k1 + k2) / m1, dt * k2 / m1, 1 - dt * (d1 + d2) / m1, dt * d2 / m1],
        [dt * k2 / m2, -dt * (k2 + k3) / m2, dt * d2 / m2, 1 - dt * (d2 + d3) / m2]])
    return A, np.array([1 / m1, 1 / m2])


def draw_episode(seed, mode="train", dt=DT):
    """One episode's randomness, in the official order. Returns dict(A, invm, targets (100, 2), x0 (4,), task)."""
    rs = np.random.RandomState(int(seed))
    mass = (20.0, 20.0) if mode == "train" else (30.0, 30.0)
    task = np.array([rs.uniform(4.0, 4.0), rs.uniform(4.0, 4.0), rs.uniform(4.0, 4.0),
                     rs.uniform(2.0, 2.0), rs.uniform(2.0, 2.0), rs.uniform(2.0, 2.0),
                     rs.uniform(*mass), rs.uniform(*mass)])
    cols = []
    for _ in range(2):
        num = rs.randint(1, 2 + 1)
        t = np.array([])
        for i in range(num):
            if i == num - 1:
                t = np.append(t, np.ones(T_MAX - len(t)) * rs.uniform(-1.5, 1.5))
            else:
                t = np.append(t, np.ones(T_MAX // num) * rs.uniform(-1.5, 1.5))
        cols.append(t)
    targets = np.array(cols).T
    pos = np.array([rs.uniform(-0.25, 0.25) for _ in range(2)])
    vel = np.array([rs.uniform(-0.1, 0.1) for _ in range(2)])
    A, invm = dynamics_from_task(task, dt)
    return dict(A=A, invm=invm, targets=targets, x0=np.concatenate([pos, vel]), task=task)


def step_reward32(next_pos, next_tgt):
    """DMSD reward of one step, float32 arithmetic as in dmsd_reward.get_reward (numpy path). Inputs (n, 2) float64; returns (n,) float32."""
    p = np.asarray(next_pos).astype(np.float32)
    t = np.asarray(next_tgt).astype(np.float32)
    return np.sum(-np.log(np.abs(p - t) + np.float32(0.01)), axis=-1)


def rollout(policy, seeds, mode="train", dt=DT, record=False):
    """Batched episodes. Returns returns (n,) float64 (python-float sums of float32 step rewards, as the official loop), and, if record, observation/action histories."""
    eps = [draw_episode(s, mode, dt) for s in seeds]
    n = len(eps)
    A = np.stack([e["A"] for e in eps])
    invm = np.stack([e["invm"] for e in eps])
    T = np.stack([e["targets"] for e in eps])
    x = np.stack([e["x0"] for e in eps])
    ret = np.zeros(n)
    hist = {"obs": [], "act": [], "next_obs": [], "reward": []} if record else None
    for t in range(T_MAX):
        obs = np.concatenate([x[:, :2], x[:, 2:], T[:, t]], axis=1)
        a = np.clip(np.asarray(policy(obs), dtype=np.float64), -1.0, 1.0)
        f = (a + 1) / 2 * (FORCE_HI - FORCE_LO) + FORCE_LO
        xn = np.einsum("nij,nj->ni", A, x)
        xn[:, 2:] += dt * invm * f
        tn = T[:, min(t + 1, T_MAX - 1)]
        r32 = step_reward32(xn[:, :2], tn)
        ret += r32.astype(np.float64)
        if record:
            hist["obs"].append(obs); hist["act"].append(a)
            hist["next_obs"].append(np.concatenate([xn, tn], axis=1)); hist["reward"].append(r32)
        x = xn
    if record:
        return ret, {k: np.stack(v, axis=1) for k, v in hist.items()}
    return ret


def score_seed_set(policy, s, n=1000, mode="train"):
    """Normalised per-episode scores of the evaluation set of seed s (PREREGISTRATION item 2)."""
    return normalize(rollout(policy, [ep_seed(s, i) for i in range(n)], mode))
