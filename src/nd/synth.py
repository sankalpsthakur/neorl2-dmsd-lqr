"""Synthetic stand-in for the DMSD dataset (dry run and tests only; NEVER a result). Same column layout as the Hub files.
Behaviour: a noisy saturated PD law with a random gain per episode, collected in the numpy replica."""
import numpy as np
from .env import rollout
from .data import COLS


class NoisyPD:
    def __init__(self, seed, n, noise=0.15):
        rs = np.random.RandomState(seed)
        self.kp = rs.uniform(1.0, 4.0, size=(n, 1)); self.kd = rs.uniform(0.5, 3.0, size=(n, 1)); self.noise = noise; self.rs = rs

    def __call__(self, obs):
        a = self.kp * (obs[:, 4:6] - obs[:, :2]) - self.kd * obs[:, 2:4]
        return np.clip(a + self.noise * self.rs.randn(*a.shape), -1, 1)


def make(n_eps, seed=0, noise=0.15, mode="train"):
    """Returns dict of column arrays (float32) for n_eps episodes of 100 steps; episode seeds are 900,000,000 + seed*100,000 + i."""
    seeds = [900_000_000 + 100_000 * seed + i for i in range(n_eps)]
    ret, h = rollout(NoisyPD(seed, n_eps, noise), seeds, mode, record=True)
    n = n_eps * 100
    cols = {}
    obs = h["obs"].reshape(n, 6); nxt = h["next_obs"].reshape(n, 6); act = h["act"].reshape(n, 2)
    for i in range(6):
        cols[f"obs_{i}"] = obs[:, i].astype(np.float32); cols[f"next_obs_{i}"] = nxt[:, i].astype(np.float32)
    for i in range(2):
        cols[f"action_{i}"] = act[:, i].astype(np.float32)
    cols["reward"] = h["reward"].reshape(n).astype(np.float32)
    trunc = np.zeros(n, np.float32); trunc[99::100] = 1.0
    cols["done"] = np.zeros(n, np.float32); cols["truncated"] = trunc
    assert list(cols) and set(COLS) == set(cols)
    return cols, ret
