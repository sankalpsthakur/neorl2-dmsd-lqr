"""Dataset handling (numpy only; parquet is read in the kernels with pandas). Columns follow the Hub files: obs_0..5, action_0..1, next_obs_0..5, reward, done, truncated."""
import numpy as np

COLS = [f"obs_{i}" for i in range(6)] + [f"action_{i}" for i in range(2)] + [f"next_obs_{i}" for i in range(6)] + ["reward", "done", "truncated"]


def from_columns(cols):
    """cols: mapping column name -> 1-d array. Returns dict(obs, action, next_obs, reward, done, truncated)."""
    g = lambda names: np.stack([np.asarray(cols[n], np.float64) for n in names], axis=1)
    return dict(obs=g(COLS[:6]), action=g(COLS[6:8]), next_obs=g(COLS[8:14]), reward=np.asarray(cols["reward"], np.float64),
                done=np.asarray(cols["done"], np.float64), truncated=np.asarray(cols["truncated"], np.float64))


def episode_slices(d, length=100):
    """Episode boundaries: by done/truncated flags if any is set, else consecutive blocks of `length` whose next_obs equals the next obs.
    Returns (list of (start, stop), how)."""
    ends = np.where(np.logical_or(d["done"] > 0, d["truncated"] > 0))[0]
    n = len(d["reward"])
    if len(ends):
        starts = np.concatenate([[0], ends[:-1] + 1]); stops = ends + 1
        sl = list(zip(starts.tolist(), stops.tolist()))
        if stops[-1] != n:
            sl.append((int(stops[-1]), n))
        return sl, "flags"
    return [(i, min(i + length, n)) for i in range(0, n, length)], "blocks"


def data_score_raw(d):
    """report_result.py: sum(reward) / number of trajectories (done or truncated)."""
    k = int(np.sum(np.logical_or(d["done"] > 0, d["truncated"] > 0)))
    return float(np.sum(d["reward"]) / k), k


def val_episodes(d, length=100):
    """x0 (n,4), targets (n,L,2) (obs target at each step), next_targets (n,L,2) (next_obs target, which the reward uses)."""
    sl, how = episode_slices(d, length)
    sl = [(a, b) for a, b in sl if b - a == length]
    x0 = np.stack([d["obs"][a, :4] for a, b in sl])
    tg = np.stack([d["obs"][a:b, 4:6] for a, b in sl])
    ntg = np.stack([d["next_obs"][a:b, 4:6] for a, b in sl])
    return x0, tg, ntg


def reward_from_next_obs(next_obs):
    p = next_obs[:, :2].astype(np.float32); t = next_obs[:, 4:6].astype(np.float32)
    return np.sum(-np.log(np.abs(p - t) + np.float32(0.01)), axis=1).astype(np.float64)


def continuity(d, sl):
    """Maximum |next_obs[t] - obs[t+1]| inside episodes (should be 0)."""
    m = 0.0
    for a, b in sl:
        if b - a > 1:
            m = max(m, float(np.max(np.abs(d["next_obs"][a:b - 1] - d["obs"][a + 1:b]))))
    return m
