"""Adapter to the OFFICIAL NeoRL2 package, used only inside Kaggle kernels (needs internet, gymnasium, torch, matplotlib).
The package is not pip-installed (setup.py pins numpy 1.26.4 and gymnasium 0.29.1): the five source files needed by `gym.make("DMSD")` are fetched at the pinned
commit, hashed against protocol/sources.sha256 and put on sys.path. A hash mismatch stops the kernel."""
import hashlib, os, sys, urllib.request
import numpy as np

REPO = "polixir/NeoRL2"
COMMIT = "b50d91d0a65916a89342a4f6978188b17cb2d97a"
# repository path -> file name inside protocol/sources/
FILES = {"neorl2/__init__.py": "neorl2___init__.py", "neorl2/envs/__init__.py": None, "neorl2/envs/dmsd.py": "neorl2_envs_dmsd.py", "neorl2/envs/info.py": "neorl2_envs_info.py",
         "neorl2/envs/base.py": "neorl2_envs_base.py", "neorl2/envs/reward/dmsd_reward.py": "neorl2_envs_reward_dmsd_reward.py"}
EMPTY_SHA = hashlib.sha256(b"").hexdigest()


def parse_sha_file(text):
    out = {}
    for line in text.splitlines():
        if line.strip():
            h, name = line.split(None, 1)
            out[name.strip().lstrip("*")] = h
    return out


def install(sha_file_text, dest="/tmp/neorl2_src"):
    """Fetch, verify, put on sys.path. Returns dict path -> sha256 actually installed."""
    expect = parse_sha_file(sha_file_text)
    got = {}
    for path, local in FILES.items():
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}"
        data = urllib.request.urlopen(url, timeout=60).read()
        h = hashlib.sha256(data).hexdigest()
        want = EMPTY_SHA if local is None else expect[local]
        if h != want:
            raise SystemExit(f"HASH MISMATCH for {path} at {COMMIT}: got {h}, pinned {want}: stopping (PREREGISTRATION section 5)")
        os.makedirs(os.path.dirname(os.path.join(dest, path)), exist_ok=True)
        open(os.path.join(dest, path), "wb").write(data)
        got[path] = h
    sys.path.insert(0, dest)
    return got


def versions():
    out = {"python": sys.version.split()[0], "numpy": np.__version__}
    for m in ("gymnasium", "torch", "pandas", "matplotlib"):
        try:
            out[m] = __import__(m).__version__
        except Exception as e:                      # reported, not fatal here
            out[m] = f"import failed: {e}"
    return out


def make_env(mode="train"):
    import gymnasium as gym
    import neorl2  # noqa: F401  (registers DMSD)
    return gym.make("DMSD") if mode == "train" else gym.make("DMSD", mode=mode)


def returns(policy, seeds, mode="train", record_first=0):
    """Official-loop returns (python-float sums of the env reward) with every episode seeded through env.reset(seed=...). Returns (returns (n,), obs histories of the first `record_first` episodes (k, 100, 6))."""
    env = make_env(mode)
    out = []; hist = []
    for j, sd in enumerate(seeds):
        obs, _ = env.reset(seed=int(sd))
        R = 0.0; h = []
        while True:
            h.append(np.array(obs, dtype=np.float64))
            a = np.asarray(policy(np.asarray(obs, dtype=np.float64)[None]), dtype=np.float64)[0]
            obs, r, term, trunc, _ = env.step(a)
            R += r
            if term or trunc:
                break
        out.append(R)
        if j < record_first:
            hist.append(np.stack(h))
    return np.array(out), (np.stack(hist) if hist else None)
