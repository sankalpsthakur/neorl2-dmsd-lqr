"""Behaviour cloning for the G0c scale anchor (torch, kernels only): MLP 6-256-256-2 with tanh output, Adam 3e-4, batch 256, final epoch, observation standardisation."""
import numpy as np


def train_bc(obs, act, epochs=30, batch=256, lr=3e-4, seed=0, hidden=256, threads=4):
    import torch
    torch.set_num_threads(threads); torch.manual_seed(seed)
    rs = np.random.RandomState(seed)
    mu = obs.mean(0); sd = obs.std(0) + 1e-6
    X = torch.tensor((obs - mu) / sd, dtype=torch.float32); Y = torch.tensor(act, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(6, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 2), torch.nn.Tanh())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(X); losses = []
    for ep in range(epochs):
        perm = torch.tensor(rs.permutation(n))
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = torch.mean((net(X[idx]) - Y[idx]) ** 2)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(idx)
        losses.append(tot / n)

    def policy(o):
        with torch.no_grad():
            return net(torch.tensor((np.atleast_2d(o) - mu) / sd, dtype=torch.float32)).numpy().astype(np.float64)
    return policy, losses
