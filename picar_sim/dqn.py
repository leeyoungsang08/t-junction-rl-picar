"""Masked (Double) DQN with two interchangeable backends.

* torch  : used automatically when PyTorch is installed (Colab / your GPU PC, device auto -> cuda).
* numpy  : dependency-free fallback (this cloud workspace, Raspberry Pi / central PC inference).
Both save the same portable .npz policy (W1,b1,W2,b2,W3,b3 + metadata), so a policy trained on a GPU
runs on the real central PC with numpy only.
"""
import json
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:  # pragma: no cover - depends on environment
    HAVE_TORCH = False


class Replay:
    def __init__(self, capacity, obs_size, n_actions, rng):
        self.o = np.zeros((capacity, obs_size), np.float32)
        self.o2 = np.zeros_like(self.o)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)
        self.m2 = np.zeros((capacity, n_actions), bool)
        self.cap, self.n, self.i, self.rng = capacity, 0, 0, rng

    def add(self, o, a, r, o2, done, m2):
        j = self.i
        self.o[j], self.a[j], self.r[j], self.o2[j], self.d[j], self.m2[j] = o, a, r, o2, done, m2
        self.i = (j + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, b):
        idx = self.rng.integers(0, self.n, b)
        return self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx], self.m2[idx]


# ----------------------------------------------------------------------------- numpy backend
class NumpyMLP:
    def __init__(self, sizes, rng):
        self.p = {}
        for k, (i, o) in enumerate(zip(sizes, sizes[1:]), 1):
            self.p[f"W{k}"] = (rng.standard_normal((i, o)) * np.sqrt(2.0 / i)).astype(np.float32)
            self.p[f"b{k}"] = np.zeros(o, np.float32)
        self.L = len(sizes) - 1

    def forward(self, x, cache=False):
        hs = [x]
        for k in range(1, self.L + 1):
            x = x @ self.p[f"W{k}"] + self.p[f"b{k}"]
            if k < self.L:
                x = np.maximum(x, 0)
            hs.append(x)
        return (x, hs) if cache else x

    def backward(self, hs, dout):
        g = {}
        for k in range(self.L, 0, -1):
            g[f"W{k}"] = hs[k - 1].T @ dout
            g[f"b{k}"] = dout.sum(0)
            if k > 1:
                dout = (dout @ self.p[f"W{k}"].T) * (hs[k - 1] > 0)
        return g


class NumpyAgent:
    def __init__(self, obs_size, n_actions, cfg, rng):
        self.cfg, self.nA = cfg, n_actions
        sizes = [obs_size, *cfg["hidden"], n_actions]
        self.online, self.target = NumpyMLP(sizes, rng), NumpyMLP(sizes, rng)
        self.sync()
        self.m = {k: np.zeros_like(v) for k, v in self.online.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.online.p.items()}
        self.t = 0
        self.device = "numpy-cpu"

    def sync(self):
        self.target.p = {k: v.copy() for k, v in self.online.p.items()}

    def q(self, obs):
        return self.online.forward(np.asarray(obs, np.float32)[None])[0]

    def update(self, batch):
        o, a, r, o2, d, m2 = batch
        c = self.cfg
        q2_t = self.target.forward(o2)
        if c.get("double", True):
            q2_o = np.where(m2, self.online.forward(o2), -np.inf)
            best = q2_o.argmax(1)
        else:
            best = np.where(m2, q2_t, -np.inf).argmax(1)
        y = r + c["gamma"] * (1 - d) * q2_t[np.arange(len(a)), best]
        q, hs = self.online.forward(o, cache=True)
        err = q[np.arange(len(a)), a] - y
        dq = np.zeros_like(q)
        dq[np.arange(len(a)), a] = np.clip(err, -1, 1) / len(a)            # Huber gradient
        g = self.online.backward(hs, dq)
        norm = np.sqrt(sum((x ** 2).sum() for x in g.values()))
        scale = min(1.0, c["grad_clip"] / (norm + 1e-8))
        self.t += 1
        b1, b2, lr = 0.9, 0.999, c["lr"]
        for k in self.online.p:
            gk = g[k] * scale
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            mh, vh = self.m[k] / (1 - b1 ** self.t), self.v[k] / (1 - b2 ** self.t)
            self.online.p[k] -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
        ae = np.abs(err)
        return float(np.mean(np.where(ae < 1, 0.5 * err ** 2, ae - 0.5)))

    def weights(self):
        return {k: v.copy() for k, v in self.online.p.items()}

    def state(self):
        return {"online": self.weights(), "target": {k: v.copy() for k, v in self.target.p.items()},
                "m": self.m, "v": self.v, "t": self.t}

    def load_state(self, s):
        self.online.p, self.target.p, self.m, self.v, self.t = s["online"], s["target"], s["m"], s["v"], s["t"]


# ----------------------------------------------------------------------------- torch backend
if HAVE_TORCH:
    class TorchAgent:
        def __init__(self, obs_size, n_actions, cfg, rng):
            self.cfg, self.nA = cfg, n_actions
            dev = cfg.get("device", "auto")
            self.device = torch.device("cuda" if dev == "auto" and torch.cuda.is_available() else
                                       ("cpu" if dev == "auto" else dev))
            torch.manual_seed(int(rng.integers(1 << 31)))
            layers, i = [], obs_size
            for h in cfg["hidden"]:
                layers += [nn.Linear(i, h), nn.ReLU()]
                i = h
            layers.append(nn.Linear(i, n_actions))
            self.online = nn.Sequential(*layers).to(self.device)
            self.target = nn.Sequential(*[type(l)(l.in_features, l.out_features) if isinstance(l, nn.Linear) else nn.ReLU()
                                          for l in layers]).to(self.device)
            self.sync()
            self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg["lr"])
            self.t = 0

        def sync(self):
            self.target.load_state_dict(self.online.state_dict())

        @torch.no_grad()
        def q(self, obs):
            return self.online(torch.as_tensor(obs, device=self.device)[None])[0].cpu().numpy()

        def update(self, batch):
            o, a, r, o2, d, m2 = (torch.as_tensor(x, device=self.device) for x in batch)
            c = self.cfg
            with torch.no_grad():
                q2_t = self.target(o2)
                src = self.online(o2) if c.get("double", True) else q2_t
                best = src.masked_fill(~m2, -float("inf")).argmax(1, keepdim=True)
                y = r + c["gamma"] * (1 - d) * q2_t.gather(1, best).squeeze(1)
            q = self.online(o).gather(1, a[:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(q, y)
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.online.parameters(), c["grad_clip"])
            self.opt.step()
            self.t += 1
            return float(loss)

        def weights(self):
            lin = [l for l in self.online if isinstance(l, nn.Linear)]
            out = {}
            for k, l in enumerate(lin, 1):
                out[f"W{k}"] = l.weight.detach().cpu().numpy().T.copy()
                out[f"b{k}"] = l.bias.detach().cpu().numpy().copy()
            return out

        def state(self):
            return {"online": self.online.state_dict(), "target": self.target.state_dict(),
                    "opt": self.opt.state_dict(), "t": self.t}

        def load_state(self, s):
            self.online.load_state_dict(s["online"])
            self.target.load_state_dict(s["target"])
            self.opt.load_state_dict(s["opt"])
            self.t = s["t"]


def make_agent(obs_size, n_actions, cfg, rng, backend="auto"):
    if backend == "torch" or (backend == "auto" and HAVE_TORCH):
        if not HAVE_TORCH:
            raise RuntimeError("PyTorch not installed")
        return TorchAgent(obs_size, n_actions, cfg, rng)
    return NumpyAgent(obs_size, n_actions, cfg, rng)


# ----------------------------------------------------------------------------- portable policy
def save_policy(path, weights, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **weights, meta=np.array(json.dumps(meta)))


class Policy:
    """numpy-only inference used by evaluation and by the real central PC."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        self.p = {k: z[k] for k in z.files if k != "meta"}
        self.L = len([k for k in self.p if k.startswith("W")])

    def q(self, obs):
        x = np.asarray(obs, np.float32)
        for k in range(1, self.L + 1):
            x = x @ self.p[f"W{k}"] + self.p[f"b{k}"]
            if k < self.L:
                x = np.maximum(x, 0)
        return x

    def act(self, obs, mask):
        return int(np.where(mask, self.q(obs), -np.inf).argmax())
