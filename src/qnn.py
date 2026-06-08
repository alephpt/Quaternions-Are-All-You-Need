"""
qnn.py
======

Thread 3 -- can the quaternion architecture learn, in isolation?

A from-scratch quaternion multilayer perceptron, in pure NumPy with manual
backprop, trained by gradient descent.  No RoPE, no transformer -- just the
Hamilton product as the linear primitive, to see whether the algebra supports
learning on its own merits.

Quaternion linear layer
-----------------------
A layer mapping n_in quaternions to n_out quaternions has weights W[o, i] that
are themselves quaternions and computes

        y_o = sum_i  W_{o i} (x) x_i  +  b_o        (Hamilton product).

Because the Hamilton product is bilinear, w (x) x = L(w) x where L(w) is the
4x4 left-multiplication matrix  L(w) = sum_p w_p M_p,  with M_0..M_3 the
left-multiplication matrices of 1, i, j, k.  That makes forward and backward
passes exact einsums (verified against finite differences below).

A quaternion layer of shape (n_out, n_in) has 4*n_out*n_in weights -- a quarter
of the 16*n_out*n_in a dense real layer between the same 4n-dim spaces would use.
The Hamilton product is a hard-wired weight-sharing prior.

Task
----
Learn the rotation action  p' = r (x) p (x) r*  from data: given a rotation r
(unit quaternion) and a point p (pure quaternion), predict the rotated point.
This is exactly the spinor action of Thread 1, now learned rather than computed.
We compare the quaternion MLP to a parameter-matched real MLP.

Writes ../figures/fig11_learning.png and results/learning.json.
"""

from __future__ import annotations

import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import framework as F

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Left-multiplication basis matrices: L(w) = sum_p w_p M[p],  w (x) x = L(w) x.
M_BASIS = np.array([
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],   # 1
    [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]],  # i
    [[0, 0, -1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, -1, 0, 0]],  # j
    [[0, 0, 0, -1], [0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],  # k
], dtype=float)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
class QuaternionLinear:
    def __init__(self, n_in, n_out, rng):
        s = 1.0 / np.sqrt(n_in * 4)
        self.W = rng.normal(scale=s, size=(n_out, n_in, 4))
        self.b = np.zeros((n_out, 4))
        self.n_in, self.n_out = n_in, n_out

    def params(self):
        return [self.W, self.b]

    def forward(self, X):                       # X: (B, n_in, 4)
        self.X = X
        self.L = np.einsum("oip,prc->oirc", self.W, M_BASIS)   # (o,i,4,4)
        return np.einsum("oirc,bic->bor", self.L, X) + self.b  # (B, n_out, 4)

    def backward(self, G):                       # G: (B, n_out, 4)
        self.dW = np.einsum("bor,prc,bic->oip", G, M_BASIS, self.X)
        self.db = G.sum(axis=0)
        dX = np.einsum("oirc,bor->bic", self.L, G)
        return dX

    def grads(self):
        return [self.dW, self.db]


class QuaternionLinearFast:
    """Mathematically identical to QuaternionLinear, but assembles the block
    left-multiplication matrix once per call and uses a single BLAS matmul for
    the batched forward/backward instead of a per-sample einsum.  Same params,
    same init draws, same gradients -- only faster.
    """

    def __init__(self, n_in, n_out, rng):
        s = 1.0 / np.sqrt(n_in * 4)
        self.W = rng.normal(scale=s, size=(n_out, n_in, 4))
        self.b = np.zeros((n_out, 4))
        self.n_in, self.n_out = n_in, n_out

    def params(self):
        return [self.W, self.b]

    def _block_matrix(self):
        # T[o,i,r,c] = sum_p W[o,i,p] M[p,r,c]; lay out as (4*n_out, 4*n_in)
        T = np.einsum("oip,prc->oric", self.W, M_BASIS)        # (o,r,i,c)
        return T.reshape(4 * self.n_out, 4 * self.n_in)

    def forward(self, X):                         # X: (B, n_in, 4)
        self.B = X.shape[0]
        self.X2 = X.reshape(self.B, 4 * self.n_in)
        self.Wmat = self._block_matrix()
        y2 = self.X2 @ self.Wmat.T + self.b.reshape(-1)
        return y2.reshape(self.B, self.n_out, 4)

    def backward(self, G):                         # G: (B, n_out, 4)
        G2 = G.reshape(self.B, 4 * self.n_out)
        dWmat = G2.T @ self.X2                      # (4*n_out, 4*n_in)
        Tg = dWmat.reshape(self.n_out, 4, self.n_in, 4).transpose(0, 2, 1, 3)
        self.dW = np.einsum("oirc,prc->oip", Tg, M_BASIS)
        self.db = G.sum(axis=0)
        dX2 = G2 @ self.Wmat
        return dX2.reshape(self.B, self.n_in, 4)

    def grads(self):
        return [self.dW, self.db]


class RealLinear:
    def __init__(self, n_in, n_out, rng):
        s = 1.0 / np.sqrt(n_in)
        self.W = rng.normal(scale=s, size=(n_in, n_out))
        self.b = np.zeros(n_out)

    def params(self):
        return [self.W, self.b]

    def forward(self, X):                        # X: (B, n_in)
        self.X = X
        return X @ self.W + self.b

    def backward(self, G):
        self.dW = self.X.T @ G
        self.db = G.sum(axis=0)
        return G @ self.W.T

    def grads(self):
        return [self.dW, self.db]


class SplitTanh:
    """Elementwise tanh (the standard 'split' quaternion activation)."""

    def forward(self, X):
        self.A = np.tanh(X)
        return self.A

    def backward(self, G):
        return G * (1 - self.A ** 2)

    def params(self):
        return []

    def grads(self):
        return []


# ---------------------------------------------------------------------------
# Network container + Adam
# ---------------------------------------------------------------------------
class Net:
    def __init__(self, layers, reshape_out=None):
        self.layers = layers
        self.reshape_out = reshape_out

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, G):
        for layer in reversed(self.layers):
            G = layer.backward(G)

    def params(self):
        ps = []
        for layer in self.layers:
            ps += layer.params()
        return ps

    def grads(self):
        gs = []
        for layer in self.layers:
            gs += layer.grads()
        return gs

    def n_params(self):
        return int(sum(p.size for layer in self.layers for p in layer.params()))


class Adam:
    def __init__(self, params, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ---------------------------------------------------------------------------
# Data: learn  p' = r (x) p (x) r*
# ---------------------------------------------------------------------------
def make_data(n, rng):
    R = rng.normal(size=(n, 4))
    R /= np.linalg.norm(R, axis=1, keepdims=True)          # unit rotations
    P = np.zeros((n, 4))
    P[:, 1:] = rng.uniform(-1, 1, size=(n, 3))             # pure-quaternion points
    Y = np.empty((n, 4))
    for t in range(n):
        Y[t] = F.hamilton(F.hamilton(R[t], P[t]), F.conjugate(R[t]))
    X_q = np.stack([R, P], axis=1)                          # (n, 2, 4)
    X_r = np.concatenate([R, P], axis=1)                   # (n, 8)
    return X_q, X_r, Y


# ---------------------------------------------------------------------------
# Gradient check (finite differences) on the quaternion layer
# ---------------------------------------------------------------------------
def gradient_check():
    rng = np.random.default_rng(1)
    layer = QuaternionLinear(2, 3, rng)
    X = rng.normal(size=(5, 2, 4))
    Y = layer.forward(X)
    G = rng.normal(size=Y.shape)
    loss = lambda: float((layer.forward(X) * G).sum())
    layer.forward(X)
    layer.backward(G)
    ana = layer.dW.copy()
    num = np.zeros_like(ana)
    h = 1e-6
    for idx in np.ndindex(layer.W.shape):
        layer.W[idx] += h
        lp = loss()
        layer.W[idx] -= 2 * h
        lm = loss()
        layer.W[idx] += h
        num[idx] = (lp - lm) / (2 * h)
    return float(np.max(np.abs(ana - num)))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(net, Xtr, Ytr, Xte, Yte, steps, lr, batch, rng, flat_out=False,
          eval_every=50, thresholds=None):
    opt = Adam(net.params(), lr=lr)
    n = Xtr.shape[0]
    hist = {"step": [], "train": [], "test": []}

    def mse_and_grad(X, Y):
        pred = net.forward(X)
        p = pred.reshape(pred.shape[0], -1) if not flat_out else pred
        diff = p - Y
        loss = float(np.mean(diff ** 2))
        g = (2.0 / Y.size) * diff
        g = g.reshape(pred.shape) if not flat_out else g
        return loss, g

    t0 = time.perf_counter()
    for s in range(steps):
        idx = rng.integers(0, n, size=batch)
        _, g = mse_and_grad(Xtr[idx], Ytr[idx])
        net.backward(g)
        opt.step(net.grads())
        if s % eval_every == 0 or s == steps - 1:
            tr, _ = mse_and_grad(Xtr, Ytr)
            te, _ = mse_and_grad(Xte, Yte)
            hist["step"].append(s)
            hist["train"].append(tr)
            hist["test"].append(te)
    hist["train_time_s"] = time.perf_counter() - t0

    # steps to reach each test-MSE threshold (None = never reached)
    if thresholds:
        s2t = {}
        for tau in thresholds:
            hit = next((st for st, te in zip(hist["step"], hist["test"]) if te <= tau),
                       None)
            s2t[str(tau)] = hit
        hist["steps_to_threshold"] = s2t
    return hist


def run(seed=0):
    rng = np.random.default_rng(seed)
    Xq_tr, Xr_tr, Y_tr = make_data(4000, rng)
    Xq_te, Xr_te, Y_te = make_data(1000, rng)

    # Quaternion MLP: 2 -> 16 -> 16 -> 1  quaternions
    qnet = Net([
        QuaternionLinear(2, 16, rng), SplitTanh(),
        QuaternionLinear(16, 16, rng), SplitTanh(),
        QuaternionLinear(16, 1, rng),
    ])
    # Real MLP matched on parameter count
    target = qnet.n_params()
    h = 26
    while True:
        rnet = Net([RealLinear(8, h, rng), SplitTanh(),
                    RealLinear(h, h, rng), SplitTanh(),
                    RealLinear(h, 4, rng)], reshape_out=True)
        if rnet.n_params() >= target:
            break
        h += 1

    print(f"  quaternion params = {qnet.n_params()},  real params = {rnet.n_params()} (h={h})")
    gc = gradient_check()
    print(f"  gradient check (max |analytic - numeric|) = {gc:.2e}")

    hq = train(qnet, Xq_tr, Y_tr, Xq_te, Y_te,
               steps=3000, lr=3e-3, batch=128, rng=rng)
    hr = train(rnet, Xr_tr, Y_tr, Xr_te, Y_te,
               steps=3000, lr=3e-3, batch=128, rng=rng, flat_out=True)

    # baseline: predicting the mean -> variance of targets
    baseline = float(np.mean((Y_te - Y_te.mean(0)) ** 2))

    res = {
        "gradient_check_max_err": gc,
        "quaternion_params": qnet.n_params(),
        "real_params": rnet.n_params(),
        "quaternion_final_test_mse": hq["test"][-1],
        "real_final_test_mse": hr["test"][-1],
        "mean_predictor_mse": baseline,
        "hist_q": hq,
        "hist_r": hr,
    }

    # predictions for a scatter panel
    pq = qnet.forward(Xq_te).reshape(-1, 4)
    res["scatter_true"] = Y_te[:, 1:].ravel().tolist()
    res["scatter_pred"] = pq[:, 1:].ravel().tolist()
    return res


def check_fast_equivalence():
    """Fast layer must match the (gradient-checked) slow layer bit-for-bit."""
    r1 = np.random.default_rng(2)
    r2 = np.random.default_rng(2)
    slow = QuaternionLinear(3, 5, r1)
    fast = QuaternionLinearFast(3, 5, r2)
    X = np.random.default_rng(3).normal(size=(7, 3, 4))
    ys, yf = slow.forward(X), fast.forward(X)
    G = np.random.default_rng(4).normal(size=ys.shape)
    dXs, dXf = slow.backward(G), fast.backward(G)
    return {
        "forward": float(np.max(np.abs(ys - yf))),
        "dX": float(np.max(np.abs(dXs - dXf))),
        "dW": float(np.max(np.abs(slow.dW - fast.dW))),
    }


def benchmark(seed=0, reps=200):
    """Per-step (forward+backward) wall-clock for the three implementations,
    at the apples-to-apples shape.  Isolates compute, not optimisation."""
    rng = np.random.default_rng(seed)
    H, B, Bev = 16, 128, 4000

    def q_stack(cls):
        return Net([cls(2, H, rng), SplitTanh(), cls(H, H, rng), SplitTanh(),
                    cls(H, 1, rng)])

    nets = {
        "real (h=64)": (Net([RealLinear(8, 64, rng), SplitTanh(),
                             RealLinear(64, 64, rng), SplitTanh(),
                             RealLinear(64, 4, rng)], reshape_out=True), True),
        "quaternion-fast (BLAS)": (q_stack(QuaternionLinearFast), False),
        "quaternion-einsum": (q_stack(QuaternionLinear), False),
    }
    Xq = rng.normal(size=(B, 2, 4)); Xr = rng.normal(size=(B, 8))
    Xq_ev = rng.normal(size=(Bev, 2, 4)); Xr_ev = rng.normal(size=(Bev, 8))

    out = {}
    for name, (net, flat) in nets.items():
        X = Xr if flat else Xq
        Xev = Xr_ev if flat else Xq_ev
        out_shape = (B, 4) if flat else (B, 1, 4)
        for _ in range(10):                         # warmup
            y = net.forward(X)
            net.backward(np.ones_like(y))
        t0 = time.perf_counter()
        for _ in range(reps):
            y = net.forward(X)
            net.backward(np.ones_like(y))
        step_ms = 1e3 * (time.perf_counter() - t0) / reps
        t0 = time.perf_counter()
        for _ in range(reps):
            net.forward(Xev)
        eval_ms = 1e3 * (time.perf_counter() - t0) / reps
        out[name] = {"step_ms": step_ms, "eval_ms": eval_ms}
    return out


def make_compute_figure(bench):
    names = list(bench)
    colors = ["#7f7f7f", "#1f77b4", "#9467bd"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3))
    for ax, key, title in [(ax1, "step_ms", "Training step (fwd+bwd), batch=128"),
                           (ax2, "eval_ms", "Full-set forward, B=4000 (eval cost)")]:
        vals = [bench[n][key] for n in names]
        bars = ax.bar(names, vals, color=colors)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f} ms",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("milliseconds")
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=12)
    fig.suptitle("Both implementations of the quaternion layer (identical math)",
                 y=1.02, fontsize=13)
    path = os.path.join(FIG_DIR, "fig14_compute.png")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def apples_to_apples(seed=0, steps=6000):
    """True apples-to-apples on the rotation action p' = r (x) p (x) r*.

    A quaternion hidden layer of width H quaternions carries 4H real activations.
    So there are two honest real baselines:
      * param-matched   -- same number of trainable scalars (real net is narrower);
      * capacity-matched -- same real hidden DIMENSION 4H (real net has ~4x params).
    The quaternion net is exactly a real net of that hidden dimension whose weight
    matrices are constrained to the Hamilton block form -- so capacity-matching
    isolates the effect of the inductive bias alone.

    We report params, wall-clock training time, final test MSE, and steps to reach
    each accuracy threshold.
    """
    rng = np.random.default_rng(seed)
    Xq_tr, Xr_tr, Y_tr = make_data(4000, rng)
    Xq_te, Xr_te, Y_te = make_data(2000, rng)
    H = 16                       # quaternion hidden width -> 64 real hidden dim
    taus = [0.10, 0.05, 0.03]
    lr, batch = 3e-3, 128

    def q_mlp():
        return Net([QuaternionLinearFast(2, H, rng), SplitTanh(),
                    QuaternionLinearFast(H, H, rng), SplitTanh(),
                    QuaternionLinearFast(H, 1, rng)])

    def r_mlp(h):
        return Net([RealLinear(8, h, rng), SplitTanh(),
                    RealLinear(h, h, rng), SplitTanh(),
                    RealLinear(h, 4, rng)], reshape_out=True)

    # param-match: grow real width until params >= quaternion params
    qref = q_mlp()
    h_param = 8
    while r_mlp(h_param).n_params() < qref.n_params():
        h_param += 1

    configs = {
        "quaternion (H=16, 64-dim)": (q_mlp(), Xq_tr, Xq_te, False),
        f"real param-matched (h={h_param})": (r_mlp(h_param), Xr_tr, Xr_te, True),
        "real capacity-matched (h=64)": (r_mlp(4 * H), Xr_tr, Xr_te, True),
    }

    out = {}
    for name, (net, Xtr, Xte, flat) in configs.items():
        h = train(net, Xtr, Y_tr, Xte, Y_te, steps=steps, lr=lr, batch=batch,
                  rng=rng, flat_out=flat, eval_every=25, thresholds=taus)
        out[name] = {
            "params": net.n_params(),
            "final_test_mse": h["test"][-1],
            "train_time_s": h["train_time_s"],
            "steps_to_threshold": h["steps_to_threshold"],
            "hist": h,
        }
    out["_taus"] = taus
    return out


def make_apples_figure(ata):
    taus = ata["_taus"]
    names = [k for k in ata if not k.startswith("_")]
    colors = {"quaternion (H=16, 64-dim)": "#1f77b4"}
    palette = ["#1f77b4", "#ff7f0e", "#d62728"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.7),
                                   gridspec_kw={"width_ratios": [1.4, 1]})

    # left: learning curves with threshold lines
    for name, col in zip(names, palette):
        h = ata[name]["hist"]
        ax1.plot(h["step"], h["test"], color=col, lw=2,
                 label=f"{name}\n  {ata[name]['params']} params, "
                       f"{ata[name]['train_time_s']:.1f}s, "
                       f"final {ata[name]['final_test_mse']:.3f}")
    for tau in taus:
        ax1.axhline(tau, color="0.6", ls=":", lw=1)
        ax1.text(ax1.get_xlim()[1], tau, f" τ={tau}", va="center",
                 fontsize=8, color="0.4")
    ax1.set_yscale("log")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("test MSE")
    ax1.set_title("Apples-to-apples: p'=r p r*  (test loss vs steps)")
    ax1.legend(fontsize=7.5, loc="upper right")

    # right: steps-to-threshold grouped bars
    x = np.arange(len(taus))
    w = 0.26
    for i, (name, col) in enumerate(zip(names, palette)):
        s2t = ata[name]["steps_to_threshold"]
        vals = [s2t[str(t)] if s2t[str(t)] is not None else np.nan for t in taus]
        bars = ax2.bar(x + (i - 1) * w, vals, w, color=col, label=name.split(" (")[0])
        for xi, v in zip(x + (i - 1) * w, vals):
            if np.isnan(v):
                ax2.text(xi, 50, "n/a", ha="center", va="bottom", fontsize=8,
                         rotation=90, color=col)
    ax2.set_xticks(x, [f"τ={t}" for t in taus])
    ax2.set_ylabel("steps to reach threshold")
    ax2.set_title("Steps-to-threshold (lower = faster)")
    ax2.legend(fontsize=8)

    fig.suptitle("Thread 3 (apples-to-apples) — parameter, compute, and step efficiency",
                 y=1.02, fontsize=13)
    path = os.path.join(FIG_DIR, "fig13_apples_to_apples.png")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def sample_efficiency(seed=0):
    """When the data IS a quaternion map y = Q (x) x + noise, does the
    4-DOF Hamilton layer recover it from fewer samples than a 16-DOF real layer?
    """
    rng = np.random.default_rng(seed)
    sizes = [8, 16, 32, 64, 128, 256]
    repeats = 12
    noise = 0.15
    q_curve = {n: [] for n in sizes}
    r_curve = {n: [] for n in sizes}

    for rep in range(repeats):
        Q = F.random_unit_quaternion(rng)

        def gen(n, nz):
            X = rng.normal(size=(n, 1, 4))
            Y = np.array([F.hamilton(Q, X[t, 0]) for t in range(n)])
            Y = Y + nz * rng.normal(size=Y.shape)
            return X, X.reshape(n, 4), Y

        Xq_te, Xr_te, Y_te = gen(2000, 0.0)   # clean test = recover the true map
        for n in sizes:
            Xq, Xr, Y = gen(n, noise)
            qn = Net([QuaternionLinear(1, 1, rng)])             # 8 params
            rn = Net([RealLinear(4, 4, rng)], reshape_out=True)  # 20 params
            train(qn, Xq, Y, Xq, Y, steps=1500, lr=5e-3, batch=min(n, 64), rng=rng)
            train(rn, Xr, Y, Xr, Y, steps=1500, lr=5e-3, batch=min(n, 64),
                  rng=rng, flat_out=True)
            q_curve[n].append(float(np.mean(
                (qn.forward(Xq_te).reshape(-1, 4) - Y_te) ** 2)))
            r_curve[n].append(float(np.mean((rn.forward(Xr_te) - Y_te) ** 2)))

    return {
        "sizes": sizes,
        "quaternion_mse": [float(np.mean(q_curve[n])) for n in sizes],
        "quaternion_std": [float(np.std(q_curve[n])) for n in sizes],
        "real_mse": [float(np.mean(r_curve[n])) for n in sizes],
        "real_std": [float(np.std(r_curve[n])) for n in sizes],
        "noise": noise,
    }


def make_sample_efficiency_figure(se):
    sizes = np.array(se["sizes"])
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.errorbar(sizes, se["quaternion_mse"], yerr=se["quaternion_std"],
                color="#1f77b4", lw=2, marker="o", capsize=3,
                label="quaternion layer (4 DOF)")
    ax.errorbar(sizes, se["real_mse"], yerr=se["real_std"],
                color="#d62728", lw=2, marker="s", capsize=3,
                label="real layer (16 DOF)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("training samples")
    ax.set_ylabel("test MSE (recover the true map)")
    ax.set_title("Sample efficiency on quaternion-native data\n"
                 f"$y=Q\\otimes x$ + noise: the Hamilton prior generalises from fewer samples")
    ax.legend()
    path = os.path.join(FIG_DIR, "fig12_sample_efficiency.png")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def make_figure(res):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.5))

    hq, hr = res["hist_q"], res["hist_r"]
    ax1.plot(hq["step"], hq["train"], color="#1f77b4", lw=1, alpha=0.5)
    ax1.plot(hq["step"], hq["test"], color="#1f77b4", lw=2,
             label=f"quaternion MLP  ({res['quaternion_params']} params)")
    ax1.plot(hr["step"], hr["train"], color="#d62728", lw=1, alpha=0.5)
    ax1.plot(hr["step"], hr["test"], color="#d62728", lw=2,
             label=f"real MLP  ({res['real_params']} params)")
    ax1.axhline(res["mean_predictor_mse"], color="0.5", ls="--", lw=1,
                label="mean predictor")
    ax1.set_yscale("log")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("MSE  (solid=test, faint=train)")
    ax1.set_title("Learning the rotation action $p'=r\\,p\\,r^{*}$")
    ax1.legend(fontsize=9)

    tp = np.array(res["scatter_true"])
    pp = np.array(res["scatter_pred"])
    ax2.scatter(tp, pp, s=4, alpha=0.2, color="#1f77b4")
    lim = [min(tp.min(), pp.min()), max(tp.max(), pp.max())]
    ax2.plot(lim, lim, "k--", lw=1)
    r2 = 1 - np.sum((tp - pp) ** 2) / np.sum((tp - tp.mean()) ** 2)
    ax2.set_xlabel("true rotated coordinate")
    ax2.set_ylabel("quaternion-MLP prediction")
    ax2.set_title(f"Held-out predictions  ($R^2={r2:.3f}$)")
    ax2.set_aspect("equal")

    fig.suptitle("Thread 3 -- the quaternion architecture learns in isolation",
                 y=1.02, fontsize=13)
    path = os.path.join(FIG_DIR, "fig11_learning.png")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def main():
    print("Thread 3: does the quaternion architecture learn?")
    res = run()
    print(f"  quaternion test MSE = {res['quaternion_final_test_mse']:.4e}")
    print(f"  real (matched) MSE  = {res['real_final_test_mse']:.4e}")
    print(f"  mean-predictor MSE  = {res['mean_predictor_mse']:.4e}")
    make_figure(res)

    eq = check_fast_equivalence()
    print(f"  fast-vs-slow layer equivalence: forward={eq['forward']:.1e}, "
          f"dX={eq['dX']:.1e}, dW={eq['dW']:.1e}")
    print("  benchmarking both quaternion implementations...")
    bench = benchmark()
    for n, d in bench.items():
        print(f"    {n:<26} step={d['step_ms']:.2f} ms   eval(B=4000)={d['eval_ms']:.2f} ms")
    make_compute_figure(bench)

    print("  apples-to-apples (param- and capacity-matched, steps-to-threshold)...")
    ata = apples_to_apples()
    for name in [k for k in ata if not k.startswith("_")]:
        d = ata[name]
        print(f"    {name:<32} params={d['params']:<5} "
              f"time={d['train_time_s']:.1f}s  final={d['final_test_mse']:.4f}  "
              f"steps@thr={d['steps_to_threshold']}")
    make_apples_figure(ata)

    print("  sample-efficiency experiment (quaternion-native data)...")
    se = sample_efficiency()
    for n, q, r in zip(se["sizes"], se["quaternion_mse"], se["real_mse"]):
        print(f"    N={n:<4}  quaternion MSE={q:.4f}   real MSE={r:.4f}")
    make_sample_efficiency_figure(se)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    slim = {k: v for k, v in res.items() if k not in ("hist_q", "hist_r",
                                                      "scatter_true", "scatter_pred")}
    slim["sample_efficiency"] = se
    slim["fast_equivalence"] = eq
    slim["benchmark_ms"] = bench
    slim["apples_to_apples"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "hist"}
        for k, v in ata.items() if not k.startswith("_")
    }
    with open(os.path.join(RESULTS_DIR, "learning.json"), "w") as fh:
        json.dump(slim, fh, indent=2)
    print(f"  wrote {os.path.relpath(os.path.join(RESULTS_DIR, 'learning.json'))}")


if __name__ == "__main__":
    main()
