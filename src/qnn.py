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
def train(net, Xtr, Ytr, Xte, Yte, steps, lr, batch, rng, flat_out=False):
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

    for s in range(steps):
        idx = rng.integers(0, n, size=batch)
        _, g = mse_and_grad(Xtr[idx], Ytr[idx])
        net.backward(g)
        opt.step(net.grads())
        if s % 50 == 0 or s == steps - 1:
            tr, _ = mse_and_grad(Xtr, Ytr)
            te, _ = mse_and_grad(Xte, Yte)
            hist["step"].append(s)
            hist["train"].append(tr)
            hist["test"].append(te)
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

    print("  sample-efficiency experiment (quaternion-native data)...")
    se = sample_efficiency()
    for n, q, r in zip(se["sizes"], se["quaternion_mse"], se["real_mse"]):
        print(f"    N={n:<4}  quaternion MSE={q:.4f}   real MSE={r:.4f}")
    make_sample_efficiency_figure(se)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    slim = {k: v for k, v in res.items() if k not in ("hist_q", "hist_r",
                                                      "scatter_true", "scatter_pred")}
    slim["sample_efficiency"] = se
    with open(os.path.join(RESULTS_DIR, "learning.json"), "w") as fh:
        json.dump(slim, fh, indent=2)
    print(f"  wrote {os.path.relpath(os.path.join(RESULTS_DIR, 'learning.json'))}")


if __name__ == "__main__":
    main()
