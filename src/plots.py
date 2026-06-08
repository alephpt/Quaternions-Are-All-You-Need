"""
plots.py
========

Generates every figure used in *Quaternions Are All You Need*.

Figures (written to ../figures):

  fig1_involution.png      The Euler unit-circle involution  1<->0, -1<->pi,
                           i<->pi/2, -i<->3pi/2, with an antipodal pair
                           (equal energy e, opposite phase sigma).
  fig2_associativity.png   Residual histograms of (a b) c vs a (b c) for both
                           operators in both domains.
  fig3_distributivity.png  Residual histograms of a(b+c) vs ab+ac.
  fig4_law_matrix.png      Heatmap summary: max residual per law x operator/domain.
  fig5_cayley.png          Cayley table of the Hamilton product on {1,i,j,k}.
  fig6_non_contradiction.png  q q* lands on the real "1"-ray; imaginary residue ~0.
  fig7_tradeoff.png        Associativity vs commutativity tradeoff in H.
  fig8_ijk_sphere.png      From the circle to the 3-sphere: i,j,k axes and the
                           orbit of i under random unit-quaternion conjugation.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm

import framework as F
import verification as V

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Consistent palette
C_HAM = "#1f77b4"     # Hamilton  (blue)
C_PHASE = "#d62728"   # phase-additive (red)
C_ACCENT = "#2ca02c"  # green accent
plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.bbox": "tight",
})


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


# ---------------------------------------------------------------------------
# Fig 1 -- the Euler involution on the unit circle
# ---------------------------------------------------------------------------
def fig_involution(rng):
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    th = np.linspace(0, 2 * np.pi, 512)
    ax.plot(np.cos(th), np.sin(th), color="0.6", lw=1.3, zorder=1)

    # landmarks  point <-> phase angle
    for label, (angle, _pt) in F.LANDMARKS.items():
        x, y = np.cos(angle), np.sin(angle)
        ax.plot([x], [y], "o", color="0.2", ms=7, zorder=4)
        ax.annotate(f"{label}\n(angle = {angle/np.pi:.2g}$\\pi$)",
                    (x, y), textcoords="offset points",
                    xytext=(12 * np.sign(x + 1e-9), 12 * np.sign(y + 1e-9)),
                    ha="center", fontsize=9)

    # an antipodal pair x and x* : equal energy, opposite phase
    alpha = 0.7
    x = F.euler(alpha)
    xc = F.conjugate(x)  # reflection across the real axis : sigma(x*) = -sigma(x)
    for q, col, name in [(x, C_HAM, "x"), (xc, C_PHASE, "x*  (=y)")]:
        ax.annotate("", xy=(q[1], q[0] if False else q[0]),
                    xytext=(0, 0))  # placeholder, drawn below
    # draw as radial arrows (cos = real on x-axis, sin = i on y-axis)
    ax.annotate("", xy=(np.cos(alpha), np.sin(alpha)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_HAM, lw=2))
    ax.annotate("", xy=(np.cos(-alpha), np.sin(-alpha)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_PHASE, lw=2))
    ax.text(np.cos(alpha) + 0.04, np.sin(alpha) + 0.04, "$x$",
            color=C_HAM, fontsize=13)
    ax.text(np.cos(-alpha) + 0.04, np.sin(-alpha) - 0.10, "$y=x^{*}$",
            color=C_PHASE, fontsize=13)

    ax.text(0.0, -1.32,
            r"$e(x)=e(y)$  with  $\sigma(x)=-\sigma(y)$" "\n"
            r"deviation $\sigma(x)+\sigma(y)=0 \Rightarrow$ product on the $1$-ray",
            ha="center", fontsize=10)

    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.55, 1.45)
    ax.set_xlabel("real axis  (1)")
    ax.set_ylabel("imaginary axis  (i)")
    ax.set_title("The Euler involution: realising the imaginaries on $S^1$")
    _save(fig, "fig1_involution.png")


# ---------------------------------------------------------------------------
# Fig 2 / 3 -- residual histograms
# ---------------------------------------------------------------------------
def _residual_hist(fig_name, title, residual_fn, rng, n=20000):
    operators = {"Hamilton": (F.hamilton, C_HAM),
                 "phase-additive": (F.phase_add, C_PHASE)}
    domains = {"complex  $\\mathbb{C}$": F.random_complex,
               "quaternion  $\\mathbb{H}$": F.random_quaternion}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (dom_name, sampler) in zip(axes, domains.items()):
        for op_name, (op, col) in operators.items():
            res = residual_fn(op, sampler, rng, n)
            res = np.clip(res, 1e-18, None)
            bins = np.logspace(-18, 2, 60)
            ax.hist(res, bins=bins, alpha=0.6, color=col,
                    label=f"{op_name}  (max {res.max():.1e})")
        ax.set_xscale("log")
        ax.axvspan(1e-18, 1e-10, color="0.85", zorder=0)
        ax.set_title(dom_name)
        ax.set_xlabel("residual  (log scale)")
        ax.legend(fontsize=8, loc="upper center")
    axes[0].set_ylabel("count")
    fig.text(0.135, 0.86, "machine-precision\nband", fontsize=8, color="0.4")
    fig.suptitle(title, y=1.02, fontsize=13)
    _save(fig, fig_name)


# ---------------------------------------------------------------------------
# Fig 4 -- law x (operator/domain) heatmap
# ---------------------------------------------------------------------------
def fig_law_matrix(metrics):
    laws = ["associativity", "distributivity", "commutativity"]
    cols = ["hamilton/complex", "hamilton/quaternion",
            "phase_add/complex", "phase_add/quaternion"]
    col_labels = ["Hamilton\n$\\mathbb{C}$", "Hamilton\n$\\mathbb{H}$",
                  "phase-add\n$\\mathbb{C}$", "phase-add\n$\\mathbb{H}$"]

    M = np.array([[max(metrics["laws"][law][c]["max"], 1e-18) for c in cols]
                  for law in laws])

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    im = ax.imshow(M, cmap="RdYlGn_r", norm=LogNorm(vmin=1e-16, vmax=1e2),
                   aspect="auto")
    ax.set_xticks(range(len(cols)), col_labels)
    ax.set_yticks(range(len(laws)), laws)
    for i in range(len(laws)):
        for j in range(len(cols)):
            val = M[i, j]
            txt = "PASS" if val < 1e-8 else "FAIL"
            ax.text(j, i, f"{val:.0e}\n{txt}", ha="center", va="center",
                    fontsize=9,
                    color="black" if val < 1e-8 else "white", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("max residual")
    ax.set_title("Which laws survive? (green = holds, red = broken)")
    _save(fig, "fig4_law_matrix.png")


# ---------------------------------------------------------------------------
# Fig 5 -- Cayley table of the Hamilton product
# ---------------------------------------------------------------------------
def fig_cayley():
    basis = {"1": F.ONE, "i": F.I, "j": F.J, "k": F.K}
    names = list(basis)
    n = len(names)
    # encode each product as signed basis index for colouring
    label = np.empty((n, n), dtype=object)
    color = np.zeros((n, n))
    for r, a in enumerate(names):
        for c, b in enumerate(names):
            p = F.hamilton(basis[a], basis[b])
            idx = int(np.argmax(np.abs(p)))
            sign = np.sign(p[idx])
            sym = names[idx]
            label[r, c] = f"{'-' if sign < 0 else ''}{sym}"
            color[r, c] = sign * (idx + 1)

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    im = ax.imshow(color, cmap="coolwarm", vmin=-4, vmax=4)
    ax.set_xticks(range(n), names)
    ax.set_yticks(range(n), names)
    ax.set_xlabel("right factor")
    ax.set_ylabel("left factor")
    for r in range(n):
        for c in range(n):
            ax.text(c, r, label[r, c], ha="center", va="center",
                    fontsize=15, fontweight="bold")
    ax.set_title("Cayley table of the Hamilton product\n"
                 "(note $ij=k$ but $ji=-k$: non-commutative)")
    _save(fig, "fig5_cayley.png")


# ---------------------------------------------------------------------------
# Fig 6 -- non-contradiction:  q q* on the real 1-ray
# ---------------------------------------------------------------------------
def fig_non_contradiction(rng, n=2000):
    reals, imags, norms2 = [], [], []
    for _ in range(n):
        q = F.random_quaternion(rng)
        p = F.hamilton(q, F.conjugate(q))
        reals.append(p[0])
        imags.append(np.linalg.norm(p[1:]))
        norms2.append(F.e(q) ** 2)

    reals = np.array(reals)
    imags = np.array(imags)
    norms2 = np.array(norms2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    ax1.scatter(norms2, reals, s=8, alpha=0.4, color=C_HAM,
                label=r"$\mathrm{Re}(qq^{*})$")
    lim = [0, norms2.max() * 1.05]
    ax1.plot(lim, lim, "k--", lw=1, label=r"$y=|q|^2$")
    ax1.set_xlabel(r"$|q|^2 = e(q)^2$")
    ax1.set_ylabel(r"real part of $q\,q^{*}$")
    ax1.set_title(r"$q\,q^{*}=|q|^2$: lands exactly on the $1$-ray")
    ax1.legend()

    ax2.hist(np.clip(imags, 1e-18, None), bins=np.logspace(-18, 0, 50),
             color=C_ACCENT, alpha=0.8)
    ax2.set_xscale("log")
    ax2.axvspan(1e-18, 1e-10, color="0.85", zorder=0)
    ax2.set_xlabel(r"$\|\mathrm{Im}(q\,q^{*})\|$  (log scale)")
    ax2.set_ylabel("count")
    ax2.set_title("imaginary residue $\\approx 0$\n(no contradiction)")
    fig.suptitle("Non-contradiction: the antipode product is a positive real",
                 y=1.02, fontsize=13)
    _save(fig, "fig6_non_contradiction.png")


# ---------------------------------------------------------------------------
# Fig 7 -- associativity vs commutativity tradeoff in H
# ---------------------------------------------------------------------------
def fig_tradeoff(metrics):
    laws = ["associativity", "distributivity", "commutativity"]
    ham = [metrics["laws"][l]["hamilton/quaternion"]["max"] for l in laws]
    phase = [metrics["laws"][l]["phase_add/quaternion"]["max"] for l in laws]
    ham = [max(v, 1e-18) for v in ham]
    phase = [max(v, 1e-18) for v in phase]

    x = np.arange(len(laws))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(x - w / 2, ham, w, color=C_HAM, label="Hamilton product")
    ax.bar(x + w / 2, phase, w, color=C_PHASE, label="phase-additive")
    ax.set_yscale("log")
    ax.axhline(1e-10, color="0.4", ls="--", lw=1)
    ax.text(2.35, 2e-10, "consistency\nthreshold", fontsize=8, color="0.4")
    ax.set_xticks(x, laws)
    ax.set_ylabel("max residual in $\\mathbb{H}$  (log scale)")
    ax.set_title("The tradeoff in $\\mathbb{H}$: only Hamilton keeps\n"
                 "associativity AND distributivity")
    ax.legend()
    _save(fig, "fig7_tradeoff.png")


# ---------------------------------------------------------------------------
# Fig 8 -- from the circle to the 3-sphere
# ---------------------------------------------------------------------------
def fig_ijk_sphere(rng, n=1500):
    # orbit of i under conjugation by random unit quaternions: u i u*  -> a sphere
    pts = np.empty((n, 3))
    for t in range(n):
        u = F.random_unit_quaternion(rng)
        r = F.hamilton(F.hamilton(u, F.I), F.conjugate(u))
        pts[t] = r[1:]

    fig = plt.figure(figsize=(6.6, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4, alpha=0.25, color=C_HAM)

    for vec, name, col in [(np.array([1, 0, 0]), "i", "#d62728"),
                           (np.array([0, 1, 0]), "j", "#2ca02c"),
                           (np.array([0, 0, 1]), "k", "#9467bd")]:
        ax.quiver(0, 0, 0, *vec, color=col, lw=2.5, arrow_length_ratio=0.12)
        ax.text(*(vec * 1.18), name, color=col, fontsize=14, fontweight="bold")

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("From the circle to the 3-sphere:\n"
                 "the orbit $u\\,i\\,u^{*}$ fills the imaginary sphere")
    _save(fig, "fig8_ijk_sphere.png")


# ---------------------------------------------------------------------------
def main():
    print("Generating figures...")
    rng = np.random.default_rng(7)
    metrics = V.run(n=20000, seed=0)

    fig_involution(rng)
    _residual_hist("fig2_associativity.png",
                   "Associativity residual  $\\|(ab)c-a(bc)\\|$",
                   V.associativity_residuals, rng)
    _residual_hist("fig3_distributivity.png",
                   "Distributivity residual  $\\|a(b{+}c)-(ab{+}ac)\\|$",
                   V.distributivity_residuals, rng)
    fig_law_matrix(metrics)
    fig_cayley()
    fig_non_contradiction(rng)
    fig_tradeoff(metrics)
    fig_ijk_sphere(rng)
    print("Done.")


if __name__ == "__main__":
    main()
