"""
dirac.py
========

Thread 2 -- the Cayley table is the atomic cell of the Dirac algebra.

The quaternion units map to the Pauli matrices,

        1 -> I,   i -> -i sigma_x,   j -> -i sigma_y,   k -> -i sigma_z,

which is an algebra isomorphism  H ~= su(2)  (and the unit quaternions are
exactly SU(2)).  The Dirac algebra of spacetime is then built from these cells:
the spacetime Clifford algebra Cl(1,3) is isomorphic to M_2(H), the 2x2 matrices
over the quaternions.  So the quaternionic Cayley table is not "more complete"
than Dirac -- it is the 2x2 building block Dirac is assembled from.

This module verifies, numerically:
  * the quaternion -> Pauli homomorphism:  mat(a (x) b) = mat(a) mat(b);
  * unit quaternions are SU(2):  U U^dagger = I,  det U = 1;
  * the Dirac gamma matrices satisfy the Clifford relation
        {gamma_mu, gamma_nu} = 2 eta_{mu nu} I_4,    eta = diag(+,-,-,-);
and writes ../figures/fig10_dirac.png.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import framework as F

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

# Pauli matrices
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """H -> M_2(C) via 1->I, i->-i sx, j->-i sy, k->-i sz."""
    w, x, y, z = q
    return w * I2 - 1j * (x * SX + y * SY + z * SZ)


def dirac_gammas() -> dict:
    """Standard (Dirac) representation of the gamma matrices, signature (+,-,-,-)."""
    g0 = np.block([[I2, np.zeros((2, 2))], [np.zeros((2, 2)), -I2]])
    gk = lambda s: np.block([[np.zeros((2, 2)), s], [-s, np.zeros((2, 2))]])
    return {"g0": g0, "g1": gk(SX), "g2": gk(SY), "g3": gk(SZ)}


def verify(n: int = 20000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)

    # (1) algebra homomorphism mat(a b) = mat(a) mat(b)
    homo_err = 0.0
    for _ in range(n):
        a, b = F.random_quaternion(rng), F.random_quaternion(rng)
        lhs = quaternion_to_matrix(F.hamilton(a, b))
        rhs = quaternion_to_matrix(a) @ quaternion_to_matrix(b)
        homo_err = max(homo_err, np.linalg.norm(lhs - rhs))

    # (2) unit quaternions are SU(2)
    su2_unitary = 0.0
    su2_det = 0.0
    for _ in range(n):
        U = quaternion_to_matrix(F.random_unit_quaternion(rng))
        su2_unitary = max(su2_unitary, np.linalg.norm(U @ U.conj().T - I2))
        su2_det = max(su2_det, abs(np.linalg.det(U) - 1.0))

    # (3) Dirac Clifford relation {g_mu, g_nu} = 2 eta_{mu nu} I_4
    g = dirac_gammas()
    eta = np.diag([1.0, -1.0, -1.0, -1.0])
    keys = ["g0", "g1", "g2", "g3"]
    clifford_err = 0.0
    anticomm = np.zeros((4, 4))
    for mu in range(4):
        for nu in range(4):
            ac = g[keys[mu]] @ g[keys[nu]] + g[keys[nu]] @ g[keys[mu]]
            target = 2 * eta[mu, nu] * np.eye(4)
            clifford_err = max(clifford_err, np.linalg.norm(ac - target))
            anticomm[mu, nu] = np.real(ac[0, 0]) / 2  # = eta_{mu nu}

    return {
        "quaternion->Pauli homomorphism (err)": homo_err,
        "unit quaternion is unitary (err)": su2_unitary,
        "unit quaternion det==1 (err)": su2_det,
        "Dirac Clifford relation (err)": clifford_err,
        "recovered_metric": anticomm,
    }


def make_figure(res: dict):
    g = dirac_gammas()
    keys = ["g0", "g1", "g2", "g3"]
    titles = [r"$\gamma^0$", r"$\gamma^1$", r"$\gamma^2$", r"$\gamma^3$"]

    fig = plt.figure(figsize=(11.5, 5.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])

    # top row: the four Dirac gammas (real + imaginary encoded as signed magnitude)
    for idx, (k, t) in enumerate(zip(keys, titles)):
        ax = fig.add_subplot(gs[0, idx])
        M = g[k]
        disp = np.real(M) + np.imag(M)  # entries are purely real or purely imag
        ax.imshow(disp, cmap="coolwarm", vmin=-1, vmax=1)
        for r in range(4):
            for c in range(4):
                val = M[r, c]
                if abs(val) > 1e-9:
                    s = f"{val.real:+.0f}" if abs(val.imag) < 1e-9 else f"{val.imag:+.0f}i"
                    ax.text(c, r, s, ha="center", va="center", fontsize=9)
        # outline the 2x2 quaternionic blocks
        for (x0, y0) in [(-.5, -.5), (1.5, -.5), (-.5, 1.5), (1.5, 1.5)]:
            ax.add_patch(plt.Rectangle((x0, y0), 2, 2, fill=False,
                                       edgecolor="0.2", lw=1.4))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(t)
    fig.text(0.5, 0.52, r"Dirac gammas = 2$\times$2 blocks of quaternion (Pauli) cells "
                        r"$\;\;\Rightarrow\;\; Cl(1,3)\cong M_2(\mathbb{H})$",
             ha="center", fontsize=11)

    # bottom left: recovered Minkowski metric from the anticommutator
    ax = fig.add_subplot(gs[1, 0:2])
    M = res["recovered_metric"]
    im = ax.imshow(M, cmap="RdBu", vmin=-1, vmax=1)
    for r in range(4):
        for c in range(4):
            ax.text(c, r, f"{M[r, c]:+.0f}", ha="center", va="center", fontsize=12,
                    fontweight="bold")
    ax.set_xticks(range(4), [r"$\nu{=}0$", "1", "2", "3"])
    ax.set_yticks(range(4), [r"$\mu{=}0$", "1", "2", "3"])
    ax.set_title(r"$\frac{1}{2}\{\gamma^\mu,\gamma^\nu\}=\eta^{\mu\nu}=$diag$(+,-,-,-)$")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # bottom right: the quaternion Cayley table as Pauli matrices (text)
    ax = fig.add_subplot(gs[1, 2:4])
    ax.axis("off")
    lines = [
        r"$\mathbb{H}\to\mathfrak{su}(2)$  (algebra isomorphism):",
        r"   $1\mapsto I,\quad i\mapsto -i\sigma_x,$",
        r"   $j\mapsto -i\sigma_y,\quad k\mapsto -i\sigma_z$",
        "",
        f"homomorphism error:   {res['quaternion->Pauli homomorphism (err)']:.1e}",
        f"unit q is SU(2):       {res['unit quaternion is unitary (err)']:.1e}",
        f"Dirac Clifford error:  {res['Dirac Clifford relation (err)']:.1e}",
    ]
    ax.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", fontsize=11,
            family="monospace" if False else None)

    fig.suptitle("Thread 2 -- the Cayley table is the atomic cell of the Dirac algebra",
                 y=1.0, fontsize=13)
    path = os.path.join(FIG_DIR, "fig10_dirac.png")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def main():
    print("Thread 2: quaternions -> Pauli -> Dirac")
    res = verify()
    for k, v in res.items():
        if isinstance(v, float):
            print(f"  {k:<40} {v:.3e}")
    print("  recovered metric eta =", np.diag(res["recovered_metric"]).tolist())
    make_figure(res)


if __name__ == "__main__":
    main()
