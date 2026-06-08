"""
spinor.py
=========

Thread 1 -- the 2*pi / 4*pi doubling.

The complex unit circle is 2*pi-periodic:  e^{i(theta+2pi)} = e^{i theta}.
Quaternions are 4*pi-periodic.  A rotation by angle phi about a unit axis n is

        q(phi) = cos(phi/2) + n sin(phi/2)            (note the HALF angle)

so q(2pi) = -1  (not +1), and only q(4pi) = +1.  Yet q and -q act identically
on vectors:  R_q(v) = q v q* = (-q) v (-q)*.  The quaternion (the "spinor") is
4*pi-periodic while its rotation action on space is 2*pi-periodic.  This is the
double cover SU(2) -> SO(3): the belt trick / orientation entanglement, and the
exact reason the phase functional sigma(q) = (phi/2) n sees the HALF angle.

This module demonstrates all of that numerically and writes ../figures/fig9_spinor.png.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import framework as F

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def rotor(phi: float, axis: np.ndarray) -> np.ndarray:
    """Unit quaternion rotating by angle phi about a unit axis (half-angle)."""
    n = axis / np.linalg.norm(axis)
    return np.array([np.cos(phi / 2), *(n * np.sin(phi / 2))])


def rotate_vector(q: np.ndarray, v3: np.ndarray) -> np.ndarray:
    """Apply the rotation R_q(v) = q v q* to a 3-vector v3."""
    v = np.array([0.0, *v3])
    out = F.hamilton(F.hamilton(q, v), F.conjugate(q))
    return out[1:]


def demonstrate() -> dict:
    axis = np.array([0.0, 0.0, 1.0])  # rotate about k
    v0 = np.array([1.0, 0.0, 0.0])    # a test vector

    # (1) q(2pi) = -1 and q(4pi) = +1
    q_2pi = rotor(2 * np.pi, axis)
    q_4pi = rotor(4 * np.pi, axis)
    err_2pi = np.linalg.norm(q_2pi - (-F.ONE))
    err_4pi = np.linalg.norm(q_4pi - F.ONE)

    # (2) q and -q give the SAME rotation of space
    rng = np.random.default_rng(0)
    same = 0.0
    for _ in range(5000):
        q = F.random_unit_quaternion(rng)
        v = rng.normal(size=3)
        same = max(same, np.linalg.norm(rotate_vector(q, v) - rotate_vector(-q, v)))

    # (3) the rotation action returns after 2pi, the spinor only after 4pi
    v_2pi = rotate_vector(q_2pi, v0)
    action_return_2pi = np.linalg.norm(v_2pi - v0)

    return {
        "q(2pi) == -1 (err)": err_2pi,
        "q(4pi) == +1 (err)": err_4pi,
        "R_q == R_{-q} (max err)": same,
        "vector returns at 2pi (err)": action_return_2pi,
    }


def make_figure():
    axis = np.array([0.0, 0.0, 1.0])
    v0 = np.array([1.0, 0.0, 0.0])
    phis = np.linspace(0, 4 * np.pi, 600)

    q_real = np.cos(phis / 2)                       # real part of the spinor
    # overlap of the rotated vector with its start = cos(phi): 2pi-periodic
    overlap = np.array([rotate_vector(rotor(p, axis), v0) @ v0 for p in phis])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))

    ax1.plot(phis / np.pi, q_real, color="#1f77b4", lw=2,
             label=r"spinor  $\mathrm{Re}\,q(\varphi)=\cos(\varphi/2)$")
    ax1.plot(phis / np.pi, overlap, color="#d62728", lw=2, ls="--",
             label=r"space action  $R_q(v)\cdot v=\cos\varphi$")
    for x, lab, col in [(0, "+1", "#1f77b4"), (2, "$-1$ (spinor)", "#1f77b4"),
                        (4, "+1", "#1f77b4")]:
        ax1.axvline(x, color="0.8", lw=1)
    ax1.scatter([2], [np.cos(np.pi)], color="#1f77b4", zorder=5)
    ax1.annotate(r"$q(2\pi)=-1$", (2, -1), textcoords="offset points",
                 xytext=(8, 10), color="#1f77b4")
    ax1.scatter([4], [1], color="#1f77b4", zorder=5)
    ax1.annotate(r"$q(4\pi)=+1$", (4, 1), textcoords="offset points",
                 xytext=(-90, -18), color="#1f77b4")
    ax1.set_xlabel(r"rotation angle  $\varphi / \pi$")
    ax1.set_ylabel("value")
    ax1.set_title("The 4$\\pi$ spinor vs the 2$\\pi$ rotation of space")
    ax1.legend(fontsize=9, loc="lower right")

    # right: phase functional sigma sees the HALF angle (slope 1/2)
    sig = np.array([np.linalg.norm(F.sigma(rotor(p, axis))) for p in phis])
    ax2.plot(phis / np.pi, sig / np.pi, color="#2ca02c", lw=2,
             label=r"$\|\sigma(q)\|$  (phase functional)")
    ax2.plot(phis / np.pi, (phis / 2) / np.pi, color="0.5", lw=1, ls=":",
             label=r"slope $\frac{1}{2}$:  $\varphi/2$")
    ax2.set_xlabel(r"rotation angle  $\varphi / \pi$")
    ax2.set_ylabel(r"$\|\sigma\| / \pi$")
    ax2.set_title(r"$\sigma$ reads the half-angle: the $2\pi\!\to\!4\pi$ factor")
    ax2.legend(fontsize=9, loc="upper left")
    # sigma is the principal value in [0, pi] -> it folds back; annotate
    ax2.annotate("principal branch folds\nat $\\varphi=2\\pi$", (2.4, 0.7),
                 fontsize=8, color="0.4")

    fig.suptitle("Thread 1 -- the $2\\pi/4\\pi$ doubling (double cover SU(2)$\\to$SO(3))",
                 y=1.03, fontsize=13)
    path = os.path.join(FIG_DIR, "fig9_spinor.png")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def main():
    print("Thread 1: the 2pi/4pi doubling")
    for k, v in demonstrate().items():
        print(f"  {k:<34} {v:.3e}")
    make_figure()


if __name__ == "__main__":
    main()
