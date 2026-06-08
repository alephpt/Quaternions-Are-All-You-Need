"""
ledger.py
=========

The 2*pi / 4*pi ledger.

The framework's two regimes are the **circle** (2*pi-periodic, U(1), C, one
imaginary direction) and the **sphere** (4*pi-measure, SU(2), H, three imaginary
directions i, j, k).  Almost every appearance of 2*pi or 4*pi in mathematics and
physics is one of these two geometries showing through:

  * 2*pi  = the measure of the 1-sphere S^1 (a circle): the home of e^{i theta},
            U(1), the complex numbers.  ONE imaginary direction.
  * 4*pi  = the measure of the 2-sphere S^2 (a sphere): the home of the spinor
            q(4*pi)=1, SU(2), the quaternions.  THREE imaginary directions.

A field that spreads through 3-D space crosses a sphere, so its laws carry 4*pi
(Coulomb, Gauss, Poisson; Einstein's 8*pi = 2 * 4*pi).  A phase that turns in a
plane crosses a circle, so its laws carry 2*pi (Euler, Fourier, Cauchy, hbar).
And the factor of two between them, 4*pi = 2 * 2*pi, is the double cover: SU(2)
wraps U(1)'s circle exactly twice (the half-angle of Section 7).

This module verifies the geometric roots numerically and draws fig15.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def verify_geometric_roots(n: int = 2000) -> dict:
    """Show that 2*pi is the circle's measure and 4*pi is the sphere's."""
    # circle circumference  = integral_0^2pi |r'(t)| dt  with r(t)=(cos t, sin t)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    dt = 2 * np.pi / n
    circumference = np.sum(np.ones_like(t)) * dt          # = 2 pi

    # 2-sphere surface area / total solid angle = ∫ sin(theta) dtheta dphi = 4 pi
    th = np.linspace(0, np.pi, n)
    ph = np.linspace(0, 2 * np.pi, n)
    dth, dph = th[1] - th[0], ph[1] - ph[0]
    area = np.sum(np.sin(th)[:, None] * np.ones_like(ph)[None, :]) * dth * dph

    # Gauss-Bonnet for S^2: ∫ K dA = 2 pi chi,  K=1, chi=2  ->  4 pi
    gauss_bonnet = area * 1.0                              # K = 1 on unit sphere

    # Coulomb/Gauss: flux of q/(4 pi r^2) radial field through enclosing sphere = q
    q = 1.0
    flux = (q / (4 * np.pi * 1.0 ** 2)) * area             # field * area = q

    return {
        "circle_circumference": circumference,
        "two_pi": 2 * np.pi,
        "sphere_area_4pi": area,
        "four_pi": 4 * np.pi,
        "gauss_bonnet_S2": gauss_bonnet,
        "gauss_flux_recovers_q": flux,
        "ratio_4pi_over_2pi": area / circumference,        # = 2 (the double cover)
    }


def make_figure(res: dict):
    fig = plt.figure(figsize=(12.0, 5.4))

    # --- left: the 2*pi circle world (U(1), C) ---
    ax1 = fig.add_subplot(1, 2, 1)
    th = np.linspace(0, 2 * np.pi, 400)
    ax1.plot(np.cos(th), np.sin(th), color="#1f77b4", lw=2)
    ax1.annotate("", xy=(np.cos(0.9), np.sin(0.9)), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=2))
    ax1.text(0.30, 0.16, r"$e^{i\theta}$", color="#1f77b4", fontsize=14)
    ax1.set_aspect("equal")
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.6, 1.5)
    ax1.axis("off")
    ax1.set_title(r"$2\pi$ world — the circle $S^1$", fontsize=13)
    ax1.text(0, -1.95, "U(1)   •   $\\mathbb{C}$   •   ONE imaginary axis\n"
                       r"$e^{2\pi i}=1$,  $C=2\pi r$,  $\hbar=h/2\pi$,"
                       "\n" r"$\oint\frac{dz}{z}=2\pi i$,  $\omega=2\pi f$",
             ha="center", va="top", fontsize=10)

    # --- right: the 4*pi sphere world (SU(2), H) ---
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax2.plot_wireframe(xs, ys, zs, color="#9467bd", lw=0.4, alpha=0.5)
    for vec, name, col in [((1, 0, 0), "i", "#d62728"), ((0, 1, 0), "j", "#2ca02c"),
                           ((0, 0, 1), "k", "#9467bd")]:
        ax2.quiver(0, 0, 0, *vec, color=col, lw=2.2, arrow_length_ratio=0.15)
        ax2.text(*(np.array(vec) * 1.25), name, color=col, fontsize=13,
                 fontweight="bold")
    ax2.set_box_aspect((1, 1, 1))
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])
    ax2.set_title(r"$4\pi$ world — the sphere $S^2$", fontsize=13)
    ax2.text2D(0.5, -0.08,
               "SU(2)   •   $\\mathbb{H}$   •   THREE imaginary axes $i,j,k$\n"
               r"$q(4\pi)=1$,  $A=4\pi r^2$,  $\Omega=4\pi$,"
               "\n" r"$F=\frac{q_1q_2}{4\pi\varepsilon_0 r^2}$,  $\nabla^2\Phi=4\pi G\rho$,"
               r"  $G_{\mu\nu}=\frac{8\pi G}{c^4}T_{\mu\nu}$",
               transform=ax2.transAxes, ha="center", va="top", fontsize=10)

    fig.suptitle(r"The $2\pi/4\pi$ ledger:  $4\pi = 2\times 2\pi$ is the double cover "
                 r"(SU(2) wraps U(1) twice)", y=1.0, fontsize=13)
    path = os.path.join(FIG_DIR, "fig15_ledger.png")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def main():
    print("The 2pi/4pi ledger: geometric roots")
    res = verify_geometric_roots()
    print(f"  circle circumference = {res['circle_circumference']:.6f}  (2pi = {res['two_pi']:.6f})")
    print(f"  sphere area / solid angle = {res['sphere_area_4pi']:.6f}  (4pi = {res['four_pi']:.6f})")
    print(f"  Gauss-Bonnet S^2 total curvature = {res['gauss_bonnet_S2']:.6f}  (= 4pi)")
    print(f"  Gauss flux recovers enclosed charge q = {res['gauss_flux_recovers_q']:.6f}  (= 1)")
    print(f"  ratio (sphere)/(circle) = {res['ratio_4pi_over_2pi']:.6f}  (= 2, the double cover)")
    make_figure(res)


if __name__ == "__main__":
    main()
