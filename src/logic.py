"""
logic.py
========

Part III -- the logical substrate.

The framework opened with a *logic*:  e(x)=e(y) when sigma(x)=-sigma(y).  This
module shows that the logic is literally Boolean.  The four minterms of two
Boolean variables A, B are the four landmarks of the unit circle:

    AB    (A and B,        agreement / XNOR=1)  ->   1   (angle 0)
    A!B   (A and not B,    disagreement/ XOR=1)  ->   i   (angle pi/2)
    !A!B  (not A, not B,    agreement / XNOR=1)  ->  -1   (angle pi)
    !AB   (not A and B,    disagreement/ XOR=1)  ->  -i   (angle 3pi/2)

so that

  * { A!B , !AB } are CONJUGATES        = the imaginary axis { i, -i };
  * { AB , !A!B } are COMPLEMENTS        = the real axis      { 1, -1 };

and the two natural Boolean involutions realise the two algebraic ones:

  * swap A<->B          = complex conjugation  (fixes 1,-1; flips i<->-i);
  * complement A,B      = negation / antipode  (1<->-1 and i<->-i).

The disagreement bit XOR(A,B) is exactly the "is it imaginary?" / grade bit, and
it ADDS mod 2 under multiplication -- the Z_2 grading.  This is the n=1 shadow of
the general fact (Cayley-Dickson / twisted group algebra of Z_2^n):

    e_a . e_b  =  (-1)^{phi(a,b)} e_{a XOR b},

the product index is the BITWISE XOR of the indices and the sign is a Boolean
cocycle phi.  We verify this for H (basis 1,i,j,k indexed by 00,01,10,11) and draw
fig16.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import framework as F

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

# minterm (a, b) with a = polarity of A, b = polarity of B  ->  complex landmark
MINTERM_TO_C = {
    (1, 1): 1 + 0j,     # AB    -> 1
    (1, 0): 0 + 1j,     # A!B   -> i
    (0, 1): 0 - 1j,     # !AB   -> -i
    (0, 0): -1 + 0j,    # !A!B  -> -1
}
MINTERM_LABEL = {(1, 1): "AB", (1, 0): "A!B", (0, 1): "!AB", (0, 0): "!A!B"}


def swap(m):           # A <-> B
    return (m[1], m[0])


def complement(m):     # A -> !A, B -> !B
    return (1 - m[0], 1 - m[1])


def parity(m):         # XOR(A, B): 1 = disagreement = imaginary axis
    return m[0] ^ m[1]


def verify_boolean_circle() -> dict:
    """Conjugation = swap; negation = complement; XOR = the imaginary grade."""
    swap_is_conj = 0.0
    comp_is_neg = 0.0
    parity_is_imag = 0.0
    for m, z in MINTERM_TO_C.items():
        swap_is_conj = max(swap_is_conj, abs(np.conj(z) - MINTERM_TO_C[swap(m)]))
        comp_is_neg = max(comp_is_neg, abs(-z - MINTERM_TO_C[complement(m)]))
        # parity bit == 1  iff  z is imaginary (real part 0)
        parity_is_imag = max(parity_is_imag, abs(parity(m) - (1 if abs(z.real) < 1e-9 else 0)))

    # XOR adds mod 2 under multiplication of the units (the Z_2 grading)
    xor_adds = 0.0
    for m1 in MINTERM_TO_C:
        for m2 in MINTERM_TO_C:
            prod = MINTERM_TO_C[m1] * MINTERM_TO_C[m2]
            grade_prod = 1 if abs(prod.real) < 1e-9 else 0
            xor_adds = max(xor_adds, abs(grade_prod - (parity(m1) ^ parity(m2))))

    return {
        "swap == conjugation (err)": swap_is_conj,
        "complement == negation (err)": comp_is_neg,
        "XOR == imaginary grade (err)": parity_is_imag,
        "grade adds via XOR under x (err)": xor_adds,
    }


# --- the general Cayley-Dickson / Z_2^n statement, verified on H ----------------
H_BASIS = {0: F.ONE, 1: F.I, 2: F.J, 3: F.K}   # integer index = bit pattern 00,01,10,11
H_NAME = {0: "1", 1: "i", 2: "j", 3: "k"}


def verify_xor_index() -> dict:
    """For H: e_a . e_b = sign * e_{a XOR b}.  Index is bitwise XOR; sign is Boolean."""
    index_err = 0
    xor_table = np.zeros((4, 4), dtype=int)
    sign_table = np.zeros((4, 4), dtype=int)
    for a in range(4):
        for b in range(4):
            prod = F.hamilton(H_BASIS[a], H_BASIS[b])
            support = int(np.argmax(np.abs(prod)))
            sign = int(np.sign(prod[support]))
            xor_table[a, b] = a ^ b
            sign_table[a, b] = sign
            if support != (a ^ b):
                index_err += 1

    # conjugation: sign is -1 exactly on the nonzero (imaginary) indices
    conj_is_boolean = 0
    for a in range(4):
        qc = F.conjugate(H_BASIS[a])
        expected = -1 if a != 0 else 1
        if int(np.sign(qc[int(np.argmax(np.abs(qc)))])) != expected:
            conj_is_boolean += 1

    return {
        "index_is_xor_violations": index_err,         # must be 0
        "conjugation_boolean_violations": conj_is_boolean,
        "xor_table": xor_table,
        "sign_table": sign_table,
    }


def rotated_conjugation(z, theta):
    """Conjugation across the axis at angle theta:  C_theta(z) = e^{2 i theta} conj(z).

    Fixes the diameter at angle theta; flips the perpendicular diameter (theta+pi/2).
    theta=0 is ordinary conjugation (fixes 1,-1; flips i,-i); theta=pi/2 swaps the
    roles (fixes i,-i; flips 1,-1).  Every theta is a valid involution.
    """
    return np.exp(2j * theta) * np.conj(z)


def verify_rotation(n: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    # involution for arbitrary axis and arbitrary point
    invol = 0.0
    fixed_axis = 0.0
    for _ in range(n):
        theta = rng.uniform(0, np.pi)
        z = rng.normal() + 1j * rng.normal()
        invol = max(invol, abs(rotated_conjugation(rotated_conjugation(z, theta), theta) - z))
        on_axis = np.exp(1j * theta) * abs(z)            # a point on the theta-diameter
        fixed_axis = max(fixed_axis, abs(rotated_conjugation(on_axis, theta) - on_axis))

    # theta = 0 : fixes {1,-1}, flips {i,-i}
    c0_real = max(abs(rotated_conjugation(1, 0) - 1), abs(rotated_conjugation(-1, 0) + 1))
    c0_imag = max(abs(rotated_conjugation(1j, 0) + 1j), abs(rotated_conjugation(-1j, 0) - 1j))
    # theta = 90 deg : fixes {i,-i}, flips {1,-1}  (the user's rotation)
    h = np.pi / 2
    c90_imag = max(abs(rotated_conjugation(1j, h) - 1j), abs(rotated_conjugation(-1j, h) + 1j))
    c90_real = max(abs(rotated_conjugation(1, h) + 1), abs(rotated_conjugation(-1, h) - 1))
    return {
        "involution any axis (err)": invol,
        "theta-diameter is fixed (err)": fixed_axis,
        "theta=0 fixes {1,-1} (err)": c0_real,
        "theta=0 flips {i,-i} (err)": c0_imag,
        "theta=90 fixes {i,-i} (err)": c90_imag,
        "theta=90 flips {1,-1} (err)": c90_real,
    }


def make_rotation_figure():
    thetas = [0.0, np.pi / 4, np.pi / 2]
    titles = [r"$\theta=0^\circ$: real $\{1,-1\}$ fixed",
              r"$\theta=45^\circ$: a mixed axis",
              r"$\theta=90^\circ$: imaginary $\{i,-i\}$ fixed"]
    pts = {1 + 0j: "AB=1", 0 + 1j: "A!B=i", -1 + 0j: "!A!B=-1", 0 - 1j: "!AB=-i"}

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    th = np.linspace(0, 2 * np.pi, 400)
    for ax, theta, title in zip(axes, thetas, titles):
        ax.plot(np.cos(th), np.sin(th), color="0.75", lw=1.2)
        # fixed diameter (green) and flipped diameter (red)
        d = np.array([np.cos(theta), np.sin(theta)])
        p = np.array([np.cos(theta + np.pi / 2), np.sin(theta + np.pi / 2)])
        ax.plot([-1.3 * d[0], 1.3 * d[0]], [-1.3 * d[1], 1.3 * d[1]],
                color="#2ca02c", lw=2, label="fixed axis (mirror)")
        ax.plot([-1.3 * p[0], 1.3 * p[0]], [-1.3 * p[1], 1.3 * p[1]],
                color="#d62728", lw=1.4, ls="--", label="flipped diameter")
        for z, lab in pts.items():
            img = rotated_conjugation(z, theta)
            moved = abs(img - z) > 1e-9
            col = "#d62728" if moved else "#2ca02c"
            ax.plot([z.real], [z.imag], "o", color=col, ms=8, zorder=5)
            ax.annotate(lab, (z.real, z.imag), textcoords="offset points",
                        xytext=(11 * (z.real if z.real else 0.3),
                                14 * (z.imag if z.imag else 0.6)),
                        ha="center", fontsize=9, color=col, fontweight="bold")
            if moved:
                ax.annotate("", xy=(img.real, img.imag), xytext=(z.real, z.imag),
                            arrowprops=dict(arrowstyle="-|>", color="#d62728",
                                            lw=1.1, alpha=0.6,
                                            connectionstyle="arc3,rad=0.3"))
        ax.set_aspect("equal"); ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.axis("off")
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle(r"Conjugation is a choice of axis: $C_\theta(z)=e^{2i\theta}\,\bar z$"
                 r" — rotating $90^\circ$ swaps which pair is 'real'",
                 y=1.02, fontsize=12.5)
    path = os.path.join(FIG_DIR, "fig17_rotation.png")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def make_figure(res_h: dict):
    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])

    # --- panel 1: the four minterms on the unit circle ---
    ax = fig.add_subplot(gs[0, 0])
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="0.7", lw=1.2)
    real_col, imag_col = "#1f77b4", "#d62728"
    for m, z in MINTERM_TO_C.items():
        col = imag_col if parity(m) else real_col
        ax.plot([z.real], [z.imag], "o", color=col, ms=9, zorder=5)
        zexpr = {1: "1", -1: "-1", 1j: "i", -1j: "-i"}[z]
        ax.annotate(f"{MINTERM_LABEL[m]} = {zexpr}", (z.real, z.imag),
                    textcoords="offset points",
                    xytext=(14 * (z.real if z.real else 0.2 * np.sign(z.imag + .1)),
                            16 * (z.imag if z.imag else 0.6)),
                    ha="center", color=col, fontsize=11, fontweight="bold")
    # conjugation = swap A<->B  : i <-> -i (vertical)
    ax.annotate("", xy=(0, -1), xytext=(0, 1),
                arrowprops=dict(arrowstyle="<|-|>", color=imag_col, lw=1.6,
                                connectionstyle="arc3,rad=0.25"))
    ax.text(0.46, 0.0, "conjugation\n= swap A$\\leftrightarrow$B", color=imag_col,
            fontsize=8.5, ha="center", va="center")
    # negation = complement : through origin
    ax.annotate("", xy=(-1, 0), xytext=(1, 0),
                arrowprops=dict(arrowstyle="<|-|>", color=real_col, lw=1.6))
    ax.text(0.0, -0.5, "negation = complement A,B", color=real_col,
            fontsize=8.5, ha="center")
    ax.set_aspect("equal"); ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7)
    ax.axis("off")
    ax.set_title("Minterms of (A,B) = unit-circle landmarks\n"
                 "agreement (XNOR) = real • disagreement (XOR) = imaginary",
                 fontsize=10)

    # --- panel 2: H product index = a XOR b ---
    ax2 = fig.add_subplot(gs[0, 1])
    xt = res_h["xor_table"]
    ax2.imshow(xt, cmap="Blues", vmin=0, vmax=3)
    for a in range(4):
        for b in range(4):
            ax2.text(b, a, H_NAME[xt[a, b]], ha="center", va="center",
                     fontsize=13, fontweight="bold")
    ax2.set_xticks(range(4), [H_NAME[i] for i in range(4)])
    ax2.set_yticks(range(4), [H_NAME[i] for i in range(4)])
    ax2.set_xlabel("b  (index $\\to$ bits)")
    ax2.set_ylabel("a")
    ax2.set_title("$\\mathbb{H}$ product index $= a \\oplus b$\n(bitwise XOR: 1,i,j,k = 00,01,10,11)",
                  fontsize=10)

    # --- panel 3: the Boolean sign cocycle ---
    ax3 = fig.add_subplot(gs[0, 2])
    st = res_h["sign_table"]
    ax3.imshow(st, cmap="RdBu", vmin=-1, vmax=1)
    for a in range(4):
        for b in range(4):
            ax3.text(b, a, "$+$" if st[a, b] > 0 else "$-$", ha="center",
                     va="center", fontsize=15, fontweight="bold",
                     color="black" if st[a, b] > 0 else "white")
    ax3.set_xticks(range(4), [H_NAME[i] for i in range(4)])
    ax3.set_yticks(range(4), [H_NAME[i] for i in range(4)])
    ax3.set_xlabel("b"); ax3.set_ylabel("a")
    ax3.set_title("sign $=(-1)^{\\phi(a,b)}$\nthe Boolean cocycle", fontsize=10)

    fig.suptitle("Part III — the logical substrate: multiplication is XOR + a Boolean sign",
                 y=1.04, fontsize=13)
    path = os.path.join(FIG_DIR, "fig16_logic.png")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def main():
    print("Part III: the logical substrate")
    for k, v in verify_boolean_circle().items():
        print(f"  {k:<38} {v:.2e}")
    res_h = verify_xor_index()
    print(f"  H: product index = a XOR b violations    {res_h['index_is_xor_violations']}")
    print(f"  H: conjugation Boolean violations         {res_h['conjugation_boolean_violations']}")
    print("  H XOR index table (rows/cols = 1,i,j,k):")
    for a in range(4):
        print("    " + "  ".join(H_NAME[res_h["xor_table"][a, b]] for b in range(4)))
    print("  rotated conjugation C_theta(z) = e^{2i theta} conj(z):")
    for k, v in verify_rotation().items():
        print(f"    {k:<32} {v:.2e}")
    make_figure(res_h)
    make_rotation_figure()


if __name__ == "__main__":
    main()
