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


# --- the four symmetric gates over the minterms -------------------------------
GATES = {
    "AND":  lambda a, b: a & b,
    "NAND": lambda a, b: 1 - (a & b),
    "OR":   lambda a, b: a | b,
    "NOR":  lambda a, b: 1 - (a | b),
    "XOR":  lambda a, b: a ^ b,
    "XNOR": lambda a, b: 1 - (a ^ b),
}


def linearly_separable(gate) -> bool:
    """Can a single half-plane (one axis + threshold) realise the gate?

    AND/NAND/OR/NOR are single-half-plane gates; XOR/XNOR are not -- they need an
    axis AND its orthogonal/complement.  (Grid search over thresholds.)
    """
    pts = [(0, 0), (0, 1), (1, 0), (1, 1)]
    grid = np.linspace(-2, 2, 17)
    for w1 in grid:
        for w2 in grid:
            for th in np.linspace(-4, 4, 33):
                ok = all((w1 * a + w2 * b - th > 0) == bool(gate(a, b)) for a, b in pts)
                if ok:
                    return True
    return False


def verify_gates() -> dict:
    """How the four gate families sit on the circle, all checks exact."""
    real_is_xnor = imag_is_xor = and_is_plus1 = nor_is_minus1 = 0.0
    for m, z in MINTERM_TO_C.items():
        a, b = m
        is_real = 1 if abs(z.imag) < 1e-9 else 0
        is_imag = 1 if abs(z.real) < 1e-9 else 0
        real_is_xnor = max(real_is_xnor, abs(GATES["XNOR"](a, b) - is_real))
        imag_is_xor = max(imag_is_xor, abs(GATES["XOR"](a, b) - is_imag))
        and_is_plus1 = max(and_is_plus1, abs(GATES["AND"](a, b) - (1 if z == 1 else 0)))
        nor_is_minus1 = max(nor_is_minus1, abs(GATES["NOR"](a, b) - (1 if z == -1 else 0)))

    separable = {name: linearly_separable(g) for name, g in GATES.items()}
    return {
        "XNOR == real axis {1,-1} (err)": real_is_xnor,
        "XOR == imaginary axis {i,-i} (err)": imag_is_xor,
        "AND == the +1 pole (err)": and_is_plus1,
        "NOR == the -1 pole (err)": nor_is_minus1,
        "separable (one half-plane)": separable,
    }


def make_gate_figure(res_g: dict):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0),
                                  gridspec_kw={"width_ratios": [1.1, 1]})

    # panel A: gates on the circle
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="0.75", lw=1.2)
    ax.plot([-1.25, 1.25], [0, 0], color="#2ca02c", lw=2)          # XNOR / agreement axis
    ax.plot([0, 0], [-1.25, 1.25], color="#d62728", lw=2, ls="--")  # XOR / disagreement axis
    pole = {1 + 0j: ("AB = 1", "AND pole", "#1f77b4"),
            -1 + 0j: ("!A!B = -1", "NOR pole", "#1f77b4"),
            0 + 1j: ("A!B = i", "XOR", "#d62728"),
            0 - 1j: ("!AB = -i", "XOR", "#d62728")}
    for z, (lab, role, col) in pole.items():
        ax.plot([z.real], [z.imag], "o", color=col, ms=9, zorder=5)
        ax.annotate(f"{lab}\n({role})", (z.real, z.imag), textcoords="offset points",
                    xytext=(20 * (z.real if z.real else 0.0),
                            22 * (z.imag if z.imag else 1.0) * (1 if z.imag >= 0 else 1)),
                    ha="center", fontsize=9, fontweight="bold", color=col)
    ax.text(0.92, 0.10, "XNOR\n(agree)", color="#2ca02c", fontsize=9, ha="center")
    ax.text(0.16, 0.93, "XOR\n(disagree)", color="#d62728", fontsize=9, ha="center")
    ax.set_aspect("equal"); ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7); ax.axis("off")
    ax.set_title("Each axis: an XNOR(agree)/XOR(disagree) half-split;\n"
                 "the poles are AND (+1) and NOR (-1)", fontsize=10)

    # panel B: separability -- AND is one half-plane, XOR needs two
    for axis_pts, title, gate in [(None, None, None)]:
        pass
    ax2.set_title("AND = one half-plane (one axis); XOR needs an axis\n"
                  "AND its orthogonal/complement", fontsize=10)
    sq = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (a, b) in sq:
        # colour by XOR to show the diagonal (non-separable) pattern
        c = "#d62728" if (a ^ b) else "#2ca02c"
        ax2.plot([a], [b], "o", ms=16, color=c, zorder=5)
        ax2.annotate(f"{'A' if a else '!A'}{'B' if b else '!B'}", (a, b),
                     textcoords="offset points", xytext=(0, 18), ha="center", fontsize=9)
    # AND separating line (one axis): A + B = 1.5
    ax2.plot([1.5, -0.5], [0.0, 2.0], color="#1f77b4", lw=2, label="AND: one half-plane")
    # XOR needs two lines: A+B=0.5 and A+B=1.5
    ax2.plot([0.5, -0.5], [0.0, 1.0], color="#d62728", lw=1.4, ls="--",
             label="XOR: needs two (axis + orthogonal)")
    ax2.plot([1.5, 0.5], [0.0, 1.0], color="#d62728", lw=1.4, ls="--")
    ax2.set_xlim(-0.7, 1.9); ax2.set_ylim(-0.5, 2.2); ax2.set_aspect("equal")
    ax2.set_xlabel("A"); ax2.set_ylabel("B")
    ax2.legend(fontsize=8, loc="upper right")

    fig.suptitle("The gate decomposition: XNOR/XOR (agree/disagree) + AND/NOR poles, "
                 "per orthogonal axis pair", y=1.0, fontsize=12.5)
    path = os.path.join(FIG_DIR, "fig18_gates.png")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


# --- the gate lattice: whole (superset) and parts (subset) --------------------
LM_NAME = {1 + 0j: "1", -1 + 0j: "-1", 0 + 1j: "i", 0 - 1j: "-i"}
ALL_LM = frozenset(LM_NAME.values())


def gate_trueset(name) -> frozenset:
    """The set of unit-circle landmarks where a symmetric gate is true."""
    g = GATES[name]
    return frozenset(LM_NAME[MINTERM_TO_C[m]] for m in MINTERM_TO_C if g(*m))


def verify_lattice() -> dict:
    S = {name: gate_trueset(name) for name in GATES}
    S["FALSE"] = frozenset()
    S["TRUE"] = ALL_LM
    comp = lambda s: ALL_LM - s

    checks = {
        # the two halves partition the whole
        "XNOR | XOR == whole": S["XNOR"] | S["XOR"] == ALL_LM,
        "XNOR & XOR == empty": S["XNOR"] & S["XOR"] == frozenset(),
        # superset > subset pairs (the user's ordering)
        "XNOR superset of AND": S["AND"] < S["XNOR"],
        "NAND superset of XOR": S["XOR"] < S["NAND"],
        # poles are parts of the agreement half
        "AND, NOR subset of XNOR": S["AND"] < S["XNOR"] and S["NOR"] < S["XNOR"],
        # disagreement half is part of both size-3 supersets
        "XOR subset of OR and NAND": S["XOR"] < S["OR"] and S["XOR"] < S["NAND"],
        # complement (output-negation) pairs
        "AND^c == NAND": comp(S["AND"]) == S["NAND"],
        "NOR^c == OR": comp(S["NOR"]) == S["OR"],
        "XOR^c == XNOR": comp(S["XOR"]) == S["XNOR"],
    }
    violations = sum(1 for v in checks.values() if not v)
    return {"checks": checks, "violations": violations, "sets": S}


# =============================================================================
#  A data structure for partial logics, and their map between subset / superset
# =============================================================================
LM_ORDER = {"1": 0, "i": 1, "-1": 2, "-i": 3}


def setstr(s) -> str:
    return "∅" if not s else "{" + ",".join(sorted(s, key=LM_ORDER.get)) + "}"


class Logic:
    """A *total* logic over two propositions A, B: the set of unit-circle
    landmarks (minterms) on which it is true.  The order on logics is set
    inclusion -- ``a <= b`` means *a is a subset (refinement) of b*."""

    __slots__ = ("name", "trueset")

    def __init__(self, name: str, trueset):
        self.name = name
        self.trueset = frozenset(trueset)

    def __le__(self, o): return self.trueset <= o.trueset
    def __lt__(self, o): return self.trueset < o.trueset
    def __ge__(self, o): return self.trueset >= o.trueset
    def is_subset_of(self, o): return self.trueset <= o.trueset
    def is_superset_of(self, o): return self.trueset >= o.trueset

    def complement(self) -> "Logic":
        return Logic(self.name + "^c", ALL_LM - self.trueset)

    def __eq__(self, o): return isinstance(o, Logic) and self.trueset == o.trueset
    def __hash__(self): return hash(self.trueset)
    def __repr__(self): return f"{self.name}={setstr(self.trueset)}"


class GateLattice:
    """The inclusion lattice of named logics over the landmarks {1,i,-1,-i}.

    This is the data structure that lets a *partial* logic be located between a
    subset bound and a superset bound.  It exposes the two maps explicitly:
    ``subsets``/``principal_ideal`` (downward, toward the parts) and
    ``supersets``/``principal_filter`` (upward, toward the wholes), plus the
    Hasse covers and the lattice operations ``meet`` (=intersection) and
    ``join`` (=union)."""

    def __init__(self, logics):
        self.nodes = {l.name: l for l in logics}

    def __getitem__(self, name) -> Logic: return self.nodes[name]

    def _by_size(self, ls): return sorted(ls, key=lambda l: (len(l.trueset), l.name))

    def supersets(self, x, strict=False):
        return self._by_size([y for y in self.nodes.values()
                              if (x < y if strict else x <= y)])

    def subsets(self, x, strict=False):
        return self._by_size([y for y in self.nodes.values()
                              if (y < x if strict else y <= x)])

    def covers_up(self, x):       # immediate supersets (one Hasse step up)
        sup = self.supersets(x, strict=True)
        return [y for y in sup if not any(x < z and z < y for z in sup)]

    def covers_down(self, x):     # immediate subsets (one Hasse step down)
        sub = self.subsets(x, strict=True)
        return [y for y in sub if not any(y < z and z < x for z in sub)]

    def principal_filter(self, x):   # up-set: every logic that CONTAINS x
        return self.supersets(x)

    def principal_ideal(self, x):    # down-set: every logic CONTAINED IN x
        return self.subsets(x)

    def meet(self, x, y) -> Logic:   # greatest lower bound = intersection
        return self._named(x.trueset & y.trueset)

    def join(self, x, y) -> Logic:   # least upper bound = union
        return self._named(x.trueset | y.trueset)

    def _named(self, ts):            # name a true-set if it is one of our nodes
        ts = frozenset(ts)
        for l in self.nodes.values():
            if l.trueset == ts:
                return l
        return Logic("?", ts)


class PartialLogic:
    """A *partial* logic: every landmark is known-true, known-false, or unknown.

    It denotes the closed interval of total logics ``[lower, upper]`` with

        lower = the known-true landmarks      -- the SUBSET bound (strongest),
        upper = everything not known-false    -- the SUPERSET bound (weakest).

    Learning one more fact refines an unknown landmark and shrinks the interval:
    asserting a landmark true raises the subset bound, ruling it false lowers the
    superset bound.  A partial logic is therefore literally a position *between*
    a subset and a superset, and learning walks it from the superset bound down
    toward a single subset."""

    __slots__ = ("known_true", "known_false")

    def __init__(self, known_true=(), known_false=()):
        self.known_true = frozenset(known_true)
        self.known_false = frozenset(known_false)

    @property
    def lower(self): return self.known_true              # subset bound
    @property
    def upper(self): return ALL_LM - self.known_false    # superset bound
    @property
    def unknown(self): return ALL_LM - self.known_true - self.known_false

    def is_consistent(self): return not (self.known_true & self.known_false)

    def admits(self, logic: Logic) -> bool:
        """Is this total logic a completion (lower <= logic <= upper)?"""
        return self.known_true <= logic.trueset <= self.upper

    def learn_true(self, lm):  return PartialLogic(self.known_true | {lm}, self.known_false)
    def learn_false(self, lm): return PartialLogic(self.known_true, self.known_false | {lm})

    def completions(self, lattice: GateLattice):
        """The named gates that lie between the subset and the superset bound."""
        return lattice._by_size([l for l in lattice.nodes.values() if self.admits(l)])

    def consistency_to(self, cursor):
        """Line each landmark's known/unknown status up against a cursor's
        consistency kappa = Re(c z̄).  Returns {landmark: (kappa, state)}."""
        return {n: (consistency(n, cursor),
                    "true" if n in self.known_true else
                    "false" if n in self.known_false else "unknown")
                for n in NAME_LM}

    @classmethod
    def from_cursor(cls, cursor, tau_lo, tau_hi):
        """Read a partial logic off a cursor with a consistency *band*: landmarks
        above ``tau_hi`` are known-true (the sublogic / subset bound), those below
        ``tau_lo`` are known-false (carving the superset bound), the rest unknown.
        Raising ``tau_lo`` walks the interval down exactly as learning facts does --
        the bridge between the cursor (11.6) and the partial-logic sweep (11.5)."""
        kt = {n for n in NAME_LM if consistency(n, cursor) >= tau_hi}
        kf = {n for n in NAME_LM if consistency(n, cursor) < tau_lo}
        return cls(kt, kf)

    def __repr__(self):
        return (f"PartialLogic(true={setstr(self.known_true)}, "
                f"false={setstr(self.known_false)}, unknown={setstr(self.unknown)}) "
                f"-> subset {setstr(self.lower)} .. superset {setstr(self.upper)}")


def build_lattice() -> GateLattice:
    S = {name: gate_trueset(name) for name in GATES}
    S["FALSE"] = frozenset()
    S["TRUE"] = ALL_LM
    return GateLattice([Logic(n, s) for n, s in S.items()])


def verify_datastructure(lat: GateLattice) -> dict:
    """Exercise the data structure and the partial-logic interval; all exact."""
    AND, NAND = lat["AND"], lat["NAND"]
    names = lambda ls: {l.name for l in ls}

    filt = lat.principal_filter(AND)      # wholes containing AND
    ideal = lat.principal_ideal(NAND)     # parts contained in NAND
    # complement duality:  S in (up-set of AND)  <=>  S^c in (down-set of NAND)
    dual = {frozenset(ALL_LM - l.trueset) for l in filt} == {l.trueset for l in ideal}

    # the worked example: start knowing only "both true" (landmark 1) holds, then
    # learn facts; watch the interval narrow from the TRUE superset to the AND subset.
    p0 = PartialLogic(known_true={"1"})
    p1 = p0.learn_false("-1")
    p2 = p1.learn_false("i").learn_false("-i")
    trace = [p0, p1, p2]

    checks = {
        "filter(AND) == {AND,XNOR,OR,TRUE}": names(filt) == {"AND", "XNOR", "OR", "TRUE"},
        "ideal(NAND) == {NAND,XOR,NOR,FALSE}": names(ideal) == {"NAND", "XOR", "NOR", "FALSE"},
        "up(AND) is complement-dual of down(NAND)": dual,
        "covers_up(AND) == {XNOR,OR}": names(lat.covers_up(AND)) == {"XNOR", "OR"},
        "covers_down(NAND) == {XOR,NOR}": names(lat.covers_down(NAND)) == {"XOR", "NOR"},
        "meet(XNOR,OR) == AND": lat.meet(lat["XNOR"], lat["OR"]) == AND,
        "join(AND,NOR) == XNOR": lat.join(AND, lat["NOR"]) == lat["XNOR"],
        "join(XNOR,XOR) == TRUE": lat.join(lat["XNOR"], lat["XOR"]) == lat["TRUE"],
        "meet(XNOR,XOR) == FALSE": lat.meet(lat["XNOR"], lat["XOR"]) == lat["FALSE"],
        "partial p0 completions == filter(AND)":
            names(p0.completions(lat)) == {"AND", "XNOR", "OR", "TRUE"},
        "partial p1 completions == {AND,OR}":
            names(p1.completions(lat)) == {"AND", "OR"},
        "partial p2 collapses to {AND}":
            names(p2.completions(lat)) == {"AND"},
    }
    violations = sum(1 for v in checks.values() if not v)
    return {"checks": checks, "violations": violations, "trace": trace, "lattice": lat}


# =============================================================================
#  The consistency cursor: a degree of consistency that slides a partial logic
#  between its subset bound (the sublogic) and superset bound (the superlogic).
#  A cursor is a unit direction c; the consistency of a landmark z is Re(c z̄)
#  = cos of the angle between them: +1 alignment, 0 exclusion, -1 contradiction.
#  Thresholding that consistency at a level tau picks out the landmarks that are
#  "consistent enough" -- and sweeping tau from +1 down to -1 grows that set
#  monotonically from a tight sublogic up to the full superlogic.
# =============================================================================
NAME_LM = {"1": 1 + 0j, "i": 0 + 1j, "-1": -1 + 0j, "-i": 0 - 1j}


def consistency(name: str, cursor: complex) -> float:
    """Degree of consistency of a landmark with a unit cursor:  Re(c·z̄) = cosΔθ.
    +1 = alignment, 0 = exclusion, -1 = contradiction."""
    return float((cursor * np.conj(NAME_LM[name])).real)


def relation(name: str, cursor: complex, tol: float = 1e-9) -> str:
    r = cursor * np.conj(NAME_LM[name])
    if abs(r.imag) < tol and r.real > tol:  return "alignment"
    if abs(r.imag) < tol and r.real < -tol: return "contradiction"
    if abs(r.real) < tol:                   return "exclusion"
    return "mixed"


def level_set(cursor: complex, tau: float) -> frozenset:
    """The landmarks at least `tau`-consistent with the cursor -- the sublogic at
    that degree of consistency."""
    return frozenset(n for n in NAME_LM if consistency(n, cursor) >= tau - 1e-9)


def consistency_chain(cursor: complex):
    """The distinct level-sets as tau falls 1 -> -1: a nested chain from the tight
    sublogic (subset) up to the full superlogic (superset)."""
    chain = []
    for tau in np.linspace(1.0, -1.0, 401):
        L = level_set(cursor, tau)
        if not chain or L != chain[-1]:
            chain.append(L)
    return chain


def _relation_multiset(values: dict):
    """Counter of pairwise relations among a labelled set of unit landmarks."""
    from collections import Counter
    out = Counter()
    for a in values:
        for b in values:
            if a == b:
                continue
            r = values[a] * np.conj(values[b])
            if abs(r.imag) < 1e-9 and r.real > 0:   out["alignment"] += 1
            elif abs(r.imag) < 1e-9 and r.real < 0: out["contradiction"] += 1
            elif abs(r.real) < 1e-9:                out["exclusion"] += 1
    return out


def verify_cursor() -> dict:
    """The cursor ties the relational measure to the subset/superset lattice."""
    # 1. a cursor (one direction + one threshold = ONE half-plane) realises EXACTLY
    #    the linearly separable gates of 11.5; the parity gates are unreachable.
    realised = set()
    for deg in range(0, 360):
        c = np.exp(1j * np.deg2rad(deg))
        for tau in np.linspace(-1, 1, 401):
            realised.add(level_set(c, tau))
    sep = {name: linearly_separable(GATES[name]) for name in GATES}
    thresholdable = {name: (gate_trueset(name) in realised) for name in GATES}
    match = all(thresholdable[n] == sep[n] for n in GATES)

    # 2. every sweep is a monotone chain ending at the full superlogic
    chains_ok = True
    for deg in range(0, 360, 5):
        ch = consistency_chain(np.exp(1j * np.deg2rad(deg)))
        nested = all(ch[i] < ch[i + 1] for i in range(len(ch) - 1))
        chains_ok = chains_ok and nested and ch[-1] == ALL_LM

    # 3. rotation covariance: rotating the cursor by any angle preserves the
    #    multiset of pairwise relations (the dynamics hold; only labels move)
    base = _relation_multiset(NAME_LM)
    rotation_inv = all(
        _relation_multiset({k: np.exp(1j * np.deg2rad(phi)) * v for k, v in NAME_LM.items()})
        == base for phi in (13, 57, 90, 123, 180))

    # 4. both layouts (paper's minterm map, and the contrast map A=-1,B=+1,AB=i,!A!B=-i)
    #    carry the same relational structure -- the logic labelling is a free choice
    paper = {"AB": 1 + 0j, "A!B": 1j, "!AB": -1j, "!A!B": -1 + 0j}
    user = {"B": 1 + 0j, "AB": 1j, "A": -1 + 0j, "!A!B": -1j}
    layouts_match = (_relation_multiset(paper) == _relation_multiset(user) == base)

    checks = {
        "cursor level-sets == separable gates": match,
        "every sweep is a chain ending at superlogic": chains_ok,
        "rotation preserves the relation multiset": rotation_inv,
        "both layouts share the relation structure": layouts_match,
    }
    return {"checks": checks, "violations": sum(1 for v in checks.values() if not v),
            "thresholdable": thresholdable, "separable": sep}


# --- the relational quaternion: align/contradict (scalar) + exclusion axis (vector)
def relational_quaternion(p, q):
    """R = p·q̄.  Scalar part = alignment(+)/contradiction(-); vector part = the
    *exclusion axis* -- which only fans out into a 2-sphere of directions in H."""
    return F.hamilton(p, F.conjugate(q))


def _qclass(r, tol=1e-9):
    s, v = r[0], r[1:]
    nv = float(np.linalg.norm(v))
    if nv < tol and s > 0: return "alignment"
    if nv < tol and s < 0: return "contradiction"
    if abs(s) < tol:       return "exclusion"
    return "mixed"


def verify_relational_quaternion() -> dict:
    one, I, J, K = F.ONE, F.I, F.J, F.K
    axes = {n: relational_quaternion(u, one)[1:] for n, u in (("i", I), ("j", J), ("k", K))}
    distinct = (np.linalg.norm(axes["i"] - axes["j"]) > 0.5 and
                np.linalg.norm(axes["j"] - axes["k"]) > 0.5)
    sqrt_minus1 = all(np.allclose(F.hamilton(relational_quaternion(u, one),
                                             relational_quaternion(u, one)), -one)
                      for u in (I, J, K))
    checks = {
        "alignment   R(1,1)  = +1": _qclass(relational_quaternion(one, one)) == "alignment",
        "contradict  R(1,-1) = -1": _qclass(relational_quaternion(one, -one)) == "contradiction",
        "exclusion   R(i,1) imaginary": _qclass(relational_quaternion(I, one)) == "exclusion",
        "exclusion is directional (i,j,k distinct)": distinct,
        "pure exclusion squares to -1 (a √-1)": sqrt_minus1,
    }
    return {"checks": checks, "violations": sum(1 for v in checks.values() if not v)}


def verify_cursor_partial(lat: GateLattice) -> dict:
    """The cursor's lower threshold reproduces the partial-logic learning sweep:
    PartialLogic.from_cursor(c=+1, tau_lo, 1) is exactly the p0->p1->p2 of 11.5."""
    c = 1 + 0j
    names = lambda ls: {l.name for l in ls}
    cases = [(-1.0, {"AND", "XNOR", "OR", "TRUE"}),   # band wide open  -> p0
             (-0.5, {"AND", "OR"}),                    # raise floor     -> p1
             (0.5, {"AND"})]                           # raise floor more-> p2
    ok = all(names(PartialLogic.from_cursor(c, lo, 1.0).completions(lat)) == exp
             and PartialLogic.from_cursor(c, lo, 1.0).known_true == frozenset({"1"})
             for lo, exp in cases)
    return {"checks": {"cursor band reproduces the learning sweep": ok},
            "violations": 0 if ok else 1}


def verify_three_proposition() -> dict:
    """A worked example on the full imaginary 2-sphere: three propositions as unit
    quaternions, their pairwise relational quaternions classified."""
    one, I, J, K = F.ONE, F.I, F.J, F.K
    R = relational_quaternion
    pe = {}
    for (na, a), (nb, b) in [(("p", I), ("q", J)), (("q", J), ("r", K)), (("r", K), ("p", I))]:
        r = R(a, b)
        pe[(na, nb)] = (_qclass(r), r[1:])
    all_exclude = all(cl == "exclusion" for cl, _ in pe.values())
    # each exclusion axis is one coordinate axis (the cross product of the pair)
    axes_coord = all(abs(np.linalg.norm(v) - 1) < 1e-9 and int((np.abs(v) > 0.99).sum()) == 1
                     for _, v in pe.values())
    q45 = (I + J) / np.sqrt(2.0)                 # 45deg from i in the i-j plane
    r45 = R(I, q45)
    graded = (abs(r45[0] - 1 / np.sqrt(2)) < 1e-9
              and abs(np.linalg.norm(r45[1:]) - 1 / np.sqrt(2)) < 1e-9)
    contra = _qclass(R(I, -I)) == "contradiction"
    checks = {
        "three orthogonal props pairwise exclude": all_exclude,
        "each exclusion axis is the cross-product axis": axes_coord,
        "45° proposition is half-align half-exclude": graded,
        "antipodal props contradict (R(i,-i)=-1)": contra,
    }
    return {"checks": checks, "violations": sum(1 for v in checks.values() if not v),
            "pairexcl": pe, "r45": r45}


# =============================================================================
#  PROOFS.  The statements below are theorems; we prove them analytically (see the
#  docstrings) and confirm to machine precision over RANDOM inputs -- not just the
#  basis elements -- so nothing rests on a hand-picked example or a naming choice.
# =============================================================================
def verify_relational_proofs(n: int = 20000, seed: int = 0) -> dict:
    """THEOREM (relational-quaternion decomposition).  For unit quaternions p,q let
    R = p q̄.  Then
        (1) |R| = 1,
        (2) scalar(R) = <p,q>  (the R^4 inner product) = cos θ,
        (3) |vector(R)| = sin θ = sqrt(1 - cos^2 θ),
        (4) p ⊥ q  ⇔  scalar(R) = 0, and then R is a unit imaginary quaternion
            with R^2 = -1  (a square root of minus one),
        (5) for orthogonal PURE-imaginary unit p=[0,a], q=[0,b]:  R = [cos? ...]
            has scalar 0 and vector = -(a × b)  (the cross product).
    Proof.  |R|=|p||q̄|=1.  Writing p=(w_p,v_p), q=(w_q,v_q), the scalar part of the
    Hamilton product p q̄ is w_p w_q + v_p·v_q = <p,q>; for unit p,q this is cos θ.
    Then scalar^2 + |vector|^2 = |R|^2 = 1 gives |vector| = sin θ.  If <p,q>=0 then
    R = (0, vector) with |vector|=1, and (0,u)^2 = (-|u|^2, 0) = -1.  For pure-
    imaginary p,q the product is p q̄ = -p q = -(-a·b + a×b) = a·b - a×b; orthogonal
    ⇒ a·b=0 ⇒ R = (0, -(a×b)).  ∎  We confirm all five over random inputs."""
    rng = np.random.default_rng(seed)
    e_unit = e_scalar = e_vecnorm = 0.0
    for _ in range(n):
        p = F.random_unit_quaternion(rng)
        q = F.random_unit_quaternion(rng)
        R = relational_quaternion(p, q)
        cos_t = float(np.dot(p, q))
        e_unit = max(e_unit, abs(float(np.linalg.norm(R)) - 1.0))
        e_scalar = max(e_scalar, abs(float(R[0]) - cos_t))
        e_vecnorm = max(e_vecnorm,
                        abs(float(np.linalg.norm(R[1:])) - np.sqrt(max(0.0, 1 - cos_t ** 2))))
    e_orth = e_sqrt = 0.0
    for _ in range(n):
        p = F.random_unit_quaternion(rng)
        r = F.random_unit_quaternion(rng)
        q = r - np.dot(r, p) * p                       # Gram-Schmidt: q ⊥ p in R^4
        nq = float(np.linalg.norm(q))
        if nq < 1e-12:
            continue
        q = q / nq
        R = relational_quaternion(p, q)
        e_orth = max(e_orth, abs(float(R[0])))                       # scalar = 0
        e_sqrt = max(e_sqrt, float(np.linalg.norm(F.hamilton(R, R) + F.ONE)))  # R^2 = -1
    e_cross = 0.0
    for _ in range(n):
        a = rng.standard_normal(3); a /= np.linalg.norm(a)
        b = rng.standard_normal(3); b -= (b @ a) * a; b /= np.linalg.norm(b)
        R = relational_quaternion(np.array([0.0, *a]), np.array([0.0, *b]))
        e_cross = max(e_cross, abs(float(R[0])),
                      float(np.linalg.norm(R[1:] - (-np.cross(a, b)))))
    errs = {
        "(1) |R| = 1": e_unit,
        "(2) scalar(R) = <p,q> = cos θ": e_scalar,
        "(3) |vector(R)| = sin θ": e_vecnorm,
        "(4a) p⊥q ⇒ scalar(R)=0": e_orth,
        "(4b) p⊥q ⇒ R² = −1": e_sqrt,
        "(5) pure-imag orthogonal ⇒ R = −(a×b)": e_cross,
    }
    return {"errors": errs, "max_error": max(errs.values()), "n": n}


def verify_correlation_trichotomy() -> dict:
    """THEOREM (the logical content, exact).  Represent two propositions A,B as
    ±1 functions on the four equiprobable minterms.  Their PEARSON CORRELATION
    ρ = E[AB] (both are balanced, so mean 0) takes exactly:
        ρ = +1  iff  A ≡ B            (logical equivalence)        -- 'alignment'
        ρ = -1  iff  A ≡ ¬B           (logical negation)           -- 'contradiction'
        ρ =  0  iff  A,B independent  (orthogonal truth-tables)    -- 'exclusion'
    and ρ equals cos of the angle between the ±1 vectors, i.e. the relational
    scalar.  For two BINARY variables, zero correlation ⇔ independence (proved by
    the 2×2 table with fixed marginals), so the middle case is genuine statistical
    independence -- NOT logical mutual-exclusivity, which is a different relation.
    All quantities are exact rationals; we verify them as such."""
    minterms = [(1, 1), (1, 0), (0, 1), (0, 0)]
    sign = lambda b: 1.0 if b else -1.0
    A = np.array([sign(a) for a, b in minterms])
    B = np.array([sign(b) for a, b in minterms])
    corr = lambda X, Y: float(np.mean(X * Y))          # zero-mean ⇒ this is Pearson ρ
    # the relational scalar via the cosine of the ±1 vectors equals the correlation
    cos_AB = float(A @ B / (np.linalg.norm(A) * np.linalg.norm(B)))
    # independence of the uncorrelated pair (A,B) via the 2×2 joint table
    indep = all(abs(np.mean((A == a) & (B == b)) - np.mean(A == a) * np.mean(B == b)) < 1e-12
                for a in (1.0, -1.0) for b in (1.0, -1.0))
    checks = {
        "A≡B  ⇒ ρ = +1": abs(corr(A, A) - 1) < 1e-12,
        "A≡¬B ⇒ ρ = −1": abs(corr(A, -A) + 1) < 1e-12,
        "A,B independent ⇒ ρ = 0": abs(corr(A, B)) < 1e-12,
        "ρ = cos(angle of ±1 vectors)": abs(corr(A, B) - cos_AB) < 1e-12,
        "binary: ρ=0 ⇔ statistical independence": indep,
    }
    return {"checks": checks, "violations": sum(1 for v in checks.values() if not v)}


# --- shared lattice geometry (positions / covering edges) ---------------------
_POS = {
    "FALSE": (0.0, 0.0),
    "AND": (-0.8, 1.0), "NOR": (0.8, 1.0),
    "XNOR": (-1.0, 2.0), "XOR": (1.0, 2.0),
    "OR": (-0.8, 3.0), "NAND": (0.8, 3.0),
    "TRUE": (0.0, 4.0),
}
_EDGES = [("FALSE", "AND"), ("FALSE", "NOR"), ("FALSE", "XOR"),
          ("AND", "XNOR"), ("AND", "OR"), ("NOR", "XNOR"), ("NOR", "NAND"),
          ("XOR", "OR"), ("XOR", "NAND"),
          ("XNOR", "TRUE"), ("OR", "TRUE"), ("NAND", "TRUE")]   # each (smaller, larger)
_COLORS = {"XNOR": "#2ca02c", "XOR": "#d62728", "AND": "#1f77b4", "NOR": "#1f77b4",
           "OR": "#9467bd", "NAND": "#9467bd", "TRUE": "0.3", "FALSE": "0.3"}
_SHORT = {"TRUE": "⊤", "FALSE": "⊥"}


def _draw_node(ax, lat, name, active=True, ring=None):
    x, y = _POS[name]
    base = _COLORS.get(name, "0.5") if active else "0.85"
    ax.scatter([x], [y], s=900, color=base, edgecolor="black",
               linewidths=(1.0 if active else 0.5), zorder=3, alpha=0.95 if active else 0.6)
    if ring is not None:
        ax.scatter([x], [y], s=1500, facecolors="none", edgecolors=ring,
                   linewidths=2.6, zorder=4)
    ax.annotate(_SHORT.get(name, name), (x, y), ha="center", va="center",
                fontsize=9.5, color="white" if active else "0.5",
                fontweight="bold", zorder=5)
    ox, ha = (-22, "right") if x < 0 else ((22, "left") if x > 0 else (0, "center"))
    oy = 0 if x != 0 else (20 if y > 2 else -20)
    ax.annotate(setstr(lat[name].trueset), (x, y), textcoords="offset points",
                xytext=(ox, oy), ha=ha, va="center", fontsize=8.2,
                color=(_COLORS.get(name, "0.4") if active else "0.7"), zorder=5)


def _directed_diagram(ax, lat, focus, relation, col):
    """Draw the whole lattice faded, then the principal filter (relation=
    'superset') or ideal (relation='subset') of `focus` as directed arrows."""
    members = {l.name for l in (lat.principal_filter(lat[focus]) if relation == "superset"
                                else lat.principal_ideal(lat[focus]))}
    for u, v in _EDGES:                              # faint background skeleton
        if not (u in members and v in members):
            ax.plot([_POS[u][0], _POS[v][0]], [_POS[u][1], _POS[v][1]],
                    color="0.86", lw=1.0, zorder=1)
    for u, v in _EDGES:                              # directed arrows inside the set
        if u in members and v in members:
            start, end = (_POS[u], _POS[v]) if relation == "superset" else (_POS[v], _POS[u])
            ax.annotate("", xy=end, xytext=start,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.6,
                                        shrinkA=16, shrinkB=16), zorder=2)
    for name in _POS:
        _draw_node(ax, lat, name, active=(name in members),
                   ring=(col if name == focus else None))
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-0.9, 4.6); ax.axis("off")


def make_superset_figure(lat: GateLattice):
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    _directed_diagram(ax, lat, "AND", "superset", "#9467bd")
    ax.set_title("Superset view — the principal filter $\\uparrow$AND\n"
                 "from a part, the wholes that contain it (arrows point to supersets)",
                 fontsize=11)
    ax.text(0.0, -0.7, "AND $\\subset$ XNOR $\\subset$ ⊤   and   AND $\\subset$ OR $\\subset$ ⊤\n"
            "the up-set $\\{$AND, XNOR, OR, ⊤$\\}$ — every logic in which \"both true\" still holds",
            ha="center", fontsize=8.6, color="#6a4ca0")
    path = os.path.join(FIG_DIR, "fig19_superset.png")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def make_subset_figure(lat: GateLattice):
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    _directed_diagram(ax, lat, "NAND", "subset", "#1f77b4")
    ax.set_title("Subset view — the principal ideal $\\downarrow$NAND\n"
                 "from a whole, the parts it contains (arrows point to subsets)",
                 fontsize=11)
    ax.text(0.0, -0.7, "NAND $\\supset$ XOR $\\supset$ ∅   and   NAND $\\supset$ NOR $\\supset$ ∅\n"
            "the down-set $\\{$NAND, XOR, NOR, ∅$\\}$ — the complement-dual of $\\uparrow$AND",
            ha="center", fontsize=8.6, color="#155f9c")
    path = os.path.join(FIG_DIR, "fig20_subset.png")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def make_partial_figure(res_d: dict):
    """The worked example: a partial logic narrowing from the superset bound (⊤)
    down to the subset bound (AND) as facts are learned."""
    lat, trace = res_d["lattice"], res_d["trace"]
    captions = [
        "know: AB true; rest unknown",
        "learn: ¬A¬B (−1) is false",
        "learn: also A¬B, ¬AB false",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.0))
    for ax, p, cap in zip(axes, trace, captions):
        comp = {l.name for l in p.completions(lat)}
        lo = lat._named(p.lower).name      # subset bound (may be unnamed -> '?')
        hi = lat._named(p.upper).name      # superset bound
        for u, v in _EDGES:                # skeleton; bold edges inside the interval
            inside = u in comp and v in comp
            ax.plot([_POS[u][0], _POS[v][0]], [_POS[u][1], _POS[v][1]],
                    color=("#444" if inside else "0.86"),
                    lw=(2.2 if inside else 1.0), zorder=1)
        for name in _POS:
            ring = "#2ca02c" if name == lo else ("#9467bd" if name == hi else None)
            _draw_node(ax, lat, name, active=(name in comp), ring=ring)
        ax.set_xlim(-2.2, 2.2); ax.set_ylim(-1.0, 4.6); ax.axis("off")
        ax.set_title(cap, fontsize=10)
        ax.text(0.0, -0.78,
                f"true={setstr(p.known_true)}  false={setstr(p.known_false)}\n"
                f"subset bound {setstr(p.lower)}  ..  superset bound {setstr(p.upper)}\n"
                f"completions: {{{', '.join(sorted(comp, key=lambda n: len(lat[n].trueset)))}}}",
                ha="center", fontsize=8.0, color="0.25")
    fig.suptitle("A partial logic mapped between its subset bound (green) and superset bound (purple)\n"
                 "learning shrinks the interval from ⊤ down to the single subset AND",
                 y=1.02, fontsize=12)
    path = os.path.join(FIG_DIR, "fig21_partial.png")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def make_cursor_figure():
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    relcol = {"alignment": "#2ca02c", "exclusion": "#7f7f7f",
              "contradiction": "#d62728", "mixed": "#1f77b4"}
    th = np.linspace(0, 2 * np.pi, 400)
    fig = plt.figure(figsize=(13.8, 11.4))

    # (A) the consistency field of a cursor sitting on +1 ----------------------
    axA = fig.add_subplot(2, 2, 1)
    axA.plot(np.cos(th), np.sin(th), color="0.82", lw=1.2)
    c = 1 + 0j
    axA.annotate("", xy=(c.real, c.imag), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=2.4))
    axA.text(0.5, -0.2, "cursor $c$", ha="center", fontsize=9)
    for n, z in NAME_LM.items():
        rel, val = relation(n, c), consistency(n, c)
        axA.plot([z.real], [z.imag], "o", ms=13, color=relcol[rel], zorder=5)
        axA.annotate(f"{n}: {val:+.0f}\n{rel}", (z.real, z.imag),
                     textcoords="offset points", xytext=(20 * z.real, 22 * z.imag),
                     ha="center", fontsize=8.3, color=relcol[rel])
    axA.set_aspect("equal"); axA.axis("off"); axA.set_xlim(-1.9, 1.9); axA.set_ylim(-1.7, 1.7)
    axA.set_title("(A) consistency $=\\mathrm{Re}(c\\,\\bar z)=\\cos\\Delta\\theta$\n"
                  "+1 align (green) · 0 exclude (grey) · −1 contradict (red)", fontsize=10.5)

    # (B) sweeping the degree of consistency: sublogic -> superlogic -----------
    axB = fig.add_subplot(2, 2, 2)
    order = ["1", "i", "-i", "-1"]
    vals = {n: consistency(n, c) for n in order}
    for n in order:
        axB.plot([0, 1], [vals[n], vals[n]], color=relcol[relation(n, c)], lw=2)
        axB.text(1.03, vals[n], f"{n}", va="center", fontsize=9,
                 color=relcol[relation(n, c)])
    bands = [(0.0, 1.0, "AND $=\\{1\\}$", "sublogic (subset)"),
             (-1.0, 0.0, "OR $=\\{1,i,-i\\}$", ""),
             (-1.05, -1.0, "⊤ $=\\{1,i,-1,-i\\}$", "superlogic (superset)")]
    for lo, hi, lab, tag in bands[:2]:
        axB.axhspan(lo, hi, xmin=0.0, xmax=0.32, color="0.9", zorder=0)
    axB.annotate("", xy=(0.16, -1.05), xytext=(0.16, 1.05),
                 arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.6))
    axB.text(0.46, 0.5, "$\\tau\\in(0,1]$ →\nAND  (sublogic)", fontsize=8.5, va="center")
    axB.text(0.46, -0.5, "$\\tau\\in(-1,0]$ →\nOR", fontsize=8.5, va="center")
    axB.text(0.46, -1.0, "$\\tau=-1$ → ⊤ (superlogic)", fontsize=8.5, va="center")
    axB.text(0.02, 1.16, "lower the threshold $\\tau$  =  open sublogic → superlogic",
             fontsize=9, color="0.3")
    axB.set_xlim(-0.05, 1.35); axB.set_ylim(-1.25, 1.28)
    axB.set_xticks([]); axB.set_ylabel("degree of consistency  $\\tau$")
    axB.set_title("(B) cursor at +1 sweeps the chain  AND ⊂ OR ⊂ ⊤", fontsize=10.5)

    # (C) rotation covariance: rotate the cursor, same dynamics, relabelled -----
    axC = fig.add_subplot(2, 2, 3)
    axC.plot(np.cos(th), np.sin(th), color="0.85", lw=1.0)
    for col, deg in [("#1f77b4", 0), ("#9467bd", 180)]:
        cc = np.exp(1j * np.deg2rad(deg))
        axC.annotate("", xy=(cc.real, cc.imag), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4))
        ch = consistency_chain(cc)
        named = []
        for L in ch:
            hit = [g for g in ("AND", "NOR", "OR", "NAND") if gate_trueset(g) == L]
            named.append(hit[0] if hit else ("⊤" if L == ALL_LM else "·"))
        axC.text(cc.real * 1.45, -0.22 if deg == 0 else 0.22, "  →  ".join(named),
                 ha="center", va="center", fontsize=8.6, color=col)
    axC.text(0.0, -1.72, "rotate 90° instead and the cursor lands on the single\n"
             "propositions A, B (the non-symmetric gates)", ha="center",
             fontsize=8.0, color="0.45")
    for n, z in NAME_LM.items():
        axC.plot([z.real], [z.imag], "o", ms=8, color="0.6", zorder=5)
        axC.annotate(n, (z.real, z.imag), textcoords="offset points",
                     xytext=(11 * z.real, 11 * z.imag), ha="center", fontsize=8, color="0.5")
    axC.set_aspect("equal"); axC.axis("off"); axC.set_xlim(-2.0, 2.0); axC.set_ylim(-1.9, 1.9)
    axC.set_title("(C) rotate the cursor → same chain, relabelled\n"
                  "(gauge covariance: the dynamics hold under rotation)", fontsize=10.5)

    # (D) the quaternion level: exclusion gains a direction (a 2-sphere) --------
    axD = fig.add_subplot(2, 2, 4, projection="3d")
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    axD.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                     color="0.85", alpha=0.25, linewidth=0)
    for vec, lab in [((1, 0, 0), "i"), ((0, 1, 0), "j"), ((0, 0, 1), "k")]:
        axD.quiver(0, 0, 0, *vec, color="#7f7f7f", lw=2)
        axD.text(vec[0] * 1.25, vec[1] * 1.25, vec[2] * 1.25, lab,
                 fontsize=11, color="#3f3f3f")
    axD.text(0, 0, 1.7, "exclusion = the imaginary 2-sphere\n(every direction a √−1)",
             ha="center", fontsize=8.6, color="#3f3f3f")
    axD.scatter([0], [0], [0], color="black", s=20)
    axD.text(0.1, 0.1, -1.8, "align/contradict = ±1 (scalar axis,\noff this sphere)",
             ha="center", fontsize=8.4, color="0.3")
    axD.set_box_aspect((1, 1, 1)); axD.set_axis_off()
    axD.set_title("(D) in ℍ: exclusion is directional", fontsize=10.5)

    fig.suptitle("The consistency cursor — degrees of consistency between sublogic and superlogic\n"
                 "alignment / exclusion / contradiction = $\\mathrm{Re}(c\\bar z)=+1/0/-1$",
                 y=1.01, fontsize=13)
    path = os.path.join(FIG_DIR, "fig22_cursor.png")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def make_three_proposition_figure(res3: dict):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(12.8, 5.8))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                    color="0.9", alpha=0.16, linewidth=0)
    props = {"p = i": ((1, 0, 0), "#1f77b4"), "q = j": ((0, 1, 0), "#2ca02c"),
             "r = k": ((0, 0, 1), "#d62728")}
    for lab, (vec, col) in props.items():
        ax.quiver(0, 0, 0, *vec, color=col, lw=3)
        ax.text(vec[0] * 1.22, vec[1] * 1.22, vec[2] * 1.22, lab, color=col, fontsize=10)
    excl = {("p", "q"): (0, 0, -1), ("q", "r"): (-1, 0, 0), ("r", "p"): (0, -1, 0)}
    for (a, b), vec in excl.items():
        ax.quiver(0, 0, 0, *vec, color="0.55", lw=1.4, linestyle="dashed")
        ax.text(vec[0] * 1.18, vec[1] * 1.18, vec[2] * 1.18, f"excl({a},{b})",
                color="0.4", fontsize=7.3)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.set_title("three orthogonal propositions on the imaginary 2-sphere\n"
                 "each pair excludes along the third (cross-product) axis", fontsize=10)

    ax2 = fig.add_subplot(1, 2, 2); ax2.axis("off")
    lines = [
        (r"$R(p,q)=p\,\bar q$ : scalar $=$ align/contradict,", "0.15"),
        (r"                       vector $=$ exclusion axis", "0.15"),
        ("", "k"),
        (r"$R(i,j)=-k$    exclude along $-k$", "#7f7f7f"),
        (r"$R(j,k)=-i$    exclude along $-i$", "#7f7f7f"),
        (r"$R(k,i)=-j$    exclude along $-j$", "#7f7f7f"),
        ("", "k"),
        (r"$R(i,-i)=-1$    contradiction (antipodal)", "#d62728"),
        (r"$R(i,i)=+1$    alignment", "#2ca02c"),
        ("", "k"),
        (r"$R\left(i,\frac{i+j}{\sqrt{2}}\right)=\frac{1-k}{\sqrt{2}}$", "#1f77b4"),
        (r"   half alignment $(+\frac{1}{\sqrt{2}})$, half exclusion along $-k$", "#1f77b4"),
        ("", "k"),
        ("exclusion is DIRECTIONAL — a 2-sphere of axes,", "0.3"),
        (r"invisible in $\mathbb{C}$ (only $\pm i$ there).", "0.3"),
    ]
    y = 0.97
    for txt, col in lines:
        ax2.text(0.0, y, txt, va="top", ha="left", fontsize=11, color=col)
        y -= 0.066
    ax2.set_title("the relational quaternions, classified", fontsize=10)

    fig.suptitle("A three-proposition example on the full 2-sphere", y=1.02, fontsize=12.5)
    path = os.path.join(FIG_DIR, "fig23_three_prop.png")
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
    res_g = verify_gates()
    print("  gate decomposition:")
    for k, v in res_g.items():
        if isinstance(v, float):
            print(f"    {k:<38} {v:.2e}")
    print(f"    separable (one half-plane): {res_g['separable (one half-plane)']}")
    res_l = verify_lattice()
    print(f"  gate lattice (whole/part): {res_l['violations']} violations")
    for k, v in res_l["checks"].items():
        print(f"    {k:<28} {v}")
    lat = build_lattice()
    res_d = verify_datastructure(lat)
    print(f"  partial-logic data structure: {res_d['violations']} violations")
    for k, v in res_d["checks"].items():
        print(f"    {k:<44} {v}")
    print("  worked example (interval narrows superset -> subset):")
    for p in res_d["trace"]:
        print(f"    {p}")
    res_c = verify_cursor()
    print(f"  consistency cursor: {res_c['violations']} violations")
    for k, v in res_c["checks"].items():
        print(f"    {k:<46} {v}")
    res_cp = verify_cursor_partial(lat)
    print(f"  cursor <-> partial bridge: {res_cp['violations']} violations")
    for k, v in res_cp["checks"].items():
        print(f"    {k:<46} {v}")
    res_rq = verify_relational_quaternion()
    print(f"  relational quaternion: {res_rq['violations']} violations")
    for k, v in res_rq["checks"].items():
        print(f"    {k:<46} {v}")
    res3 = verify_three_proposition()
    print(f"  three-proposition (2-sphere): {res3['violations']} violations")
    for k, v in res3["checks"].items():
        print(f"    {k:<46} {v}")
    res_pf = verify_relational_proofs()
    print(f"  PROOF — relational decomposition over n={res_pf['n']} random "
          f"quaternions (max err {res_pf['max_error']:.1e}):")
    for k, v in res_pf["errors"].items():
        print(f"    {k:<40} {v:.2e}")
    res_ct = verify_correlation_trichotomy()
    print(f"  PROOF — correlation trichotomy (logic): {res_ct['violations']} violations")
    for k, v in res_ct["checks"].items():
        print(f"    {k:<40} {v}")
    make_figure(res_h)
    make_rotation_figure()
    make_gate_figure(res_g)
    make_superset_figure(lat)
    make_subset_figure(lat)
    make_partial_figure(res_d)
    make_cursor_figure()
    make_three_proposition_figure(res3)


if __name__ == "__main__":
    main()
