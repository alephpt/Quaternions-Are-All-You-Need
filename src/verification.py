"""
verification.py
===============

Numerical verification suite for *Quaternions Are All You Need*.

Runs Monte-Carlo experiments that measure the residuals of three
algebraic laws -- associativity, distributivity and non-contradiction --
for the two operators (Hamilton product vs. phase-additive operator) in
two domains (the complex i-plane vs. the full quaternions).

Residuals on the order of machine epsilon (~1e-16) confirm a law; macroscopic
residuals (order 1) falsify it.  Results are printed as a table and written
to ``results/metrics.json``.
"""

from __future__ import annotations

import json
import os

import numpy as np

import framework as F

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ---------------------------------------------------------------------------
# Residual collectors
# ---------------------------------------------------------------------------
def associativity_residuals(op, sampler, rng, n: int) -> np.ndarray:
    """| (a op b) op c  -  a op (b op c) |  over n random triples."""
    out = np.empty(n)
    for t in range(n):
        a, b, c = sampler(rng), sampler(rng), sampler(rng)
        lhs = op(op(a, b), c)
        rhs = op(a, op(b, c))
        out[t] = np.linalg.norm(lhs - rhs)
    return out


def distributivity_residuals(op, sampler, rng, n: int) -> np.ndarray:
    """| a op (b + c)  -  (a op b + a op c) |  over n random triples."""
    out = np.empty(n)
    for t in range(n):
        a, b, c = sampler(rng), sampler(rng), sampler(rng)
        lhs = op(a, b + c)
        rhs = op(a, b) + op(a, c)
        out[t] = np.linalg.norm(lhs - rhs)
    return out


def commutativity_residuals(op, sampler, rng, n: int) -> np.ndarray:
    """| a op b  -  b op a |  over n random pairs."""
    out = np.empty(n)
    for t in range(n):
        a, b = sampler(rng), sampler(rng)
        out[t] = np.linalg.norm(op(a, b) - op(b, a))
    return out


# ---------------------------------------------------------------------------
# Non-contradiction checks (the framework's self-consistency)
# ---------------------------------------------------------------------------
def non_contradiction(rng, n: int) -> dict:
    """Verify the structural identities that keep the framework consistent."""

    # (1) Fundamental quaternion relations  i^2 = j^2 = k^2 = ijk = -1
    fundamentals = {
        "i^2": F.hamilton(F.I, F.I),
        "j^2": F.hamilton(F.J, F.J),
        "k^2": F.hamilton(F.K, F.K),
        "ijk": F.hamilton(F.hamilton(F.I, F.J), F.K),
    }
    fund_err = max(np.linalg.norm(v - (-F.ONE)) for v in fundamentals.values())

    # (2) Euler landmarks of the involution:  E(0)=1, E(pi)=-1, E(pi/2)=i, ...
    landmark_err = max(
        np.linalg.norm(F.euler(theta) - point)
        for theta, point in F.LANDMARKS.values()
    )

    # (3) The governing law realised by conjugation:
    #     e(q*) = e(q)            (equal energy)
    #     sigma(q*) = -sigma(q)   (opposite phase)
    e_err = 0.0
    sigma_err = 0.0
    # (4) q q* = |q|^2  is a non-negative real (no imaginary residue -> no
    #     contradiction): the product of an element with its antipode lands
    #     exactly on the "1 = angle 0" landmark, scaled by the energy.
    qqstar_imag = 0.0
    qqstar_real_err = 0.0
    # (5) exp/log round trip (the multiplicative <-> additive bridge).
    explog_err = 0.0

    for _ in range(n):
        q = F.random_quaternion(rng)
        qc = F.conjugate(q)
        e_err = max(e_err, abs(F.e(qc) - F.e(q)))
        sigma_err = max(sigma_err, np.linalg.norm(F.sigma(qc) + F.sigma(q)))

        prod = F.hamilton(q, qc)
        qqstar_imag = max(qqstar_imag, np.linalg.norm(prod[1:]))
        qqstar_real_err = max(qqstar_real_err, abs(prod[0] - F.e(q) ** 2))

        u = F.random_unit_quaternion(rng)
        explog_err = max(explog_err, np.linalg.norm(F.qexp(F.qlog(u)) - u))

    return {
        "fundamental_relations_max_err": fund_err,
        "euler_landmarks_max_err": landmark_err,
        "conj_energy_max_err": e_err,
        "conj_phase_max_err": sigma_err,
        "qqstar_imaginary_residue_max": qqstar_imag,
        "qqstar_real_eq_norm2_max_err": qqstar_real_err,
        "exp_log_roundtrip_max_err": explog_err,
        "fundamentals": {k: v.tolist() for k, v in fundamentals.items()},
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def summarise(res: np.ndarray) -> dict:
    return {
        "max": float(res.max()),
        "mean": float(res.mean()),
        "median": float(np.median(res)),
    }


def run(n: int = 20000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)

    operators = {"hamilton": F.hamilton, "phase_add": F.phase_add}
    domains = {"complex": F.random_complex, "quaternion": F.random_quaternion}

    metrics = {"n_samples": n, "seed": seed, "laws": {}}

    for law_name, fn in [
        ("associativity", associativity_residuals),
        ("distributivity", distributivity_residuals),
        ("commutativity", commutativity_residuals),
    ]:
        metrics["laws"][law_name] = {}
        for op_name, op in operators.items():
            for dom_name, sampler in domains.items():
                res = fn(op, sampler, rng, n)
                metrics["laws"][law_name][f"{op_name}/{dom_name}"] = summarise(res)

    metrics["non_contradiction"] = non_contradiction(rng, n)
    return metrics


def _fmt(x: float) -> str:
    return f"{x:.3e}"


def print_table(metrics: dict) -> None:
    print(f"\nMonte-Carlo verification  (n = {metrics['n_samples']}, seed = {metrics['seed']})")
    print("=" * 78)
    cols = ["hamilton/complex", "hamilton/quaternion",
            "phase_add/complex", "phase_add/quaternion"]
    header = f"{'law':<16}" + "".join(f"{c:>16}" for c in
                                       ["ham/C", "ham/H", "phase/C", "phase/H"])
    for law_name, table in metrics["laws"].items():
        print(f"\n{law_name}  (max residual)")
        print(header)
        row = f"{'':<16}"
        for c in cols:
            row += f"{_fmt(table[c]['max']):>16}"
        print(row)

    print("\nnon-contradiction (max residual)")
    print("-" * 78)
    nc = metrics["non_contradiction"]
    for k, v in nc.items():
        if isinstance(v, float):
            print(f"  {k:<40} {_fmt(v)}")
    print("\n  fundamental products (should each equal [-1, 0, 0, 0]):")
    for k, v in nc["fundamentals"].items():
        vec = np.array(v)
        print(f"    {k:<6} = [{vec[0]:+.3f} {vec[1]:+.3f} {vec[2]:+.3f} {vec[3]:+.3f}]")


def main() -> None:
    metrics = run()
    print_table(metrics)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nWrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
