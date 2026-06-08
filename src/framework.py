"""
framework.py
============

Core algebra for the paper *Quaternions Are All You Need*.

A quaternion is represented as a length-4 ``numpy`` array ``[w, x, y, z]``
where ``w`` is the real (scalar) part and ``(x, y, z)`` is the imaginary
(vector) part spanning the basis ``i, j, k``.  Complex numbers embed as
quaternions with ``y = z = 0`` (the *i-plane*).

Two functionals organise the framework:

* ``e(q)``       -- the *energy*  (norm)        :  e(q) = |q|
* ``sigma(q)``   -- the *phase*   (vector log)  :  sigma(q) = Im(log q)

and the governing law of the framework is the antipodal involution

        e(x) = e(y)        whenever        sigma(x) = -sigma(y),

which is realised concretely by quaternion conjugation ``q -> q*``.

Two binary operations are compared throughout the paper:

* ``hamilton(a, b)``     -- the Hamilton product (the "correct" extension).
* ``phase_add(a, b)``    -- the naive phase-additive operator
                            h_+(x, y) = exp(log x + log y).

In the complex plane the two operators coincide and are associative,
commutative and distributive.  In the quaternions the phase-additive
operator *breaks* associativity (Baker--Campbell--Hausdorff: the
generators i, j, k do not commute), whereas the Hamilton product retains
associativity, distributivity and non-contradiction.  That contrast is
the empirical heart of the paper.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Basis elements
# ---------------------------------------------------------------------------
ONE = np.array([1.0, 0.0, 0.0, 0.0])
I = np.array([0.0, 1.0, 0.0, 0.0])
J = np.array([0.0, 0.0, 1.0, 0.0])
K = np.array([0.0, 0.0, 0.0, 1.0])

CONJ_SIGN = np.array([1.0, -1.0, -1.0, -1.0])


# ---------------------------------------------------------------------------
# Elementary quaternion arithmetic
# ---------------------------------------------------------------------------
def hamilton(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product  a (x) b  of two quaternions.

    Associative and distributive over addition, but *not* commutative.
    """
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def conjugate(a: np.ndarray) -> np.ndarray:
    """Quaternion conjugate q* = w - xi - yj - zk."""
    return a * CONJ_SIGN


def norm(a: np.ndarray) -> float:
    """Euclidean norm |q| = sqrt(q q*)."""
    return float(np.sqrt(np.dot(a, a)))


def inverse(a: np.ndarray) -> np.ndarray:
    """Multiplicative inverse q^{-1} = q* / |q|^2."""
    n2 = float(np.dot(a, a))
    return conjugate(a) / n2


# ---------------------------------------------------------------------------
# The two organising functionals:  energy e(.)  and phase sigma(.)
# ---------------------------------------------------------------------------
def e(a: np.ndarray) -> float:
    """Energy functional e(q) = |q| (the magnitude / "realised" part)."""
    return norm(a)


def sigma(a: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Phase functional sigma(q) = Im(log q), a 3-vector in span{i, j, k}.

    For q = |q|(cos(theta) + n sin(theta)) with unit axis n, we have
    sigma(q) = theta * n.  Conjugation negates it:  sigma(q*) = -sigma(q).
    """
    vec = a[1:]
    vnorm = np.sqrt(np.dot(vec, vec))
    if vnorm < eps:
        return np.zeros(3)
    n = a[0] / norm(a)
    n = np.clip(n, -1.0, 1.0)
    theta = np.arccos(n)
    return theta * (vec / vnorm)


# ---------------------------------------------------------------------------
# Quaternion exponential / logarithm  (the bridge  1 <-> 0,  -1 <-> pi)
# ---------------------------------------------------------------------------
def qlog(a: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Principal quaternion logarithm log q = ln|q| + sigma(q)."""
    n = norm(a)
    vec = a[1:]
    vnorm = np.sqrt(np.dot(vec, vec))
    out = np.zeros(4)
    out[0] = np.log(n) if n > eps else 0.0
    if vnorm < eps:
        return out
    theta = np.arccos(np.clip(a[0] / n, -1.0, 1.0))
    out[1:] = theta * (vec / vnorm)
    return out


def qexp(a: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Quaternion exponential exp(q) = e^w (cos|v| + (v/|v|) sin|v|)."""
    w = a[0]
    vec = a[1:]
    vnorm = np.sqrt(np.dot(vec, vec))
    ew = np.exp(w)
    out = np.zeros(4)
    if vnorm < eps:
        out[0] = ew
        return out
    out[0] = ew * np.cos(vnorm)
    out[1:] = ew * np.sin(vnorm) * (vec / vnorm)
    return out


# ---------------------------------------------------------------------------
# The two operators under comparison
# ---------------------------------------------------------------------------
def phase_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Naive phase-additive operator h_+(x, y) = exp(log x + log y).

    *Adds the phases* of the two arguments.  In the complex i-plane this
    is exactly the complex product (associative, commutative).  In the
    quaternions it is commutative but NOT associative, because i, j, k do
    not commute (Baker--Campbell--Hausdorff).
    """
    return qexp(qlog(a) + qlog(b))


# ---------------------------------------------------------------------------
# Euler map  E(theta) = e^{i theta}  on the i-plane unit circle
# ---------------------------------------------------------------------------
def euler(theta: float) -> np.ndarray:
    """Unit-circle point E(theta) = cos(theta) + i sin(theta) as a quaternion."""
    return np.array([np.cos(theta), np.sin(theta), 0.0, 0.0])


# Named landmarks of the involution  (point on circle  <->  phase angle)
LANDMARKS = {
    "1": (0.0, ONE),                 #  1  lives at angle 0
    "i": (np.pi / 2, I),             #  i  lives at angle pi/2
    "-1": (np.pi, -ONE),             # -1  lives at angle pi
    "-i": (3 * np.pi / 2, -I),       # -i  lives at angle 3pi/2
}


# ---------------------------------------------------------------------------
# Random sampling helpers (used by the verification + plotting suites)
# ---------------------------------------------------------------------------
def random_quaternion(rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """A random quaternion with Gaussian components."""
    return rng.normal(scale=scale, size=4)


def random_unit_quaternion(rng: np.random.Generator) -> np.ndarray:
    """A random unit quaternion (point on the 3-sphere)."""
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def random_complex(rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """A random element of the complex i-plane (y = z = 0)."""
    q = np.zeros(4)
    q[:2] = rng.normal(scale=scale, size=2)
    return q
