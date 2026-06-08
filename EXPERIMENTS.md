# Experiments — three threads

Working notes (not the paper). Each thread is a self-contained, reproducible
experiment. Run:

```bash
python src/spinor.py     # Thread 1 -> figures/fig9_spinor.png
python src/dirac.py      # Thread 2 -> figures/fig10_dirac.png
python src/qnn.py        # Thread 3 -> figures/fig11_learning.png, fig12_sample_efficiency.png
```

---

## Thread 1 — the 2π / 4π doubling (`src/spinor.py`)

**Claim.** The complex circle is 2π-periodic; quaternions are 4π-periodic, and the
phase functional `σ` reads the *half* angle. This is the double cover
SU(2) → SO(3) — the same fact as your "−1 ↔ π", seen from the rotation side.

A rotation by φ about axis **n** is `q(φ) = cos(φ/2) + n·sin(φ/2)` (half-angle).

| check | result |
|---|---|
| `q(2π) = −1` | err 1.2e-16 |
| `q(4π) = +1` | err 2.4e-16 |
| `R_q = R_{−q}` (q and −q rotate space identically) | err 0 (exact) |
| vector returns to start after 2π | err 2.4e-16 |

**Finding.** Confirmed exactly. The spinor needs **4π** to return to identity; its
action on space returns after **2π**. `σ(q(φ)) = (φ/2)·n` — the half-angle is where
your `2π → 4π` factor lives.

![spinor](figures/fig9_spinor.png)

---

## Thread 2 — the Cayley table is the atomic cell of Dirac (`src/dirac.py`)

**Claim (refined).** The quaternion units *are* the Pauli matrices
(`1→I, i→−iσx, j→−iσy, k→−iσz`), an algebra isomorphism ℍ ≅ 𝔰𝔲(2). The Dirac
algebra of spacetime is `Cl(1,3) ≅ M₂(ℍ)` — 2×2 matrices over the quaternions. So
the Cayley table isn't *more complete* than Dirac; it's the **2×2 building block
Dirac is assembled from**.

| check | result |
|---|---|
| quaternion → Pauli homomorphism `mat(a⊗b)=mat(a)mat(b)` | err 4.4e-15 |
| unit quaternions are SU(2) (`UU† = I`) | err 8.0e-16 |
| `det U = 1` | err 6.8e-16 |
| Dirac Clifford relation `{γμ,γν} = 2ημν` | err 0 (exact) |
| recovered metric `η` | `diag(+1,−1,−1,−1)` |

**Finding.** Confirmed. The Minkowski metric `diag(+,−,−,−)` falls out of the
anticommutator of gammas built from quaternion (Pauli) blocks.

![dirac](figures/fig10_dirac.png)

---

## Thread 3 — can the architecture learn, in isolation? (`src/qnn.py`)

A from-scratch quaternion MLP (Hamilton-product linear layers + split-tanh),
pure NumPy, manual backprop. **Gradient check vs finite differences: 5.3e-10** —
backprop is correct. No RoPE, no transformer.

### 3a. It learns — task `p' = r ⊗ p ⊗ r*` (learn the rotation action)

| model | params | test MSE | R² |
|---|---|---|---|
| quaternion MLP | 1348 | 3.2e-2 | **0.897** |
| real MLP (matched) | 1399 | 6.6e-3 | — |
| mean predictor | — | 2.6e-1 | 0 |

**Finding — honest.** The quaternion net **learns** (R²=0.90, ~8× better than the
mean predictor). But on this *dense* task a parameter-matched real MLP does
**better** (MSE 6.6e-3 vs 3.2e-2). Why: the rotation *sandwich* `r p r*` is not
naturally a chain of left-multiplications, so the Hamilton prior is a poor fit
here. Quaternion nets are **not** universally superior — and we should say so.

![learning](figures/fig11_learning.png)

### 3b. Where the prior actually pays — sample efficiency on quaternion-native data

Task `y = Q ⊗ x + noise` for fixed unknown `Q`. The quaternion layer (4 DOF) has
*exactly* the right structure; the real layer (16 DOF) must estimate it.

| train samples | quaternion MSE | real MSE | quaternion advantage |
|---:|---|---|---:|
| 8   | 0.0058 | 0.0473 | **8.2×** |
| 16  | 0.0027 | 0.0123 | 4.6× |
| 32  | 0.0012 | 0.0040 | 3.3× |
| 64  | 0.0008 | 0.0018 | 2.3× |
| 128 | 0.0005 | 0.0011 | 2.2× |
| 256 | 0.0002 | 0.0005 | 2.5× |

**Finding.** When the data really is quaternion-structured, the Hamilton prior
generalises from **far fewer samples** (8× at N=8), with the gap narrowing as data
grows — the signature of a correct, restrictive inductive bias.

![sample efficiency](figures/fig12_sample_efficiency.png)

---

## Honest summary

- **Thread 1 (2π/4π):** real and exact; the deepest of the three. The half-angle
  /double-cover is the genuinely non-trivial structure.
- **Thread 2 (Dirac):** real; the quaternion Cayley table is the atomic cell of the
  Dirac algebra (`Cl(1,3) ≅ M₂(ℍ)`), not a superset of it.
- **Thread 3 (learning):** the architecture **does** learn in isolation
  (verified backprop, R²=0.90), and its weight-sharing prior gives a real
  **sample-efficiency** advantage on quaternion-structured data — but it is not a
  free win on arbitrary tasks. That conditional, measurable advantage is the
  honest contribution to chase.

**Open question for the paper's thesis:** the strongest, most defensible claim
emerging from these experiments is *not* "quaternions beat real nets," but
"**the 2π→4π / SU(2) structure is a correct and sample-efficient inductive bias for
data with rotational/spinorial structure**." That ties Thread 1 (the structure),
Thread 2 (why it's the right algebra), and Thread 3b (it measurably helps) into one
line — without overclaiming.
