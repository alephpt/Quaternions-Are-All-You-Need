# Quaternions Are All You Need

### Associativity, Distributivity, and Non-Contradiction of the Realised-Imaginary Framework

**Richard I Christopher** &nbsp;·&nbsp; [rchris@neotec.dev](mailto:rchris@neotec.dev)

*Draft — work in progress.*

**Abstract.** Descartes named them *imaginary* and Euler made them turn: numbers
that carry both magnitude *and* direction, neither of which a single real number
can hold. We formalise this intuition as a small framework built on two
functionals — an *energy* $e(\cdot)$ and a *phase* $\sigma(\cdot)$ — governed by a
single involution, $e(x)=e(y)$ whenever $\sigma(x)=-\sigma(y)$, realised
concretely by conjugation. On the complex unit circle the framework is exact: the
point $1$ sits at phase $0$ and the point $-1$ at phase $\pi$, and the naive rule
"multiply by adding phases" is associative, commutative and distributive. We then
ask what survives when the single imaginary axis is extended to the three axes
$i,j,k$. We prove, and confirm empirically over $2\times10^{4}$ Monte-Carlo trials,
that the naive phase-additive rule **breaks associativity and distributivity** in
four dimensions, while the **Hamilton product preserves associativity,
distributivity and non-contradiction** — at the sole cost of commutativity. The
quaternions are not one option among many for realising the imaginaries in higher
dimension; up to the laws we demand, they are the only one. *Quaternions are all
you need.*

---

## 1. Introduction

When Descartes coined "imaginary" he meant it dismissively: a quantity that
appears in the algebra but cannot be *realised* as a length on the number line.
Euler supplied the missing realisation. His identity

$$ e^{i\theta} = \cos\theta + i\sin\theta $$

does not place the imaginary unit *on* the real line; it places it *around* it.
The "unrealised" content of $i$ is exactly the content a real number is missing:
**direction**. A real number is a signed magnitude; a complex number is a magnitude
together with a phase. The imaginary axis is the bookkeeping device that lets a
single object carry both.

This paper takes that reading literally and asks a structural question. The
complex numbers realise *one* imaginary direction. Physical space has three. If we
insist on extending the construction so that magnitude and direction continue to
compose lawfully, **which algebraic laws can we keep, and which must we give up?**

We answer with both proof and measurement. Section 2 fixes the framework: the two
functionals $e$ and $\sigma$, the governing involution, the Euler map, and the two
candidate composition operators. Section 3 states the three laws under test —
associativity, distributivity, non-contradiction — and proves the decisive
theorems. Section 4 describes the numerical methodology; Section 5 reports the
results and the figures. Section 6 draws the conclusion the title advertises.

The framing is deliberately layered. The geometric/philosophical reading (the
"realised imaginaries") is *motivation*; the theorems are *what is proven*; the
Monte-Carlo suite is *independent empirical confirmation* of the theorems on
pseudo-random inputs. We are explicit throughout about which is which.

---

## 2. The framework

### 2.1 Representation

A quaternion is written

$$ q = w + x\,i + y\,j + z\,k, \qquad (w,x,y,z)\in\mathbb{R}^4, $$

with real (scalar) part $w$ and imaginary (vector) part $\mathbf v=(x,y,z)$. The
complex numbers embed as the **$i$-plane** $\{\,y=z=0\,\}$. In code each quaternion
is the array `[w, x, y, z]` (`src/framework.py`).

### 2.2 Two functionals: energy and phase

We organise everything around two maps.

* **Energy** (the *realised* magnitude):
  $$ e(q) = |q| = \sqrt{w^2+x^2+y^2+z^2}. $$
* **Phase** (the *unrealised* direction), defined as the imaginary part of the
  quaternion logarithm. For $q=|q|\,(\cos\theta + \mathbf n\sin\theta)$ with unit
  axis $\mathbf n$,
  $$ \sigma(q) = \operatorname{Im}\log q = \theta\,\mathbf n \in \operatorname{span}\{i,j,k\}. $$

Energy answers "how much"; phase answers "which way, and how far around".

### 2.3 The governing involution

The single law that animates the framework is

$$ \boxed{\,e(x)=e(y)\quad\text{whenever}\quad \sigma(x)=-\sigma(y).\,} $$

Two elements of opposite phase carry equal energy. This is not an extra axiom to
be bolted on; it is realised by **conjugation** $q\mapsto q^{*}=w-x i-y j-z k$,
which negates the vector part and fixes the magnitude:

$$ e(q^{*})=e(q), \qquad \sigma(q^{*})=-\sigma(q). $$

We call the pair $(x,y)=(q,q^{*})$ *antipodal*. Their *phase deviation*
$\sigma(x)+\sigma(y)$ vanishes, and — as Section 3.3 shows — their product lands
exactly on the real $1$-ray.

### 2.4 The Euler map and its landmarks

Restricting to the $i$-plane, the Euler map $E(\theta)=\cos\theta + i\sin\theta$
sends phase angles to unit-circle points. Its landmarks are the bridge between the
*multiplicative* world of points and the *additive* world of angles:

| point | $1$ | $i$ | $-1$ | $-i$ |
|------:|:---:|:---:|:----:|:----:|
| phase angle | $0$ | $\pi/2$ | $\pi$ | $3\pi/2$ |

The point $1$ realises the angle $0$ and $-1$ realises the angle $\pi$: the
multiplicative identity sits at the additive identity, and the multiplicative
involution $x\mapsto -x$ is the additive shift $\theta\mapsto\theta+\pi$. This is
the precise sense in which, on the circle, "$1=0$ and $-1=\pi$."

![The Euler involution on the unit circle](../figures/fig1_involution.png)

*Figure 1. The realised imaginaries on $S^1$. Landmarks pair each point with its
phase angle. The antipodal pair $x$ (blue) and $y=x^{*}$ (red) have equal energy
and opposite phase; their phase deviation is zero.*

### 2.5 Two candidate operators

How should two such objects compose? We compare exactly two rules.

* **Phase-additive operator** — the naive realisation of "multiply by adding
  phases":
  $$ h_{+}(x,y) = \exp\!\big(\log x + \log y\big). $$
  In the $i$-plane this *is* complex multiplication, because $\log$ of commuting
  elements adds: $h_{+}(e^{i\alpha},e^{i\beta})=e^{i(\alpha+\beta)}$. It is
  commutative by construction (addition commutes).

* **Hamilton product** — the quaternion multiplication
  $$ (w_1+\mathbf v_1)(w_2+\mathbf v_2) =
     \big(w_1w_2-\mathbf v_1\!\cdot\!\mathbf v_2\big)
     + \big(w_1\mathbf v_2 + w_2\mathbf v_1 + \mathbf v_1\!\times\!\mathbf v_2\big). $$

The cross-product term is the entire story. It vanishes in the $i$-plane (parallel
vectors), so there the two operators agree. Off the plane it does not, and the two
operators part ways.

---

## 3. The three laws and the decisive theorems

We test three laws. For operator $\ast$ and elements $a,b,c$:

* **Associativity:** $(a\ast b)\ast c = a\ast(b\ast c)$.
* **Distributivity:** $a\ast(b+c) = (a\ast b)+(a\ast c)$.
* **Non-contradiction:** the structural identities of the framework hold
  simultaneously without producing a contradiction (Section 3.3).

### 3.1 The Hamilton product keeps associativity and distributivity

**Theorem 1 (Associativity).** *The Hamilton product is associative on
$\mathbb{H}$.*

*Proof.* $\mathbb{H}$ is the real Clifford-type algebra generated by $i,j,k$ with
$i^2=j^2=k^2=ijk=-1$. Multiplication is defined by bilinear extension of the basis
products, and a bilinear product on a finite-dimensional space is associative iff
it is associative on basis elements. Direct computation of the $4^3=64$ basis
triples (equivalently, verifying $i(jk)=(ij)k$ and its permutations) confirms
associativity on generators; bilinearity propagates it to all of $\mathbb{H}$.
$\square$

**Theorem 2 (Distributivity).** *The Hamilton product distributes over addition.*

*Proof.* Immediate from bilinearity: each output component is a fixed bilinear form
in the input components, and bilinear forms are additive in each argument. $\square$

The Cayley table makes the generator relations — and the source of
non-commutativity — visible at a glance.

![Cayley table of the Hamilton product](../figures/fig5_cayley.png)

*Figure 2. $ij=k$ but $ji=-k$: associativity and distributivity hold, commutativity
does not.*

### 3.2 The phase-additive operator breaks both in $\mathbb{H}$

**Theorem 3 (Failure of phase-additivity).** *In the $i$-plane $h_{+}$ coincides
with complex multiplication and is therefore associative and distributive. In
$\mathbb{H}$ it is in general neither.*

*Proof.* On the $i$-plane all logarithms lie in the commutative subalgebra
$\mathbb{R}\oplus\mathbb{R}i$, so $\log x+\log y=\log(xy)$ and
$h_{+}(x,y)=xy$; the complex product is associative and distributive. In
$\mathbb{H}$, $\log a$ and $\log b$ generally do not commute, so by the
Baker–Campbell–Hausdorff formula
$$ \exp(\log a)\exp(\log b)=\exp\!\Big(\log a+\log b+\tfrac12[\log a,\log b]+\cdots\Big), $$
the bracket $[\log a,\log b]=\log a\,\log b-\log b\,\log a$ is generically nonzero,
so $h_{+}(a,b)=\exp(\log a+\log b)\neq ab$. Concretely $h_{+}$ drops the cross-product
term, so it is **not bilinear**; a non-bilinear product cannot satisfy
distributivity, and associativity fails along with it. A single witness suffices to
refute the universal laws, and Section 5 exhibits whole populations of them. $\square$

The contrast is exactly the BCH commutator: when $a,b$ nearly commute the residual
is tiny, and when they do not it is order one. This produces the *bimodal* residual
distribution seen in Figure 4.

### 3.3 Non-contradiction

The framework asserts several identities at once. Non-contradiction is the claim
that they are mutually consistent — that no two of them can be played against each
other to derive a falsehood. The load-bearing identities are:

1. **Fundamental relations:** $i^2=j^2=k^2=ijk=-1$.
2. **Euler landmarks:** $E(0)=1,\ E(\pi)=-1,\ E(\pi/2)=i,\ E(3\pi/2)=-i$.
3. **Antipode product:** $q\,q^{*}=|q|^2$ — a *non-negative real*. The product of
   an element with its opposite-phase antipode has **zero imaginary residue** and
   lands on the $1$-ray (phase $0$) with energy $e(q)^2$.
4. **Conjugation involution:** $e(q^{*})=e(q)$ and $\sigma(q^{*})=-\sigma(q)$.
5. **Exp/Log bridge:** $\exp(\log u)=u$ for unit $u$ — the multiplicative and
   additive descriptions agree.

**Theorem 4 (Non-contradiction).** *Identities 1–5 hold simultaneously in
$\mathbb{H}$.*

*Proof sketch.* $\mathbb{H}$ is an associative real division algebra (Frobenius),
so the multiplicative and additive structures are jointly consistent by
construction; (1) is the defining relation, (3) is $q q^{*}=\sum q_\mu^2$ by direct
expansion, (4) is the definition of conjugation, and (2),(5) are evaluations of the
convergent exp/log series. No identity contradicts another because all are theorems
of one consistent algebra. $\square$

Crucially, **non-contradiction does not require commutativity.** That $ij\neq ji$
is not a contradiction; it is a true, consistent fact of the algebra. The framework
gives up commutativity and keeps everything else — and that is the whole point.

---

## 4. Empirical methodology

Proof tells us the laws hold identically; measurement tells us they hold *in
practice*, on inputs we did not choose, to the precision of real arithmetic. We
treat each law as a hypothesis and measure its **residual** — the norm of the
difference between the two sides — over pseudo-random inputs.

* **Operators:** Hamilton product and phase-additive operator.
* **Domains:** the complex $i$-plane ($y=z=0$) and the full quaternions; components
  drawn i.i.d. Gaussian; unit quaternions sampled uniformly on $S^3$.
* **Trials:** $n=2\times10^{4}$ random triples per (law, operator, domain) cell,
  seed $0$ (`src/verification.py`).
* **Decision rule:** a residual below $10^{-10}$ is at machine-precision and counts
  as the law **holding**; a residual of order $1$ or more counts as the law
  **failing**. (Double precision carries $\approx 16$ digits; accumulated rounding
  over a handful of products sits near $10^{-14}$.)

All figures and numbers below are reproduced by `python src/verification.py` and
`python src/plots.py`. Figure residual maxima are computed on an independent draw
(seed $7$) and therefore differ from the canonical table at the last digit; both
agree on the order of magnitude, which is all the decision rule uses.

---

## 5. Results

### 5.1 The summary matrix

![Law survival matrix](../figures/fig4_law_matrix.png)

*Figure 3. Maximum residual for each law across operator and domain. Green = holds
(machine precision), red = broken (order one). The single red column on the right is
the phase-additive operator in $\mathbb{H}$.*

The canonical maxima (seed $0$, $n=2\times10^4$):

| law | Hamilton / $\mathbb{C}$ | Hamilton / $\mathbb{H}$ | phase-add / $\mathbb{C}$ | phase-add / $\mathbb{H}$ |
|---|---|---|---|---|
| associativity  | $5.4\times10^{-15}$ | $1.4\times10^{-14}$ | $9.1\times10^{-13}$ | $\mathbf{5.4\times10^{1}}$ |
| distributivity | $3.6\times10^{-15}$ | $5.3\times10^{-15}$ | $3.5\times10^{-11}$ | $\mathbf{2.3\times10^{1}}$ |
| commutativity  | $0$ | $\mathbf{3.0\times10^{1}}$ | $0$ | $0$ |

Read the table as a ledger of what each operator costs. The Hamilton product pays
**only** in commutativity (one bold entry, in $\mathbb{H}$) and keeps associativity
and distributivity to machine precision everywhere. The phase-additive operator
buys commutativity but pays in **both** associativity and distributivity the moment
it leaves the plane.

### 5.2 Associativity

![Associativity residuals](../figures/fig2_associativity.png)

*Figure 4. Residual $\lVert(ab)c-a(bc)\rVert$. In $\mathbb{C}$ (left) both operators
sit in the machine-precision band. In $\mathbb{H}$ (right) Hamilton stays there
while the phase-additive operator splits off a second mode near $10$ — exactly the
BCH commutator population of Theorem 3.*

### 5.3 Distributivity

![Distributivity residuals](../figures/fig3_distributivity.png)

*Figure 5. Residual $\lVert a(b{+}c)-(ab{+}ac)\rVert$. Same verdict: distributivity is
a machine-precision fact for the Hamilton product and a macroscopic failure for the
non-bilinear phase-additive operator in $\mathbb{H}$.*

### 5.4 Non-contradiction

![Non-contradiction](../figures/fig6_non_contradiction.png)

*Figure 6. The antipode product. Left: $\operatorname{Re}(q q^{*})$ tracks $|q|^2$
on the line $y=x$ over thousands of random $q$. Right: the imaginary residue
$\lVert\operatorname{Im}(q q^{*})\rVert$ sits entirely in the machine-precision band
— the product of an element with its opposite-phase antipode is a positive real, on
the $1$-ray, with no contradictory imaginary remainder.*

The structural identities verify to machine precision or exactly:

| identity | max residual |
|---|---|
| $i^2=j^2=k^2=ijk=-1$ | $0$ (exact) |
| Euler landmarks | $1.8\times10^{-16}$ |
| $e(q^{*})=e(q)$ | $0$ (exact) |
| $\sigma(q^{*})=-\sigma(q)$ | $0$ (exact) |
| $\operatorname{Im}(q q^{*})=0$ | $1.3\times10^{-15}$ |
| $\operatorname{Re}(q q^{*})=|q|^2$ | $7.1\times10^{-15}$ |
| $\exp(\log u)=u$ | $1.3\times10^{-15}$ |

### 5.5 The tradeoff, and the extension from circle to sphere

![Tradeoff in H](../figures/fig7_tradeoff.png)

*Figure 7. In $\mathbb{H}$, only the Hamilton product keeps both associativity and
distributivity below the consistency threshold; the phase-additive operator buys its
commutativity at their expense.*

![From the circle to the 3-sphere](../figures/fig8_ijk_sphere.png)

*Figure 8. The geometric reason the cross term cannot be wished away. Conjugating
$i$ by random unit quaternions, $u\,i\,u^{*}$, sweeps out the entire imaginary
2-sphere: in $\mathbb{H}$ "direction" is genuinely three-dimensional, and any lawful
composition must move points around that sphere — which is precisely what the
Hamilton product's cross term does and the phase-additive rule cannot.*

---

## 6. Discussion and conclusion

The experiments confirm, on inputs we did not choose, what the theorems guarantee.
Realising the imaginaries in one dimension is forgiving: on the unit circle, adding
phases and multiplying points are the same act, and every law holds. The forgiveness
ends in higher dimension. Three imaginary directions do not commute, and *something*
must give:

* keep **commutativity** by adding phases, and you **lose associativity and
  distributivity** — the algebra stops being an algebra (Theorem 3, Figures 3–5);
* keep **associativity and distributivity** with the Hamilton product, and you lose
  **only commutativity** — a true, non-contradictory fact, not an inconsistency
  (Theorems 1, 2, 4, Figures 3, 6, 7).

Distributivity and associativity are the laws that make a *ring*; without them there
is no well-defined arithmetic to speak of, no factoring, no linear structure. Among
the candidates for extending the realised imaginaries to three directions, the
Hamilton product is the one that keeps them. By Frobenius's theorem this is no
accident: $\mathbb{R}$, $\mathbb{C}$ and $\mathbb{H}$ are the *only* finite-dimensional
associative real division algebras. Once you demand magnitude-and-direction,
associativity, distributivity, and non-contradiction in the smallest dimension that
holds physical space, the quaternions are not *a* choice. They are the *only* choice.

Quaternions are all you need.

---

## Appendix A. Reproducibility

```bash
pip install -r requirements.txt
python src/verification.py     # prints the residual table, writes results/metrics.json
python src/plots.py            # writes figures/fig1..fig8 .png
```

* `src/framework.py` — the algebra: Hamilton product, conjugation, $e$, $\sigma$,
  quaternion `exp`/`log`, the phase-additive operator, the Euler map.
* `src/verification.py` — Monte-Carlo residual suite and non-contradiction checks.
* `src/plots.py` — every figure in this paper.
* `results/metrics.json` — machine-readable record of all residuals.

## Appendix B. Notation

| symbol | meaning |
|---|---|
| $e(q)$ | energy / norm $|q|$ |
| $\sigma(q)$ | phase $=\operatorname{Im}\log q=\theta\,\mathbf n$ |
| $q^{*}$ | conjugate (opposite phase, equal energy) |
| $E(\theta)$ | Euler map $\cos\theta+i\sin\theta$ |
| $h_{+}$ | phase-additive operator $\exp(\log x+\log y)$ |
| $(\cdot)$ | Hamilton product |
| $[a,b]$ | commutator $ab-ba$ |
