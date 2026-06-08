# Quaternions Are All You Need

Empirical evidence and proofs that the *realised-imaginary* framework —
energy $e(\cdot)$, phase $\sigma(\cdot)$, and the involution
$e(x)=e(y)\Leftrightarrow\sigma(x)=-\sigma(y)$ — is **associative**,
**distributive**, and **non-contradictory** under the Hamilton product, and
that the naive *phase-additive* alternative breaks associativity and
distributivity the moment it leaves the complex plane.

> **Thesis.** Extending the imaginaries from one direction (the complex unit
> circle, where $1\leftrightarrow 0$ and $-1\leftrightarrow\pi$) to three
> ($i,j,k$), *something* must give. Add phases and you keep commutativity but
> lose associativity + distributivity; use the Hamilton product and you keep
> associativity + distributivity, losing only commutativity. By Frobenius's
> theorem that tradeoff is forced — so **quaternions are all you need**.

## The paper

[`paper/paper.md`](paper/paper.md) — full write-up with definitions, theorems,
proofs, methodology, results, and all figures.

## Quick start

```bash
pip install -r requirements.txt
python src/verification.py     # Monte-Carlo residual table -> results/metrics.json
python src/plots.py            # all figures -> figures/
```

## Results at a glance

Maximum residual over $2\times10^4$ random trials (machine precision $\approx10^{-15}$):

| law | Hamilton / ℂ | Hamilton / ℍ | phase-add / ℂ | phase-add / ℍ |
|---|---|---|---|---|
| associativity  | 5.4e-15 | 1.4e-14 | 9.1e-13 | **5.4e+01** |
| distributivity | 3.6e-15 | 5.3e-15 | 3.5e-11 | **2.3e+01** |
| commutativity  | 0 | **3.0e+01** | 0 | 0 |

Bold = law broken. All non-contradiction identities ($i^2=j^2=k^2=ijk=-1$,
$qq^{*}=|q|^2$, $e(q^{*})=e(q)$, $\sigma(q^{*})=-\sigma(q)$) hold to machine
precision or exactly.

![Law survival matrix](figures/fig4_law_matrix.png)

## Layout

```
src/framework.py      core algebra (Hamilton & phase-additive operators, e, sigma, exp/log)
src/verification.py   Monte-Carlo law-residual + non-contradiction suite
src/plots.py          generates figures/fig1..fig8
paper/paper.md        the paper
figures/              generated figures
results/metrics.json  machine-readable residuals
```
