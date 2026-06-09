"""
word_relations.py
=================

Grade real word pairs into the truthiness structure of Section 11.7, with the
cells/relation MEASURED from data rather than guessed.  Two graders:

  * embedding cosine  -- cos(theta) between pretrained GloVe vectors.  This is the
    relational measure Re(c zbar)=cos(theta) of Section 11.6 directly, and (Levy &
    Goldberg 2014) word2vec/SGNS implicitly factorises the shifted PMI matrix, so
    the embedding inner product is ~ PMI ~ log of the coupling gamma.
  * co-occurrence    -- over the text8 corpus (~17M tokens of Wikipedia), at three
    window scales (10/30/100 tokens ~ window/sentence/paragraph).  Each window is
    an instance; the four minterm masses are the contingency table, from which we
    report PMI = log P(A,B)/(P(A)P(B)) and the Section 11.7 quantities.

Requires `gensim` (pip install gensim) and network access to the gensim-data
models on GitHub.  Writes ../results/word_relations.json.

Honest caveats, borne out by the numbers:
  - embedding cosine measures DISTRIBUTIONAL similarity; it cannot separate
    synonyms from antonyms (both share contexts), e.g. induce/deduce score
    positive though they are opposite directions of inference.
  - for RARE words the contingency is dominated by the empty corner p00~1, so the
    resultant z and covariance gamma are non-discriminative; PMI (base-rate
    normalised) is the meaningful signal there.  The framework's z/gamma are most
    discriminative for BALANCED predicates (cf. the number-theory examples in
    logic.py:truthiness_examples).
"""

from __future__ import annotations

import contextlib
import json
import os

import numpy as np

FORMS = {
    "empathy": {"empathy", "empathic", "empathetic", "empathize", "empathise"},
    "sympathy": {"sympathy", "sympathetic", "sympathies", "sympathize", "sympathise"},
    "illusion": {"illusion", "illusions", "illusory"},
    "delusion": {"delusion", "delusions", "delusional"},
    "induce": {"induce", "induced", "induces", "induction", "inductive", "inducing"},
    "deduce": {"deduce", "deduced", "deduces", "deduction", "deductive", "deducing"},
}
PAIRS = [("empathy", "sympathy"), ("illusion", "delusion"), ("induce", "deduce")]
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "word_relations.json")


def _load(name):
    """Load a gensim-data resource with the download progress bar silenced."""
    import gensim.downloader as api
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        return api.load(name)


def embedding_cosines(model="glove-wiki-gigaword-50"):
    kv = _load(model)
    return {f"{a}/{b}": float(kv.similarity(a, b)) for a, b in PAIRS}, model


def cooccurrence(windows=(10, 30, 100)):
    tokens = [t for chunk in _load("text8") for t in chunk]
    n = len(tokens)
    inv = {f: c for c, fs in FORMS.items() for f in fs}
    pres = {c: np.zeros(n, bool) for c in FORMS}
    for i, t in enumerate(tokens):
        c = inv.get(t)
        if c is not None:
            pres[c][i] = True
    counts = {c: int(pres[c].sum()) for c in FORMS}

    def table(A, B, w):
        m = (n // w) * w
        a = A[:m].reshape(-1, w).any(1)
        b = B[:m].reshape(-1, w).any(1)
        M = len(a)
        return np.array([(a & b).sum(), (a & ~b).sum(), (~a & b).sum(), (~a & ~b).sum()]) / M

    def quantities(p):
        p11, p10, p01, p00 = p
        PA, PB = p11 + p10, p11 + p01
        z = [float(PA + PB - 1), float(PA - PB)]
        cov = float(p11 * p00 - p10 * p01)
        pmi = float(np.log(p11 / (PA * PB))) if p11 > 0 and PA * PB > 0 else None
        detM = float(np.sqrt(p11 * p00) - np.sqrt(p10 * p01))
        return {"cells": [float(x) for x in p], "P(A)": float(PA), "P(B)": float(PB),
                "gamma=Cov": cov, "PMI": pmi, "z": z, "detM": detM}

    out = {}
    for a, b in PAIRS:
        out[f"{a}/{b}"] = {f"win{w}": quantities(table(pres[a], pres[b], w)) for w in windows}
    return counts, out, n


def main():
    cos, model = embedding_cosines()
    counts, cooc, n = cooccurrence()
    res = {
        "embedding": {"model": model, "cosine": cos,
                      "note": "cosine = relational cos(theta); ~ shifted PMI (Levy-Goldberg)"},
        "corpus": {"name": "text8 (Wikipedia, %d tokens)" % n, "raw_counts": counts},
        "cooccurrence": cooc,
    }
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=2)

    print(f"corpus text8: {n:,} tokens; raw counts {counts}\n")
    print(f"{'pair':20} {'cosine':>7} | {'PMI@10':>7} {'PMI@30':>7} {'PMI@100':>8}")
    for a, b in PAIRS:
        pr = f"{a}/{b}"
        g = lambda w: cooc[pr][w]["PMI"]
        fmt = lambda v: f"{v:+7.2f}" if v is not None else "   -inf"
        print(f"{pr:20} {cos[pr]:+7.3f} | {fmt(g('win10'))} {fmt(g('win30'))} {fmt(g('win100')):>8}")
    print(f"\nwrote {os.path.relpath(RESULTS)}")


if __name__ == "__main__":
    main()
