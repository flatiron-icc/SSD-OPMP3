"""Independent W matrices for the tests: the paper's Eqs. 7, 9 and 11a in a
plain spin-orbital basis, from a run's semicanonical orbitals and AO (or
density-fitted) integrals, without PySCF's CC code.  The same-spin and
opposite-spin parts (Eqs. 17-18) are the terms of Eq. 11a whose k has the
spin of i or the other spin.  Small molecules only: the spin-orbital
<ab||cd> block is held in memory.
"""
from __future__ import annotations

import numpy as np
from pyscf import ao2mo


def _phys(eri, a, b, c, d) -> np.ndarray:
    """<pq|rs> = (pr|qs) for spin-orbital sets a, b, c, d, each a pair
    (AO coefficients, spin labels)."""
    g = ao2mo.general(eri, (a[0], c[0], b[0], d[0]), compact=False)
    g = g.reshape(a[0].shape[1], c[0].shape[1], b[0].shape[1], d[0].shape[1])
    g = g.transpose(0, 2, 1, 3)
    same_pr = a[1][:, None, None, None] == c[1][None, None, :, None]
    same_qs = b[1][None, :, None, None] == d[1][None, None, None, :]
    return g * (same_pr & same_qs)


def _anti(eri, a, b, c, d) -> np.ndarray:
    """<pq||rs> = <pq|rs> - <pq|sr>."""
    return _phys(eri, a, b, c, d) - _phys(eri, a, b, d, c).transpose(0, 1, 3, 2)


def independent_w(eri, coeff, nocc, eps, ncore: int) -> dict[str, np.ndarray]:
    """The eight W{2,3}{alpha,beta}_{ss,os} matrices.

    eri: packed AO integrals (mol.intor("int2e", aosym="s8"), or a DF
    get_eri()); coeff, nocc, eps: (alpha, beta) pairs of semicanonical MO
    coefficients (occupied first), occupied counts and orbital energies;
    ncore: frozen occupied orbitals per spin (the lowest ones).
    """
    oc, os_, oe, vc, vs, ve = [], [], [], [], [], []
    for s in (0, 1):
        c, n, e = np.asarray(coeff[s]), int(nocc[s]), np.asarray(eps[s])
        oc.append(c[:, ncore:n]); os_.append(np.full(n - ncore, s)); oe.append(e[ncore:n])
        vc.append(c[:, n:]);      vs.append(np.full(c.shape[1] - n, s)); ve.append(e[n:])
    o = (np.hstack(oc), np.concatenate(os_))
    v = (np.hstack(vc), np.concatenate(vs))
    eo, ev = np.concatenate(oe), np.concatenate(ve)

    oovv = _anti(eri, o, o, v, v)
    vvvv = _anti(eri, v, v, v, v)
    oooo = _anti(eri, o, o, o, o)
    ovvo = _anti(eri, o, v, v, o)
    D = (ev[None, None, :, None] + ev[None, None, None, :]
         - eo[:, None, None, None] - eo[None, :, None, None])

    c1 = -oovv / D                                                   # Eq. 7
    ring = np.einsum("kbcj,ikac->ijab", ovvo, c1, optimize=True)     # <kb||cj> c_ikac
    c2 = -(0.5 * np.einsum("abcd,ijcd->ijab", vvvv, c1, optimize=True)
           + 0.5 * np.einsum("klij,klab->ijab", oooo, c1, optimize=True)
           + ring - ring.transpose(1, 0, 2, 3) - ring.transpose(0, 1, 3, 2)
           + ring.transpose(1, 0, 3, 2)) / D                         # Eq. 9

    out = {}
    for n, c in ((2, c1), (3, c2)):
        for s, spin in enumerate(("alpha", "beta")):
            i = np.where(o[1] == s)[0]
            for kind, kspin in (("ss", s), ("os", 1 - s)):
                k = (o[1] == kspin).astype(float)
                x = 0.125 * np.einsum("ikab,jkab,k->ij", c[i], oovv[i], k, optimize=True)  # Eq. 11a
                out[f"W{n}{spin}_{kind}"] = x + x.T
    return out


def tr_pade(w2: np.ndarray, w3: np.ndarray, x: float = 1.0) -> float:
    """Tr[W2 (W2 - x W3)^-1 W2] (Eq. 13); 0 for an empty or zero block."""
    if w2.size == 0 or not np.any(w2):
        return 0.0
    return float(np.trace(w2 @ np.linalg.solve(w2 - x * w3, w2)))


def resummed(w: dict[str, np.ndarray], x_ss: float) -> dict[str, float]:
    """OP-MP3, SOP-MP3 and SSD-OPMP3(x_ss) correlation energies (Eqs. 14, 21, 22)."""
    def p(kinds, x=1.0):
        return sum(tr_pade(sum(w[f"W2{s}_{k}"] for k in kinds),
                           sum(w[f"W3{s}_{k}"] for k in kinds), x)
                   for s in ("alpha", "beta"))
    return {"OP-MP3": p(["ss", "os"]),
            "SOP-MP3": p(["ss"]) + p(["os"]),
            "SSD-OPMP3": p(["ss"], x_ss) + p(["os"])}
