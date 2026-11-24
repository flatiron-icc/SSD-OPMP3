"""Resummation of the second- and third-order correlation contributions.

Notation follows the paper. W2 and W3 are the occupied-orbital
matrices W^(2), W^(3) that partition the MP2 and MP3 correlation energies
over occupied orbitals, so that E2 = Tr W2 and E3 = Tr W3. With

    P(A, B) = A (A - B)^-1 A

the orbital Padé resummation with the third order scaled by x is

    E_c = Tr P(W2, x W3),                                       OP-MP3(x)

and the scalar Padé resummation of the energies themselves is

    E_c^[2,3] = E2^2 / (E2 - x E3).

How Tr P is evaluated here. Let Z = W2^+ W3, where W2^+ is the
pseudo-inverse of the symmetric W2 (its eigendecomposition with the zero
eigenvalues left out), with eigendecomposition Z = V diag(zeta) V^-1, and
put c_i = (V^-1 W2 V)_ii. Since W2 - x W3 = W2 (1 - x Z),

    Tr P(W2, x W3) = Tr[ W2 (1 - x Z)^-1 W2^+ W2 ]
                   = Tr[ (1 - x Z)^-1 W2 ]                 (cyclic trace)
                   = Tr[ V (1 - x diag(zeta))^-1 V^-1 W2 ]
                   = Tr[ (1 - x diag(zeta))^-1 (V^-1 W2 V) ]
                   = sum_i c_i / (1 - x zeta_i).

For a non-singular W2 the pseudo-inverse is the inverse and this is the
paper's P(W2, x W3) exactly. W2 is singular when an occupied orbital
carries no correlation, which is the frozen-core case: a frozen orbital
has a zero row and column in both W2 and W3 (it enters no amplitude), so
P is defined on the range of W2 and the frozen orbital drops out, with
zeta = 0 and c = 0. The pair is decomposed once; a fit then varies x at
no further cost. The scalar Padé is the same expression on the 1x1
matrices [[E2]], [[E3]]: zeta = E3/E2 and c = E2 give
E2 / (1 - x E3/E2) = E2^2 / (E2 - x E3). Because sum_i c_i = Tr W2 = E2
and sum_i c_i zeta_i = Tr(W2 Z) = E3, expanding
c_i / (1 - x zeta_i) = c_i (1 + x zeta_i + x^2 zeta_i^2 + ...) shows that
the resummation reproduces E2 + x E3 through first order in x and
supplies the higher orders as a geometric series.

The resummations set aside in the paper (--alt-resummations only) replace
1 / (1 - z) by another g(z) with g(0) = g'(0) = 1, E_c = sum_i c_i g(x zeta_i):
Padé–Borel g(z) = f(3/z)/2 with f(w) = w^3 (e^-w Ei(w) - 1/w - 1/w^2);
exponential e^z; Landau–Zener (e^(2z) - 1)/(2z); logarithmic
-ln(1 - 2z)/(2z).

The spectra of many systems are concatenated and summed segment-wise, so
a fit objective over a dataset is one vectorised call.  A pair with an
empty sector, Tr W2 = 0 or W3 = 0 carries its value instead of a spectrum.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eig, pinvh, solve
from scipy.special import exp1, expi

from blocks import ZERO_TOL, is_zero_block, tr


# ---------------------------------------------------------------------------
# One (W2, W3) pair
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Spectrum:
    """What a resummation needs from one (W2, W3) pair, independent of x.

    zeta, c: eigenvalues of W2^+ W3 and the weights (V^-1 W2 V)_ii; None
             for the plain MP series or a degenerate pair.
    e2, e3:  Tr W2 and Tr W3.
    value:   the resummed energy of a degenerate pair (0 for an empty
             sector or Tr W2 = 0; E2 when W3 = 0), else None.
    """
    zeta:  np.ndarray | None
    c:     np.ndarray | None
    e2:    float
    e3:    float
    value: float | None = None


def traces(w2: np.ndarray, w3: np.ndarray) -> Spectrum:
    """E2 and E3 only: all the plain MP series needs."""
    return Spectrum(None, None, tr(w2), tr(w3))


def spectrum(w2: np.ndarray, w3: np.ndarray) -> Spectrum:
    """Decompose W2^+ W3, or record the value of a degenerate pair.

    W2^+ is the pseudo-inverse of the symmetric W2 from its
    eigendecomposition (scipy.linalg.pinvh), which is the inverse when W2
    is non-singular and restricts P to the range of W2 otherwise (frozen
    core: see the module docstring).
    """
    if w2.size == 0:
        return Spectrum(None, None, 0.0, 0.0, value=0.0)
    e2, e3 = tr(w2), tr(w3)
    if abs(e2) < ZERO_TOL:
        return Spectrum(None, None, e2, e3, value=0.0)
    if is_zero_block(w3):
        return Spectrum(None, None, e2, e3, value=e2)
    zeta, V = eig(pinvh(w2) @ w3)
    c = np.diag(solve(V, w2 @ V))
    return Spectrum(zeta, c, e2, e3)


# ---------------------------------------------------------------------------
# Many systems at once
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Spectra:
    """The spectra of one (W2, W3) pair over an ordered list of systems.

    active:  indices of the systems that carry a spectrum.
    zeta, c: those spectra, concatenated; offsets: segment starts.
    base:    per-system value, holding the degenerate pairs' values (0 for
             active systems).
    e2, e3:  per-system traces.
    """
    active:  np.ndarray
    zeta:    np.ndarray
    c:       np.ndarray
    offsets: np.ndarray
    base:    np.ndarray
    e2:      np.ndarray
    e3:      np.ndarray


def stack(spectra: Sequence[Spectrum]) -> Spectra:
    active  = [i for i, s in enumerate(spectra) if s.zeta is not None]
    zetas   = [np.ravel(spectra[i].zeta).astype(complex) for i in active]
    lengths = [z.size for z in zetas]
    return Spectra(
        active=np.array(active, dtype=np.intp),
        zeta=np.concatenate(zetas or [np.zeros(0, complex)]),
        c=np.concatenate([np.ravel(spectra[i].c).astype(complex)
                          for i in active] or [np.zeros(0, complex)]),
        offsets=np.array([0, *np.cumsum(lengths[:-1])], dtype=np.intp)
                if active else np.zeros(0, dtype=np.intp),
        base=np.array([s.value or 0.0 for s in spectra]),
        e2=np.array([s.e2 for s in spectra]),
        e3=np.array([s.e3 for s in spectra]),
    )


def _per_system(sp: Spectra, values: np.ndarray) -> np.ndarray:
    """Degenerate pairs' values plus the segment sums of `values`."""
    out = sp.base.copy()
    if sp.active.size:
        out[sp.active] = np.add.reduceat(values, sp.offsets)
    return out


def _resum(sp: Spectra, g: Callable[[np.ndarray], np.ndarray],
           x: float) -> np.ndarray:
    """sum_i Re[c_i g(x zeta_i)] per system. Overflow yields inf silently;
    the fit maps non-finite energies to an infinite objective."""
    with np.errstate(all="ignore"):
        return _per_system(sp, np.real(sp.c * g(x * sp.zeta)))


# ---------------------------------------------------------------------------
# The resummations: (Spectra, x) -> E_c per system
# ---------------------------------------------------------------------------

# Relative distance to a Padé pole |1 - x zeta| below which the system's
# energy is reported as inf rather than a huge finite number.
POLE_REL_TOL: float = 1.0e-13


def mp_series(sp: Spectra, x: float) -> np.ndarray:
    """No resummation: E2 + x E3."""
    return sp.e2 + x * sp.e3


def pade(sp: Spectra, x: float) -> np.ndarray:
    """sum_i c_i / (1 - x zeta_i) = Tr P(W2, x W3); inf if on a pole."""
    z     = x * sp.zeta
    denom = 1.0 - z
    with np.errstate(all="ignore"):
        out = _per_system(sp, np.real(sp.c / denom))
    if sp.active.size:
        on_pole = np.abs(denom) < POLE_REL_TOL * np.maximum(1.0, np.abs(z))
        hits = np.add.reduceat(on_pole.astype(np.intp), sp.offsets)
        out[sp.active] = np.where(hits > 0, np.inf, out[sp.active])
    return out


def pade_borel(sp: Spectra, x: float) -> np.ndarray:
    return _resum(sp, _g_borel, x)


def exponential(sp: Spectra, x: float) -> np.ndarray:
    return _resum(sp, np.exp, x)


def landau_zener(sp: Spectra, x: float) -> np.ndarray:
    """sum_i c_i (e^(2 x zeta_i) - 1) / (2 x zeta_i)."""
    return _resum(sp, _g_landau_zener, 2.0 * x)


def logarithmic(sp: Spectra, x: float) -> np.ndarray:
    """sum_i c_i [-ln(1 - 2 x zeta_i)] / (2 x zeta_i)."""
    return _resum(sp, _g_logarithmic, 2.0 * x)


# ---------------------------------------------------------------------------
# The functions g of the other resummations, vectorised over complex arguments
# ---------------------------------------------------------------------------

_SMALL: float = 1.0e-10        # |u| below which the u -> 0 limit is used


def _g_landau_zener(u: np.ndarray) -> np.ndarray:
    """(e^u - 1) / u, with its limit 1 at u = 0; called with u = 2z."""
    return np.where(np.abs(u) < _SMALL, 1.0, np.expm1(u) / u)


def _log1p(u: np.ndarray) -> np.ndarray:
    """ln(1 + u) for complex u, accurate for small |u| (numpy's complex
    log1p is not: it evaluates log(1 + u) and loses the leading digits)."""
    x, y = u.real, u.imag
    return 0.5 * np.log1p(2.0 * x + x * x + y * y) + 1j * np.arctan2(y, 1.0 + x)


def _g_logarithmic(u: np.ndarray) -> np.ndarray:
    """-ln(1 - u) / u, with its series 1 + u/2 + u^2/3 near u = 0; called
    with u = 2z."""
    return np.where(np.abs(u) < _SMALL, 1.0 + u / 2.0 + u * u / 3.0,
                    -_log1p(-u) / u)


# f(w) for Padé–Borel: the small-|w| limit -w - w^2, the direct formula,
# and beyond |w| = 40 (where the direct form cancels) the asymptotic series
# sum_k (k+2)!/w^k.  z = 0 maps to w = inf, where f = 2.
_F_SMALL:  float = 1.0e-150
_F_DIRECT: float = 40.0
_F_ASYMP_COEF: np.ndarray = np.array(
    [float(math.factorial(k + 2)) for k in range(30)][::-1])
_REAL_TOL: float = 1.0e-12


def _ei(w: np.ndarray) -> np.ndarray:
    """Ei(w) on the real axis (principal value), -E1(-w) off it."""
    out = np.empty(w.shape, dtype=complex)
    real = np.abs(w.imag) < _REAL_TOL * np.maximum(np.abs(w.real), 1.0)
    out[real] = expi(w[real].real)
    out[~real] = -exp1(-w[~real])
    return out


def _f(w: np.ndarray) -> np.ndarray:
    out = np.empty_like(w)
    a = np.abs(w)
    small, asym = a < _F_SMALL, a >= _F_DIRECT
    direct = ~small & ~asym
    ws = w[small]
    out[small] = -ws - ws * ws
    wd = w[direct]
    out[direct] = wd**3 * (np.exp(-wd) * _ei(wd) - 1.0 / wd - 1.0 / (wd * wd))
    out[asym] = np.polyval(_F_ASYMP_COEF, 1.0 / w[asym])
    return out


def _g_borel(z: np.ndarray) -> np.ndarray:
    out = np.full(z.shape, 1.0, dtype=complex)          # z = 0: f(inf)/2 = 1
    nz = z != 0.0
    out[nz] = 0.5 * _f(3.0 / z[nz])
    return out
