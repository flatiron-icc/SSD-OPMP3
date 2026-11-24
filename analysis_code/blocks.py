"""Assemble the (W2, W3) block pairs the resummations act on.

The paper partitions the n-th order correlation energy over occupied
orbitals as E_c^(n) = Tr W^(n), with W^(n) block-diagonal in spin and each
spin block the sum of a same-spin (SS) and an opposite-spin (OS) part.
The eight input arrays are those parts, W{2,3}{alpha,beta}_{ss,os}. The
resummations act on three block-diagonal pairs built from them:

    W2,    W3        SS + OS per spin, the full W^(2), W^(3)    (OP-MP3)
    W2_ss, W3_ss     same-spin part per spin                    (SOP-MP3, SS)
    W2_os, W3_os     opposite-spin part per spin                (SOP-MP3, OS)
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag


# Absolute tolerance below which a W block counts as empty.
ZERO_TOL: float = 1.0e-14

SPINS = ("alpha", "beta")
PAIRS = (("", ("ss", "os")), ("_ss", ("ss",)), ("_os", ("os",)))


def tr(matrix: np.ndarray) -> float:
    return float(np.trace(matrix).real)


def is_zero_block(m: np.ndarray) -> bool:
    return m.size == 0 or float(np.max(np.abs(m))) < ZERO_TOL


def _pack(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Block-diagonal (W2, W3) over the spin sectors where either is
    non-zero; frozen-orbital zero rows inside a sector stay (handled by the
    pseudo-inverse in resummation.spectrum)."""
    kept = [(a, b) for a, b in pairs
            if not (is_zero_block(a) and is_zero_block(b))]
    if not kept:
        return np.zeros((0, 0)), np.zeros((0, 0))
    a_parts, b_parts = zip(*kept)
    return block_diag(*a_parts), block_diag(*b_parts)


def combine_blocks(w: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """{"W2", "W3", "W2_ss", "W3_ss", "W2_os", "W3_os"} from the eight inputs."""
    out: dict[str, np.ndarray] = {}
    for tag, comps in PAIRS:
        out[f"W2{tag}"], out[f"W3{tag}"] = _pack([
            (sum(w[f"W2{spin}_{c}"] for c in comps),
             sum(w[f"W3{spin}_{c}"] for c in comps))
            for spin in SPINS
        ])
    return out
