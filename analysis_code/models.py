"""The correlation-energy models of the paper.

A model is a sum of contributions, each one (W2 block, W3 block) pair of
blocks.py resummed with its third order scaled by x:

    OP-MP3(x)            Tr P(W2, x W3),   P(A, B) = A (A - B)^-1 A
    SOP-MP3(x_ss, x_os)  Tr P(W2_ss, x_ss W3_ss) + Tr P(W2_os, x_os W3_os)
    SSD-OPMP3(x_ss)      SOP-MP3(x_ss, 1)
    MP2.X                E2 + x E3

A ModelKey names a model: its resummation ("MP" for the plain series,
"Padé", or a set-aside alternative), its scaling (`contributions`), whether
it acts on the W matrices (orbital) or on their traces (scalar, printed
without the O: P-MP3, SP-MP3), and the fixed (x_ss, x_os) of SOP-MP3 and
SSD-OPMP3.  `precompute` decomposes each system once; `evaluate` then
applies any parameter values.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

import resummation as rs
from blocks import tr
from config import (
    SCS_MP2_OS,
    SCS_MP2_SS,
    SCS_MP3_E3,
    SOS_MP2_OS,
    X_FIXED_DEFAULT,
)

# ---------------------------------------------------------------------------
# Resummations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Resummation:
    """A resummation: `decompose` a block pair once, `resum` at any x.
    `stem` builds the printed names ("P" -> P-MP3, OP-MP3, SOP-MP3, ...)."""
    name:      str
    decompose: Callable[[np.ndarray, np.ndarray], rs.Spectrum]
    resum:     Callable[[rs.Spectra, float], np.ndarray]
    stem:      str = ""


MP   = "MP"
PADE = "Padé"

RESUMMATIONS: tuple[Resummation, ...] = (
    Resummation(MP,             rs.traces,   rs.mp_series),
    Resummation(PADE,           rs.spectrum, rs.pade,         "P"),
    Resummation("Padé–Borel",   rs.spectrum, rs.pade_borel,   "Borel"),
    Resummation("exponential",  rs.spectrum, rs.exponential,  "Exp"),
    Resummation("Landau-Zener", rs.spectrum, rs.landau_zener, "LZ"),
    Resummation("logarithmic",  rs.spectrum, rs.logarithmic,  "Log"),
)
_BY_NAME = {r.name: r for r in RESUMMATIONS}

# The alternatives enter no result of the paper and are evaluated only when
# asked for (--alt-resummations).
OTHER_RESUMMATIONS = tuple(r.name for r in RESUMMATIONS if r.name not in (MP, PADE))


def resummation(name: str) -> Resummation:
    return _BY_NAME[name]


def _resummed(alt: bool) -> tuple[str, ...]:
    return (PADE, *(OTHER_RESUMMATIONS if alt else ()))


# ---------------------------------------------------------------------------
# Scalings and their contributions
# ---------------------------------------------------------------------------

UNSCALED       = "E2+E3"           # MP3, OP-MP3
SECOND_ORDER   = "E2"              # MP2
SCALED         = "E2+x*E3"         # MP2.X, OP-MP3(x)
SPIN_SEPARATED = "ss+os fixed"     # SOP-MP3 at fixed (x_ss, x_os); SSD-OPMP3 at (x_ss, 1)
SCALED_SS_OS   = "x_ss,x_os"       # SOP-MP3(x_ss, x_os)
SCALED_SS      = "x_ss"            # x_os = 1
OS_ONLY        = "x_os"            # x_ss = 0
SCS_MP2        = "SCS-MP2"         # literature spin-component scalings (MP only)
SOS_MP2        = "SOS-MP2"
SCS_MP3        = "SCS-MP3"

# Parameter-free MP-series models. The spin-separated MP series at (1, 1)
# is MP3 itself, so the MP series has no SOP-MP3 counterpart.
MP_DEFAULTS = (SECOND_ORDER, UNSCALED, SCS_MP2, SOS_MP2, SCS_MP3)
FITTED      = (SCALED, SCALED_SS_OS, SCALED_SS, OS_ONLY)

# Fit parameters, named as in the paper.
X, X_SS, X_OS = "x", "x_ss", "x_os"


@dataclass(frozen=True)
class Contribution:
    """One (W2 block, W3 block) pair entering a model.

    scale multiplies the third-order block: a number, or the name of a
    fit parameter. weight multiplies the whole contribution; it is 1
    except in the literature spin-component-scaled models, whose
    coefficients sit on the SECOND order (SCS-MP2 = p_ss E2_ss +
    p_os E2_os), which a scale on the third order cannot express.
    """
    w2:     str
    w3:     str
    scale:  str | float
    weight: float = 1.0


_ALL, _SS, _OS = ("W2", "W3"), ("W2_ss", "W3_ss"), ("W2_os", "W3_os")


def contributions(scaling: str,
                  x_fixed: tuple[float, float] | None = None) -> list[Contribution]:
    C = Contribution
    if scaling == UNSCALED:
        return [C(*_ALL, 1.0)]
    if scaling == SECOND_ORDER:
        return [C(*_ALL, 0.0)]
    if scaling == SCALED:
        return [C(*_ALL, X)]
    if scaling == SPIN_SEPARATED:
        x_ss, x_os = x_fixed
        return [C(*_SS, x_ss), C(*_OS, x_os)]
    if scaling == SCALED_SS_OS:
        return [C(*_SS, X_SS), C(*_OS, X_OS)]
    if scaling == SCALED_SS:
        return [C(*_SS, X_SS), C(*_OS, 1.0)]
    if scaling == OS_ONLY:
        return [C(*_SS, 0.0), C(*_OS, X_OS)]
    if scaling == SCS_MP2:
        return [C(*_SS, 0.0, SCS_MP2_SS), C(*_OS, 0.0, SCS_MP2_OS)]
    if scaling == SOS_MP2:
        return [C(*_OS, 0.0, SOS_MP2_OS)]
    if scaling == SCS_MP3:
        # SCS-MP2 plus a quarter of E3, written as (E2 + E3) - E2.
        return [C(*_SS, 0.0, SCS_MP2_SS), C(*_OS, 0.0, SCS_MP2_OS),
                C(*_ALL, 1.0, SCS_MP3_E3), C(*_ALL, 0.0, -SCS_MP3_E3)]
    raise ValueError(f"Unknown scaling {scaling!r}")


# ---------------------------------------------------------------------------
# Model identity and paper names
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelKey:
    resummation: str
    scaling:     str
    orbital:     bool                       = True
    x_fixed:     tuple[float, float] | None = None

    def display_name(self) -> str:
        """The paper's name for this model: OP-MP3, SOP-MP3,
        SSD-OPMP3(x_ss) and their fitted forms; the scalar variants drop
        the O of "orbital" (P-MP3, SP-MP3, SSD-PMP3)."""
        if self.resummation == MP:
            return _MP_NAMES[self.scaling]
        stem = ("O" if self.orbital else "") + resummation(self.resummation).stem
        base = f"{stem}-MP3"
        ssd  = f"SSD-{stem}MP3"          # the paper drops the hyphen here
        if self.scaling == SPIN_SEPARATED:
            x_ss, x_os = self.x_fixed
            if (x_ss, x_os) == (1.0, 1.0):
                return f"S{base}"
            tag = f"x_ss={x_ss:g}" + (f";x_os={x_os:g}" if x_os != 1.0 else "")
            return f"{ssd}({tag})"
        return {
            UNSCALED:     base,
            SCALED:       f"{base}(x)",
            SCALED_SS_OS: f"S{base}(x_ss,x_os)",
            SCALED_SS:    f"{ssd}(x_ss)",
            OS_ONLY:      f"S{base}(x_ss=0;x_os)",
        }[self.scaling]


_MP_NAMES = {
    SECOND_ORDER: "MP2", UNSCALED: "MP3", SCS_MP2: "SCS-MP2",
    SOS_MP2: "SOS-MP2", SCS_MP3: "SCS-MP3",
    SCALED: "MP2.X", SCALED_SS_OS: "MP2.X(x_ss,x_os)",
    SCALED_SS: "MP2.X(x_ss;x_os=1)", OS_ONLY: "MP2.X(x_ss=0;x_os)",
}


def is_scalar_name(name: str) -> bool:
    """True for the printed name of a scalar (non-orbital) resummed model:
    P-MP3, SP-MP3, SSD-PMP3(...) and the same forms of the other
    resummations. Lets a writer that only has names mark the scalar block."""
    base = name.split("(")[0]
    return any(base in (f"{s}-MP3", f"S{s}-MP3", f"SSD-{s}MP3")
               for s in (r.stem for r in RESUMMATIONS if r.stem))


def params_of(key: ModelKey) -> tuple[str, ...]:
    """The fit parameters a model needs: the symbolic scales it contains."""
    return tuple(dict.fromkeys(
        c.scale for c in contributions(key.scaling, key.x_fixed)
        if isinstance(c.scale, str)))


def default_model_keys(
    alt:     bool = False,
    x_fixed: tuple[float, float] = X_FIXED_DEFAULT,
) -> list[ModelKey]:
    """The parameter-free models: the MP-series rows, then per resummation
    and orbital/scalar variant OP-MP3, SOP-MP3 and SSD-OPMP3 at `x_fixed`."""
    keys = [ModelKey(MP, s) for s in MP_DEFAULTS]
    for name in _resummed(alt):
        for orbital in (True, False):
            keys += [ModelKey(name, UNSCALED, orbital),
                     ModelKey(name, SPIN_SEPARATED, orbital, (1.0, 1.0)),
                     ModelKey(name, SPIN_SEPARATED, orbital, x_fixed)]
    return keys


def fit_model_keys(alt: bool = False) -> list[ModelKey]:
    """The models with fitted parameters."""
    return ([ModelKey(MP, s) for s in FITTED]
            + [ModelKey(name, s, orbital) for name in _resummed(alt)
               for orbital in (True, False) for s in FITTED])


# ---------------------------------------------------------------------------
# Precompute and evaluate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Model:
    """One model, decomposed over an ordered list of systems."""
    key:       ModelKey
    sys_names: tuple[str, ...]
    parts:     tuple[tuple[Contribution, rs.Spectra], ...]


# (system, decompose, W2 block, W3 block, orbital) -> Spectrum. Every
# resummation and scaling reuses the same few decompositions, so a memo
# shared across models avoids repeating them.
Memo = dict[tuple, rs.Spectrum]


def precompute(
    key:     ModelKey,
    systems: Mapping[str, Mapping[str, np.ndarray]],
    memo:    Memo | None = None,
) -> Model:
    """Decompose every system's blocks for `key` ({name: blocks}), memoised."""
    r     = resummation(key.resummation)
    memo  = {} if memo is None else memo
    parts = []
    for contrib in contributions(key.scaling, key.x_fixed):
        spectra = []
        for name, blocks in systems.items():
            mk = (name, r.decompose, contrib.w2, contrib.w3, key.orbital)
            if mk not in memo:
                w2, w3 = blocks[contrib.w2], blocks[contrib.w3]
                if not key.orbital:               # scalar: E2^2 / (E2 - x E3)
                    w2, w3 = np.array([[tr(w2)]]), np.array([[tr(w3)]])
                memo[mk] = r.decompose(w2, w3)
            spectra.append(memo[mk])
        parts.append((contrib, rs.stack(spectra)))
    return Model(key, tuple(systems), tuple(parts))


def evaluate(model: Model, params: Mapping[str, float]) -> np.ndarray:
    """Correlation energy per system, in model.sys_names order."""
    resum = resummation(model.key.resummation).resum
    total = np.zeros(len(model.sys_names))
    for contrib, spectra in model.parts:
        x = (float(params[contrib.scale]) if isinstance(contrib.scale, str)
             else contrib.scale)
        total += contrib.weight * resum(spectra, x)
    return total
