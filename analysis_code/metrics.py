"""Reaction energies, error statistics, and the result containers.

With per-system total energies ordered like the stoichiometry matrix S,
the signed reaction errors against the reference are one matvec,
errors = S @ totals - e_ref (Hartree); statistics are in kcal/mol.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from config import BOUND_ACTIVE_TOL, KCAL_PER_HA, MAX_EXCLUDED_NAMED, PARAM_BOUNDS
from models import ModelKey

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stoichiometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReactionMatrix:
    rxn_names: tuple[str, ...]
    sys_names: tuple[str, ...]
    S:         np.ndarray          # (n_rxn, n_sys) stoichiometric coefficients
    e_ref:     np.ndarray          # (n_rxn,) reference reaction energies, Ha

    def errors(self, totals: np.ndarray) -> np.ndarray:
        return self.S @ totals - self.e_ref


def build_reaction_matrix(rxns: dict, refs: Mapping[str, float],
                          sys_names: Sequence[str]) -> ReactionMatrix:
    """S and e_ref = S @ (per-structure reference energies)."""
    names = tuple(sys_names)
    index = {n: i for i, n in enumerate(names)}
    rxn_names = tuple(rxns)
    S     = np.zeros((len(rxn_names), len(names)))
    e_ref = np.zeros(len(rxn_names))
    for r, rxn_name in enumerate(rxn_names):
        rxn = rxns[rxn_name]
        for struct, count in zip(rxn["structs"], rxn["counts"]):
            S[r, index[struct]] += float(count)
            e_ref[r]            += float(count) * refs[struct]
    return ReactionMatrix(rxn_names, names, S, e_ref)


# ---------------------------------------------------------------------------
# Error statistics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Metrics:
    """MAE, RMSE, MSE (mean signed error) and the worst reaction, kcal/mol;
    non-finite errors (a Padé pole) are excluded and counted."""
    mae:           float
    rmse:          float
    mse:           float
    max_abs_err:   float
    max_err_rxn:   str
    max_err_ref:   float           # reference energy at the worst reaction
    max_err_pred:  float           # predicted energy there
    n_excluded:    int = 0
    excluded_rxns: tuple[str, ...] = ()

    @classmethod
    def from_errors(cls, errors_ha: np.ndarray, rmat: ReactionMatrix) -> Metrics:
        finite = np.isfinite(errors_ha)
        bad    = np.flatnonzero(~finite)
        excluded = tuple(rmat.rxn_names[i] for i in bad[:MAX_EXCLUDED_NAMED])
        if not finite.any():
            nan = float("nan")
            return cls(nan, nan, nan, nan, "", nan, nan, bad.size, excluded)
        errs  = errors_ha[finite] * KCAL_PER_HA
        i_max = int(np.argmax(np.abs(errs)))
        r_max = int(np.flatnonzero(finite)[i_max])
        ref   = rmat.e_ref[r_max] * KCAL_PER_HA
        return cls(
            mae=float(np.mean(np.abs(errs))),
            rmse=float(np.sqrt(np.mean(errs ** 2))),
            mse=float(np.mean(errs)),
            max_abs_err=float(abs(errs[i_max])),
            max_err_rxn=rmat.rxn_names[r_max],
            max_err_ref=ref,
            max_err_pred=ref + float(errs[i_max]),
            n_excluded=int(bad.size),
            excluded_rxns=excluded,
        )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """One row of a results table: its printed name, statistics and the
    signed errors (Ha). Models are named by ModelKey.display_name(); the
    CCSD baseline read from a file is named "CCSD"."""
    name:    str
    metrics: Metrics
    errors:  np.ndarray

    @classmethod
    def from_totals(cls, name: str, rmat: ReactionMatrix,
                    totals: np.ndarray) -> ModelResult:
        errors = rmat.errors(totals)
        return cls(name, Metrics.from_errors(errors, rmat), errors)


@dataclass
class FitResult:
    """Fitted parameters, in-sample statistics, and the parameters that
    ended on an edge of PARAM_BOUNDS (at_bound)."""
    key:       ModelKey
    params:    dict[str, float]
    metrics:   Metrics
    converged: bool = True
    at_bound:  tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetTables:
    """Everything one dataset produced, as the report writer consumes it."""
    dataset:     str
    rxns:        dict                          # {name: {structs, counts, ...}}
    rmat:        ReactionMatrix
    n_systems:   int
    default:     dict[ModelKey, ModelResult]
    baseline:    ModelResult | None = None     # CCSD, from --ccsd-file
    sources:     tuple[str, ...] = ()          # W-matrix trees, in order
    xc:          str = ""
    fit_eval:    dict[ModelKey, ModelResult] | None = None
    fits:        dict[ModelKey, FitResult] | None = None
    fit_dataset: str | None = None
    notes:       tuple[str, ...] = ()          # extra header lines

    @property
    def fitting_on(self) -> bool:
        return self.fit_eval is not None and self.fit_dataset is not None

    @property
    def default_rows(self) -> list[ModelResult]:
        """The CCSD baseline, if any, then the parameter-free models."""
        return ([self.baseline] if self.baseline else []) + list(self.default.values())


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

def active_bounds(params: Mapping[str, float]) -> tuple[str, ...]:
    """Parameters within BOUND_ACTIVE_TOL of an edge of PARAM_BOUNDS."""
    lo, hi = PARAM_BOUNDS
    return tuple(sorted(n for n, v in params.items()
                        if abs(v - lo) <= BOUND_ACTIVE_TOL
                        or abs(v - hi) <= BOUND_ACTIVE_TOL))


def log_bound_summary(fits: Mapping[ModelKey, FitResult]) -> None:
    bad   = [r for r in fits.values() if not r.converged]
    bound = [r for r in fits.values() if r.at_bound]
    if bad:
        log.warning("  %d / %d fit(s) did not converge", len(bad), len(fits))
    if not bound:
        log.info("  all %d fit(s) converged to interior optima", len(fits))
        return
    counts: dict[str, int] = {}
    for r in bound:
        for name in r.at_bound:
            counts[name] = counts.get(name, 0) + 1
    log.warning(
        "  %d / %d fit(s) rest on a PARAM_BOUNDS edge (%s): their "
        "unconstrained optima are unphysical.",
        len(bound), len(fits),
        ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
    )
