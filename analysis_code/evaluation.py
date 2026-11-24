"""Evaluate and fit models over a dataset.

Every model's total energy is E_HF + E_S^(2) + E_c; the reaction errors
follow from the stoichiometry matrix. The fit minimises the RMSE of those
errors over the fit dataset, inside PARAM_BOUNDS.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from config import BRENT_XATOL, GRID_STEP, KCAL_PER_HA, NM_OPTIONS, PARAM_BOUNDS
from data import SystemData
from metrics import FitResult, ModelResult, ReactionMatrix, active_bounds
from models import Memo, Model, ModelKey, evaluate, fit_model_keys, params_of, precompute

log = logging.getLogger(__name__)


def _check_order(rmat: ReactionMatrix, system_data: Mapping[str, SystemData]):
    if rmat.sys_names != tuple(system_data):
        raise ValueError("system order of the reaction matrix and the loaded "
                         "systems differ")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def base_energies(system_data: Mapping[str, SystemData]) -> np.ndarray:
    """E_HF + E_S^(2) per system, in dict order."""
    return np.array([sd.e_hf + sd.e_nbs for sd in system_data.values()])


def precompute_models(
    keys:        Sequence[ModelKey],
    system_data: Mapping[str, SystemData],
) -> dict[ModelKey, Model]:
    """One Model per key over every system, sharing one decomposition memo."""
    blocks = {name: sd.blocks for name, sd in system_data.items()}
    memo: Memo = {}
    return {key: precompute(key, blocks, memo) for key in keys}


def evaluate_models(
    rmat:        ReactionMatrix,
    system_data: Mapping[str, SystemData],
    keys:        Sequence[ModelKey] | Mapping[ModelKey, Mapping[str, float]],
    models:      Mapping[ModelKey, Model] | None = None,
) -> dict[ModelKey, ModelResult]:
    """Each key at its parameters -> ModelResult.

    `keys` is either a {key: params} mapping or a plain sequence of
    parameter-free keys. `models` may be prebuilt.
    """
    _check_order(rmat, system_data)
    params_by_key = (dict(keys) if isinstance(keys, Mapping)
                     else {k: {} for k in keys})
    models = dict(models or {})
    models.update(precompute_models(
        [k for k in params_by_key if k not in models], system_data))
    base = base_energies(system_data)
    return {
        key: ModelResult.from_totals(key.display_name(), rmat,
                                     base + evaluate(models[key], p))
        for key, p in params_by_key.items()
    }


def evaluate_single(
    key:    ModelKey,
    sd:     SystemData,
    params: Mapping[str, float],
    memo:   Memo | None = None,
) -> tuple[float, float]:
    """(E_c, E_total) for one model on one system."""
    ec = float(evaluate(precompute(key, {sd.name: sd.blocks}, memo), params)[0])
    return ec, sd.e_hf + sd.e_nbs + ec


def ccsd_baseline(
    rmat:        ReactionMatrix,
    ccsd_totals: Mapping[str, float],
) -> ModelResult | None:
    """The CCSD row from per-structure CCSD totals; None if any is missing."""
    missing = [n for n in rmat.sys_names if n not in ccsd_totals]
    if missing:
        log.warning("  CCSD row skipped: %d / %d system(s) lack a total (%s%s)",
                    len(missing), len(rmat.sys_names),
                    ", ".join(missing[:5]), "..." if len(missing) > 5 else "")
        return None
    totals = np.array([float(ccsd_totals[n]) for n in rmat.sys_names])
    return ModelResult.from_totals("CCSD", rmat, totals)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _objective(model: Model, names: Sequence[str], base: np.ndarray,
               rmat: ReactionMatrix):
    """RMSE (kcal/mol) as a function of the parameter tuple; inf if any
    reaction energy is non-finite (a Padé pole, an overflow)."""
    def rmse(values: Sequence[float]) -> float:
        errs = rmat.errors(base + evaluate(model, dict(zip(names, values))))
        if not np.all(np.isfinite(errs)):
            return float("inf")
        return float(np.sqrt(np.mean((errs * KCAL_PER_HA) ** 2)))
    return rmse


def _minimise(rmse, dim: int, label: str) -> tuple[tuple[float, ...], bool]:
    """Constrained minimum within PARAM_BOUNDS: bounded Brent in 1-D,
    grid-seeded bounded Nelder-Mead in 2-D. Returns (params, converged)."""
    lo, hi = PARAM_BOUNDS
    if dim == 1:
        res = minimize_scalar(lambda v: rmse((v,)), bounds=PARAM_BOUNDS,
                              method="bounded", options={"xatol": BRENT_XATOL})
        best = (float(res.x),)
    elif dim == 2:
        grid = np.linspace(lo, hi, int(round((hi - lo) / GRID_STEP)) + 1)
        seed = min(((a, b) for a in grid for b in grid), key=rmse)
        res  = minimize(rmse, x0=np.array(seed), method="Nelder-Mead",
                        bounds=[PARAM_BOUNDS] * 2, options=NM_OPTIONS)
        best = tuple(float(min(max(v, lo), hi)) for v in res.x)
    else:
        raise ValueError(f"unsupported fit dimension {dim}")
    if not res.success:
        log.warning("  fit did not converge (%s): %s", label, res.message)
    return best, bool(res.success)


def fit_models(
    rmat:        ReactionMatrix,
    system_data: Mapping[str, SystemData],
    alt:         bool = False,
) -> tuple[dict[ModelKey, FitResult], dict[ModelKey, Model]]:
    """Fit every parametric model on `rmat`'s dataset.

    Returns the fits and the Models used, so the caller can reuse the
    decompositions when applying the fitted parameters to other datasets.
    """
    _check_order(rmat, system_data)
    keys   = fit_model_keys(alt)
    models = precompute_models(keys, system_data)
    base   = base_energies(system_data)
    fits: dict[ModelKey, FitResult] = {}
    for key in keys:
        log.info("  fitting %s", key.display_name())
        names = params_of(key)
        rmse  = _objective(models[key], names, base, rmat)
        best, converged = _minimise(rmse, len(names), key.display_name())
        params = dict(zip(names, best))
        result = ModelResult.from_totals(
            key.display_name(), rmat, base + evaluate(models[key], params))
        fits[key] = FitResult(key, params, result.metrics, converged,
                              active_bounds(params))
    return fits, models
