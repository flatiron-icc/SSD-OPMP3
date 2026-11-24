"""CSV and text-report writers."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from config import KCAL_PER_HA
from metrics import DatasetTables, FitResult, ModelResult, ReactionMatrix
from models import MP, X, X_OS, X_SS, ModelKey, is_scalar_name

PARAM_COLUMNS: tuple[str, ...] = (X, X_SS, X_OS)

# Written on the line before every block of scalar (non-orbital) models.
SCALAR_TAG = "[Scalar Models]"

# Report column widths.
_VAL_W, _ABS_W, _PAR_W = 10, 12, 9
_MIN_NAME_W, _BANNER_W = 40, 110


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def write_dataset_errors_csv(
    outfile: Path,
    rows:    Sequence[ModelResult],
    rmat:    ReactionMatrix,
) -> None:
    """reaction, reference (kcal/mol), one signed-error column per row."""
    with open(outfile, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["reaction", "reference (kcal/mol)"]
                   + [f"{r.name} error (kcal/mol)" for r in rows])
        for i, rxn in enumerate(rmat.rxn_names):
            w.writerow([rxn, f"{rmat.e_ref[i] * KCAL_PER_HA:.6f}"]
                       + [f"{r.errors[i] * KCAL_PER_HA:.6f}" for r in rows])


def write_fits_csv(outfile: Path, fits: Mapping[ModelKey, FitResult]) -> None:
    """One row per fitted model: parameters and in-sample statistics."""
    with open(outfile, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "resummation", "orbital", "scaling",
                    *PARAM_COLUMNS,
                    "fit_MAE_kcal_per_mol", "fit_RMSE_kcal_per_mol",
                    "fit_MSE_kcal_per_mol", "fit_max_abs_err_kcal_per_mol",
                    "max_err_reaction", "max_err_ref_kcal_per_mol",
                    "max_err_pred_kcal_per_mol", "converged", "at_bound"])
        for key, r in fits.items():
            m = r.metrics
            w.writerow([
                key.display_name(), key.resummation,
                "yes" if key.orbital else "no", key.scaling,
                *(f"{r.params[p]:.6f}" if p in r.params else ""
                  for p in PARAM_COLUMNS),
                f"{m.mae:.6f}", f"{m.rmse:.6f}", f"{m.mse:.6f}",
                f"{m.max_abs_err:.6f}", m.max_err_rxn,
                f"{m.max_err_ref:.6f}", f"{m.max_err_pred:.6f}",
                "yes" if r.converged else "NO", " ".join(r.at_bound),
            ])


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def _num(value: float, width: int = _VAL_W) -> str:
    return f"{value:>{width}.4f}" if np.isfinite(value) else f"{'nan':>{width}}"


def name_width(names: Sequence[str]) -> int:
    """Label column wide enough for every row name."""
    return max(_MIN_NAME_W, max(map(len, names), default=0) + 2)


def _row(res: ModelResult, name_w: int, fit: FitResult | None,
         with_params: bool) -> str:
    m = res.metrics
    row = f"   {res.name:<{name_w - 3}}"
    if with_params:
        params = fit.params if fit else {}
        row += "".join(f"{params[p]:>{_PAR_W}.4f}" if p in params
                       else f"{'—':>{_PAR_W}}" for p in PARAM_COLUMNS)
    row += (f"{_num(m.mae)}{_num(m.rmse)}{_num(m.mse)}"
            f"{_num(m.max_abs_err)}{_num(m.max_err_ref, _ABS_W)}"
            f"{_num(m.max_err_pred, _ABS_W)}  {m.max_err_rxn}")
    if fit and fit.at_bound:
        row += f"   [bound: {' '.join(fit.at_bound)}]"
    if m.n_excluded:
        more = ", ..." if m.n_excluded > len(m.excluded_rxns) else ""
        row += (f"   [!! {m.n_excluded} non-finite: "
                f"{', '.join(m.excluded_rxns)}{more}]")
    return row + "\n"


def _section(fh, title: str, subtitle: str,
             baseline: ModelResult | None,
             table: Mapping[ModelKey, ModelResult], name_w: int,
             fits: Mapping[ModelKey, FitResult] | None) -> None:
    """One table: the baseline row first, then the models, a blank line
    between groups of (resummation, orbital/scalar). A block of scalar
    models (P-MP3, SP-MP3, ...) is preceded by the SCALAR_TAG line."""
    fh.write(f"{title}\n {'-' * (len(title) - 1)}\n")
    if subtitle:
        fh.write(subtitle + "\n")
    params_hdr = ("".join(f"{p:>{_PAR_W}}" for p in PARAM_COLUMNS)
                  if fits is not None else "")
    fh.write(f"   {'Model':<{name_w - 3}}{params_hdr}"
             f"{'MAE':>{_VAL_W}}{'RMSE':>{_VAL_W}}{'MSE':>{_VAL_W}}"
             f"{'Max|Err|':>{_VAL_W}}{'Ref':>{_ABS_W}}{'Pred':>{_ABS_W}}"
             f"  Max-error reaction\n")
    if baseline is not None:
        fh.write(_row(baseline, name_w, None, fits is not None) + "\n")
    prev = None
    for key, res in table.items():
        group = (key.resummation, key.orbital)
        if prev is not None and group != prev:
            fh.write("\n")
        if group != prev and not key.orbital and key.resummation != MP:
            fh.write(f"   {SCALAR_TAG}\n")
        prev = group
        fh.write(_row(res, name_w, fits.get(key) if fits else None,
                      fits is not None))
    fh.write("\n")


def write_text_report(outfile: Path, t: DatasetTables) -> None:
    rows = t.default_rows + list((t.fit_eval or {}).values())
    name_w = name_width([r.name for r in rows])
    with open(outfile, "w", encoding="utf-8") as fh:
        fh.write("=" * _BANNER_W + "\n W-matrix energy model benchmark\n")
        fh.write(f" Dataset:          {t.dataset}\n")
        for i, src in enumerate(t.sources):
            fh.write(f"{' W-matrix sources:' if i == 0 else '':<19}{src}\n")
        if t.xc:
            fh.write(f" Reference (xc):   {t.xc}\n")
        fh.write(f" # systems loaded: {t.n_systems}\n"
                 f" # reactions:      {len(t.rmat.rxn_names)}\n"
                 " Errors are predicted minus CCSD(T) reference, in kcal/mol.\n"
                 " MAE = mean absolute error, RMSE = root-mean-square error,"
                 " MSE = mean signed error,\n"
                 " Max|Err| = largest absolute error, with the corresponding"
                 " reaction; Ref / Pred are\n"
                 " the reference / predicted energy there. A [!! n non-finite]"
                 " tag means those\n"
                 " reactions were excluded from that row's statistics.\n")
        for line in t.notes:
            fh.write(f" {line}\n")
        if t.fitting_on:
            fh.write(f" Fit parameters were fitted on {t.fit_dataset}. A "
                     "[bound: ...] tag marks parameters resting\n on a "
                     "PARAM_BOUNDS edge, whose unconstrained optimum is"
                     " unphysical.\n")
        fh.write("=" * _BANNER_W + "\n\n")

        _section(fh, " Default-parameter results (kcal/mol)", "", t.baseline,
                 t.default, name_w, None)
        if t.fitting_on:
            _section(fh, " Fitted-parameter results (kcal/mol)",
                     f" (parameters fitted on {t.fit_dataset})", None,
                     t.fit_eval, name_w, t.fits)

        worst = list(dict.fromkeys(
            r.metrics.max_err_rxn for r in rows if r.metrics.max_err_rxn))
        if worst:
            fh.write(" Worst-case reactions (structures and counts)\n"
                     " --------------------------------------------\n")
            for name in worst:
                rxn = t.rxns[name]
                parts = ", ".join(
                    f"{'+' if c >= 0 else '-'}{abs(c)} * {s}"
                    for s, c in zip(rxn["structs"], rxn["counts"]))
                fh.write(f"   {name}\n     {parts}\n")
            fh.write("\n")


# ---------------------------------------------------------------------------
# Single-molecule report
# ---------------------------------------------------------------------------

def write_single_job_report(
    outfile:  Path,
    prefix:   str,
    params:   Mapping[str, float | None],
    energies: Mapping[str, float],
    models:   Sequence[tuple[str, float, float]],
) -> None:
    """Header, parameters, base energies, then (name, E_c, E_total) rows."""
    name_w = name_width([n for n, _, _ in models])
    val_w  = 22
    banner = "=" * (name_w + 2 * val_w)
    with open(outfile, "w", encoding="utf-8") as fh:
        fh.write(f"{banner}\n W-matrix energy model analysis\n"
                 f" Prefix:    {prefix}\n Energies in Hartree.\n{banner}\n\n")
        fh.write(" Parameters\n ----------\n")
        for name in PARAM_COLUMNS:
            v = params.get(name)
            fh.write(f"   {name:<10}{'(unset)' if v is None else f'{v:.6f}'}\n")
        fh.write("\n Base energies\n -------------\n")
        for label, value in energies.items():
            fh.write(f"   {label:<{name_w - 3}}{value:>{val_w}.15f}\n")
        fh.write("\n Models (E_c and E_total, Hartree)\n"
                 " ---------------------------------\n"
                 f"   {'':<{name_w - 3}}{'E_c':>{val_w}}{'E_total':>{val_w}}\n")
        prev_scalar = False
        for name, ec, total in models:
            scalar = is_scalar_name(name)
            if scalar and not prev_scalar:
                fh.write(f"   {SCALAR_TAG}\n")
            prev_scalar = scalar
            fh.write(f"   {name:<{name_w - 3}}{ec:>{val_w}.15f}"
                     f"{total:>{val_w}.15f}\n")
