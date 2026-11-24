"""Benchmark the W-matrix energy models on reaction datasets.

For one reference determinant (--xc): load every system's W matrices,
evaluate each model's total energy E_HF + E_S^(2) + E_c, form reaction
energies through the stoichiometry, and score them against the CCSD(T)
reference in kcal/mol. The parametric models are fitted on --fit-dataset
and applied to every dataset (in-sample there, out-of-sample elsewhere).
With --ccsd-file, a CCSD row from those totals heads every table.

Per dataset: <dataset>.report.txt, <dataset>.default.csv and, with
fitting, <dataset>.fit_eval.csv; once per run: fits.csv.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cli import (
    add_source_args,
    add_verbosity_args,
    configure_logging,
    resolve_sources,
)
from config import DEFAULT_FIT_DATASET, X_FIXED_DEFAULT
from data import load_dataset, load_yaml
from evaluation import (
    ccsd_baseline,
    evaluate_models,
    fit_models,
    precompute_models,
)
from metrics import DatasetTables, build_reaction_matrix, log_bound_summary
from models import OTHER_RESUMMATIONS, default_model_keys
from output import write_dataset_errors_csv, write_fits_csv, write_text_report

log = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    add_source_args(p)
    p.add_argument("--rxn-file", required=True, type=Path,
                   help="Reaction table: {name: {Dataset, structs, counts}}.")
    p.add_argument("--refs-file", required=True, type=Path,
                   help="{structure: CCSD(T) total energy (Ha)}.")
    p.add_argument("--ccsd-file", type=Path,
                   help="{structure: CCSD total energy}; adds a CCSD row.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where the per-dataset reports and CSVs and fits.csv "
                        "are written; created if absent.")
    p.add_argument("--datasets", nargs="+", default=[DEFAULT_FIT_DATASET],
                   metavar="NAME",
                   help="Datasets to evaluate, as spelled in the reaction "
                        f"table (default: {DEFAULT_FIT_DATASET}).")
    p.add_argument("--fit-dataset", default=DEFAULT_FIT_DATASET, metavar="NAME",
                   help="Dataset the parametric models are fitted on; added "
                        f"to the run if absent (default: {DEFAULT_FIT_DATASET}).")
    p.add_argument("--no-fit", action="store_true",
                   help="Skip the fits and the fitted-parameter tables.")
    p.add_argument("--alt-resummations", action="store_true",
                   help="Also evaluate and fit the resummations the paper "
                        "set aside: " + ", ".join(OTHER_RESUMMATIONS) + ".")
    p.add_argument("--x-ss", type=float, default=X_FIXED_DEFAULT[0],
                   help="Same-spin damping x_ss of SSD-OPMP3 (default "
                        f"{X_FIXED_DEFAULT[0]:g}). The paper adopts one value "
                        "per reference determinant, e.g. 0.45 for kOOMP2, "
                        "0.50 for PBE0.")
    p.add_argument("--x-os", type=float, default=X_FIXED_DEFAULT[1],
                   help=f"x_os of SSD-OPMP3 (default {X_FIXED_DEFAULT[1]:g}).")
    add_verbosity_args(p)
    return p


def main() -> int:
    args = _parser().parse_args()
    configure_logging(args)
    srcs = resolve_sources(args)

    do_fit   = not args.no_fit
    datasets = list(args.datasets)
    if do_fit and args.fit_dataset not in datasets:
        datasets.insert(0, args.fit_dataset)

    rxns_by_ds, refs, system_data = load_dataset(
        args.rxn_file, args.refs_file, datasets, srcs, args.xc)
    rmats = {ds: build_reaction_matrix(rxns_by_ds[ds], refs, tuple(system_data))
             for ds in datasets}
    ccsd_totals = load_yaml(args.ccsd_file) if args.ccsd_file else None

    fits, fit_models_ = {}, {}
    if do_fit:
        log.info("Fitting on %s ...", args.fit_dataset)
        fits, fit_models_ = fit_models(rmats[args.fit_dataset], system_data,
                                       args.alt_resummations)
        log_bound_summary(fits)

    default_keys   = default_model_keys(args.alt_resummations,
                                        (args.x_ss, args.x_os))
    default_models = precompute_models(default_keys, system_data)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        rmat = rmats[ds]
        log.info("Evaluating %s (%d reactions) ...", ds, len(rmat.rxn_names))
        tables = DatasetTables(
            dataset=ds, rxns=rxns_by_ds[ds], rmat=rmat,
            n_systems=len(system_data),
            default=evaluate_models(rmat, system_data, default_keys,
                                    default_models),
            baseline=(ccsd_baseline(rmat, ccsd_totals)
                      if ccsd_totals is not None else None),
            sources=tuple(map(str, srcs)), xc=args.xc,
            fit_eval=(evaluate_models(rmat, system_data,
                                      {k: f.params for k, f in fits.items()},
                                      fit_models_) if fits else None),
            fits=fits or None,
            fit_dataset=args.fit_dataset if do_fit else None,
        )
        stem = ds.replace("/", "_")
        write_text_report(args.output_dir / f"{stem}.report.txt", tables)
        write_dataset_errors_csv(args.output_dir / f"{stem}.default.csv",
                                 tables.default_rows, rmat)
        if tables.fit_eval:
            write_dataset_errors_csv(args.output_dir / f"{stem}.fit_eval.csv",
                                     list(tables.fit_eval.values()), rmat)
        log.info("  -> %s/%s.{report.txt,default.csv%s}", args.output_dir,
                 stem, ",fit_eval.csv" if tables.fit_eval else "")

    if fits:
        write_fits_csv(args.output_dir / "fits.csv", fits)
        log.info("Fits summary -> %s", args.output_dir / "fits.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
