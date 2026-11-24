"""Evaluate every W-matrix energy model for one molecule.

    python compute_resummed_energies.py DIR --x-ss 0.45
    python compute_resummed_energies.py DIR --x 0.7890 --x-ss 0.3809 --x-os 1.0253 --json

DIR is the directory of one molecule's W-matrix files (or the prefix
<dir>/<name>).  E_c and E_total = E_HF + E_S^(2) + E_c of every parameter-free
model, plus the one-parameter models at --x and the two-parameter ones at
(--x-ss, --x-os) when given, are written in Hartree to <name>.analysis.out.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from blocks import tr
from cli import add_verbosity_args, configure_logging
from config import X_FIXED_DEFAULT
from data import SystemData, load_system
from evaluation import evaluate_single
from models import (
    OTHER_RESUMMATIONS,
    SCALED,
    SCALED_SS_OS,
    X,
    X_OS,
    X_SS,
    Memo,
    ModelKey,
    default_model_keys,
    fit_model_keys,
)
from output import write_single_job_report

log = logging.getLogger(__name__)

_MARKER = ".W2alpha_ss.npy"      # the file whose stem names the system


def resolve_prefix(path: Path) -> Path:
    """<dir>/<name> for a system directory; `path` itself if a prefix."""
    if not path.is_dir():
        return path
    hits = sorted(path.glob(f"*{_MARKER}"))
    if len(hits) != 1:
        raise SystemExit(f"error: {path} holds {len(hits)} '*{_MARKER}' "
                         "files; expected exactly one system")
    return path / hits[0].name[: -len(_MARKER)]


def model_keys(x_ss: float, x: float | None, x_os: float | None,
               alt: bool) -> list[ModelKey]:
    """The parameter-free models, plus the fitted ones whose parameters
    were given: the one-parameter models (MP2.X, OP-MP3(x), P-MP3(x)) for
    --x, the two-parameter ones (MP2.X, SOP-MP3, SP-MP3 in x_ss, x_os) for
    --x-os."""
    keys   = default_model_keys(alt, (x_ss, 1.0))
    fitted = fit_model_keys(alt)
    for scaling, value in ((SCALED, x), (SCALED_SS_OS, x_os)):
        if value is not None:
            keys += [k for k in fitted if k.scaling == scaling]
    return keys


def header_energies(sd: SystemData) -> dict[str, float]:
    """E_HF, E_S^(2) and the E2 / E3 traces, for the report header."""
    return {"E_HF": sd.e_hf, "E_NBS": sd.e_nbs,
            **{f"Tr({k})": tr(sd.blocks[k]) for k in
               ("W2", "W2_ss", "W2_os", "W3", "W3_ss", "W3_os")}}


def single_molecule_rows(prefix: Path, x_ss: float, x: float | None = None,
                         x_os: float | None = None, alt: bool = False) -> dict:
    sd = load_system(prefix)
    params = {X_SS: x_ss, **({X: x} if x is not None else {}),
              **({X_OS: x_os} if x_os is not None else {})}
    memo: Memo = {}
    rows = [(k.display_name(), *evaluate_single(k, sd, params, memo))
            for k in model_keys(x_ss, x, x_os, alt)]
    return {"prefix": str(prefix), "parameters": {X: x, X_SS: x_ss, X_OS: x_os},
            "energies": header_energies(sd), "models": rows}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("path", type=Path,
                   help="The directory of one molecule's W-matrix files (holding "
                        "exactly one *.W2alpha_ss.npy), or the file prefix "
                        "<dir>/<name>.")
    p.add_argument("--x-ss", type=float, default=X_FIXED_DEFAULT[0],
                   help=f"Same-spin damping of SSD-OPMP3 (default "
                        f"{X_FIXED_DEFAULT[0]:g}; 0.45 for kOOMP2, 0.50 PBE0).")
    p.add_argument("--x", type=float,
                   help="Also print the one-parameter fitted models (MP2.X, OP-MP3(x), P-MP3(x)) at this x.")
    p.add_argument("--x-os", type=float,
                   help="Also print the two-parameter fitted models (MP2.X, SOP-MP3, SP-MP3 in x_ss, x_os) at (--x-ss, --x-os).")
    p.add_argument("--alt-resummations", action="store_true",
                   help="Also print the resummations the paper set aside: "
                        + ", ".join(OTHER_RESUMMATIONS) + ".")
    p.add_argument("--json", action="store_true",
                   help="Also write the same content as JSON.")
    p.add_argument("-o", "--output", type=Path,
                   help="Report path (default <prefix>.analysis.out).")
    add_verbosity_args(p)
    args = p.parse_args()
    configure_logging(args)

    prefix = resolve_prefix(args.path)
    result = single_molecule_rows(prefix, args.x_ss, args.x, args.x_os,
                                  args.alt_resummations)
    report = args.output or prefix.with_name(prefix.name + ".analysis.out")
    write_single_job_report(report, result["prefix"], result["parameters"],
                            result["energies"], result["models"])
    log.info("Analysis report = %s", report)
    if args.json:
        json_file = report.with_suffix(".json")
        json_file.write_text(json.dumps({
            **{k: result[k] for k in ("prefix", "parameters", "energies")},
            "models": [{"name": n, "E_c": ec, "E_total": et}
                       for n, ec, et in result["models"]],
        }, indent=2), encoding="utf-8")
        log.info("Analysis JSON   = %s", json_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
