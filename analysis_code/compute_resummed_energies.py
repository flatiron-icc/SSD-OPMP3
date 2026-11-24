"""Evaluate every W-matrix energy model for one molecule.

    python compute_resummed_energies.py DIR --x-ss 0.45
    python compute_resummed_energies.py DIR --x 0.7890 --x-ss 0.3809 --x-os 1.0253 --json

DIR is the directory of one molecule's W-matrix files (or the prefix
<dir>/<name>).  E_c and E_total = E_HF + E_S^(2) + E_c of every parameter-free
model, plus the models scaled by --x and by (--x-ss, --x-os) when those are
given, are written in Hartree to <name>.analysis.out.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from blocks import tr
from config import X_FIXED_DEFAULT
from data import SystemData, load_system
from models import (
    OTHER_RESUMMATIONS,
    RESUMMATIONS,
    SCALED,
    SCALED_SS_OS,
    X,
    X_OS,
    X_SS,
    Memo,
    ModelKey,
    default_model_keys,
    evaluate,
    is_scalar_name,
    precompute,
    scaled_model_keys,
)

log = logging.getLogger(__name__)

_MARKER = ".W2alpha_ss.npy"      # the file whose stem names the system

PARAM_COLUMNS = (X, X_SS, X_OS)
SCALAR_TAG    = "[Scalar Models]"  # written before every block of scalar models
_MIN_NAME_W   = 40                 # report column widths
_VAL_W        = 22


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
    """The parameter-free models, plus the models scaled by --x (MP2.X,
    OP-MP3(x), P-MP3(x)) and by (--x-ss, --x-os) (MP2.X, SOP-MP3, SP-MP3 in
    x_ss, x_os) when those are given.  Ordered by resummation, the orbital
    models of each before its scalar ones."""
    keys   = default_model_keys(alt, (x_ss, 1.0))
    scaled = scaled_model_keys(alt)
    for scaling, value in ((SCALED, x), (SCALED_SS_OS, x_os)):
        if value is not None:
            keys += [k for k in scaled if k.scaling == scaling]
    order = [r.name for r in RESUMMATIONS]
    return sorted(keys, key=lambda k: (order.index(k.resummation), not k.orbital))


def evaluate_single(key: ModelKey, sd: SystemData, params: Mapping[str, float],
                    memo: Memo | None = None) -> tuple[float, float]:
    """(E_c, E_total) of one model."""
    ec = float(evaluate(precompute(key, {sd.name: sd.blocks}, memo), params)[0])
    return ec, sd.e_hf + sd.e_nbs + ec


def header_energies(sd: SystemData) -> dict[str, float]:
    """E_HF, E_S^(2) and the E2 / E3 traces, for the report header."""
    return {"E_HF": sd.e_hf, "E_S^(2)": sd.e_nbs,
            **{f"Tr({k})": tr(sd.blocks[k]) for k in
               ("W2", "W2_ss", "W2_os", "W3", "W3_ss", "W3_os")}}


def single_molecule_rows(prefix: Path, x_ss: float, x: float | None = None,
                         x_os: float | None = None, alt: bool = False) -> dict:
    """The prefix, parameters, header energies and (name, E_c, E_total)
    rows of one molecule."""
    sd = load_system(prefix)
    params = {X_SS: x_ss, **({X: x} if x is not None else {}),
              **({X_OS: x_os} if x_os is not None else {})}
    memo: Memo = {}
    rows = [(k.display_name(), *evaluate_single(k, sd, params, memo))
            for k in model_keys(x_ss, x, x_os, alt)]
    return {"prefix": str(prefix), "parameters": {X: x, X_SS: x_ss, X_OS: x_os},
            "energies": header_energies(sd), "models": rows}


def write_report(
    outfile:  Path,
    prefix:   str,
    params:   Mapping[str, float | None],
    energies: Mapping[str, float],
    models:   Sequence[tuple[str, float, float]],
) -> None:
    """Header, parameters, base energies, then (name, E_c, E_total) rows:
    each block of scalar models is preceded by SCALAR_TAG, and a blank line
    separates it from the models after it."""
    name_w = max(_MIN_NAME_W, max((len(n) for n, _, _ in models), default=0) + 2)
    banner = "=" * (name_w + 2 * _VAL_W)
    with open(outfile, "w", encoding="utf-8") as fh:
        fh.write(f"{banner}\n W-matrix energy model analysis\n"
                 f" Prefix:    {prefix}\n Energies in Hartree.\n{banner}\n\n")
        fh.write(" Parameters\n ----------\n")
        for name in PARAM_COLUMNS:
            v = params.get(name)
            fh.write(f"   {name:<10}{'(unset)' if v is None else f'{v:.6f}'}\n")
        fh.write("\n Base energies\n -------------\n")
        for label, value in energies.items():
            fh.write(f"   {label:<{name_w - 3}}{value:>{_VAL_W}.15f}\n")
        fh.write("\n Models (E_c and E_total, Hartree)\n"
                 " ---------------------------------\n"
                 f"   {'':<{name_w - 3}}{'E_c':>{_VAL_W}}{'E_total':>{_VAL_W}}\n")
        prev_scalar = False
        for name, ec, total in models:
            scalar = is_scalar_name(name)
            if scalar and not prev_scalar:
                fh.write(f"   {SCALAR_TAG}\n")
            elif prev_scalar and not scalar:
                fh.write("\n")
            prev_scalar = scalar
            fh.write(f"   {name:<{name_w - 3}}{ec:>{_VAL_W}.15f}"
                     f"{total:>{_VAL_W}.15f}\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("path", type=Path,
                   help="The directory of one molecule's W-matrix files (holding "
                        "exactly one *.W2alpha_ss.npy), or the file prefix "
                        "<dir>/<name>.")
    p.add_argument("--x-ss", type=float, default=X_FIXED_DEFAULT[0],
                   help=f"x_ss of SSD-OPMP3 and of the models scaled by (x_ss, x_os) "
                        f"(default {X_FIXED_DEFAULT[0]:g}; 0.45 for kOOMP2, 0.50 PBE0).")
    p.add_argument("--x", type=float,
                   help="Also print the models scaled by this x: MP2.X, OP-MP3(x), P-MP3(x).")
    p.add_argument("--x-os", type=float,
                   help="Also print the models scaled by (--x-ss, --x-os): "
                        "MP2.X, SOP-MP3 and SP-MP3 in x_ss, x_os.")
    p.add_argument("--alt-resummations", action="store_true",
                   help="Also print the resummations the paper set aside: "
                        + ", ".join(OTHER_RESUMMATIONS) + ".")
    p.add_argument("--json", action="store_true",
                   help="Also write the same content as JSON.")
    p.add_argument("-o", "--output", type=Path,
                   help="Report path (default <prefix>.analysis.out).")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Debug-level logging.")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="Warnings and errors only.")
    args = p.parse_args()
    level = (logging.WARNING if args.quiet else
             logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr,
                        force=True)

    prefix = resolve_prefix(args.path)
    result = single_molecule_rows(prefix, args.x_ss, args.x, args.x_os,
                                  args.alt_resummations)
    report = args.output or prefix.with_name(prefix.name + ".analysis.out")
    write_report(report, result["prefix"], result["parameters"],
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
