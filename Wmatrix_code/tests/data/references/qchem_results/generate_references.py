#!/usr/bin/env python3
"""Generate reference JSON files from Q-Chem output files.

Input files must follow the naming convention:

    <mol>_<functional>_<basis>_<tag>.out

e.g.  h2o_pbe0_cc-pvdz_base.out
      cn_pbe0_cc-pvdz_df-restricted.out

Usage
-----
    python generate_references.py *.out
    python generate_references.py --outdir tests/data/references h2o*.out
    python generate_references.py --overwrite cn_pbe0_cc-pvdz_base.out

Outputs are written as <outdir>/<mol>_<functional>_<basis>_<tag>.json;
the default outdir is the parent of this script's directory
(tests/data/references/).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Regex patterns
#
# Numbers in Q-Chem output use either "au" or "a.u." as the unit label
# and may have leading whitespace before the sign.  All patterns capture
# only the numeric part; the unit is left unconsumed.
# ---------------------------------------------------------------------------

# SCF energy — appears in both the MP2 summary block and the MP3 block;
# the last occurrence is used.
_RE_SCF = re.compile(
    r'SCF\s+energy\s*=\s*([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)'
)

# Spin-component doubles
_RE_AAAA = re.compile(
    r'\baaaa\b\s+correlation energy\s*=\s*([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)'
)
_RE_ABAB = re.compile(
    r'\babab\b\s+correlation energy\s*=\s*([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)'
)
_RE_BBBB = re.compile(
    r'\bbbbb\b\s+correlation energy\s*=\s*([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)'
)

# Non-Brillouin singles correction
_RE_NBS = re.compile(
    r'non-Brillouin singles\s*=\s*([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)'
)

# MP3 correction
_RE_MP3 = re.compile(
    r'MP3 correlation energy\s*=\s*([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)'
)

# Total MP2 correlation energy (includes NBS) — used for internal
# validation only.
_RE_TOTAL_MP2_CORR = re.compile(
    r'Total\s+MP2\s+correlation energy\s*=\s*([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)'
)

_CHECK_TOL = 1e-7   # tolerance for internal consistency validation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _floats(pattern: re.Pattern, text: str) -> list[float]:
    return [float(m) for m in pattern.findall(text)]


def _last(vals: list[float], label: str, path: Path) -> float:
    if not vals:
        raise ValueError(f"'{label}' not found in {path.name}")
    return vals[-1]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_output(text: str, path: Path) -> dict:
    """Parse one Q-Chem output file and return the reference data dict.

    Raises ValueError with a descriptive message if any required quantity
    cannot be found or if the internal consistency check fails.
    """
    e_hf  = _last(_floats(_RE_SCF,  text), "SCF energy",           path)
    aaaa  = _last(_floats(_RE_AAAA, text), "aaaa correlation energy", path)
    abab  = _last(_floats(_RE_ABAB, text), "abab correlation energy", path)
    bbbb  = _last(_floats(_RE_BBBB, text), "bbbb correlation energy", path)
    e_nbs = _last(_floats(_RE_NBS,  text), "non-Brillouin singles", path)

    tr_w2_ss = aaaa + bbbb
    tr_w2_os = abab
    tr_w2    = tr_w2_ss + tr_w2_os

    tr_w3 = _last(_floats(_RE_MP3, text), "MP3 correlation energy", path)
    total_corr_vals = _floats(_RE_TOTAL_MP2_CORR, text)

    # Internal consistency check: tr_w2 + e_nbs must equal Q-Chem's printed
    # total correlation energy (which includes NBS).
    if total_corr_vals:
        total_corr = total_corr_vals[-1]
        computed   = tr_w2 + e_nbs
        if abs(computed - total_corr) > _CHECK_TOL:
            raise ValueError(
                f"{path.name}: consistency failure — "
                f"tr_w2 + e_nbs = {computed:.10f}  but  "
                f"Q-Chem total correlation = {total_corr:.10f}  "
                f"(diff = {abs(computed - total_corr):.2e})"
            )
    else:
        print(f"  WARN  {path.name}: total correlation energy not found; "
              "skipping consistency check", file=sys.stderr)

    return {
        "e_hf":     e_hf,
        "e_nbs":    e_nbs,
        "tr_w2_ss": tr_w2_ss,
        "tr_w2_os": tr_w2_os,
        "tr_w2":    tr_w2,
        "tr_w3":    tr_w3,
    }


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_filename(path: Path) -> tuple[str, str, str, str]:
    """(mol, functional, basis, tag) of e.g. h2o_pbe0_cc-pvdz_field-z-restricted.out."""
    stem  = path.stem
    parts = stem.split("_", 3)
    if len(parts) != 4:
        raise ValueError(
            f"Cannot parse '{path.name}'. "
            "Expected: <mol>_<functional>_<basis>_<tag>.out"
        )
    return tuple(parts)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def process(path: Path, outdir: Path, overwrite: bool) -> None:
    mol, functional, basis, tag = parse_filename(path)
    dest = outdir / f"{mol}_{functional}_{basis}_{tag}.json"

    if dest.is_file() and not overwrite:
        print(f"  SKIP  {dest.name}  (use --overwrite to replace)")
        return

    text   = path.read_text(encoding="utf-8", errors="replace")
    record = parse_output(text, path)

    # Include metadata for human readability; not required by the test suite.
    record.update({"mol": mol, "functional": functional,
                   "basis": basis, "tag": tag})

    dest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(f"  WROTE {dest.name}")
    print(f"    e_hf     = {record['e_hf']:.12f}")
    print(f"    e_nbs    = {record['e_nbs']:.12f}")
    print(f"    tr_w2_ss = {record['tr_w2_ss']:.12f}")
    print(f"    tr_w2_os = {record['tr_w2_os']:.12f}")
    print(f"    tr_w2    = {record['tr_w2']:.12f}")
    print(f"    tr_w3    = {record['tr_w3']:.12f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate reference JSONs from Q-Chem output files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "outputs", nargs="+",
        help="Q-Chem output files named <mol>_<functional>_<basis>_<tag>.out",
    )
    parser.add_argument(
        "--outdir", default=None,
        help=(
            "Directory to write JSON files "
            "(default: tests/data/references/ relative to this script)"
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing JSON files",
    )
    args = parser.parse_args()

    if args.outdir is None:
        # Script lives at tests/data/references/qchem_results/; JSONs go one level up.
        outdir = Path(__file__).resolve().parent.parent
    else:
        outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    for fname in args.outputs:
        path = Path(fname)
        print(f"Processing {path.name} ...")
        if not path.is_file():
            print(f"  ERROR  file not found: {path}", file=sys.stderr)
            errors.append(fname)
            continue
        try:
            process(path, outdir, args.overwrite)
        except ValueError as exc:
            print(f"  ERROR  {exc}", file=sys.stderr)
            errors.append(fname)

    if errors:
        print(f"\n{len(errors)} file(s) failed:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
