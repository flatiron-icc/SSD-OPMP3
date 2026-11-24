"""Input: one molecule's W matrices and reference energies.

Read are the eight <name>.W{2,3}{alpha,beta}_{ss,os}.npy and, from
<name>.semicanonical.out (or <name>.final.out), E_HF, the HF energy of the
reference density, and E_S^(2), the non-Brillouin singles energy.  A
model's total energy is E_HF + E_S^(2) + E_c.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from blocks import combine_blocks


@dataclass(frozen=True)
class SystemData:
    """blocks: combine_blocks output. e_hf, e_nbs: see the module doc."""
    name:   str
    blocks: dict[str, np.ndarray]
    e_hf:   float
    e_nbs:  float


_W_LABELS = tuple(f"W{n}{spin}_{comp}" for n in (2, 3)
                  for spin in ("alpha", "beta") for comp in ("ss", "os"))
_FLOAT  = r"([+-]?[0-9]*\.?[0-9]+(?:[Ee][+-]?[0-9]+)?)"
# <name>.semicanonical.out
_RE_HF  = re.compile(r"HF energy from density\s*:\s*" + _FLOAT)
_RE_NBS = re.compile(r"Non-Brillouin singles E2\s*:\s*" + _FLOAT)
# <name>.final.out (the run summary; same values, other labels)
_RE_HF_SUMMARY  = re.compile(r"Reference energy \(HF on [^)]*density\)\s*=\s*" + _FLOAT)
_RE_NBS_SUMMARY = re.compile(r"Non-Brillouin singles correction\s*=\s*" + _FLOAT)


def reference_energies(prefix: Path) -> tuple[float, float]:
    """(E_HF, E_S^(2)) of one system, from <name>.semicanonical.out if that
    file exists, else from <name>.final.out."""
    report  = prefix.with_name(f"{prefix.name}.semicanonical.out")
    summary = prefix.with_name(f"{prefix.name}.final.out")
    if report.is_file():
        text, re_hf, re_nbs = report.read_text(), _RE_HF, _RE_NBS
    elif summary.is_file():
        text, re_hf, re_nbs = summary.read_text(), _RE_HF_SUMMARY, _RE_NBS_SUMMARY
    else:
        raise FileNotFoundError(f"Neither {report.name} nor {summary.name} found for {prefix}")
    hf, nbs = re_hf.search(text), re_nbs.search(text)
    if hf is None or nbs is None:
        raise ValueError(f"Could not parse E_HF and E_S^(2) for {prefix}")
    return float(hf.group(1)), float(nbs.group(1))


def load_system(prefix: str | Path) -> SystemData:
    """Read one system; FileNotFoundError / ValueError on bad inputs."""
    p     = Path(prefix)
    files = {lbl: p.with_name(f"{p.name}.{lbl}.npy") for lbl in _W_LABELS}
    missing = [str(f) for f in files.values() if not f.is_file()]
    if missing:
        raise FileNotFoundError("Missing W-matrix files:\n" + "\n".join(missing))
    e_hf, e_nbs = reference_energies(p)
    return SystemData(
        name=p.name,
        blocks=combine_blocks({lbl: np.load(f) for lbl, f in files.items()}),
        e_hf=e_hf,
        e_nbs=e_nbs,
    )
