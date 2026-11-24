"""Input: W-matrix trees, reaction tables, reference energies.

A system is <base>/<xc>/<name>.<ext>/<name>.* (ext "xyz" or "molden") or,
in a flat tree, <base>/<xc>/<name>.*.  Read are the eight
<name>.W{2,3}{alpha,beta}_{ss,os}.npy and, from <name>.semicanonical.out
(or <name>.final.out), E_HF, the HF energy of the reference density, and
E_S^(2), the non-Brillouin singles energy.  A model's total energy is
E_HF + E_S^(2) + E_c.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from blocks import combine_blocks
from config import LOAD_WORKERS


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Where systems live
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockSource:
    """One directory to search and its layout.

    sys_ext "xyz" or "molden": one directory per system,
    <base>/<xc>/<name>.<ext>/<name>.*.  sys_ext None: a flat tree,
    <base>/<xc>/<name>.*.
    """
    base_dir: Path
    sys_ext:  str | None

    def prefix_for(self, xc: str, sys_name: str) -> Path:
        if self.sys_ext is None:
            return self.base_dir / xc / sys_name
        return self.base_dir / xc / f"{sys_name}.{self.sys_ext}" / sys_name

    def __str__(self) -> str:
        return (f"{self.base_dir} (flat)" if self.sys_ext is None
                else f"{self.base_dir} (*.{self.sys_ext})")


def sources(xyz_dirs: Sequence[Path] = (),
            molden_dirs: Sequence[Path] = (),
            flat_dirs: Sequence[Path] = ()) -> list[BlockSource]:
    """Trees in search order: molden, then xyz, then flat, each in the order
    given; the first tree holding a system supplies it."""
    return ([BlockSource(Path(d), "molden") for d in molden_dirs]
            + [BlockSource(Path(d), "xyz") for d in xyz_dirs]
            + [BlockSource(Path(d), None) for d in flat_dirs])


# ---------------------------------------------------------------------------
# One system
# ---------------------------------------------------------------------------

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

# Errors on which a system is skipped with a warning.
LOAD_FAILURES = (OSError, ValueError, EOFError)


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
        raise ValueError(f"Could not parse HF / NBS energies for {prefix}")
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


def _load_from_sources(name: str, srcs: Sequence[BlockSource], xc: str,
                       ) -> tuple[SystemData | None, BlockSource | None, str]:
    """First source that has the system; else a warning naming each attempt."""
    attempts = []
    for src in srcs:
        prefix = src.prefix_for(xc, name)
        try:
            return load_system(prefix), src, ""
        except LOAD_FAILURES as exc:
            attempts.append(f"{prefix}: {exc}")
    return None, None, (f"  skipping {name!r}: not found in any of "
                        f"{len(srcs)} source(s).\n    " + "\n    ".join(attempts))


def load_systems(
    srcs:      Sequence[BlockSource],
    xc:        str,
    sys_names: Sequence[str],
    workers:   int = LOAD_WORKERS,
) -> dict[str, SystemData]:
    """Load the systems in `sys_names` order (the column order of the
    stoichiometry matrix), skipping any not found."""
    if not srcs:
        raise ValueError("no W-matrix sources given")
    names = list(sys_names)

    def load(n: str):
        return _load_from_sources(n, srcs, xc)

    if workers > 1 and len(names) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(load, names))
    else:
        outcomes = [load(n) for n in names]

    out: dict[str, SystemData] = {}
    per_source: dict[BlockSource, int] = {}
    for name, (sd, src, warning) in zip(names, outcomes):
        if sd is None:
            log.warning("%s", warning)
            continue
        out[name] = sd
        per_source[src] = per_source.get(src, 0) + 1
    log.info("  loaded %d / %d systems", len(out), len(names))
    if len(srcs) > 1:
        for src in srcs:
            log.info("  -> %d system(s) from %s", per_source.get(src, 0), src)
    return out


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    """A top-level YAML mapping (reactions, or {structure: energy})."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def require_present(label: str, missing: Sequence[str]) -> None:
    """RuntimeError naming up to ten of `missing`, if any."""
    if missing:
        head = ", ".join(missing[:10]) + ("..." if len(missing) > 10 else "")
        raise RuntimeError(f"{label}: {len(missing)} ({head})")


def load_dataset(
    rxn_file:  Path,
    refs_file: Path,
    datasets:  Sequence[str],
    srcs:      Sequence[BlockSource],
    xc:        str,
) -> tuple[dict[str, dict], dict[str, float], dict[str, SystemData]]:
    """Reactions per dataset, CCSD(T) totals and loaded systems.  Every
    structure needs a total; reactions with an unloaded system are dropped."""
    all_rxns = load_yaml(rxn_file)
    rxns_by_ds = {ds: {n: r for n, r in all_rxns.items() if r["Dataset"] == ds}
                  for ds in datasets}
    require_present(f"No reactions in {rxn_file} for datasets",
                    [ds for ds, r in rxns_by_ds.items() if not r])
    for ds, r in rxns_by_ds.items():
        log.info("  %s: %d reactions", ds, len(r))

    refs = load_yaml(refs_file)
    sys_names = sorted({s for r in rxns_by_ds.values() for rxn in r.values()
                        for s in rxn["structs"]})
    require_present("References missing for systems",
                    [n for n in sys_names if n not in refs])

    log.info("Loading %d systems for xc=%s from: %s", len(sys_names), xc,
             "; ".join(map(str, srcs)))
    system_data = load_systems(srcs, xc, sys_names)
    for ds, r in rxns_by_ds.items():
        dropped = [n for n, rxn in r.items()
                   if any(s not in system_data for s in rxn["structs"])]
        for n in dropped:
            del r[n]
        if dropped:
            log.warning("  [%s] dropped %d reaction(s) with unloaded systems",
                        ds, len(dropped))
    require_present("All reactions dropped for dataset(s)",
                    [ds for ds, r in rxns_by_ds.items() if not r])
    return rxns_by_ds, refs, system_data
