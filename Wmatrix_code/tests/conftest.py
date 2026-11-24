"""Shared fixtures of the W-matrix test suite.

A Variant is one workflow run (molecule, functional, restricted, closed
shell, density fitting, field, frozen core, basis); ALL_VARIANTS lists
them and the parameter lists below are filtered from it.

The ``run`` fixture does one Q-Chem SCF per (molecule, functional,
restricted, basis), which is that base variant's run, and restarts every
other variant of it from that Molden with ``--source molden``.  Without Q-Chem on PATH the SCF is done by PySCF
instead, and the ``qchem`` tests are skipped.

A closed-shell variant must reproduce its unrestricted twin (the same
Variant with closed_shell=False); this checks the closed-shell code,
including the closed-shell W assembly in pyscf_w_matrices.py.  It is compared
with the twin's Q-Chem reference file.

Restricted open shell with frozen core is not a variant: Q-Chem fails on
it, so there is no reference.

Reference files: tests/data/references/<mol>_<functional>_<basis>_<tag>.json
(basis cc-pvdz unless the Variant sets another).
"""

from __future__ import annotations

import json
import shutil
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from electric_field import parse_electric_field
from workflow import run_one_workflow
from workflow_common import (
    W2_LABELS, W3_LABELS, WorkflowConfig, ResultFileKey, derive_output_paths,
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
REFS_DIR = DATA_DIR / "references"
RUNS_DIR = Path(__file__).parent / "_runs"
BASIS    = "cc-pvdz"
AUXBASIS = "aug-cc-pvqz-ri"
THREADS  = 8
MEMORY   = 16000
FIELD_Z  = [0.0, 0.0, 0.001]
ETOL     = 1e-5

FUNCTIONALS = ("pbe0", "hf")

MOLECULES = {
    "h":   {"mult": 2},
    "he":  {"mult": 1},
    "he-triplet": {"mult": 3},
    "cn":  {"mult": 2},
    "h2o": {"mult": 1},
    "ch2": {"mult": 3},
    "ch3": {"mult": 2},
    "hco": {"mult": 2},
}

_NO_CORR = {"h"}

_QCHEM_AVAILABLE = shutil.which("qchem") is not None

# ---------------------------------------------------------------------------
# Variant
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Variant:
    mol:         str
    functional:  str  = "pbe0"
    restricted:   bool = False
    df:           bool = False
    field_z:      bool = False
    closed_shell: bool = False
    frozen_core:  str  = "0"
    basis:        str  = BASIS

    def __post_init__(self):
        # Restricted + frozen core is not a variant (module docstring).
        if self.restricted and self.frozen_core != "0":
            raise ValueError(
                f"Variant {self!r}: restricted=True with frozen_core={self.frozen_core!r} "
                "is not part of the registry."
            )
        # --closed-shell needs a singlet and overrides --restricted.
        if self.closed_shell and MOLECULES[self.mol]["mult"] != 1:
            raise ValueError(
                f"Variant {self!r}: closed_shell=True requires a multiplicity-1 "
                f"molecule, but {self.mol} has multiplicity "
                f"{MOLECULES[self.mol]['mult']}."
            )
        if self.closed_shell and self.restricted:
            raise ValueError(
                f"Variant {self!r}: closed_shell=True already implies a restricted "
                "reference; do not combine it with restricted=True."
            )

    @property
    def id(self) -> str:
        parts = [self.mol, self.functional, "RO" if self.restricted else "U"]
        if self.closed_shell:       parts.append("CS")
        if self.df:                 parts.append("df")
        if self.field_z:            parts.append("field")
        if self.frozen_core != "0": parts.append(f"fc{self.frozen_core}")
        if self.basis != BASIS:     parts.append(self.basis)
        return "-".join(parts)

    @property
    def ref_tag(self) -> str:
        # base, restricted, df, df-restricted, field-z, field-z-restricted,
        # fc-base; closed_shell does not appear (compared with its twin's file).
        parts = []
        if self.df:      parts.append("df")
        if self.field_z: parts.append("field-z")
        if self.frozen_core != "0":
            parts.append("fc" if self.frozen_core == "fc" else f"fc{self.frozen_core}")
        if self.restricted:
            parts.append("restricted")
        elif not parts or self.frozen_core != "0":
            parts.append("base")
        return "-".join(parts)

    @property
    def has_corr(self) -> bool:
        return self.mol not in _NO_CORR

    @property
    def energy_keys(self) -> dict[str, str]:
        return {
            "e_hf":     "Reference energy (HF on KS density)",
            "e_nbs":    "Non-Brillouin singles correction",
            "tr_w2_ss": "MP2 doubles correlation (same-spin)",
            "tr_w2_os": "MP2 doubles correlation (opposite-spin)",
            "tr_w2":    "MP2 doubles correlation",
            "tr_w3":    "MP3 correlation",
        }

    @property
    def semicanonical_suffixes(self) -> tuple[str, ...]:
        """Suffixes of the saved semicanonical arrays: one set if closed shell."""
        return ("",) if self.closed_shell else ("_a", "_b")

    @property
    def occupation_values(self) -> tuple[float, ...]:
        """Allowed entries of the saved occupation vectors."""
        return (0.0, 2.0) if self.closed_shell else (0.0, 1.0)

    def occupied_counts(self, paths) -> tuple[int, int]:
        """(nocc_alpha, nocc_beta) from the saved occupation vectors."""
        if self.closed_shell:
            n = int(np.load(paths.semicanonical_array("occ")).sum()) // 2
            return n, n
        return (
            int(np.load(paths.semicanonical_array("occ_a")).sum()),
            int(np.load(paths.semicanonical_array("occ_b")).sum()),
        )

    @property
    def _base_key(self) -> tuple[str, str, bool, str]:
        """Key for the shared DFT cache (mol, functional, restricted, basis)."""
        return (self.mol, self.functional, self.restricted, self.basis)

    def to_config_kwargs(self) -> dict:
        kw: dict = {
            "functional":  self.functional,
            "restricted":  self.restricted,
            "frozen_core": self.frozen_core,
        }
        if self.df:           kw["df"] = True
        if self.field_z:      kw["electric_field"] = FIELD_Z
        if self.closed_shell: kw["closed_shell"] = True
        if self.basis != BASIS: kw["basis"] = self.basis
        return kw

    def unrestricted_twin(self) -> "Variant":
        """This variant with closed_shell switched off."""
        return replace(self, closed_shell=False)

    def bare(self) -> "Variant":
        """Return the minimal base variant sharing the same DFT cache key."""
        return Variant(self.mol, self.functional, self.restricted, basis=self.basis)


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------
def _make_variants(functional: str) -> list[Variant]:
    f = functional
    variants = [
        Variant("h",         f),
        Variant("he",        f),
        # He triplet (1s2s 3S): two alpha electrons, no beta.  cc-pVTZ: in
        # cc-pVDZ both s functions are occupied and MP2/MP3 vanish.
        Variant("he-triplet", f, basis="cc-pvtz"),
        Variant("he-triplet", f, restricted=True, basis="cc-pvtz"),
        # CN doublet
        Variant("cn",        f),
        Variant("cn",        f, restricted=True),
        Variant("h2o",       f),
        Variant("h2o",       f, frozen_core="fc"),
        Variant("cn",        f, frozen_core="fc"),
        Variant("h2o",       f, df=True),
        Variant("cn",        f, df=True),
        Variant("cn",        f, restricted=True, df=True),
        Variant("h2o",       f, field_z=True),
        Variant("cn",        f, field_z=True),
        Variant("cn",        f, restricted=True, field_z=True),
        # CH2 triplet (X 3B1)
        Variant("ch2",       f),
        Variant("ch2",       f, restricted=True),
        Variant("ch2",       f, df=True),
        Variant("ch2",       f, field_z=True),
        Variant("ch2",       f, frozen_core="fc"),
        # CH3 doublet
        Variant("ch3",       f),
        Variant("ch3",       f, restricted=True),
        Variant("ch3",       f, df=True),
        Variant("ch3",       f, restricted=True, df=True),
        Variant("ch3",       f, field_z=True),
        Variant("ch3",       f, restricted=True, field_z=True),
        Variant("ch3",       f, frozen_core="fc"),
        # HCO doublet
        Variant("hco",       f),
        Variant("hco",       f, restricted=True),
        Variant("hco",       f, df=True),
        Variant("hco",       f, restricted=True, df=True),
        Variant("hco",       f, field_z=True),
        Variant("hco",       f, restricted=True, field_z=True),
        Variant("hco",       f, frozen_core="fc"),
    ]
    return variants + _closed_shell_variants(f)


# The option sets each singlet is run with closed shell.
_CLOSED_SHELL_SPEC: dict[str, list[dict]] = {
    "he":        [{}],
    "h2o":       [{}, {"frozen_core": "fc"}, {"df": True}, {"field_z": True}],
}


def _closed_shell_variants(functional: str) -> list[Variant]:
    """Closed-shell variants (singlets only) for one functional."""
    out: list[Variant] = []
    for mol, option_sets in _CLOSED_SHELL_SPEC.items():
        if MOLECULES[mol]["mult"] != 1:
            raise ValueError(f"_CLOSED_SHELL_SPEC lists non-singlet {mol!r}")
        out.extend(
            Variant(mol, functional, closed_shell=True, **opts)
            for opts in option_sets
        )
    return out


ALL_VARIANTS: list[Variant] = [v for f in FUNCTIONALS for v in _make_variants(f)]


def _param(v: Variant) -> pytest.param:
    return pytest.param(v, id=v.id)


ALL_PARAMS        = [_param(v) for v in ALL_VARIANTS]
CORR_PARAMS       = [_param(v) for v in ALL_VARIANTS if v.has_corr]
DF_CORR_PARAMS    = [_param(v) for v in ALL_VARIANTS if v.df and v.has_corr]
FIELD_PARAMS      = [_param(v) for v in ALL_VARIANTS if v.field_z]
FIELD_CORR_PARAMS = [_param(v) for v in ALL_VARIANTS if v.field_z and v.has_corr]
FC_PARAMS         = [_param(v) for v in ALL_VARIANTS if v.frozen_core != "0"]
FC_CORR_PARAMS    = [_param(v) for v in ALL_VARIANTS if v.frozen_core != "0" and v.has_corr]
CS_PARAMS         = [_param(v) for v in ALL_VARIANTS if v.closed_shell]
CS_CORR_PARAMS    = [_param(v) for v in ALL_VARIANTS if v.closed_shell and v.has_corr]

_MOLDEN_MOLS = {"h2o", "cn", "ch2", "ch3", "hco"}
MOLDEN_VARIANTS = [
    _param(v) for v in ALL_VARIANTS
    if v.mol in _MOLDEN_MOLS
    and not v.df and not v.field_z
    and v.frozen_core == "0"
]


# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--etol", default=ETOL, type=float,
        help=f"Energy tolerance in Hartree (default: {ETOL})",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "qchem: requires qchem on PATH")
    config.addinivalue_line("markers", "reference: compares against Q-Chem reference JSON")


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "qchem" in item.keywords and not _QCHEM_AVAILABLE:
        pytest.skip("qchem not on PATH")


@pytest.fixture(scope="session")
def etol(request: pytest.FixtureRequest) -> float:
    return float(request.config.getoption("--etol"))


# ---------------------------------------------------------------------------
# Q-Chem availability warning
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _qchem_availability_check():
    """Warn once per session when Q-Chem is absent and pyscf fallback is used."""
    if not _QCHEM_AVAILABLE:
        warnings.warn(
            "\n\nqchem is not on PATH.  The session-scoped 'run' fixture will "
            "use --source pyscf for all base DFT calculations; the reference "
            "JSON files were produced with Q-Chem.  Tests marked 'qchem' are "
            "automatically skipped.\n",
            UserWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
def load_reference(mol: str, functional: str, tag: str, basis: str = BASIS) -> dict | None:
    path = REFS_DIR / f"{mol}_{functional}_{basis}_{tag}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


# ---------------------------------------------------------------------------
# Orbital-invariant views of a run (W elements depend on orbital phases and
# degenerate rotations; these do not)
# ---------------------------------------------------------------------------
def w_matrices(paths) -> dict[str, np.ndarray]:
    """The eight saved W matrices of a run, by label."""
    return {lbl: np.load(paths.w_matrix(lbl)) for lbl in W2_LABELS + W3_LABELS}


def singular_values(m: np.ndarray) -> np.ndarray:
    """Singular values of a W block, descending (empty for a 0x0 block)."""
    return np.linalg.svd(m, compute_uv=False) if m.size else np.zeros(0)


def resummed_totals(result) -> dict[str, float]:
    """OP-MP3, SOP-MP3 and SSD-OPMP3 total energies of a run summary; the
    SSD-OPMP3 label carries its x_ss, so it is found by prefix."""
    e = result.energies
    ssd = [k for k in e if k.startswith("SSD-OPMP3 total energy")]
    assert len(ssd) == 1, ssd
    return {"OP-MP3": e["OP-MP3 total energy"],
            "SOP-MP3": e["SOP-MP3 total energy"],
            "SSD-OPMP3": e[ssd[0]]}


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------
def make_config(xyz: Path, **kw) -> WorkflowConfig:
    """Build a WorkflowConfig for a single-file run."""
    ef       = kw.pop("electric_field", None)
    source   = kw.pop("source", "qchem")
    # molden-direct runs no SCF and rejects --functional.
    func     = kw.pop("functional", None if source == "molden-direct" else "pbe0")
    default_basis = None if source in ("molden", "molden-direct") else BASIS
    basis    = kw.pop("basis", default_basis)
    threads  = kw.pop("threads", THREADS)
    df       = kw.pop("df", False)
    default_auxbasis = AUXBASIS if df else None
    auxbasis = kw.pop("auxbasis", default_auxbasis)
    return WorkflowConfig(
        input_path=xyz.resolve(),
        source=source,
        functional=func,
        basis=basis,
        threads=threads,
        memory=MEMORY,
        frozen_core=kw.pop("frozen_core", "0"),
        df=df,
        auxbasis=auxbasis,
        restricted=kw.pop("restricted", False),
        qc_mp3=False,
        workers=None,
        electric_field=parse_electric_field(ef),
        closed_shell=kw.pop("closed_shell", False),
    )


def make_batch_config(d: Path, **kw) -> WorkflowConfig:
    """Build a WorkflowConfig for a directory-mode run."""
    df = kw.pop("df", False)
    default_auxbasis = AUXBASIS if df else None
    return WorkflowConfig(
        input_path=d.resolve(),
        source=kw.pop("source", "pyscf"),
        functional=kw.pop("functional", "pbe0"),
        basis=kw.pop("basis", BASIS),
        threads=kw.pop("threads", THREADS),
        memory=MEMORY,
        frozen_core=kw.pop("frozen_core", "0"),
        df=df,
        auxbasis=kw.pop("auxbasis", default_auxbasis),
        restricted=kw.pop("restricted", False),
        qc_mp3=False,
        workers=kw.pop("workers", None),
        electric_field=None,
        closed_shell=kw.pop("closed_shell", False),
    )


# ---------------------------------------------------------------------------
# Session-scoped run directory
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _prepare_runs_dir():
    if RUNS_DIR.exists():
        shutil.rmtree(RUNS_DIR)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Session-scoped workflow runner
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def run(_prepare_runs_dir, _qchem_availability_check):
    """Run a Variant, restarting from the cached base SCF (--source qchem,
    or pyscf without Q-Chem) of its (mol, functional, restricted, basis)."""
    result_cache: dict[Variant, tuple] = {}
    molden_cache: dict[tuple[str, str, bool, str], Path] = {}

    def _run(v: Variant) -> tuple:
        if v in result_cache:
            return result_cache[v]

        base_key = v._base_key

        if base_key not in molden_cache:
            base_v = v.bare()
            work = RUNS_DIR / base_v.id
            work.mkdir(parents=True, exist_ok=True)
            xyz = work / f"{base_v.mol}.xyz"
            shutil.copy(DATA_DIR / f"{base_v.mol}.xyz", xyz)

            kw = base_v.to_config_kwargs()
            if _QCHEM_AVAILABLE:
                base_result = run_one_workflow(make_config(xyz, **kw))
                molden_key  = ResultFileKey.QCHEM_MOLDEN
            else:
                kw["source"] = "pyscf"
                base_result = run_one_workflow(make_config(xyz, **kw))
                molden_key  = ResultFileKey.PYSCF_DFT_MOLDEN

            base_paths = derive_output_paths(xyz)
            result_cache[base_v] = (base_result, base_paths)
            molden_cache[base_key] = Path(base_result.files[molden_key])

        if v in result_cache:
            return result_cache[v]

        base_molden = molden_cache[base_key]
        work = RUNS_DIR / v.id
        work.mkdir(parents=True, exist_ok=True)
        molden_dst = work / base_molden.name
        shutil.copy(base_molden, molden_dst)

        kw = v.to_config_kwargs()
        kw["source"] = "molden"
        result = run_one_workflow(make_config(molden_dst, **kw))
        paths  = derive_output_paths(molden_dst)
        result_cache[v] = (result, paths)
        return result, paths

    return _run


# ---------------------------------------------------------------------------
# PySCF-direct runner
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def pyscf_run(_prepare_runs_dir):
    """Run a Variant with source=pyscf, cached independently from ``run``."""
    cache: dict[Variant, tuple] = {}
    d = RUNS_DIR / "_pyscf"
    d.mkdir(parents=True, exist_ok=True)

    def _run(v: Variant) -> tuple:
        if v in cache:
            return cache[v]
        work = d / v.id
        work.mkdir(parents=True, exist_ok=True)
        xyz = work / f"{v.mol}.xyz"
        shutil.copy(DATA_DIR / f"{v.mol}.xyz", xyz)
        result = run_one_workflow(make_config(xyz, source="pyscf", **v.to_config_kwargs()))
        paths  = derive_output_paths(xyz)
        cache[v] = (result, paths)
        return result, paths

    return _run


# ---------------------------------------------------------------------------
# Molden round-trip runner
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def molden_run(run):
    """Molden restart from the main pipeline's PySCF DFT Molden, cached."""
    cache: dict[Variant, tuple] = {}
    d = RUNS_DIR / "_molden"
    d.mkdir(parents=True, exist_ok=True)

    def _run(v: Variant) -> tuple:
        if v in cache:
            return cache[v]
        parent_result, _ = run(v)
        src  = Path(parent_result.files[ResultFileKey.PYSCF_DFT_MOLDEN])
        work = d / v.id
        work.mkdir(parents=True, exist_ok=True)
        dst  = work / src.name
        shutil.copy(src, dst)
        result = run_one_workflow(
            make_config(dst, source="molden",
                        functional=v.functional, restricted=v.restricted,
                        closed_shell=v.closed_shell)
        )
        paths = derive_output_paths(dst)
        cache[v] = (result, paths)
        return result, paths

    return _run


# ---------------------------------------------------------------------------
# Checkpoint restart runner
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def chk_run(run):
    """--source chk on a copy of the base run's checkpoint (chk overwrites its input)."""
    cache: dict[Variant, tuple] = {}
    d = RUNS_DIR / "_chk"
    d.mkdir(parents=True, exist_ok=True)

    def _run(v: Variant) -> tuple:
        if v in cache:
            return cache[v]
        _, parent_paths = run(v)
        work = d / v.id
        work.mkdir(parents=True, exist_ok=True)
        dst = work / parent_paths.pyscf_chk.name
        shutil.copy(parent_paths.pyscf_chk, dst)
        result = run_one_workflow(
            make_config(dst, source="chk", **v.to_config_kwargs())
        )
        paths = derive_output_paths(dst)
        cache[v] = (result, paths)
        return result, paths

    return _run


# ---------------------------------------------------------------------------
# molden-direct runner (no SCF)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def direct_run(run):
    """--source molden-direct on the base run's PySCF Molden."""
    cache: dict[Variant, tuple] = {}
    d = RUNS_DIR / "_direct"
    d.mkdir(parents=True, exist_ok=True)

    def _run(v: Variant) -> tuple:
        if v in cache:
            return cache[v]
        parent_result, _ = run(v)
        src  = Path(parent_result.files[ResultFileKey.PYSCF_DFT_MOLDEN])
        work = d / v.id
        work.mkdir(parents=True, exist_ok=True)
        dst  = work / src.name
        shutil.copy(src, dst)
        kw = v.to_config_kwargs()
        kw.pop("functional")          # no SCF is run in this mode
        result = run_one_workflow(make_config(dst, source="molden-direct", **kw))
        paths  = derive_output_paths(dst)
        cache[v] = (result, paths)
        return result, paths

    return _run
