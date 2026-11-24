"""Configuration, output paths, XYZ parsing and the run summary, shared by
all source modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import numpy as np
import os
import sys
import yaml

from electric_field import ElectricField, electric_field_to_string


# Source modes.  The first four run a PySCF DFT (seeded differently);
# "molden-direct" runs no SCF at all and uses the Molden orbitals as written.
SUPPORTED_SOURCES = ("qchem", "molden", "chk", "pyscf", "molden-direct")
SCF_SOURCES       = ("qchem", "molden", "chk", "pyscf")

# Labels of the eight saved W matrices.
W2_LABELS = ("W2alpha_ss", "W2alpha_os", "W2beta_ss", "W2beta_os")
W3_LABELS = ("W3alpha_ss", "W3alpha_os", "W3beta_ss", "W3beta_os")

# Same-spin scaling x_ss of the indicative SSD-OPMP3 energy (the paper's
# Eq. 22) in the run summary: the value the paper fits for the reference
# (Fig. 3; W4-17 TAE nonMR), looked up by --functional with case, '-', '_'
# and spaces ignored and 'ω' read as 'w', and DEFAULT_X_SS otherwise.  kappa-OOMP2 (0.45) enters
# through molden-direct, without --functional, so it gets the default.
# analysis_code/ evaluates SSD-OPMP3 at any x_ss.
PAPER_X_SS = {
    "hf":     0.00,
    "spw92":  0.75,
    "pbe":    0.70,
    "r2scan": 0.60,
    "pbe0":   0.50,
    "wb97xv": 0.50,
    "wb97mv": 0.50,
}
DEFAULT_X_SS = 0.5


def ssd_x_ss(functional: str | None) -> tuple[float, str]:
    """(x_ss, where it comes from) of the summary's SSD-OPMP3 energy."""
    if not functional:
        return DEFAULT_X_SS, "default: no --functional, orbitals read from a Molden file"
    key = "".join(c for c in functional.lower().replace("ω", "w") if c not in "-_ ")
    if key in PAPER_X_SS:
        return PAPER_X_SS[key], f"fitted in the paper for {functional}, Fig. 3"
    return DEFAULT_X_SS, f"default: the paper fits no x_ss for {functional}"


class ResultFileKey(str, Enum):
    """Keys of WorkflowResult.files; the values are the summary labels."""
    FINAL_SUMMARY        = "Final summary"
    FINAL_YAML           = "Final yaml file"
    PYSCF_DFT_OUTPUT     = "PySCF DFT output"
    PYSCF_DFT_CHECKPOINT = "PySCF DFT checkpoint"
    PYSCF_DFT_MOLDEN     = "PySCF DFT Molden"
    SEMICANONICAL_OUTPUT = "Semicanonical output"
    SEMICANONICAL_MOLDEN = "Semicanonical Molden"
    QCHEM_INPUT          = "Q-Chem input"
    QCHEM_OUTPUT         = "Q-Chem output"
    QCHEM_MOLDEN         = "Q-Chem Molden"


@dataclass(frozen=True)
class SemicanonicalResult:
    """Output of semicanonicalize().  The orbital fields are (alpha, beta)
    pairs, or single arrays with 0/2 occupations when ``closed_shell``."""
    report_file:          str
    molden_file:          str
    mo_coeff:             tuple   # MO coefficient matrices
    mo_occ:               tuple   # occupation vectors
    mo_energy:            tuple   # orbital energies
    nocc:                 tuple   # occupied orbital counts
    e_hf_from_density:    float
    nuclear_field_energy: float
    e_nbs:                float
    closed_shell:         bool


@dataclass(frozen=True)
class WorkflowConfig:
    """Runtime settings for one workflow invocation."""
    input_path:    Path
    source:        str
    functional:    str | None   # None only for source == "molden-direct"
    basis:         str | None
    threads:       int
    memory:        int
    frozen_core:   str | int
    df:            bool
    auxbasis:      str | None
    restricted:    bool
    qc_mp3:        bool
    workers:       int | None
    electric_field: ElectricField | None
    closed_shell:  bool


@dataclass(frozen=True)
class WorkflowPaths:
    """Output paths: next to the input, with its stem as prefix."""
    base:                      Path
    final_output:              Path
    final_yaml:                Path
    qchem_input:               Path
    qchem_output:              Path
    qchem_molden:              Path
    pyscf_output:              Path
    pyscf_chk:                 Path
    pyscf_molden:              Path
    semicanonical_output:      Path
    semicanonical_molden:      Path

    def w_matrix(self, label: str) -> Path:
        """Return the .npy path for one W matrix."""
        return Path(f"{self.base}.{label}.npy")

    def semicanonical_array(self, label: str) -> Path:
        """Path of <stem>.semicanonical.<label>.npy, label e.g. coeff_a or coeff."""
        return Path(f"{self.base}.semicanonical.{label}.npy")


@dataclass(frozen=True)
class BatchPaths:
    """Captured-output paths for one directory-mode child job."""
    stdout: Path
    stderr: Path


@dataclass
class WorkflowResult:
    """One finished run: output paths (by ResultFileKey), W-matrix paths
    (by label), the summary energies, and the (x_ss, origin) of the
    summary's SSD-OPMP3 energy."""
    input_path:     Path
    source:         str
    electric_field: ElectricField | None
    ncore:          int
    files:          dict   # ResultFileKey -> str
    w_matrix_files: dict   # str label -> str path
    energies:       dict   # str -> float
    ssd_x_ss:       tuple[float, str] | None = None   # ssd_x_ss() of the run


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def read_xyz(xyz_file: str | Path) -> tuple[int, int, str]:
    """(charge, multiplicity, coordinate lines) of an XYZ file whose second
    line gives the charge and multiplicity as ``0 1`` (the first two tokens)
    or as ``charge=0 multiplicity=1``; any further text is ignored (README §4)."""
    xyz_path = Path(xyz_file)
    if xyz_path.suffix.lower() != ".xyz":
        raise ValueError(f"Expected an XYZ file, got {xyz_path}")

    lines = xyz_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ file is too short: {xyz_path}")

    try:
        natoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid atom count in the first line of {xyz_path}") from exc

    header_tokens = lines[1].split()
    if len(header_tokens) < 2:
        raise ValueError(
            f"The second line of {xyz_path} must begin with charge and multiplicity."
        )
    try:
        charge       = int(header_tokens[0])
        multiplicity = int(header_tokens[1])
    except ValueError:
        # key=value form: tokens other than charge=... and multiplicity=...
        # are ignored, as is further text after the two numbers above.
        values: dict[str, str] = {}
        for token in header_tokens:
            key, sep, value = token.partition("=")
            if sep and key.strip().lower() in ("charge", "multiplicity"):
                values[key.strip().lower()] = value.replace(",", "")
        try:
            charge       = int(values["charge"])
            multiplicity = int(values["multiplicity"])
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Could not parse charge and multiplicity from the second line of {xyz_path}"
            ) from exc

    if len(lines) < 2 + natoms:
        raise ValueError(
            f"XYZ file {xyz_path} declares {natoms} atoms but contains fewer "
            "coordinate lines."
        )

    geometry = "\n".join(line.rstrip() for line in lines[2 : 2 + natoms])
    return charge, multiplicity, geometry


def derive_output_paths(input_path: str | Path) -> WorkflowPaths:
    """Return the deterministic output file layout for one input file: the
    input's path without its extension, plus a suffix.  The suffix is
    appended, not set with with_suffix(), which would replace the last
    dotted part of a stem such as h2o.opt."""
    base = Path(input_path).resolve().with_suffix("")
    out  = lambda suffix: Path(f"{base}{suffix}")
    return WorkflowPaths(
        base=base,
        final_output=out(".final.out"),
        final_yaml=out(".final.yaml"),
        qchem_input=out(".in"),
        qchem_output=out(".out"),
        qchem_molden=out(".molden"),
        pyscf_output=out(".pyscf.out"),
        pyscf_chk=out(".pyscf.chk"),
        pyscf_molden=out(".pyscf.molden"),
        semicanonical_output=out(".semicanonical.out"),
        semicanonical_molden=out(".semicanonical.molden"),
    )


def derive_batch_paths(input_path: str | Path) -> BatchPaths:
    """Return the captured-output paths for one directory-mode child job
    (suffixes appended, as in derive_output_paths)."""
    base = Path(input_path).resolve().with_suffix("")
    return BatchPaths(
        stdout=Path(f"{base}.batch.stdout"),
        stderr=Path(f"{base}.batch.stderr"),
    )


def list_inputs(input_path: str | Path, source: str) -> list[Path]:
    """Return the workflow inputs implied by input_path.

    A single file is returned as a one-element list.  A directory is filtered
    by extension according to the source mode: .xyz for qchem and pyscf,
    .molden for molden and molden-direct, .chk for chk.
    """
    path = Path(input_path).resolve()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"Input path does not exist or is not accessible: {path}")
    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"Unsupported source mode: {source}")
    if source in ("molden", "molden-direct"):
        pattern = "*.molden"
    elif source == "chk":
        pattern = "*.chk"
    else:
        pattern = "*.xyz"
    files = sorted(c for c in path.glob(pattern) if c.is_file())
    if not files:
        raise ValueError(f"No input files matching {pattern} found in {path}")
    return files


def check_batch_oversubscription(workers: int | None, threads: int) -> list[str]:
    """Return human-readable warnings about likely CPU oversubscription."""
    if workers is None:
        return []
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return []
    requested = workers * threads
    if requested > cpu_count:
        return [
            f"Requested parallelism likely oversubscribes the machine: "
            f"workers * threads = {requested}, detected CPUs = {cpu_count}."
        ]
    return []


def suggested_default_workers(threads: int, ninputs: int) -> int:
    """Choose a conservative default worker count for directory mode."""
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return min(1, ninputs)
    return max(1, min(ninputs, cpu_count // max(1, int(threads))))


def resolve_frozen_core(frozen_core: str | int, mol) -> int:
    """Number of frozen orbitals per spin: the integer given, or PySCF's
    chemical core count for 'fc'."""
    text = str(frozen_core).strip().lower()
    if text == "fc":
        from pyscf.data.elements import chemcore
        return int(chemcore(mol, spinorb=False))
    try:
        ncore = int(text)
    except ValueError as exc:
        raise ValueError("--frozen-core must be a non-negative integer or 'fc'") from exc
    if ncore < 0:
        raise ValueError("--frozen-core must be non-negative")
    return ncore


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

def build_result_files(
    paths:          WorkflowPaths,
    sc:             SemicanonicalResult,
    *,
    include_pyscf:  bool,
    include_qchem:  bool,
) -> dict:
    """Collect the typed workflow-artifact paths for one run.

    ``include_pyscf`` adds the PySCF DFT stage outputs, which the
    molden-direct source mode does not produce; ``include_qchem`` adds the
    Q-Chem input, output and Molden, which only the qchem source mode produces.
    """
    files = {
        ResultFileKey.FINAL_SUMMARY: str(paths.final_output),
        ResultFileKey.FINAL_YAML:    str(paths.final_yaml),
    }
    if include_pyscf:
        files.update({
            ResultFileKey.PYSCF_DFT_OUTPUT:     str(paths.pyscf_output),
            ResultFileKey.PYSCF_DFT_CHECKPOINT: str(paths.pyscf_chk),
            ResultFileKey.PYSCF_DFT_MOLDEN:     str(paths.pyscf_molden),
        })
    files.update({
        ResultFileKey.SEMICANONICAL_OUTPUT: sc.report_file,
        ResultFileKey.SEMICANONICAL_MOLDEN: sc.molden_file,
    })
    if include_qchem:
        files.update({
            ResultFileKey.QCHEM_INPUT:  str(paths.qchem_input),
            ResultFileKey.QCHEM_OUTPUT: str(paths.qchem_output),
            ResultFileKey.QCHEM_MOLDEN: str(paths.qchem_molden),
        })
    return files


def w_matrix_file_map(w_results: dict) -> dict[str, str]:
    """Collect W-matrix .npy paths."""
    return {label: w_results["files"][label] for label in W2_LABELS + W3_LABELS}


def correlation_energy_map(
    sc:              SemicanonicalResult,
    w_results:       dict,
    reference_label: str,
) -> dict[str, float]:
    """Collect the energies reported in the final summary.

    ``reference_label`` names the reference energy, which differs between
    source modes: the SCF modes evaluate HF on the converged KS density,
    molden-direct on the density of the Molden orbitals as written.
    """
    ref_energy = sc.e_hf_from_density
    nbs        = sc.e_nbs
    doubles_ss = w_results["Tr(W2_ss)"]
    doubles_os = w_results["Tr(W2_os)"]
    doubles    = w_results["Tr(W2)"]
    mp2_corr   = doubles + nbs
    mp3_corr   = w_results["Tr(W3)"]

    return {
        "External-field nuclear contribution":      sc.nuclear_field_energy,
        reference_label:                            ref_energy,
        "Non-Brillouin singles correction":         nbs,
        "MP2 doubles correlation (same-spin)":      doubles_ss,
        "MP2 doubles correlation (opposite-spin)":  doubles_os,
        "MP2 doubles correlation":                  doubles,
        "MP2 correlation":                          mp2_corr,
        "MP3 doubles correlation (same-spin)":      w_results["Tr(W3_ss)"],
        "MP3 doubles correlation (opposite-spin)":  w_results["Tr(W3_os)"],
        "MP3 correlation":                          mp3_corr,
        "MP2 total energy":                         ref_energy + mp2_corr,
        "MP3 total energy":                         ref_energy + mp2_corr + mp3_corr,
        "OP-MP3 total energy":                      ref_energy + nbs + w_results["OP-MP3"],
        "SOP-MP3 total energy":                     ref_energy + nbs + w_results["SOP-MP3"],
        f"SSD-OPMP3 total energy (x_ss = {w_results['SSD-OPMP3 x_ss']:g})":
            ref_energy + nbs + w_results["SSD-OPMP3"],
    }


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def _pyscf_version() -> str:
    """PySCF version, recorded in every summary (CCSD internals are used)."""
    try:
        from pyscf import __version__
    except ImportError:
        return "unavailable"
    return str(__version__)


def _get_scalar_items(result: WorkflowResult, df_auxbasis: str | None = None) -> list:
    """Return the non-energy (label, value) rows of the run summary."""
    scalar_items: list[tuple[str, object]] = [
        ("Input source",         str(result.input_path)),
        ("Source mode",          result.source),
        ("Electric field",       electric_field_to_string(result.electric_field)),
        ("Frozen core orbitals", result.ncore),
        ("PySCF version",        _pyscf_version()),
    ]
    if result.ssd_x_ss is not None:
        x_ss, origin = result.ssd_x_ss
        scalar_items.append(("SSD-OPMP3 x_ss", f"{x_ss:g} ({origin})"))
    scalar_items.extend((label.value, path) for label, path in result.files.items())
    scalar_items.extend((label, path) for label, path in result.w_matrix_files.items())
    if df_auxbasis is not None:
        scalar_items.append(("DF auxiliary basis", df_auxbasis))
    return scalar_items


def format_result(result: WorkflowResult, df_auxbasis: str | None = None) -> str:
    """Return the final workflow summary as formatted plain text."""
    scalar_items: list[tuple[str, object]] = _get_scalar_items(result, df_auxbasis)

    width = max(
        [len(label) for label, _ in scalar_items]
        + [len(label) for label in result.energies]
    )

    lines = [f"{label:<{width}} = {value}" for label, value in scalar_items]
    for label, value in result.energies.items():
        lines.append(f"{label:<{width}} = {value:.15f}")
    return "\n".join(lines) + "\n"


def format_result_yaml(result: WorkflowResult, df_auxbasis: str | None = None) -> str:
    """Return the final workflow summary as a YAML document."""
    scalar_items: list[tuple[str, object]] = _get_scalar_items(result, df_auxbasis)

    data = {k: v for k, v in scalar_items}
    data["energies"] = {}
    for label, value in result.energies.items():
        data["energies"][label] = np.array(value).item()
    return yaml.dump(data)


def emit_result(
    result: WorkflowResult,
    *,
    output_file: str | Path | None = None,
    output_yaml: str | Path | None = None,
    df_auxbasis: str | None = None,
    print_stdout: bool = True,
) -> Path:
    """Write <stem>.final.out and <stem>.final.yaml (or the paths given),
    optionally echo the text to stdout, and return the .out path."""
    if output_file is None:
        try:
            output_file = result.files[ResultFileKey.FINAL_SUMMARY]
        except KeyError as exc:
            raise ValueError("Result does not define a final summary file.") from exc

    if output_yaml is None:
        try:
            output_yaml = result.files[ResultFileKey.FINAL_YAML]
        except KeyError as exc:
            raise ValueError("Result does not define a final yaml file.") from exc

    text      = format_result(result, df_auxbasis=df_auxbasis)
    yaml_text = format_result_yaml(result, df_auxbasis=df_auxbasis)

    summary_path = Path(output_file)
    summary_path.write_text(text, encoding="utf-8")
    yaml_path = Path(output_yaml)
    yaml_path.write_text(yaml_text, encoding="utf-8")

    if print_stdout:
        sys.stdout.write(text)
    return summary_path
