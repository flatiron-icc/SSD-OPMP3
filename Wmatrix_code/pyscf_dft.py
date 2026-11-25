"""The reference determinant as a PySCF mean-field object.

dft_from_molden        SCF restarted from a Molden file (--source molden, qchem)
dft_from_xyz           SCF from an XYZ file or a PySCF checkpoint (--source pyscf, chk)
reference_from_molden  the orbitals of a Molden file as written, no SCF (--source molden-direct)

Also the basis substitutions and the ECP choice (handle_basis_core, get_ecp).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
import sys

import numpy as np

from pyscf import dft, gto, lib, scf
from pyscf.lib import logger
from pyscf.tools import molden
from pyscf.data.elements import NUC, _std_symbol_without_ghost

from electric_field import ElectricField, attach_static_electric_field, electric_field_to_string
from pyscf_orbitals import (
    infer_charge_from_restricted_occ,
    infer_charge_from_unrestricted_occ,
    make_rhf_dm,
    make_uhf_dm,
    require_closed_shell_molden_payload,
    require_restricted_open_shell_molden_payload,
    require_unrestricted_molden_payload,
    write_restricted_molden_from_reference,
    write_unrestricted_molden_from_reference,
)
from workflow_common import WorkflowPaths, derive_output_paths, read_xyz

# ECP family for the Molden modes (molden, molden-direct) without --basis;
# their AO basis is the file's.
_DEFAULT_ECP_BASIS_HINT = "cc-pvdz"


def _molden_ecp_hint(basis: str | None) -> str:
    """The label that selects the ECP of a Molden input (get_ecp): --basis,
    or the cc-* family without it.  Shared by both Molden modes."""
    return basis if basis else _DEFAULT_ECP_BASIS_HINT


_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

SCF_CONV_TOL      = 1e-10
SCF_CONV_TOL_GRAD = 1e-5


@contextmanager
def _restore_stdout(*objects):
    """Restore each object's stdout on exit, so no closed log file stays attached."""
    originals = [(obj, getattr(obj, "stdout", None)) for obj in objects]
    try:
        yield
    finally:
        for obj, orig in originals:
            obj.stdout = orig


def _set_threads(threads: int) -> None:
    """Apply one thread count consistently across common BLAS/OpenMP back ends."""
    value = str(int(threads))
    for name in _THREAD_ENV_VARS:
        os.environ[name] = value
    lib.num_threads(int(threads))


def _new_dft_reference(mol, closed_shell: bool, restricted: bool):
    """The PySCF mean-field object for the requested reference: RKS, ROKS or UKS."""
    if closed_shell:
        return dft.RKS(mol)
    return dft.ROKS(mol) if restricted else dft.UKS(mol)


def _configure_mf(mf, *, functional, memory, chkfile, log_fh, electric_field) -> None:
    """Apply the settings shared by both DFT front ends (mutates mf in place)."""
    attach_static_electric_field(mf, electric_field)
    mf.xc            = functional
    mf.verbose       = logger.INFO
    mf.stdout        = log_fh
    mf.chkfile       = str(chkfile)
    mf.max_memory    = float(memory)
    mf.conv_tol      = SCF_CONV_TOL
    mf.conv_tol_grad = SCF_CONV_TOL_GRAD


def _configure_molecule_logging(mol, *, memory, log_fh) -> None:
    """Apply the logging and memory settings shared by both routes."""
    mol.max_memory = float(memory)
    mol.verbose    = logger.INFO
    mol.stdout     = log_fh


def _write_header(log_fh, title: str, items: list[tuple[str, object]]) -> None:
    """Write a fixed-format header to the PySCF log file."""
    log_fh.write("============================================================\n")
    log_fh.write(f"{title}\n")
    log_fh.write("============================================================\n")
    for label, value in items:
        log_fh.write(f"{label:<14}: {value}\n")
    log_fh.write("============================================================\n\n")
    log_fh.flush()


def _require_converged(mf) -> None:
    """Raise an explicit error if the SCF did not converge."""
    if not mf.converged:
        raise RuntimeError(
            "PySCF DFT did not converge. The semicanonicalization and "
            "W-matrix generation stages require a converged reference."
        )


def _finalize_run(log_fh, mf, molden_file: Path) -> None:
    """Check convergence, write the final Molden file, and close the log section."""
    _require_converged(mf)
    if mf.istype("UHF") or mf.istype("UKS") or mf.istype("ROKS") or mf.istype("ROHF"):
        write_unrestricted_molden_from_reference(molden_file, mf)
    else:
        write_restricted_molden_from_reference(molden_file, mf)
    log_fh.write("\n============================================================\n")
    log_fh.write("PySCF DFT finished successfully\n")
    log_fh.write("============================================================\n")
    log_fh.flush()


def _soscf(
    mf,
    *,
    mo_coeff=None,
    mo_occ=None,
    dm0=None,
    log_fh,
):
    """Converge ``mf`` with PySCF's second-order (Newton) SCF solver.

    Seeded either with orbitals and {0, 1, 2} occupations (ROKS restarts:
    the occupation pattern is kept) or with an AO density (RKS and UKS
    restarts: the orbitals are rebuilt by aufbau, which also absorbs Molden
    files with different alpha and beta orbital counts).  The solver copies
    the field-modified ``get_hcore``/``energy_nuc`` of ``mf``.  Relies on
    ``pyscf.soscf.newton_ah`` internals (checked with PySCF 2.15.1).
    """
    have_mo  = mo_coeff is not None and mo_occ is not None
    have_dm  = dm0 is not None
    if have_mo == have_dm:
        raise ValueError(
            "_soscf requires exactly one of (mo_coeff, mo_occ) or dm0."
        )

    mf_newton = mf.newton()
    mf_newton.max_cycle = 200
    # mf_newton copies the settings and field closures of mf.
    mf_newton.stdout = log_fh
    try:
        if have_mo:
            mf_newton.kernel(mo_coeff=mo_coeff, mo_occ=mo_occ)
        else:
            mf_newton.kernel(dm0=dm0)
    finally:
        # mf_newton outlives log_fh, which the caller closes.
        mf_newton.stdout = sys.stdout
    return mf_newton


def _soscf_inputs_from_molden_orbitals_restricted(
    *,
    coeff_a: np.ndarray,
    occ_a:   np.ndarray,
    occ_b:   np.ndarray,
):
    """ROKS restart inputs from the two spin blocks of a Molden file: the
    alpha orbitals as the spatial set, and alpha + beta occupations padded
    to its width (a Q-Chem file may list fewer beta orbitals)."""
    coeff_a = np.asarray(coeff_a)
    occ_a   = np.asarray(occ_a, dtype=float)
    occ_b   = np.asarray(occ_b, dtype=float)

    nmo_alpha = coeff_a.shape[1]

    occ_a_padded = np.zeros(nmo_alpha, dtype=float)
    occ_a_padded[: occ_a.size] = occ_a

    occ_b_padded = np.zeros(nmo_alpha, dtype=float)
    n_b = min(occ_b.size, nmo_alpha)
    occ_b_padded[:n_b] = occ_b[:n_b]

    return coeff_a, occ_a_padded + occ_b_padded


def dft_from_molden(
    molden_file:    str | Path,
    functional:     str,
    threads:        int,
    memory:         int,
    restricted:     bool,
    electric_field: ElectricField | None = None,
    output_paths:   WorkflowPaths | None = None,
    basis:          str | None = None,
    closed_shell:   bool = False,
):
    """Converged RKS (``closed_shell``), ROKS (``restricted``) or UKS,
    restarted from the orbitals of a Molden file.  The AO basis comes from
    the file; ``basis`` only selects the ECP family, as in
    reference_from_molden (_molden_ecp_hint; default cc-*)."""
    _set_threads(threads)

    molden_path = Path(molden_file).resolve()
    paths = derive_output_paths(molden_path) if output_paths is None else output_paths

    with open(paths.pyscf_output, "w", encoding="utf-8", buffering=1) as log_fh:
        mol, mo_energy, mo_coeff, mo_occ, _, _ = molden.load(str(molden_path))
        if closed_shell:
            _, coeff, occ = require_closed_shell_molden_payload(
                mol, mo_energy, mo_coeff, mo_occ,
            )
        else:
            (_, _), (coeff_a, coeff_b), (occ_a, occ_b) = require_unrestricted_molden_payload(
                mo_energy, mo_coeff, mo_occ,
            )
        build_with_ecp(mol, _molden_ecp_hint(basis))
        if closed_shell:
            charge = infer_charge_from_restricted_occ(mol, occ)
            mol.spin = 0
        else:
            charge = infer_charge_from_unrestricted_occ(mol, (occ_a, occ_b))
        mol.charge = charge
        mol.build()

        mf = _new_dft_reference(mol, closed_shell=closed_shell, restricted=restricted)

        with _restore_stdout(mol, mf):
            _configure_molecule_logging(mol, memory=memory, log_fh=log_fh)
            mol.build(dump_input=False, parse_arg=False)
            _configure_mf(
                mf,
                functional=functional,
                memory=memory,
                chkfile=paths.pyscf_chk,
                log_fh=log_fh,
                electric_field=electric_field,
            )
            _write_header(
                log_fh,
                "PySCF DFT restart from Molden input",
                [
                    ("Input Molden",   molden_path),
                    ("Functional",     functional),
                    ("Threads",        int(threads)),
                    ("Memory (MB)",    int(memory)),
                    ("Charge",         charge),
                    ("Spin",           mol.spin),
                    ("Electric field", electric_field_to_string(electric_field)),
                    ("Checkpoint",     paths.pyscf_chk),
                    ("Final Molden",   paths.pyscf_molden),
                ],
            )
            if closed_shell:
                # RKS: seed from the AO density built from the Molden orbitals.
                mf = _soscf(mf, dm0=make_rhf_dm(coeff, occ), log_fh=log_fh)
            elif restricted:
                # ROKS: seed from orbitals/occupations so that the occupation
                # pattern survives (no aufbau re-diagonalization).
                soscf_mo_coeff, soscf_mo_occ = _soscf_inputs_from_molden_orbitals_restricted(
                    coeff_a=coeff_a, occ_a=occ_a, occ_b=occ_b,
                )
                mf = _soscf(
                    mf,
                    mo_coeff=soscf_mo_coeff,
                    mo_occ=soscf_mo_occ,
                    log_fh=log_fh,
                )
            else:
                # UKS: seed from the AO density (the spin blocks may differ in length).
                mf = _soscf(mf, dm0=make_uhf_dm((coeff_a, coeff_b), (occ_a, occ_b)), log_fh=log_fh)
            _finalize_run(log_fh, mf, paths.pyscf_molden)

    return mf


def _atom_labels(mol) -> list[str]:
    """The distinct atom labels of a built molecule, as PySCF matches basis
    and ECP entries against them: bare symbols for a molecule built from a
    geometry (``Ag``), indexed ones for a Molden-built molecule (``Ag1``)."""
    return sorted({atom[0] for atom in mol._atom})


def get_ecp(basis, mol=None):
    """ECP for the elements beyond Kr implied by a basis label: def2-svp for
    ``def2-*``, cc-pvdz-pp for ``cc-*``, None otherwise.

    With ``mol``, a dict keyed by element symbol and by atom label (a
    Molden-built molecule has indexed labels such as ``Ag1``, which PySCF
    matches first) of the atoms beyond Kr only; the other atoms get no ECP.
    """
    if isinstance(basis, str):
        if "def2-" in basis.lower():
            ecp = "def2-svp"
        elif "cc-" in basis.lower():
            ecp = "cc-pvdz-pp"
        else:
            return None
        if mol is None:
            return ecp
        else:
            _ecp = ecp
            ecp = {}
            for label in _atom_labels(mol):
                Z = NUC[_std_symbol_without_ghost(label)]
                if Z > 36:
                    ecp[label] = _ecp
                    ecp[_std_symbol_without_ghost(label)] = _ecp
            return ecp
    else:
        return None


def handle_basis_core(basis, mol=None):
    """Per-element basis for a basis label, lowercased: H and He get
    ``cc-pvXz`` for ``cc-pcvXz``/``cc-pwcvXz``, and a ``-pp`` suffix is kept
    only for the elements beyond Kr.  A non-string basis is returned
    unchanged."""
    if isinstance(basis, str):
        pp_basis = basis.lower()
        basis = basis.lower().replace("z-pp", "z")
        hhe_basis = basis.lower()
        for letter in ["d", "t", "q", "5", "6"]:
            hhe_basis = hhe_basis.replace(f"pwcv{letter}z", f"pv{letter}z")
            hhe_basis = hhe_basis.replace(f"pcv{letter}z", f"pv{letter}z")
        dbasis = {"default": basis, "H": hhe_basis, "He": hhe_basis}
        if "z-pp" in pp_basis:  # Dunning basis with pseudo
            for label in _atom_labels(mol):
                Z = NUC[_std_symbol_without_ghost(label)]
                if Z > 36:
                    dbasis[label] = pp_basis
                    dbasis[_std_symbol_without_ghost(label)] = pp_basis
        return dbasis
    else:
        return basis


def build_with_ecp(mol, basis) -> None:
    """Attach the ECP implied by ``basis`` to ``mol`` and rebuild it.

    The ECP changes the electron count, so the build may fail on spin
    parity; the charge is then shifted by one only to get a buildable
    molecule.  The caller sets the real charge from the occupations.
    """
    mol.ecp = get_ecp(basis, mol=mol)
    if mol.ecp is None:
        return
    try:
        mol.build()
    except RuntimeError:
        mol.charge += 1
        mol.build()


def dft_from_xyz(
    geometry:       str | Path,
    functional:     str,
    basis:          str,
    threads:        int,
    memory:         int,
    restricted:     bool,
    electric_field: ElectricField | None = None,
    output_paths:   WorkflowPaths | None = None,
    use_chkfile:    bool = False,
    closed_shell:   bool = False,
):
    """Converged RKS (``closed_shell``, singlets only), ROKS (``restricted``)
    or UKS from an XYZ file, or, with ``use_chkfile``, from the geometry,
    charge, spin and orbitals of a PySCF checkpoint."""
    _set_threads(threads)

    xyz_path = Path(geometry).resolve()
    paths    = derive_output_paths(xyz_path) if output_paths is None else output_paths
    if use_chkfile:
        from pyscf.scf.chkfile import load_mol
        from pyscf.data.elements import _std_symbol
        _mol = load_mol(xyz_path)
        charge = _mol.charge
        spin = _mol.spin
        multiplicity = spin + 1
        geometry_block = []
        for symb, coord in _mol._atom:
            geometry_block.append((_std_symbol(symb), coord))
        # Internal units of PySCF for geometry
        unit = "Bohr"
    else:
        charge, multiplicity, geometry_block = read_xyz(xyz_path)
        spin = multiplicity - 1
        unit = "Angstrom"

    with open(paths.pyscf_output, "w", encoding="utf-8", buffering=1) as log_fh:
        mol = gto.Mole()
        mol.atom   = geometry_block
        mol.unit   = unit
        mol.charge = charge
        mol.spin   = spin
        mol.basis = "def2-svp"            # placeholder build, for the atom labels
        mol.build()
        mol.basis = handle_basis_core(basis, mol=mol)
        mol.ecp = get_ecp(basis, mol=mol)
        mol.build()

        if closed_shell and mol.spin != 0:
            raise ValueError(
                "closed_shell=True requires a spin-singlet molecule, but "
                f"spin = {mol.spin} (multiplicity {mol.spin + 1})."
            )
        mf = _new_dft_reference(mol, closed_shell=closed_shell, restricted=restricted)
        with _restore_stdout(mol, mf):
            _configure_molecule_logging(mol, memory=memory, log_fh=log_fh)
            mol.build()
            _configure_mf(
                mf,
                functional=functional,
                memory=memory,
                chkfile=paths.pyscf_chk,
                log_fh=log_fh,
                electric_field=electric_field,
            )
            _write_header(
                log_fh,
                "PySCF DFT direct run from XYZ input",
                [
                    ("Input XYZ",      xyz_path),
                    ("Functional",     functional),
                    ("Basis",          basis),
                    ("Threads",        int(threads)),
                    ("Memory (MB)",    int(memory)),
                    ("Charge",         charge),
                    ("Multiplicity",   multiplicity),
                    ("Spin",           spin),
                    ("Electric field", electric_field_to_string(electric_field)),
                    ("Checkpoint",     paths.pyscf_chk),
                    ("Final Molden",   paths.pyscf_molden),
                ],
            )
            if use_chkfile:
                mf.init_guess = "chkfile"
                mf.chkfile = xyz_path
            mf.kernel()
            if not mf.converged:
                mf = mf.newton()
                mf.max_cycle = 200
                mf.kernel()
            _finalize_run(log_fh, mf, paths.pyscf_molden)

    return mf


def reference_from_molden(
    molden_file:  str | Path,
    threads:      int,
    memory:       int,
    closed_shell: bool = False,
    restricted:   bool = False,
    basis:        str | None = None,
):
    """``--source molden-direct``: the orbitals of a Molden file, as written,
    in a non-iterated scf.RHF (``closed_shell``), scf.ROHF (``restricted``)
    or scf.UHF, which semicanonicalize() uses for its density.

    The file must hold the complete orbital set, with equal alpha and beta
    counts.  ``closed_shell`` and ``restricted`` check that it is a
    closed-shell or restricted open-shell determinant
    (require_closed_shell_molden_payload,
    require_restricted_open_shell_molden_payload); semicanonical orbitals
    pass.  A restricted open-shell file read as UHF gives the same numbers.
    ``basis`` only selects the ECP family, as in dft_from_molden
    (_molden_ecp_hint; default cc-*), as the AO basis comes from the file.
    """
    _set_threads(threads)

    molden_path = Path(molden_file).resolve()
    mol, mo_energy, mo_coeff, mo_occ, _, _ = molden.load(str(molden_path))
    build_with_ecp(mol, _molden_ecp_hint(basis))

    nao = int(mol.nao_nr())
    if closed_shell:
        eps, coeff, occ = require_closed_shell_molden_payload(
            mol, mo_energy, mo_coeff, mo_occ,
        )
        if coeff.shape[0] != nao:
            raise ValueError(
                f"Molden orbital coefficients do not span the AO basis of {molden_path}: "
                f"expected {nao} AO rows, found {coeff.shape[0]}."
            )
        if coeff.shape[1] != nao:
            raise ValueError(
                f"Molden file {molden_path} provides an incomplete molecular-orbital "
                f"set ({coeff.shape[1]} orbitals for {nao} AOs). The molden-direct "
                "source mode requires the full virtual space, so the file must "
                f"contain {nao} orbitals (in Q-Chem, set print_orbitals to a large "
                "number)."
            )
        charge = infer_charge_from_restricted_occ(mol, occ)
        nalpha = int(round(0.5 * float(occ.sum())))
        nbeta  = nalpha
    else:
        (eps_a, eps_b), (coeff_a, coeff_b), (occ_a, occ_b) = (
            require_unrestricted_molden_payload(mo_energy, mo_coeff, mo_occ)
        )
        if coeff_a.shape[0] != nao or coeff_b.shape[0] != nao:
            raise ValueError(
                f"Molden orbital coefficients do not span the AO basis of {molden_path}: "
                f"expected {nao} AO rows, found alpha {coeff_a.shape[0]} and "
                f"beta {coeff_b.shape[0]}."
            )
        if coeff_a.shape[1] != coeff_b.shape[1]:
            raise ValueError(
                f"Alpha and beta orbital counts differ in {molden_path} "
                f"(alpha {coeff_a.shape[1]}, beta {coeff_b.shape[1]}). The "
                "molden-direct source mode requires a common per-spin orbital count."
            )
        if coeff_a.shape[1] != nao:
            raise ValueError(
                f"Molden file {molden_path} provides an incomplete molecular-orbital "
                f"set ({coeff_a.shape[1]} orbitals per spin for {nao} AOs). The "
                "molden-direct source mode requires the full virtual space, so the "
                f"file must contain {nao} orbitals per spin (in Q-Chem, set "
                "print_orbitals to a large number)."
            )
        charge = infer_charge_from_unrestricted_occ(mol, (occ_a, occ_b))
        nalpha = int(round(float(occ_a.sum())))
        nbeta  = int(round(float(occ_b.sum())))
        if restricted:
            # Rejects a spin-polarized file; returns the shared ROHF orbitals
            # with {0, 1, 2} occupations.
            eps_ro, coeff_ro, occ_ro = require_restricted_open_shell_molden_payload(
                mol, eps_a, coeff_a, coeff_b, occ_a, occ_b,
            )

    mol.charge     = charge
    mol.spin       = nalpha - nbeta
    mol.max_memory = float(memory)
    mol.build(dump_input=False, parse_arg=False)

    if closed_shell:
        mf = scf.RHF(mol)
        mf.mo_coeff  = np.asarray(coeff)
        mf.mo_occ    = np.asarray(occ, dtype=float)
        mf.mo_energy = np.asarray(eps, dtype=float)
    elif restricted:
        mf = scf.ROHF(mol)
        mf.mo_coeff  = np.asarray(coeff_ro)
        mf.mo_occ    = np.asarray(occ_ro, dtype=float)
        mf.mo_energy = np.asarray(eps_ro, dtype=float)
    else:
        mf = scf.UHF(mol)
        mf.mo_coeff  = (np.asarray(coeff_a), np.asarray(coeff_b))
        mf.mo_occ    = (np.asarray(occ_a, dtype=float), np.asarray(occ_b, dtype=float))
        mf.mo_energy = (np.asarray(eps_a, dtype=float), np.asarray(eps_b, dtype=float))
    mf.e_tot      = 0.0
    mf.max_memory = float(memory)
    mf.chkfile    = None
    return mf
