"""Semicanonical orbitals of the HF Fock matrix of a fixed reference density.

The HF Fock matrix is built from the density of the reference determinant
(UHF, or RHF for a closed-shell reference) and its occupied-occupied and
virtual-virtual blocks are diagonalized separately, which leaves the
determinant unchanged.  The second-order non-Brillouin singles energy is
computed from the remaining occupied-virtual block.  Writes
<stem>.semicanonical.{out,molden} and the MO coefficients, occupations and
Fock matrix as <stem>.semicanonical.{coeff,occ,fock}_{a,b}.npy (one set
without the spin suffix, occupations 0/2, for a closed-shell reference).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pyscf import scf
from scipy.linalg import eigh

from electric_field import (
    ElectricField,
    attach_static_electric_field,
    electric_field_to_string,
    nuclear_electric_field_energy,
)
from pyscf_orbitals import (
    extract_spin_orbitals_and_occupations,
    write_restricted_molden,
    write_unrestricted_molden,
)
from workflow_common import SemicanonicalResult, WorkflowPaths, derive_output_paths


def _semicanonicalize_spin_block(
    c_mo:     np.ndarray,
    f_ao:     np.ndarray,
    occ_spin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Diagonalize the occupied-occupied and virtual-virtual blocks of the
    AO Fock matrix ``f_ao`` for one spin, given its 0/1 occupations
    (validated by extract_spin_orbitals_and_occupations).  Returns the
    semicanonical coefficients (occupied first), their orbital energies, the
    MO Fock matrix, the 0/1 occupations and the occupied count."""
    occ_mask = np.asarray(occ_spin) > 0.5
    c_occ    = c_mo[:, occ_mask]
    c_vir    = c_mo[:, ~occ_mask]

    if c_occ.shape[1] > 0:
        eps_occ, u_occ = eigh(c_occ.conj().T @ f_ao @ c_occ)
        c_occ_sc = c_occ @ u_occ
    else:
        eps_occ  = np.empty((0,), dtype=float)
        c_occ_sc = c_occ.copy()

    if c_vir.shape[1] > 0:
        eps_vir, u_vir = eigh(c_vir.conj().T @ f_ao @ c_vir)
        c_vir_sc = c_vir @ u_vir
    else:
        eps_vir  = np.empty((0,), dtype=float)
        c_vir_sc = c_vir.copy()

    c_sc = np.hstack((c_occ_sc, c_vir_sc))
    eps  = np.concatenate((eps_occ, eps_vir))
    f_sc = c_sc.conj().T @ f_ao @ c_sc
    occ_sc = np.concatenate((
        np.ones(c_occ_sc.shape[1],  dtype=float),
        np.zeros(c_vir_sc.shape[1], dtype=float),
    ))
    return c_sc, eps, f_sc, occ_sc, c_occ_sc.shape[1]


def nonbrillouin_singles_correction(
    f_sc_a: np.ndarray,
    f_sc_b: np.ndarray,
    eps_a:  np.ndarray,
    eps_b:  np.ndarray,
    nocc_a: int,
    nocc_b: int,
    ncore:  int,
) -> float:
    """E_S^(2) = -sum_ai |F_ai|^2 / (eps_a - eps_i), both spins, over the
    non-frozen occupied orbitals.  F_ai vanishes for an HF reference
    (Brillouin's theorem) but not for a DFT or kappa-OOMP2 one."""
    fov_a = f_sc_a[:nocc_a, nocc_a:]
    fov_b = f_sc_b[:nocc_b, nocc_b:]

    e2 = 0.0
    if nocc_a > ncore and fov_a.size:
        den_a = eps_a[:nocc_a, None] - eps_a[None, nocc_a:]
        e2 += float(np.sum((np.abs(fov_a[ncore:, :]) ** 2) / den_a[ncore:, :]).real)
    if nocc_b > ncore and fov_b.size:
        den_b = eps_b[:nocc_b, None] - eps_b[None, nocc_b:]
        e2 += float(np.sum((np.abs(fov_b[ncore:, :]) ** 2) / den_b[ncore:, :]).real)
    return e2


def write_orbital_energies(
    fh,
    label: str,
    eps:   np.ndarray,
    nocc:  int,
) -> None:
    """Write the occupied and virtual orbital energies of one spin."""
    fh.write(f"{label} occupied orbital energies\n")
    fh.write("index                 energy\n")
    fh.write("-----  ---------------------\n")
    for i, energy in enumerate(eps[:nocc], start=1):
        fh.write(f"{i:5d}  {float(energy):21.12f}\n")
    fh.write("\n")

    fh.write(f"{label} virtual orbital energies\n")
    fh.write("index                 energy\n")
    fh.write("-----  ---------------------\n")
    for a, energy in enumerate(eps[nocc:], start=nocc + 1):
        fh.write(f"{a:5d}  {float(energy):21.12f}\n")
    fh.write("\n")


def write_fov_block(
    fh,
    label: str,
    f_sc:  np.ndarray,
    nocc:  int,
) -> None:
    """Write the occupied-virtual Fock elements of one spin."""
    fov = f_sc[:nocc, nocc:]
    fh.write(f"{label} occupied-virtual Fock matrix elements\n")
    fh.write("occ    vir                 F_ov\n")
    fh.write("-----  -----  ---------------------\n")
    for i in range(fov.shape[0]):
        for a in range(fov.shape[1]):
            fh.write(f"{i + 1:5d}  {nocc + a + 1:5d}  {float(fov[i, a].real):21.12e}\n")
    fh.write("\n")


def _write_summary_header(fh, items: list[tuple[str, object]], closed_shell=False) -> None:
    """Write the fixed-format text header for a semicanonical report."""
    fh.write("============================================================\n")
    if closed_shell:
        fh.write("RHF Fock from reference density and semicanonicalization\n")
    else:
        fh.write("UHF Fock from reference density and semicanonicalization\n")
    fh.write("============================================================\n")
    for label, value in items:
        fh.write(f"{label:<30}: {value}\n")
    fh.write("============================================================\n\n")


def semicanonicalize(
    mf_dft,
    input_path:     str | Path,
    ncore:          int = 0,
    electric_field: ElectricField | None = None,
    output_paths:   WorkflowPaths | None = None,
) -> SemicanonicalResult:
    """Semicanonicalize the reference determinant of ``mf_dft`` (only its
    mol, orbitals, occupations and density are used) in the HF Fock matrix
    of its density, with the field if one is given, and write the outputs.

    The SemicanonicalResult holds single arrays for a closed-shell (RKS/RHF)
    reference and (alpha, beta) pairs otherwise.
    """
    input_path = Path(input_path).resolve()
    paths  = derive_output_paths(input_path) if output_paths is None else output_paths
    ncore  = int(ncore)
    if ncore < 0:
        raise ValueError("ncore must be non-negative")

    closed_shell = mf_dft.istype("RKS") or mf_dft.istype("RHF")
    closed_shell = closed_shell and not (mf_dft.istype("ROKS") or mf_dft.istype("ROHF"))
    if closed_shell:
        c, occ = extract_spin_orbitals_and_occupations(mf_dft)
        dm = mf_dft.make_rdm1()

        mf_hf = scf.RHF(mf_dft.mol)
        attach_static_electric_field(mf_hf, electric_field)
        mf_hf.max_memory = getattr(mf_dft, "max_memory", mf_dft.mol.max_memory)
        h1 = mf_hf.get_hcore(mf_dft.mol)
        vhf = mf_hf.get_veff(mf_dft.mol, dm)
        fock = mf_hf.get_fock(h1e=h1, vhf=vhf, dm=dm)
        nuclear_field_energy = nuclear_electric_field_energy(mf_dft.mol, electric_field)
        e_hf_from_density = float(mf_hf.energy_tot(dm=dm, h1e=h1, vhf=vhf))

        # _semicanonicalize_spin_block works in binary 0/1 occupations, so the
        # closed-shell 0/2 vector is halved on the way in and doubled on the
        # way out.
        c_sc, eps, f_sc, occ_sc, nocc = _semicanonicalize_spin_block(c, fock, 0.5 * occ)
        occ_sc[:] *= 2

        e_nbs = nonbrillouin_singles_correction(
            f_sc, f_sc, eps, eps, nocc, nocc, ncore
        )

        write_restricted_molden(
            filename=paths.semicanonical_molden,
            mol=mf_dft.mol,
            coeff=c_sc,
            eps=eps,
            occ=occ_sc,
        )

        for label, array in (
            ("coeff", c_sc),
            ("occ", occ_sc),
            ("fock", f_sc),
        ):
            np.save(paths.semicanonical_array(label), array)

        header_items = [
            ("Input source",                 input_path),
            ("Reference class",              mf_dft.__class__.__name__),
            ("External electric field",      electric_field_to_string(electric_field)),
            ("Nuclear field contribution",   f"{nuclear_field_energy:.15f}"),
            ("HF energy from density",       f"{e_hf_from_density:.15f}"),
            ("Non-Brillouin singles E2",     f"{e_nbs:.15f}"),
            ("Frozen occupied orbitals",     ncore),
            ("Occupied orbitals",            nocc),
            ("Semicanonical Molden output",  paths.semicanonical_molden),
        ]

        with open(paths.semicanonical_output, "w", encoding="utf-8") as fh:
            _write_summary_header(fh, header_items, closed_shell=True)
            write_orbital_energies(fh, "Alpha", eps, nocc)
            write_fov_block(fh, "Alpha", f_sc, nocc)

        mo_coeff = c_sc
        mo_occ = occ_sc
        mo_energy = eps
    else:
        c_a, c_b, occ_a, occ_b = extract_spin_orbitals_and_occupations(mf_dft)
        dm = mf_dft.make_rdm1()

        mf_hf = scf.UHF(mf_dft.mol)
        attach_static_electric_field(mf_hf, electric_field)
        mf_hf.max_memory = getattr(mf_dft, "max_memory", mf_dft.mol.max_memory)
        h1   = mf_hf.get_hcore(mf_dft.mol)
        vhf  = mf_hf.get_veff(mf_dft.mol, dm)
        fock_a, fock_b       = mf_hf.get_fock(h1e=h1, vhf=vhf, dm=dm)
        nuclear_field_energy = nuclear_electric_field_energy(mf_dft.mol, electric_field)
        e_hf_from_density    = float(mf_hf.energy_tot(dm=dm, h1e=h1, vhf=vhf))

        c_sc_a, eps_a, f_sc_a, occ_sc_a, nocc_a = _semicanonicalize_spin_block(c_a, fock_a, occ_a)
        c_sc_b, eps_b, f_sc_b, occ_sc_b, nocc_b = _semicanonicalize_spin_block(c_b, fock_b, occ_b)

        e_nbs = nonbrillouin_singles_correction(
            f_sc_a, f_sc_b, eps_a, eps_b, nocc_a, nocc_b, ncore,
        )

        write_unrestricted_molden(
            filename=paths.semicanonical_molden,
            mol=mf_dft.mol,
            coeff_a=c_sc_a, coeff_b=c_sc_b,
            eps_a=eps_a,    eps_b=eps_b,
            occ_a=occ_sc_a, occ_b=occ_sc_b,
        )

        for label, array in (
            ("coeff_a", c_sc_a),
            ("coeff_b", c_sc_b),
            ("occ_a",   occ_sc_a),
            ("occ_b",   occ_sc_b),
            ("fock_a",  f_sc_a),
            ("fock_b",  f_sc_b),
        ):
            np.save(paths.semicanonical_array(label), array)

        header_items = [
            ("Input source",                 input_path),
            ("Reference class",              mf_dft.__class__.__name__),
            ("External electric field",      electric_field_to_string(electric_field)),
            ("Nuclear field contribution",   f"{nuclear_field_energy:.15f}"),
            ("HF energy from density",       f"{e_hf_from_density:.15f}"),
            ("Non-Brillouin singles E2",     f"{e_nbs:.15f}"),
            ("Frozen occupied orbitals",     ncore),
            ("Alpha occupied orbitals",      nocc_a),
            ("Beta occupied orbitals",       nocc_b),
            ("Semicanonical Molden output",  paths.semicanonical_molden),
        ]

        with open(paths.semicanonical_output, "w", encoding="utf-8") as fh:
            _write_summary_header(fh, header_items)
            write_orbital_energies(fh, "Alpha", eps_a, nocc_a)
            write_orbital_energies(fh, "Beta",  eps_b, nocc_b)
            write_fov_block(fh, "Alpha", f_sc_a, nocc_a)
            write_fov_block(fh, "Beta",  f_sc_b, nocc_b)

        mo_coeff = (c_sc_a, c_sc_b)
        mo_occ = (occ_sc_a, occ_sc_b)
        mo_energy = (eps_a, eps_b)
        nocc = (nocc_a, nocc_b)

    return SemicanonicalResult(
        report_file=str(paths.semicanonical_output),
        molden_file=str(paths.semicanonical_molden),
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        nocc=nocc,
        e_hf_from_density=e_hf_from_density,
        nuclear_field_energy=nuclear_field_energy,
        e_nbs=e_nbs,
        closed_shell=closed_shell,
    )
