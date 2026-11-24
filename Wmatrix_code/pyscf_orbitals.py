"""Orbitals and occupations of a reference: Molden payload checks, conversion
of PySCF mean-field objects to explicit alpha/beta arrays (one 0/2-occupied
set for a closed-shell reference), restart densities, and Molden writers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pyscf.scf import rhf, uhf
from pyscf.tools import molden


ArrayPair = tuple[np.ndarray, np.ndarray]

_OCC_TOL = 1.0e-8

# Tolerance of the Molden spin-block checks (closed shell, restricted open
# shell): the largest deviation from 1 of the singular values of
# C_a_occ^T S C_b_occ.  Loose enough to absorb the Molden text format's
# printed precision.
_CLOSED_SHELL_TOL = 1.0e-6


def _explicit_spin_pair(name: str, obj, *, allow_spin_axis: bool) -> ArrayPair:
    """Return explicit alpha/beta arrays from a spin-resolved object."""
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        return np.asarray(obj[0]), np.asarray(obj[1])
    if allow_spin_axis:
        arr = np.asarray(obj)
        if arr.ndim >= 1 and arr.shape[0] == 2:
            return np.asarray(arr[0]), np.asarray(arr[1])
    raise ValueError(f"{name} must contain separate alpha and beta spin blocks.")


def _has_explicit_spin_occupations(obj) -> bool:
    """Return True when occupations are stored separately for alpha and beta spin."""
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        return True
    arr = np.asarray(obj)
    return arr.ndim == 2 and arr.shape[0] == 2


def _binary_spin_occupations_closed_shell(name: str, occ):
    """Validate closed-shell 0/2 occupations and return them as a float array.

    An array given in {0, 1} is doubled: a single-section Molden file cannot
    express a half-filled orbital.
    """
    occ = np.asarray(occ, dtype=float)
    bad = ~(np.isclose(occ, 0.0, atol=_OCC_TOL) | np.isclose(occ, 2.0, atol=_OCC_TOL))
    if np.any(bad):
        if (np.isclose(occ, 0.0, atol=_OCC_TOL) | np.isclose(occ, 1.0, atol=_OCC_TOL)).all():
            occ = occ * 2
        else:
            raise ValueError(f"{name} must contain only 0/2 spin occupations.")
    return occ


def _binary_spin_occupations(name: str, occ_pair: ArrayPair) -> ArrayPair:
    """Validate unrestricted occupations and return them as float arrays."""
    occ_a = np.asarray(occ_pair[0], dtype=float)
    occ_b = np.asarray(occ_pair[1], dtype=float)
    bad_a = ~(np.isclose(occ_a, 0.0, atol=_OCC_TOL) | np.isclose(occ_a, 1.0, atol=_OCC_TOL))
    bad_b = ~(np.isclose(occ_b, 0.0, atol=_OCC_TOL) | np.isclose(occ_b, 1.0, atol=_OCC_TOL))
    if np.any(bad_a) or np.any(bad_b):
        raise ValueError(f"{name} must contain only 0/1 spin occupations.")
    return occ_a, occ_b


def _validate_explicit_spin_shapes(
    coeff_a: np.ndarray,
    coeff_b: np.ndarray,
    occ_a:   np.ndarray,
    occ_b:   np.ndarray,
    eps_a:   np.ndarray | None = None,
    eps_b:   np.ndarray | None = None,
) -> None:
    """Check shape consistency for explicit alpha/beta orbital data."""
    if coeff_a.ndim != 2 or coeff_b.ndim != 2:
        raise ValueError("Orbital coefficient matrices must be two-dimensional.")
    if occ_a.ndim != 1 or occ_b.ndim != 1:
        raise ValueError("Occupation arrays must be one-dimensional.")
    if coeff_a.shape[1] != occ_a.size or coeff_b.shape[1] != occ_b.size:
        raise ValueError("Occupation lengths must match the number of molecular orbitals.")
    if eps_a is not None and np.asarray(eps_a).shape != (occ_a.size,):
        raise ValueError("Alpha orbital-energy array has inconsistent length.")
    if eps_b is not None and np.asarray(eps_b).shape != (occ_b.size,):
        raise ValueError("Beta orbital-energy array has inconsistent length.")


def _validate_explicit_shapes(
    coeff, occ, eps=None
):
    """Check shape consistency for a single (closed-shell) orbital set."""
    if coeff.ndim != 2:
        raise ValueError("Orbital coefficient matrices must be two-dimensional.")
    if occ.ndim != 1:
        raise ValueError("Occupation arrays must be one-dimensional.")
    if coeff.shape[1] != occ.size:
        raise ValueError("Occupation lengths must match the number of molecular orbitals.")
    if eps is not None and np.asarray(eps).shape != (occ.size,):
        raise ValueError("Orbital-energy array has inconsistent length.")


def require_restricted_molden_payload(
    mo_energy, mo_coeff, mo_occ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(eps, coeff, occ) of a single-section Molden payload, occupations
    0/2 (0/1 are doubled)."""
    eps = np.asarray(mo_energy, dtype=float)
    coeff = np.asarray(mo_coeff)
    occ = np.asarray(mo_occ)
    occ = _binary_spin_occupations_closed_shell("mo_occ", occ)
    return eps, coeff, occ


def require_unrestricted_molden_payload(
    mo_energy, mo_coeff, mo_occ,
) -> tuple[ArrayPair, ArrayPair, ArrayPair]:
    """(eps, coeff, occ) alpha/beta pairs of a Molden payload with explicit
    alpha and beta sections; raises otherwise."""
    eps_a,   eps_b   = _explicit_spin_pair("mo_energy", mo_energy, allow_spin_axis=False)
    coeff_a, coeff_b = _explicit_spin_pair("mo_coeff",  mo_coeff,  allow_spin_axis=False)
    occ_a,   occ_b   = _explicit_spin_pair("mo_occ",    mo_occ,    allow_spin_axis=False)

    eps_a   = np.asarray(eps_a,   dtype=float)
    eps_b   = np.asarray(eps_b,   dtype=float)
    coeff_a = np.asarray(coeff_a)
    coeff_b = np.asarray(coeff_b)
    occ_a, occ_b = _binary_spin_occupations("mo_occ", (occ_a, occ_b))
    _validate_explicit_spin_shapes(coeff_a, coeff_b, occ_a, occ_b, eps_a, eps_b)
    return (eps_a, eps_b), (coeff_a, coeff_b), (occ_a, occ_b)


def require_closed_shell_molden_payload(
    mol, mo_energy, mo_coeff, mo_occ, tol: float = _CLOSED_SHELL_TOL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(eps, coeff, 0/2 occ) of a closed-shell determinant from a Molden
    payload: a single orbital set as is, or two spin blocks with the same
    occupied space (nocc_a == nocc_b and every singular value of
    C_a,occ^T S C_b,occ is 1 to ``tol``, which does not depend on how either
    block's occupied orbitals are rotated), returned as the alpha block.  A
    spin-polarized file raises."""
    if not _has_explicit_spin_occupations(mo_occ):
        eps, coeff, occ = require_restricted_molden_payload(mo_energy, mo_coeff, mo_occ)
        _validate_explicit_shapes(coeff, occ, eps)
        return eps, coeff, occ

    (eps_a, _), (coeff_a, coeff_b), (occ_a, occ_b) = require_unrestricted_molden_payload(
        mo_energy, mo_coeff, mo_occ,
    )

    nocc_a = int(round(float(occ_a.sum())))
    nocc_b = int(round(float(occ_b.sum())))
    if nocc_a != nocc_b:
        raise ValueError(
            "Molden file is not closed shell: it has "
            f"{nocc_a} occupied alpha and {nocc_b} occupied beta orbitals. "
            "Drop --closed-shell for this input."
        )

    c_occ_a = coeff_a[:, occ_a > 0.5]
    c_occ_b = coeff_b[:, occ_b > 0.5]
    ovlp    = np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=float)
    s       = np.linalg.svd(c_occ_a.T @ ovlp @ c_occ_b, compute_uv=False)
    if s.size and float(np.max(np.abs(1.0 - s))) > tol:
        raise ValueError(
            "Molden file is not closed shell: the occupied alpha and beta "
            "orbitals span different spaces (smallest singular value of "
            f"C_a_occ^T S C_b_occ = {s.min():.6f}, tolerance {tol:.0e}). "
            "Drop --closed-shell for this input."
        )

    return eps_a, coeff_a, 2.0 * occ_a


def require_restricted_open_shell_molden_payload(
    mol, eps_a, coeff_a, coeff_b, occ_a, occ_b, tol: float = _CLOSED_SHELL_TOL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(eps, coeff, 0/1/2 occ) of a restricted open-shell determinant given
    as alpha and beta orbital blocks; a spin-polarized file raises.

    The determinant is restricted open shell when the beta occupied space
    lies inside the alpha one, i.e. when every singular value of
    P = C_a,occ^T S C_b,occ = U s V^T is 1 (to ``tol``).  This does not depend
    on how the occupied orbitals of either spin are rotated among
    themselves, so semicanonical orbitals are accepted.  The alpha occupied
    orbitals rotated by U form the shared set: the first nocc_b span the beta
    occupied space (occupation 2), the rest are singly occupied (1); the
    alpha virtuals follow (0).  ``eps`` is the diagonal of the alpha orbital
    energies in that set; nothing downstream uses it.
    """
    coeff_a = np.asarray(coeff_a)
    coeff_b = np.asarray(coeff_b)
    eps_a   = np.asarray(eps_a, dtype=float)
    occ_a, occ_b = _binary_spin_occupations("mo_occ", (occ_a, occ_b))

    nocc_a = int(round(float(occ_a.sum())))
    nocc_b = int(round(float(occ_b.sum())))
    if nocc_b > nocc_a:
        raise ValueError(
            "Molden file is not restricted open-shell in the PySCF convention: "
            f"it has more occupied beta ({nocc_b}) than alpha ({nocc_a}) orbitals. "
            "Alpha must carry the majority spin."
        )

    occ_mask = occ_a > 0.5
    c_occ_a  = coeff_a[:, occ_mask]
    c_occ_b  = coeff_b[:, occ_b > 0.5]
    ovlp     = np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=float)
    u, s, _  = np.linalg.svd(c_occ_a.T @ ovlp @ c_occ_b)   # u: (nocc_a, nocc_a)
    if s.size and float(np.max(np.abs(1.0 - s))) > tol:
        raise ValueError(
            "Molden file is not restricted open-shell: the beta occupied space is "
            "not contained in the alpha occupied space (smallest singular value "
            f"of C_a_occ^T S C_b_occ = {s.min():.6f}, tolerance {tol:.0e}). "
            "Drop --restricted for this input."
        )

    coeff = np.hstack((c_occ_a @ u, coeff_a[:, ~occ_mask]))
    eps   = np.concatenate((np.einsum("ik,i,ik->k", u, eps_a[occ_mask], u),
                            eps_a[~occ_mask]))
    occ   = np.concatenate((np.full(nocc_b, 2.0), np.ones(nocc_a - nocc_b),
                            np.zeros(int((~occ_mask).sum()))))
    return eps, coeff, occ


def infer_charge_from_restricted_occ(mol, mo_occ: np.ndarray) -> int:
    """Infer the total molecular charge from a single 0/2 occupation vector."""
    nelec_total    = float(np.sum(mo_occ))
    nuclear_charge = float(np.sum(mol.atom_charges()))
    return int(round(nuclear_charge - nelec_total))


def infer_charge_from_unrestricted_occ(mol, mo_occ: ArrayPair) -> int:
    """Infer the total molecular charge from explicit alpha/beta occupations."""
    nelec_total    = float(np.sum(mo_occ[0]) + np.sum(mo_occ[1]))
    nuclear_charge = float(np.sum(mol.atom_charges()))
    return int(round(nuclear_charge - nelec_total))


def make_uhf_dm(mo_coeff: ArrayPair, mo_occ: ArrayPair):
    """Build the UHF AO density matrix from explicit alpha/beta orbitals."""
    coeff_a = np.asarray(mo_coeff[0])
    coeff_b = np.asarray(mo_coeff[1])
    occ_a, occ_b = _binary_spin_occupations("mo_occ", mo_occ)
    _validate_explicit_spin_shapes(coeff_a, coeff_b, occ_a, occ_b)
    return uhf.make_rdm1((coeff_a, coeff_b), (occ_a, occ_b))


def make_rhf_dm(mo_coeff, mo_occ):
    """Build the RHF AO density matrix from one 0/2-occupied orbital set."""
    coeff = np.asarray(mo_coeff)
    occ = _binary_spin_occupations_closed_shell("mo_occ", mo_occ)
    return rhf.make_rdm1(coeff, occ)


def extract_spin_orbitals_and_occupations(mf_ref) -> tuple[np.ndarray, ...]:
    """(coeff_a, coeff_b, occ_a, occ_b) with 0/1 occupations for UHF/UKS and
    ROHF/ROKS (alpha holds the singly occupied orbitals), (coeff, occ) with
    0/2 occupations for RHF/RKS."""
    if mf_ref.istype("UKS") or mf_ref.istype("UHF"):
        coeff_a, coeff_b = _explicit_spin_pair(
            "mf_ref.mo_coeff", mf_ref.mo_coeff, allow_spin_axis=True,
        )
        occ_a, occ_b = _explicit_spin_pair(
            "mf_ref.mo_occ", mf_ref.mo_occ, allow_spin_axis=True,
        )
        coeff_a = np.asarray(coeff_a)
        coeff_b = np.asarray(coeff_b)
        occ_a, occ_b = _binary_spin_occupations("mf_ref.mo_occ", (occ_a, occ_b))
        _validate_explicit_spin_shapes(coeff_a, coeff_b, occ_a, occ_b)
        return coeff_a, coeff_b, occ_a, occ_b
    elif mf_ref.istype("ROKS") or mf_ref.istype("ROHF"):
        coeff = np.asarray(mf_ref.mo_coeff)
        occ   = np.asarray(mf_ref.mo_occ, dtype=float)
        bad   = ~(
            np.isclose(occ, 0.0, atol=_OCC_TOL)
            | np.isclose(occ, 1.0, atol=_OCC_TOL)
            | np.isclose(occ, 2.0, atol=_OCC_TOL)
        )
        if coeff.ndim != 2 or occ.ndim != 1 or occ.size != coeff.shape[1] or np.any(bad):
            raise ValueError(
                "Restricted-open-shell references must provide a 2D spatial coefficient "
                "matrix and a 1D occupation array in {0, 1, 2}."
            )

        if mf_ref.mol.spin < 0:
            raise ValueError(
                "mol.spin is negative, which is not a valid PySCF value. "
                "spin = 2S must be a non-negative integer."
            )

        # Standard convention: The majority of electrons are alpha.
        occ_a = (occ > 0).astype(float)   # occupied in alpha: occ == 1 or occ == 2
        occ_b = (occ == 2).astype(float)  # doubly occupied: alpha and beta both filled
        return coeff, coeff, occ_a, occ_b
    else:
        assert mf_ref.istype("RKS") or mf_ref.istype("RHF")
        coeff = np.asarray(mf_ref.mo_coeff)
        occ = np.asarray(mf_ref.mo_occ, dtype=float)
        bad = ~(
            np.isclose(occ, 0.0, atol=_OCC_TOL)
            | np.isclose(occ, 2.0, atol=_OCC_TOL)
        )
        if coeff.ndim != 2 or occ.ndim != 1 or occ.size != coeff.shape[1] or np.any(bad):
            raise ValueError(
                "Restricted-closed-shell references must provide a 2D spatial coefficient "
                "matrix and a 1D occupation array in {0, 2}."
            )
        if mf_ref.mol.spin != 0:
            raise ValueError("Closed-shell only works for spin=0!")
        return coeff, occ


def extract_spin_orbital_energies(mf_ref) -> ArrayPair | np.ndarray:
    """Return orbital energies for a PySCF reference.

    An alpha/beta pair for UHF/UKS and ROHF/ROKS; a single array for RHF/RKS.
    """
    if mf_ref.istype("UKS") or mf_ref.istype("UHF"):
        eps_a, eps_b = _explicit_spin_pair(
            "mf_ref.mo_energy", mf_ref.mo_energy, allow_spin_axis=True,
        )
        return np.asarray(eps_a, dtype=float), np.asarray(eps_b, dtype=float)
    elif mf_ref.istype("ROKS") or mf_ref.istype("ROHF"):
        eps_roothaan = np.asarray(mf_ref.mo_energy, dtype=float)
        eps_a = np.asarray(getattr(mf_ref.mo_energy, "mo_ea", eps_roothaan), dtype=float)
        eps_b = np.asarray(getattr(mf_ref.mo_energy, "mo_eb", eps_roothaan), dtype=float)
        if eps_a.shape != eps_roothaan.shape or eps_b.shape != eps_roothaan.shape:
            raise ValueError("Tagged RO orbital energies have inconsistent shapes.")
        return eps_a.copy(), eps_b.copy()
    else:
        assert mf_ref.istype("RKS") or mf_ref.istype("RHF")
        eps = np.asarray(mf_ref.mo_energy, dtype=float)
        return eps


def write_restricted_molden(
    filename: str | Path,
    mol,
    coeff: np.ndarray,
    eps:   np.ndarray,
    occ:   np.ndarray,
) -> Path:
    """Write a Molden file with a single 0/2-occupied orbital section."""
    coeff = np.asarray(coeff)
    eps = np.asarray(eps, dtype=float)
    occ = _binary_spin_occupations_closed_shell("occupations", occ)
    _validate_explicit_shapes(coeff, occ, eps)

    path = Path(filename)
    with open(path, "w", encoding="utf-8") as fh:
        molden.header(mol, fh)
        molden.orbital_coeff(mol, fh, coeff, spin="Alpha", ene=eps, occ=occ)
    return path


def write_unrestricted_molden(
    filename: str | Path,
    mol,
    coeff_a: np.ndarray,
    coeff_b: np.ndarray,
    eps_a:   np.ndarray,
    eps_b:   np.ndarray,
    occ_a:   np.ndarray,
    occ_b:   np.ndarray,
) -> Path:
    """Write a Molden file with explicit alpha and beta orbital sections."""
    coeff_a = np.asarray(coeff_a)
    coeff_b = np.asarray(coeff_b)
    eps_a   = np.asarray(eps_a, dtype=float)
    eps_b   = np.asarray(eps_b, dtype=float)
    occ_a, occ_b = _binary_spin_occupations("occupations", (occ_a, occ_b))
    _validate_explicit_spin_shapes(coeff_a, coeff_b, occ_a, occ_b, eps_a, eps_b)

    path = Path(filename)
    with open(path, "w", encoding="utf-8") as fh:
        molden.header(mol, fh)
        molden.orbital_coeff(mol, fh, coeff_a, spin="Alpha", ene=eps_a, occ=occ_a)
        molden.orbital_coeff(mol, fh, coeff_b, spin="Beta",  ene=eps_b, occ=occ_b)
    return path


def write_restricted_molden_from_reference(filename: str | Path, mf_ref) -> Path:
    """Write an RHF/RKS reference to restricted-style Molden format."""
    coeff, occ = extract_spin_orbitals_and_occupations(mf_ref)
    eps = extract_spin_orbital_energies(mf_ref)
    return write_restricted_molden(
        filename, mf_ref.mol, coeff, eps, occ
    )


def write_unrestricted_molden_from_reference(filename: str | Path, mf_ref) -> Path:
    """Write a PySCF reference to unrestricted-style Molden format."""
    coeff_a, coeff_b, occ_a, occ_b = extract_spin_orbitals_and_occupations(mf_ref)
    eps_a, eps_b = extract_spin_orbital_energies(mf_ref)
    return write_unrestricted_molden(
        filename, mf_ref.mol, coeff_a, coeff_b, eps_a, eps_b, occ_a, occ_b,
    )
