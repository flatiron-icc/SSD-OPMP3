"""Uniform static field E = (Ex, Ey, Ez), atomic units: E.r is added to the
core Hamiltonian and -sum_A Z_A E.R_A to the nuclear repulsion, both about
the origin.  Q-Chem's $multipole_field takes the opposite sign.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


ElectricField = tuple[float, float, float]

_ZERO_FIELD_TOL = 1.0e-15
_FIELD_LABELS   = ("X", "Y", "Z")


def parse_electric_field(electric_field: Iterable[float] | None) -> ElectricField | None:
    """Return a validated electric-field tuple or ``None``.

    ``None`` disables the field.  An input field whose three components are all
    numerically negligible is also treated as no field.
    """
    if electric_field is None:
        return None

    values = tuple(float(component) for component in electric_field)
    if len(values) != 3:
        raise ValueError("Electric field must have exactly three Cartesian components.")

    if max(abs(component) for component in values) <= _ZERO_FIELD_TOL:
        return None
    return values


def electric_field_to_string(electric_field: ElectricField | None) -> str:
    """Return a compact human-readable representation of the field."""
    if electric_field is None:
        return "none"
    return (
        f"({electric_field[0]:.12g}, {electric_field[1]:.12g}, "
        f"{electric_field[2]:.12g}) a.u."
    )


def qchem_multipole_field_block(electric_field: ElectricField | None) -> str:
    """Q-Chem ``$multipole_field`` block for the field, with the components
    negated for Q-Chem's opposite sign convention."""
    if electric_field is None:
        return ""

    lines = ["$multipole_field"]
    for label, value in zip(_FIELD_LABELS, electric_field):
        if not np.isclose(value, 0.0, atol=_ZERO_FIELD_TOL):
            lines.append(f"   {label} {-value:.16g}")
    lines.append("$end")
    return "\n".join(lines) + "\n"


def build_electric_field_hcore_shift(mol, electric_field: ElectricField) -> np.ndarray:
    """AO matrix of E.r, the field term added to the core Hamiltonian."""
    mol.set_common_orig((0.0, 0.0, 0.0))
    dipole_integrals = np.asarray(mol.intor("int1e_r", comp=3), dtype=float)
    shift = np.einsum(
        "x,xij->ij",
        np.asarray(electric_field, dtype=float),
        dipole_integrals,
        optimize=True,
    )
    return 0.5 * (shift + shift.T.conj())


def nuclear_electric_field_energy(mol, electric_field: ElectricField | None) -> float:
    """Return the nuclear contribution of a uniform static electric field."""
    if electric_field is None:
        return 0.0

    charges        = np.asarray(mol.atom_charges(), dtype=float)
    coords         = np.asarray(mol.atom_coords(unit="Bohr"), dtype=float)
    nuclear_dipole = np.einsum("a,ax->x", charges, coords, optimize=True)
    return float(-np.dot(np.asarray(electric_field, dtype=float), nuclear_dipole))


def attach_static_electric_field(mf, electric_field: ElectricField | None):
    """Attach a uniform static field to a PySCF mean-field object.

    Wraps ``get_hcore`` and ``energy_nuc`` in place so that the electronic
    and nuclear field contributions are both included.
    """
    if electric_field is None:
        return mf

    if getattr(mf, "_workflow_electric_field", None) == electric_field:
        return mf

    base_get_hcore  = getattr(mf, "_workflow_base_get_hcore",  mf.get_hcore)
    base_energy_nuc = getattr(mf, "_workflow_base_energy_nuc", mf.energy_nuc)
    reference_mol   = mf.mol
    cached_shift    = build_electric_field_hcore_shift(reference_mol, electric_field)
    cached_nuclear  = nuclear_electric_field_energy(reference_mol, electric_field)

    def get_hcore(mol=None):
        target = reference_mol if mol is None else mol
        if target is reference_mol:
            shift = cached_shift
        else:
            shift = build_electric_field_hcore_shift(target, electric_field)
        return base_get_hcore(target) + shift

    def energy_nuc():
        return base_energy_nuc() + cached_nuclear

    mf._workflow_base_get_hcore  = base_get_hcore
    mf._workflow_base_energy_nuc = base_energy_nuc
    mf._workflow_electric_field  = electric_field
    mf.get_hcore  = get_hcore
    mf.energy_nuc = energy_nuc
    return mf
