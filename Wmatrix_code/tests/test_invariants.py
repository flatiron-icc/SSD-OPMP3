"""Orbital-invariant checks of the W matrices and the resummed energies.

The elements of a W matrix depend on the phases and degenerate rotations of
the semicanonical occupied orbitals, so runs are compared through
invariants only: the singular values and trace of each of the eight blocks,
and the OP-MP3, SOP-MP3 and SSD-OPMP3 energies.

- each block against an independent spin-orbital evaluation of Eqs. 7, 9
  and 11a (reference_w.py) from the run's own orbitals, for one variant per
  code path (U, RO, closed shell, frozen core, DF, field, empty beta block,
  ECP);
- each closed-shell run against its unrestricted twin;
- --source molden, chk and molden-direct against the run whose orbitals
  they read;
- the x_ss of the summary's SSD-OPMP3 energy (ssd_x_ss of the functional);
- both <ab||cd> paths of mp3, AO-direct (the default when it fits) and
  stored vvvv (the fallback), against reference_w (singular values and
  trace of each block), including a spin without virtual orbitals, at
  default and at minimal memory.
"""
from __future__ import annotations

import numpy as np
import pytest
from pyscf import df as pyscf_df
from pyscf import dft, gto
from pyscf.scf.chkfile import load_mol

from conftest import (
    ALL_PARAMS, ALL_VARIANTS, AUXBASIS, CS_CORR_PARAMS, Variant,
    resummed_totals, singular_values, w_matrices,
)
import mp3
from pyscf_dft import handle_basis_core
from pyscf_w_matrices import run_pyscf_w_matrices
from reference_w import independent_w, resummed
from semicanonicalize import semicanonicalize
from workflow_common import W2_LABELS, W3_LABELS, derive_output_paths, ssd_x_ss

LABELS = W2_LABELS + W3_LABELS

REF_TOL  = 1e-10   # same orbitals: only floating-point differences
TWIN_TOL = 1e-7    # closed shell vs unrestricted: SCF convergence (test_integration.CS_ETOL)
SRC_TOL  = 1e-6    # restarts and Molden precision (test_sources.ETOL, test_molden.ETOL)

_BY_ID = {v.id: v for v in ALL_VARIANTS}
_REF_IDS = (
    "h2o-pbe0-U", "h2o-pbe0-U-CS", "h2o-hf-U-CS-fcfc", "h2o-pbe0-U-CS-df",
    "cn-pbe0-RO", "cn-pbe0-U-fcfc", "cn-hf-U-df", "ch2-hf-U",
    "ch3-pbe0-RO-df", "hco-pbe0-RO-field", "he-pbe0-U-CS",
    "he-triplet-hf-U-cc-pvtz", "he-triplet-pbe0-RO-cc-pvtz",
    "ag+-hf-U-def2-svp", "ag+-pbe0-U-CS-cc-pvdz-pp",
)
REF_PARAMS = [pytest.param(_BY_ID[i], id=i) for i in _REF_IDS]

_independent_cache: dict[Variant, dict[str, np.ndarray]] = {}


def _independent(v: Variant, result, paths) -> dict[str, np.ndarray]:
    """reference_w.independent_w for the orbitals the run saved."""
    if v not in _independent_cache:
        mol = load_mol(str(paths.pyscf_chk))
        if v.closed_shell:
            c = np.load(paths.semicanonical_array("coeff"))
            n = int(round(np.load(paths.semicanonical_array("occ")).sum() / 2))
            e = np.diag(np.load(paths.semicanonical_array("fock")))
            coeff, nocc, eps = (c, c), (n, n), (e, e)
        else:
            coeff = tuple(np.load(paths.semicanonical_array(f"coeff_{s}")) for s in "ab")
            nocc = tuple(int(round(np.load(paths.semicanonical_array(f"occ_{s}")).sum())) for s in "ab")
            eps = tuple(np.diag(np.load(paths.semicanonical_array(f"fock_{s}"))) for s in "ab")
        if v.df:
            eri = pyscf_df.DF(mol, auxbasis=handle_basis_core(AUXBASIS, mol)).get_eri()
        else:
            eri = mol.intor("int2e", aosym="s8")
        _independent_cache[v] = independent_w(eri, coeff, nocc, eps, result.ncore)
    return _independent_cache[v]


def _correlation(result, v: Variant) -> dict[str, float]:
    """Resummed correlation energies of a summary: totals - E_HF - E_S^(2)."""
    e = result.energies
    base = e[v.energy_keys["e_hf"]] + e[v.energy_keys["e_nbs"]]
    return {k: t - base for k, t in resummed_totals(result).items()}


@pytest.mark.parametrize("v", REF_PARAMS)
class TestIndependentReference:

    @pytest.mark.parametrize("label", LABELS)
    def test_singular_values(self, v, label, run):
        result, paths = run(v)
        ours = singular_values(w_matrices(paths)[label])
        ref = singular_values(_independent(v, result, paths)[label])
        np.testing.assert_allclose(ours, ref, rtol=0, atol=REF_TOL, err_msg=f"{v.id} {label}")

    @pytest.mark.parametrize("label", LABELS)
    def test_trace(self, v, label, run):
        result, paths = run(v)
        ours = np.trace(w_matrices(paths)[label])
        ref = np.trace(_independent(v, result, paths)[label])
        assert abs(ours - ref) < REF_TOL, f"{v.id} {label}: {ours:.12f} vs {ref:.12f}"

    def test_resummed_energies(self, v, run):
        result, paths = run(v)
        ours = _correlation(result, v)
        ref = resummed(_independent(v, result, paths), ssd_x_ss(v.functional)[0])
        for model, e in ref.items():
            assert abs(ours[model] - e) < REF_TOL, f"{v.id} {model}: {ours[model]:.12f} vs {e:.12f}"


@pytest.mark.parametrize("v", ALL_PARAMS)
def test_summary_reports_x_ss(v, run):
    """The SSD-OPMP3 label and the 'SSD-OPMP3 x_ss' line give ssd_x_ss of the functional."""
    result, paths = run(v)
    x_ss, origin = ssd_x_ss(v.functional)
    assert f"SSD-OPMP3 total energy (x_ss = {x_ss:g})" in result.energies
    line = next(l for l in paths.final_output.read_text().splitlines()
                if l.startswith("SSD-OPMP3 x_ss"))
    assert line.split("=", 1)[1].strip() == f"{x_ss:g} ({origin})"


@pytest.mark.parametrize("v", CS_CORR_PARAMS)
class TestClosedShellTwin:
    """Beyond the traces of test_integration: every block and the resummed energies."""

    @pytest.mark.parametrize("label", LABELS)
    def test_singular_values(self, v, label, run):
        _, paths_cs = run(v)
        _, paths_u = run(v.unrestricted_twin())
        np.testing.assert_allclose(
            singular_values(w_matrices(paths_cs)[label]),
            singular_values(w_matrices(paths_u)[label]),
            rtol=0, atol=TWIN_TOL, err_msg=f"{v.id} {label}")

    def test_resummed_energies(self, v, run):
        cs, u = resummed_totals(run(v)[0]), resummed_totals(run(v.unrestricted_twin())[0])
        for model in cs:
            assert abs(cs[model] - u[model]) < TWIN_TOL, f"{v.id} {model}"


# The variants of test_sources; all PBE0, so molden-direct (default x_ss 0.5)
# and the run it reads (PBE0: 0.5) evaluate SSD-OPMP3 at the same x_ss.
# molden_run is the restart of test_molden, chk_run and direct_run those of
# test_sources.
_SRC_PARAMS = [
    pytest.param(Variant("h2o"), id="h2o-U"),
    pytest.param(Variant("h2o", closed_shell=True), id="h2o-CS"),
    pytest.param(Variant("he", closed_shell=True), id="he-CS"),
    pytest.param(Variant("cn", restricted=True), id="cn-RO"),
    pytest.param(Variant("ch2", restricted=True), id="ch2-RO"),
]


@pytest.mark.parametrize("mode", ["molden_run", "chk_run", "direct_run"])
@pytest.mark.parametrize("v", _SRC_PARAMS)
class TestSourceModes:
    """Beyond the traces of test_molden and test_sources: every block and the
    resummed energies."""

    def test_singular_values(self, v, mode, run, request):
        _, paths_base = run(v)
        _, paths_src = request.getfixturevalue(mode)(v)
        base, src = w_matrices(paths_base), w_matrices(paths_src)
        for label in LABELS:
            np.testing.assert_allclose(
                singular_values(src[label]), singular_values(base[label]),
                rtol=0, atol=SRC_TOL, err_msg=f"{v.id} {mode} {label}")

    def test_resummed_energies(self, v, mode, run, request):
        base = resummed_totals(run(v)[0])
        src = resummed_totals(request.getfixturevalue(mode)(v)[0])
        for model in base:
            assert abs(src[model] - base[model]) < SRC_TOL, f"{v.id} {mode} {model}"


_H2O = "O 0 0 0; H 0 0 0.9579; H 0.9289588892 0 -0.2336831018"      # tests/data/h2o.xyz
_CN = "C 0 0 0; N 0 0 1.17182"                                       # tests/data/cn.xyz


@pytest.mark.parametrize("lowmem", [False, True], ids=["memory-default", "memory-minimal"])
@pytest.mark.parametrize("atom, basis, spin, cls, ncore", [
    (_H2O, "cc-pvdz", 0, dft.RKS, 1), (_CN, "cc-pvdz", 1, dft.UKS, 2),
    ("Ne 0 0 0", "sto-3g", 0, dft.RKS, 0), ("N 0 0 0; H 0 0.1 1.04", "sto-3g", 4, dft.UKS, 0),
], ids=["h2o-closed-shell-fc", "cn-unrestricted-fc", "ne-no-virtuals", "nh-quintet-no-alpha-virtuals"])
def test_vvvv_paths(atom, basis, spin, cls, ncore, lowmem, tmp_path, monkeypatch):
    """Both <ab||cd> paths of mp3 (VVVV_DIRECT True and False) against
    reference_w.independent_w: singular values and trace of each block.  In
    STO-3G every Ne orbital, and every alpha orbital of the NH quintet, is
    occupied.  memory-minimal: --memory 1 and PySCF's CC minimum lowered to
    0.2 MB, so every block sized from the memory is at its smallest, and
    the integral rows of H2O and CN are read a few orbitals at a time."""
    if lowmem:
        monkeypatch.setattr(mp3.ccsd, "MEMORYMIN", 0.2)
    mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
    mf = cls(mol)
    mf.xc = "pbe0"
    mf.kernel()
    paths = derive_output_paths(tmp_path / "m.xyz")
    sc = semicanonicalize(mf, tmp_path / "m.xyz", ncore=ncore, output_paths=paths)
    pair = (lambda x: (x, x)) if sc.closed_shell else (lambda x: x)
    ref = independent_w(mol.intor("int2e", aosym="s8"), pair(sc.mo_coeff), pair(sc.nocc),
                        pair(sc.mo_energy), ncore)
    for direct in (True, False):
        monkeypatch.setattr(mp3, "VVVV_DIRECT", direct)
        run_pyscf_w_matrices(mol, sc, tmp_path / "m.xyz", ncore, memory=1 if lowmem else 16000,
                             output_paths=paths)
        w = w_matrices(paths)
        for label in LABELS:
            np.testing.assert_allclose(singular_values(w[label]), singular_values(ref[label]), rtol=0,
                                       atol=REF_TOL, err_msg=f"VVVV_DIRECT={direct} {label}")
            assert abs(np.trace(w[label]) - np.trace(ref[label])) < REF_TOL, f"VVVV_DIRECT={direct} {label}"
