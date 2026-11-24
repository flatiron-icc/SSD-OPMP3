"""--source molden: a restart from a run's own Molden reproduces its traces;
closed-shell Molden files are accepted in both layouts, spin-polarized ones
rejected."""
from __future__ import annotations
from pathlib import Path
import pytest
from conftest import MOLDEN_VARIANTS, Variant
from pyscf_w_matrices import W2_LABELS, W3_LABELS
from workflow_common import ResultFileKey

ETOL = 1e-6


@pytest.mark.parametrize("v", MOLDEN_VARIANTS)
class TestMoldenRoundtrip:

    def test_tr_w2_ss_roundtrip(self, v, run, molden_run):
        r_ps, _ = run(v)
        r_ml, _ = molden_run(v)
        assert abs(
            r_ps.energies["MP2 doubles correlation (same-spin)"]
            - r_ml.energies["MP2 doubles correlation (same-spin)"]
        ) < ETOL

    def test_tr_w2_os_roundtrip(self, v, run, molden_run):
        r_ps, _ = run(v)
        r_ml, _ = molden_run(v)
        assert abs(
            r_ps.energies["MP2 doubles correlation (opposite-spin)"]
            - r_ml.energies["MP2 doubles correlation (opposite-spin)"]
        ) < ETOL

    def test_tr_w3_roundtrip(self, v, run, molden_run):
        r_ps, _ = run(v)
        r_ml, _ = molden_run(v)
        assert abs(
            r_ps.energies["MP3 correlation"]
            - r_ml.energies["MP3 correlation"]
        ) < ETOL

    def test_nbs_roundtrip(self, v, run, molden_run):
        r_ps, _ = run(v)
        r_ml, _ = molden_run(v)
        assert abs(
            r_ps.energies["Non-Brillouin singles correction"]
            - r_ml.energies["Non-Brillouin singles correction"]
        ) < ETOL

    def test_output_files_written(self, v, molden_run):
        r_ml, _ = molden_run(v)
        assert Path(r_ml.files[ResultFileKey.FINAL_SUMMARY]).is_file()
        assert Path(r_ml.files[ResultFileKey.SEMICANONICAL_OUTPUT]).is_file()
        for lbl in W2_LABELS + W3_LABELS:
            assert Path(r_ml.w_matrix_files[lbl]).is_file()

    def test_no_qchem_files_written(self, v, molden_run):
        r_ml, _ = molden_run(v)
        assert ResultFileKey.QCHEM_INPUT  not in r_ml.files
        assert ResultFileKey.QCHEM_OUTPUT not in r_ml.files


class TestClosedShellMoldenPayload:
    """--closed-shell accepts one 0/2-occupied orbital set or two spin blocks
    with the same occupied space, and rejects a spin-polarized file."""

    @staticmethod
    def _payload(molden_file):
        from pyscf.tools import molden as pyscf_molden
        from pyscf_orbitals import require_closed_shell_molden_payload
        mol, eps, coeff, occ, _, _ = pyscf_molden.load(str(molden_file))
        return require_closed_shell_molden_payload(mol, eps, coeff, occ)

    def test_unrestricted_singlet_molden_accepted(self, run):
        """Two spin blocks describing one closed-shell determinant."""
        result, _ = run(Variant("h2o"))
        _, coeff, occ = self._payload(result.files[ResultFileKey.PYSCF_DFT_MOLDEN])
        assert coeff.ndim == 2
        assert set(occ.tolist()) <= {0.0, 2.0}
        assert int(occ.sum()) == 10          # H2O has 10 electrons

    def test_closed_shell_molden_accepted(self, run):
        """One orbital set with 0/2 occupations."""
        result, _ = run(Variant("h2o", closed_shell=True))
        _, coeff, occ = self._payload(result.files[ResultFileKey.PYSCF_DFT_MOLDEN])
        assert coeff.ndim == 2
        assert set(occ.tolist()) <= {0.0, 2.0}
        assert int(occ.sum()) == 10

    def test_open_shell_molden_rejected_on_electron_count(self, run):
        """CN is a doublet: nocc_alpha != nocc_beta."""
        result, _ = run(Variant("cn"))
        with pytest.raises(ValueError, match="not closed shell"):
            self._payload(result.files[ResultFileKey.PYSCF_DFT_MOLDEN])

    def test_spin_polarised_molden_rejected_on_overlap(self, run, tmp_path):
        """Beta HOMO and LUMO swapped: equal counts, different occupied spaces."""
        import numpy as np
        from pyscf.tools import molden as pyscf_molden
        from pyscf_orbitals import write_unrestricted_molden

        result, _ = run(Variant("h2o"))
        src = result.files[ResultFileKey.PYSCF_DFT_MOLDEN]
        mol, eps, coeff, occ, _, _ = pyscf_molden.load(str(src))
        ea, eb = np.asarray(eps[0], float), np.asarray(eps[1], float).copy()
        ca, cb = np.asarray(coeff[0]), np.asarray(coeff[1]).copy()
        oa, ob = np.asarray(occ[0], float), np.asarray(occ[1], float)

        n = int(round(ob.sum()))              # beta HOMO = n-1, LUMO = n
        cb[:, [n - 1, n]] = cb[:, [n, n - 1]]
        eb[[n - 1, n]]    = eb[[n, n - 1]]

        swapped = tmp_path / "beta_swapped.molden"
        write_unrestricted_molden(swapped, mol, ca, cb, ea, eb, oa, ob)

        with pytest.raises(ValueError, match="span different spaces"):
            self._payload(swapped)

    def test_rotated_beta_copy_accepted(self, run, tmp_path):
        """Beta occupied orbitals rotated among themselves and one sign
        flipped: the same determinant, so accepted."""
        import numpy as np
        from pyscf.tools import molden as pyscf_molden
        from pyscf_orbitals import write_unrestricted_molden

        result, _ = run(Variant("h2o"))
        src = result.files[ResultFileKey.PYSCF_DFT_MOLDEN]
        mol, eps, coeff, occ, _, _ = pyscf_molden.load(str(src))
        ca, cb = np.asarray(coeff[0]), np.asarray(coeff[1]).copy()
        n = int(round(np.asarray(occ[1]).sum()))
        c, s = np.cos(0.3), np.sin(0.3)
        cb[:, [0, n - 1]] = cb[:, [0, n - 1]] @ np.array([[c, -s], [s, c]])
        cb[:, 1] *= -1.0

        rotated = tmp_path / "beta_rotated.molden"
        write_unrestricted_molden(rotated, mol, ca, cb, np.asarray(eps[0], float),
                                  np.asarray(eps[1], float), np.asarray(occ[0], float),
                                  np.asarray(occ[1], float))
        _, coeff_cs, occ_cs = self._payload(rotated)
        assert int(occ_cs.sum()) == 10


class TestMoldenEcpDefault:
    """Without --basis, --source molden and molden-direct attach the same
    ECP (the cc-* family) to the elements beyond Kr of a Molden file."""

    def test_both_molden_modes_attach_the_same_ecp(self, tmp_path):
        from pyscf import gto, scf
        from pyscf.tools import molden as pyscf_molden
        from conftest import MEMORY, THREADS
        from pyscf_dft import dft_from_molden, reference_from_molden
        mol = gto.M(atom="H 0 0 0; I 0 0 1.61", basis="def2-svp",
                    ecp="def2-svp", verbose=0)
        src = tmp_path / "hi.molden"
        pyscf_molden.from_scf(scf.RHF(mol).run(), str(src))
        refit = dft_from_molden(src, functional="hf", threads=THREADS, memory=MEMORY,
                                restricted=False, closed_shell=True)
        direct = reference_from_molden(src, threads=THREADS, memory=MEMORY,
                                       closed_shell=True)
        for m in (refit.mol, direct.mol):
            assert sum(m.atom_nelec_core(i) for i in range(m.natm)) == 28
            assert m.nelectron == mol.nelectron and m.charge == 0
        assert refit.mol._ecp == direct.mol._ecp


class TestMoldenErrorHandling:

    def test_single_spin_molden_rejected(self, tmp_path):
        from pyscf import dft, gto
        from pyscf.tools import molden as pyscf_molden
        from conftest import make_config
        from workflow import run_one_workflow
        mol = gto.Mole()
        mol.atom    = "H 0 0 0"
        mol.basis   = "sto-3g"
        mol.spin    = 1
        mol.verbose = 0
        mol.build()
        mf = dft.UKS(mol); mf.xc = "pbe0"; mf.verbose = 0; mf.kernel()
        mfile = tmp_path / "alpha_only.molden"
        with open(mfile, "w") as fh:
            pyscf_molden.header(mol, fh)
            pyscf_molden.orbital_coeff(mol, fh, mf.mo_coeff[0],
                                       ene=mf.mo_energy[0], occ=mf.mo_occ[0])
        with pytest.raises((ValueError, RuntimeError)):
            run_one_workflow(make_config(mfile, source="molden"))
