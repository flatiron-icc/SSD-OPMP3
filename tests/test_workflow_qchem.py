"""--source qchem against --source pyscf for H2O/PBE0 (needs qchem on PATH)."""
from __future__ import annotations
import pytest
from conftest import Variant

pytestmark = pytest.mark.qchem


@pytest.fixture(scope="module")
def h2o_qchem(run):
    return run(Variant("h2o", "pbe0"))


class TestQchemVsPyscf:

    def test_e_hf(self, h2o_qchem, pyscf_run, etol):
        r_qc, _ = h2o_qchem
        r_ps, _ = pyscf_run(Variant("h2o", "pbe0"))
        assert abs(r_qc.energies["Reference energy (HF on KS density)"]
                   - r_ps.energies["Reference energy (HF on KS density)"]) < etol

    def test_e_nbs(self, h2o_qchem, pyscf_run, etol):
        r_qc, _ = h2o_qchem
        r_ps, _ = pyscf_run(Variant("h2o", "pbe0"))
        assert abs(r_qc.energies["Non-Brillouin singles correction"]
                   - r_ps.energies["Non-Brillouin singles correction"]) < etol

    def test_tr_w2_ss(self, h2o_qchem, pyscf_run, etol):
        r_qc, _ = h2o_qchem
        r_ps, _ = pyscf_run(Variant("h2o", "pbe0"))
        assert abs(r_qc.energies["MP2 doubles correlation (same-spin)"]
                   - r_ps.energies["MP2 doubles correlation (same-spin)"]) < etol

    def test_tr_w2_os(self, h2o_qchem, pyscf_run, etol):
        r_qc, _ = h2o_qchem
        r_ps, _ = pyscf_run(Variant("h2o", "pbe0"))
        assert abs(r_qc.energies["MP2 doubles correlation (opposite-spin)"]
                   - r_ps.energies["MP2 doubles correlation (opposite-spin)"]) < etol

    def test_tr_w2(self, h2o_qchem, pyscf_run, etol):
        r_qc, _ = h2o_qchem
        r_ps, _ = pyscf_run(Variant("h2o", "pbe0"))
        assert abs(r_qc.energies["MP2 doubles correlation"]
                   - r_ps.energies["MP2 doubles correlation"]) < etol

    def test_tr_w3(self, h2o_qchem, pyscf_run, etol):
        r_qc, _ = h2o_qchem
        r_ps, _ = pyscf_run(Variant("h2o", "pbe0"))
        assert abs(r_qc.energies["MP3 correlation"]
                   - r_ps.energies["MP3 correlation"]) < etol
