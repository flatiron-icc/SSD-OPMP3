"""--source qchem against --source pyscf for H2O/PBE0 (needs qchem on PATH):
E_HF, E_S^(2), the traces, the singular values of every W block and the
resummed energies."""
from __future__ import annotations
import numpy as np
import pytest
from conftest import Variant, resummed_totals, singular_values, w_matrices
from workflow_common import W2_LABELS, W3_LABELS

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

    def test_singular_values(self, h2o_qchem, pyscf_run, etol):
        w_qc = w_matrices(h2o_qchem[1])
        w_ps = w_matrices(pyscf_run(Variant("h2o", "pbe0"))[1])
        for label in W2_LABELS + W3_LABELS:
            np.testing.assert_allclose(singular_values(w_qc[label]), singular_values(w_ps[label]),
                                       rtol=0, atol=etol, err_msg=label)

    def test_resummed_energies(self, h2o_qchem, pyscf_run, etol):
        qc = resummed_totals(h2o_qchem[0])
        ps = resummed_totals(pyscf_run(Variant("h2o", "pbe0"))[0])
        for model in qc:
            assert abs(qc[model] - ps[model]) < etol, model
