"""Every variant against its Q-Chem reference JSON (skipped if absent):
E_HF, E_S^(2) and Tr W2 to ETOL, Tr W3 to W3_ETOL (Hartree)."""
from __future__ import annotations
import pytest
from conftest import (
    ALL_PARAMS, CORR_PARAMS,
    Variant, load_reference,
)

ETOL    = 1e-5
W3_ETOL = 1e-4


def _ref(v: Variant) -> dict:
    r = load_reference(v.mol, v.functional, v.ref_tag, v.basis)
    if r is None:
        pytest.skip(
            f"no reference: {v.mol}_{v.functional}_{v.basis}_{v.ref_tag}.json"
        )
    return r


def _fmt(v, field, ours, ref):
    return f"{v.id} {field}: ours={ours:.10f} ref={ref:.10f} diff={abs(ours-ref):.2e}"



@pytest.mark.reference
@pytest.mark.parametrize("v", ALL_PARAMS)
class TestReference:

    def test_e_hf(self, v, run):
        ref = _ref(v)
        result, _ = run(v)
        our = result.energies[v.energy_keys["e_hf"]]
        assert abs(our - ref["e_hf"]) < ETOL, _fmt(v, "e_hf", our, ref["e_hf"])

    def test_e_nbs(self, v, run):
        ref = _ref(v)
        result, _ = run(v)
        our = result.energies[v.energy_keys["e_nbs"]]
        assert abs(our - ref["e_nbs"]) < ETOL, _fmt(v, "e_nbs", our, ref["e_nbs"])

    def test_json_internal_consistency(self, v):
        ref = _ref(v)
        assert abs(ref["tr_w2_ss"] + ref["tr_w2_os"] - ref["tr_w2"]) < 1e-10, (
            f"{v.id}: tr_w2_ss+tr_w2_os={ref['tr_w2_ss']+ref['tr_w2_os']:.12f} "
            f"!= tr_w2={ref['tr_w2']:.12f}"
        )


@pytest.mark.reference
@pytest.mark.parametrize("v", CORR_PARAMS)
class TestReferenceCorr:

    def test_tr_w2_ss(self, v, run):
        ref = _ref(v)
        result, _ = run(v)
        our = result.energies[v.energy_keys["tr_w2_ss"]]
        assert abs(our - ref["tr_w2_ss"]) < ETOL, _fmt(v, "tr_w2_ss", our, ref["tr_w2_ss"])

    def test_tr_w2_os(self, v, run):
        ref = _ref(v)
        result, _ = run(v)
        our = result.energies[v.energy_keys["tr_w2_os"]]
        assert abs(our - ref["tr_w2_os"]) < ETOL, _fmt(v, "tr_w2_os", our, ref["tr_w2_os"])

    def test_tr_w2(self, v, run):
        ref = _ref(v)
        result, _ = run(v)
        our = result.energies[v.energy_keys["tr_w2"]]
        assert abs(our - ref["tr_w2"]) < ETOL, _fmt(v, "tr_w2", our, ref["tr_w2"])


@pytest.mark.reference
@pytest.mark.parametrize("v", CORR_PARAMS)
class TestReferenceCorrW3:

    def test_tr_w3(self, v, run):
        ref = _ref(v)
        result, _ = run(v)
        our = result.energies[v.energy_keys["tr_w3"]]
        assert abs(our - ref["tr_w3"]) < W3_ETOL, _fmt(v, "tr_w3", our, ref["tr_w3"])
