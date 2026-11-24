"""Every variant: completion, output files, array shapes, energy identities,
closed-shell vs unrestricted, density fitting vs exact, frozen core."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from conftest import (
    ALL_PARAMS, CORR_PARAMS, CS_CORR_PARAMS, CS_PARAMS,
    DF_CORR_PARAMS, FC_CORR_PARAMS, FC_PARAMS, FIELD_CORR_PARAMS,
    FIELD_PARAMS, Variant,
)

# A closed-shell singlet and its unrestricted twin solve the same equations in
# different spin bases, so they agree to SCF-convergence precision, not to
# machine precision.
CS_ETOL = 1e-7
from pyscf_w_matrices import W2_LABELS, W3_LABELS
from workflow_common import ResultFileKey


@pytest.mark.parametrize("v", ALL_PARAMS)
class TestIntegration:

    def test_completes(self, v, run):
        run(v)

    def test_final_summary_written(self, v, run):
        result, _ = run(v)
        assert Path(result.files[ResultFileKey.FINAL_SUMMARY]).is_file()

    def test_pyscf_log_written(self, v, run):
        result, _ = run(v)
        assert Path(result.files[ResultFileKey.PYSCF_DFT_OUTPUT]).is_file()

    def test_semicanonical_report_written(self, v, run):
        result, _ = run(v)
        assert Path(result.files[ResultFileKey.SEMICANONICAL_OUTPUT]).is_file()

    def test_w_matrix_npy_files_written(self, v, run):
        result, _ = run(v)
        for lbl in W2_LABELS + W3_LABELS:
            assert Path(result.w_matrix_files[lbl]).is_file(), f"missing {lbl}"

    def test_semicanonical_arrays_written(self, v, run):
        _, paths = run(v)
        for quantity in ("coeff", "occ", "fock"):
            for s in v.semicanonical_suffixes:
                lbl = f"{quantity}{s}"
                assert paths.semicanonical_array(lbl).is_file(), f"missing {lbl}"

    def test_coeff_matrices_are_2d(self, v, run):
        _, paths = run(v)
        for s in v.semicanonical_suffixes:
            assert np.load(paths.semicanonical_array(f"coeff{s}")).ndim == 2

    def test_coeff_blocks_share_nao(self, v, run):
        _, paths = run(v)
        naos = {np.load(paths.semicanonical_array(f"coeff{s}")).shape[0]
                for s in v.semicanonical_suffixes}
        assert len(naos) == 1

    def test_occupations_binary(self, v, run):
        _, paths = run(v)
        allowed = v.occupation_values
        for s in v.semicanonical_suffixes:
            occ = np.load(paths.semicanonical_array(f"occ{s}"))
            assert occ.ndim == 1
            assert np.all(np.isin(occ, allowed)), \
                f"occ{s} outside {allowed}: {sorted(set(occ.tolist()))}"

    def test_fock_matrices_square(self, v, run):
        _, paths = run(v)
        for s in v.semicanonical_suffixes:
            f = np.load(paths.semicanonical_array(f"fock{s}"))
            assert f.ndim == 2 and f.shape[0] == f.shape[1]

    def test_coeff_occ_consistent(self, v, run):
        _, paths = run(v)
        for s in v.semicanonical_suffixes:
            c   = np.load(paths.semicanonical_array(f"coeff{s}"))
            occ = np.load(paths.semicanonical_array(f"occ{s}"))
            assert c.shape[1] == occ.size

    def test_spin_excess_matches_multiplicity(self, v, run):
        from conftest import MOLECULES
        _, paths = run(v)
        na, nb = v.occupied_counts(paths)
        assert na - nb == MOLECULES[v.mol]["mult"] - 1

    def test_tr_w2_spin_decomposition(self, v, run):
        result, _ = run(v)
        e = result.energies
        assert abs(
            e["MP2 doubles correlation (same-spin)"]
            + e["MP2 doubles correlation (opposite-spin)"]
            - e["MP2 doubles correlation"]
        ) < 1e-12

    def test_all_energies_finite(self, v, run):
        result, _ = run(v)
        for k, val in result.energies.items():
            assert np.isfinite(val), f"non-finite {k}"

    def test_mp2_corr_eq_doubles_plus_nbs(self, v, run):
        result, _ = run(v)
        e = result.energies
        assert abs(
            e["MP2 doubles correlation"] + e["Non-Brillouin singles correction"]
            - e["MP2 correlation"]
        ) < 1e-12


def test_he_same_spin_w2_zero(run):
    result, _ = run(Variant("he"))
    assert abs(result.energies["MP2 doubles correlation (same-spin)"]) < 1e-12


@pytest.mark.parametrize("v", CORR_PARAMS)
class TestCorrIntegration:

    def test_mp2_total_below_hf(self, v, run):
        """MP2 is not above HF by more than numerical noise (1e-10 Eh)."""
        result, _ = run(v)
        e = result.energies
        e_mp2, e_hf = e["MP2 total energy"], e["Reference energy (HF on KS density)"]
        assert e_mp2 <= e_hf + 1e-10, f"MP2 {e_mp2:.12f} above HF {e_hf:.12f}"


@pytest.mark.parametrize("v", CS_PARAMS)
class TestClosedShellIntegration:
    """--closed-shell must take the restricted path and reproduce its twin."""

    def test_restricted_reference_used(self, v, run):
        _, paths = run(v)
        header = paths.semicanonical_output.read_text()
        assert "RHF Fock from reference density" in header

    def test_reference_class_is_restricted(self, v, run):
        _, paths = run(v)
        line = next(l for l in paths.semicanonical_output.read_text().splitlines()
                    if "Reference class" in l)
        cls = line.split(":")[1].strip()
        assert cls.endswith(("RKS", "RHF")), cls
        assert "UKS" not in cls and "UHF" not in cls
        assert "ROKS" not in cls and "ROHF" not in cls

    def test_single_spatial_orbital_set_saved(self, v, run):
        _, paths = run(v)
        assert paths.semicanonical_array("coeff").is_file()
        assert not paths.semicanonical_array("coeff_a").is_file()

    def test_hf_energy_matches_unrestricted_twin(self, v, run):
        result_cs, _ = run(v)
        result_u,  _ = run(v.unrestricted_twin())
        assert abs(
            result_cs.energies["Reference energy (HF on KS density)"]
            - result_u.energies["Reference energy (HF on KS density)"]
        ) < CS_ETOL

    def test_nbs_matches_unrestricted_twin(self, v, run):
        result_cs, _ = run(v)
        result_u,  _ = run(v.unrestricted_twin())
        assert abs(
            result_cs.energies["Non-Brillouin singles correction"]
            - result_u.energies["Non-Brillouin singles correction"]
        ) < CS_ETOL


@pytest.mark.parametrize("v", CS_CORR_PARAMS)
class TestClosedShellCorrIntegration:

    @pytest.mark.parametrize("key", [
        "MP2 doubles correlation (same-spin)",
        "MP2 doubles correlation (opposite-spin)",
        "MP2 doubles correlation",
        "MP3 correlation",
        "MP2 total energy",
        "MP3 total energy",
    ])
    def test_energy_matches_unrestricted_twin(self, v, key, run):
        """An error in the closed-shell W assembly is only caught here."""
        result_cs, _ = run(v)
        result_u,  _ = run(v.unrestricted_twin())
        diff = abs(result_cs.energies[key] - result_u.energies[key])
        assert diff < CS_ETOL, f"{v.id} {key}: diff={diff:.3e}"


@pytest.mark.parametrize("v", FIELD_PARAMS)
class TestFieldIntegration:

    def test_field_hf_differs_from_bare(self, v, run):
        result_f, _ = run(v)
        result_0, _ = run(v.bare())
        assert abs(
            result_f.energies["Reference energy (HF on KS density)"]
            - result_0.energies["Reference energy (HF on KS density)"]
        ) > 1e-8


@pytest.mark.parametrize("v", FIELD_CORR_PARAMS)
class TestFieldCorrIntegration:

    def test_field_tr_w2_differs_from_bare(self, v, run):
        result_f, _ = run(v)
        result_0, _ = run(v.bare())
        assert abs(
            result_f.energies["MP2 doubles correlation"]
            - result_0.energies["MP2 doubles correlation"]
        ) > 1e-10


@pytest.mark.parametrize("v", DF_CORR_PARAMS)
class TestDfCorrIntegration:

    def test_df_and_conv_tr_w2_agree(self, v, run):
        result_df,   _ = run(v)
        result_conv, _ = run(v.bare())
        assert abs(
            result_df.energies["MP2 doubles correlation"]
            - result_conv.energies["MP2 doubles correlation"]
        ) < 1e-4

    def test_df_and_conv_tr_w3_agree(self, v, run):
        result_df,   _ = run(v)
        result_conv, _ = run(v.bare())
        assert abs(
            result_df.energies["MP3 correlation"]
            - result_conv.energies["MP3 correlation"]
        ) < 1e-4


@pytest.mark.parametrize("v", FC_PARAMS)
class TestFcIntegration:

    def test_frozen_core_count_positive(self, v, run):
        result, _ = run(v)
        assert result.ncore > 0


@pytest.mark.parametrize("v", FC_CORR_PARAMS)
class TestFcCorrIntegration:

    def test_fc_tr_w2_smaller_than_full(self, v, run):
        result_fc,   _ = run(v)
        result_full, _ = run(v.bare())
        assert abs(result_fc.energies["MP2 doubles correlation"]) < abs(
            result_full.energies["MP2 doubles correlation"]
        )


class TestFrozenCoreCount:
    """--frozen-core fc freezes one orbital per first-row atom."""

    def test_h2o_fc_count_is_one(self, run):
        result, _ = run(Variant("h2o", "pbe0", frozen_core="fc"))
        assert result.ncore == 1

    def test_cn_fc_count_is_two(self, run):
        result, _ = run(Variant("cn", "pbe0", frozen_core="fc"))
        assert result.ncore == 2

    def test_ch2_fc_count_is_one(self, run):
        result, _ = run(Variant("ch2", "pbe0", frozen_core="fc"))
        assert result.ncore == 1

    def test_ch3_fc_count_is_one(self, run):
        result, _ = run(Variant("ch3", "pbe0", frozen_core="fc"))
        assert result.ncore == 1

    def test_hco_fc_count_is_two(self, run):
        result, _ = run(Variant("hco", "pbe0", frozen_core="fc"))
        assert result.ncore == 2


class TestWMatrixShapes:
    """With frozen core, every W matrix is (nact, nact) for its spin: the
    frozen orbitals are dropped, not padded."""

    @pytest.mark.parametrize("v", FC_CORR_PARAMS)
    def test_w_shape_active_occ(self, v, run):
        result, paths = run(v)
        nocc_a, nocc_b = v.occupied_counts(paths)
        nact = {"alpha": nocc_a - result.ncore, "beta": nocc_b - result.ncore}
        for label in W2_LABELS + W3_LABELS:
            n = nact["alpha" if "alpha" in label else "beta"]
            w = np.load(paths.w_matrix(label))
            assert w.shape == (n, n), f"{label} shape {w.shape} != ({n},{n})"
