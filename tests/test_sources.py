"""--source chk and --source molden-direct must reproduce the run whose
checkpoint or Molden file they read, unrestricted, restricted open-shell
and closed-shell."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from conftest import Variant
from workflow_common import ResultFileKey

ETOL = 1e-6

_ENERGY_KEYS = (
    "Non-Brillouin singles correction",
    "MP2 doubles correlation (same-spin)",
    "MP2 doubles correlation (opposite-spin)",
    "MP2 doubles correlation",
    "MP3 correlation",
)

# molden-direct labels its reference energy by what it was built from.
_HF_KEYS = (
    "Reference energy (HF on KS density)",
    "Reference energy (HF on Molden density)",
)

# h2o exercises both spin paths; he is the smallest correlated closed shell;
# the RO variants carry --restricted through, which in molden-direct means the
# file is checked for being a restricted open-shell determinant.
_PARAMS = [
    pytest.param(Variant("h2o"), id="h2o-U"),
    pytest.param(Variant("h2o", closed_shell=True), id="h2o-CS"),
    pytest.param(Variant("he", closed_shell=True), id="he-CS"),
    pytest.param(Variant("cn", restricted=True), id="cn-RO"),
    pytest.param(Variant("ch2", restricted=True), id="ch2-RO"),
]


def _hf_energy(result):
    for key in _HF_KEYS:
        if key in result.energies:
            return result.energies[key]
    raise KeyError(f"no reference energy in {sorted(result.energies)}")


@pytest.mark.parametrize("v", _PARAMS)
class TestChkRestart:
    """--source chk must reproduce the run whose checkpoint it reads."""

    def test_completes(self, v, chk_run):
        result, _ = chk_run(v)
        assert result.source == "chk"

    def test_hf_energy_matches(self, v, run, chk_run):
        base, _ = run(v)
        chk,  _ = chk_run(v)
        assert abs(_hf_energy(base) - _hf_energy(chk)) < ETOL

    @pytest.mark.parametrize("key", _ENERGY_KEYS)
    def test_energy_matches(self, v, key, run, chk_run):
        base, _ = run(v)
        chk,  _ = chk_run(v)
        assert abs(base.energies[key] - chk.energies[key]) < ETOL, \
            f"{v.id} {key}"


@pytest.mark.parametrize("v", _PARAMS)
class TestMoldenDirect:
    """--source molden-direct uses the Molden orbitals as written, without an SCF."""

    def test_completes(self, v, direct_run):
        result, _ = direct_run(v)
        assert result.source == "molden-direct"

    def test_hf_energy_matches(self, v, run, direct_run):
        base,   _ = run(v)
        direct, _ = direct_run(v)
        assert abs(_hf_energy(base) - _hf_energy(direct)) < ETOL

    @pytest.mark.parametrize("key", _ENERGY_KEYS)
    def test_energy_matches(self, v, key, run, direct_run):
        base,   _ = run(v)
        direct, _ = direct_run(v)
        assert abs(base.energies[key] - direct.energies[key]) < ETOL, \
            f"{v.id} {key}"

    def test_reference_class_follows_flags(self, v, direct_run):
        """ROHF with --restricted, RHF with --closed-shell, UHF otherwise."""
        _, paths = direct_run(v)
        line = next(l for l in paths.semicanonical_output.read_text().splitlines()
                    if "Reference class" in l)
        cls = line.split(":")[1].strip()
        expected = "RHF" if v.closed_shell else ("ROHF" if v.restricted else "UHF")
        assert cls == expected, cls

    def test_no_scf_is_run(self, v, direct_run):
        """No SCF output in the PySCF log."""
        _, paths = direct_run(v)
        log = paths.pyscf_output.read_text() if paths.pyscf_output.is_file() else ""
        assert "PySCF DFT" not in log
        assert "cycle=" not in log


_MOLDEN_FILES = [
    pytest.param(ResultFileKey.PYSCF_DFT_MOLDEN, id="scf-molden"),
    pytest.param(ResultFileKey.SEMICANONICAL_MOLDEN, id="semicanonical"),
]


class TestMoldenDirectRestrictedCheck:
    """molden-direct --restricted: an RO file gives the same numbers as
    without the flag; a spin-polarized file is rejected.  Both hold for the
    SCF's orbitals and for semicanonical ones, whose occupied alpha
    orbitals mix singly and doubly occupied orbitals."""

    @staticmethod
    def _direct(run, v, tmp_path, src_key=ResultFileKey.PYSCF_DFT_MOLDEN, **kw):
        from conftest import make_config
        from workflow import run_one_workflow
        parent, _ = run(v)
        src = Path(parent.files[src_key])
        tmp_path.mkdir(parents=True, exist_ok=True)
        dst = tmp_path / src.name
        shutil.copy(src, dst)
        return run_one_workflow(make_config(dst, source="molden-direct", **kw))

    @pytest.mark.parametrize("src_key", _MOLDEN_FILES)
    @pytest.mark.parametrize("mol", ["cn", "ch2", "hco"])
    def test_ro_file_same_numbers_with_and_without_flag(self, mol, src_key, run, tmp_path):
        v   = Variant(mol, restricted=True)
        r_u = self._direct(run, v, tmp_path / "u", src_key)
        r_r = self._direct(run, v, tmp_path / "r", src_key, restricted=True)
        report = Path(r_r.files[ResultFileKey.SEMICANONICAL_OUTPUT]).read_text()
        line = next(l for l in report.splitlines() if "Reference class" in l)
        assert line.split(":")[1].strip() == "ROHF", line
        for key in _ENERGY_KEYS:
            assert abs(r_u.energies[key] - r_r.energies[key]) < 1e-9, key

    @pytest.mark.parametrize("src_key", _MOLDEN_FILES)
    @pytest.mark.parametrize("mol", ["cn", "ch2", "hco"])
    def test_spin_polarized_file_rejected(self, mol, src_key, run, tmp_path):
        with pytest.raises(ValueError, match="not restricted open-shell"):
            self._direct(run, Variant(mol), tmp_path, src_key, restricted=True)

    def test_closed_shell_singlet_file_accepted(self, run, tmp_path):
        """A UKS singlet whose spin blocks coincide is a valid RO determinant."""
        r = self._direct(run, Variant("h2o"), tmp_path, restricted=True)
        assert r.energies["MP3 correlation"] < 0.0
