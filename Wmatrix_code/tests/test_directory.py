"""Tests for directory batch mode."""
from __future__ import annotations
import shutil
import pytest
from conftest import DATA_DIR, make_batch_config
from workflow import run_many_workflows
from workflow_common import derive_batch_paths
from pyscf_w_matrices import W2_LABELS, W3_LABELS

_MOLS = ("h", "he", "h2o")


@pytest.fixture(scope="module")
def batch_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("batch")
    for mol in _MOLS:
        shutil.copy(DATA_DIR / f"{mol}.xyz", d / f"{mol}.xyz")
    return d


@pytest.fixture(scope="module")
def batch_rc(batch_dir):
    return run_many_workflows(make_batch_config(batch_dir, workers=3))


class TestBatchSuccess:

    def test_exit_zero(self, batch_rc):
        assert batch_rc == 0

    @pytest.mark.parametrize("mol", _MOLS)
    def test_final_summary(self, mol, batch_dir, batch_rc):
        assert (batch_dir / f"{mol}.final.out").is_file()

    @pytest.mark.parametrize("mol", _MOLS)
    def test_w_matrices(self, mol, batch_dir, batch_rc):
        for lbl in W2_LABELS + W3_LABELS:
            assert (batch_dir / f"{mol}.{lbl}.npy").is_file()

    @pytest.mark.parametrize("mol", _MOLS)
    def test_batch_stdout(self, mol, batch_dir, batch_rc):
        assert derive_batch_paths(batch_dir / f"{mol}.xyz").stdout.is_file()

    @pytest.mark.parametrize("mol", _MOLS)
    def test_batch_stderr(self, mol, batch_dir, batch_rc):
        assert derive_batch_paths(batch_dir / f"{mol}.xyz").stderr.is_file()


@pytest.fixture(scope="module")
def closed_shell_batch(tmp_path_factory):
    d = tmp_path_factory.mktemp("batch_cs")
    shutil.copy(DATA_DIR / "h2o.xyz", d / "h2o.xyz")
    rc = run_many_workflows(make_batch_config(d, closed_shell=True, workers=1))
    return d, rc


@pytest.fixture(scope="module")
def closed_shell_single(tmp_path_factory):
    from conftest import make_config
    from workflow import run_one_workflow
    d = tmp_path_factory.mktemp("single_cs")
    xyz = d / "h2o.xyz"
    shutil.copy(DATA_DIR / "h2o.xyz", xyz)
    return run_one_workflow(make_config(xyz, source="pyscf", closed_shell=True))


class TestBatchForwardsSettings:
    """A batch child must run with the parent's settings: compare it with
    the same single-file run."""

    ETOL = 1e-9

    def test_batch_exit_zero(self, closed_shell_batch):
        assert closed_shell_batch[1] == 0

    def test_child_used_the_closed_shell_path(self, closed_shell_batch):
        d, _ = closed_shell_batch
        report = (d / "h2o.semicanonical.out").read_text()
        assert "RHF Fock from reference density" in report, \
            "batch child ignored closed_shell and ran the unrestricted path"

    @pytest.mark.parametrize("key", [
        "Reference energy (HF on KS density)",
        "Non-Brillouin singles correction",
        "MP2 doubles correlation",
        "MP3 correlation",
    ])
    def test_child_energies_match_single_job(self, key, closed_shell_batch,
                                             closed_shell_single):
        d, _ = closed_shell_batch
        child = {}
        for line in (d / "h2o.final.out").read_text().splitlines():
            if "=" in line:
                k, val = line.rsplit("=", 1)
                try:
                    child[k.strip()] = float(val)
                except ValueError:
                    pass
        diff = abs(child[key] - closed_shell_single.energies[key])
        assert diff < self.ETOL, f"{key}: batch={child[key]} single={closed_shell_single.energies[key]}"


class TestBatchErrors:

    def test_empty_dir_raises(self, tmp_path):
        from workflow_common import list_inputs
        with pytest.raises(ValueError, match="No input files"):
            list_inputs(tmp_path, "pyscf")

    def test_missing_path_raises(self, tmp_path):
        from workflow_common import list_inputs
        with pytest.raises(ValueError, match="does not exist"):
            list_inputs(tmp_path / "ghost", "pyscf")

    def test_bad_input_nonzero_exit(self, tmp_path):
        shutil.copy(DATA_DIR / "he.xyz", tmp_path / "he.xyz")
        (tmp_path / "bad.xyz").write_text("notvalid\n")
        assert run_many_workflows(make_batch_config(tmp_path, workers=2)) != 0

    def test_good_job_completes_despite_bad_input(self, tmp_path):
        shutil.copy(DATA_DIR / "he.xyz", tmp_path / "he.xyz")
        (tmp_path / "bad.xyz").write_text("notvalid\n")
        run_many_workflows(make_batch_config(tmp_path, workers=2))
        assert (tmp_path / "he.final.out").is_file()
