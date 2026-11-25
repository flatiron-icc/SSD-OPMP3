"""Unit tests — no DFT calculations, completes in seconds."""
from __future__ import annotations
from dataclasses import fields, replace
import pytest
from conftest import DATA_DIR, FIELD_Z, make_config, AUXBASIS, BASIS
from electric_field import parse_electric_field
from run_qchem import make_qchem_input
from workflow import _build_child_command, validate_config
from workflow_common import (
    DEFAULT_X_SS, derive_batch_paths, derive_output_paths, read_xyz,
    resolve_frozen_core, ssd_x_ss,
)


class TestReadXyz:

    @pytest.mark.parametrize("mol,mult,natoms", [
        ("h", 2, 1), ("he", 1, 1), ("he-triplet", 3, 1), ("cn", 2, 2), ("h2o", 1, 3),
    ])
    def test_parses_correctly(self, mol, mult, natoms):
        charge, m, geom = read_xyz(DATA_DIR / f"{mol}.xyz")
        assert charge == 0
        assert m == mult
        assert len(geom.strip().splitlines()) == natoms

    def test_wrong_extension_raises(self, tmp_path):
        f = tmp_path / "mol.txt"
        f.write_text("1\n0 1\nH 0 0 0\n")
        with pytest.raises(ValueError, match="Expected an XYZ"):
            read_xyz(f)

    def test_non_integer_atom_count_raises(self, tmp_path):
        (tmp_path / "mol.xyz").write_text("bad\n0 1\nH 0 0 0\n")
        with pytest.raises(ValueError, match="atom count"):
            read_xyz(tmp_path / "mol.xyz")

    def test_missing_charge_raises(self, tmp_path):
        (tmp_path / "mol.xyz").write_text("1\nX\nH 0 0 0\n")
        with pytest.raises(ValueError, match="charge and multiplicity"):
            read_xyz(tmp_path / "mol.xyz")

    def test_insufficient_coord_lines_raises(self, tmp_path):
        (tmp_path / "mol.xyz").write_text("3\n0 1\nO 0 0 0\n")
        with pytest.raises(ValueError):
            read_xyz(tmp_path / "mol.xyz")

    @pytest.mark.parametrize("header", [
        "0 1 comment text",
        "charge=0 multiplicity=1",
        "charge=0 multiplicity=1 comment text",
        "comment charge=0, multiplicity=1, more",
    ])
    def test_both_header_forms_ignore_further_text(self, tmp_path, header):
        (tmp_path / "mol.xyz").write_text(f"1\n{header}\nH 0 0 0\n")
        charge, mult, _ = read_xyz(tmp_path / "mol.xyz")
        assert charge == 0 and mult == 1

    def test_key_value_form_missing_multiplicity_raises(self, tmp_path):
        (tmp_path / "mol.xyz").write_text("1\ncharge=0 comment\nH 0 0 0\n")
        with pytest.raises(ValueError, match="charge and multiplicity"):
            read_xyz(tmp_path / "mol.xyz")


class TestParseElectricField:

    def test_none_returns_none(self):
        assert parse_electric_field(None) is None

    def test_all_zero_returns_none(self):
        assert parse_electric_field([0.0, 0.0, 0.0]) is None

    def test_below_tolerance_returns_none(self):
        assert parse_electric_field([1e-16, 0.0, 0.0]) is None

    def test_valid_returns_tuple_of_floats(self):
        f = parse_electric_field(FIELD_Z)
        assert f == (0.0, 0.0, 0.001)
        assert all(isinstance(c, float) for c in f)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="three Cartesian"):
            parse_electric_field([0.0, 0.0])


@pytest.fixture(scope="module")
def h2o_mol():
    from pyscf import gto
    mol = gto.Mole()
    mol.atom    = "O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586"
    mol.basis   = "sto-3g"
    mol.verbose = 0
    mol.build()
    return mol


class TestResolveFrozenCore:

    def test_zero(self, h2o_mol):
        assert resolve_frozen_core("0", h2o_mol) == 0

    def test_integer(self, h2o_mol):
        assert resolve_frozen_core("2", h2o_mol) == 2

    def test_negative_raises(self, h2o_mol):
        with pytest.raises(ValueError, match="non-negative"):
            resolve_frozen_core("-1", h2o_mol)

    def test_fc_h2o_is_one(self, h2o_mol):
        assert resolve_frozen_core("fc", h2o_mol) == 1

    def test_invalid_string_raises(self, h2o_mol):
        with pytest.raises(ValueError):
            resolve_frozen_core("notanumber", h2o_mol)


class TestQchemInput:

    _G = "O 0 0 0\nH 0 0.757 0.586\nH 0 -0.757 0.586"

    def _make(self, **kw):
        d = dict(charge=0, multiplicity=1, geometry=self._G,
                 functional="pbe0", basis="cc-pvdz", memory=4000,
                 frozen_core="0", restricted=False, do_mp3=False,
                 electric_field=None)
        d.update(kw)
        return make_qchem_input(**d)

    def test_method_present(self):         assert "method pbe0"        in self._make()
    def test_basis_present(self):          assert "basis cc-pvdz"      in self._make()
    def test_molden_format(self):          assert "molden_format true" in self._make()
    def test_unrestricted_default(self):   assert "unrestricted true"  in self._make().lower()
    def test_roscf_when_restricted(self):  assert "roscf true"         in self._make(restricted=True).lower()
    def test_no_roscf_unrestricted(self):  assert "roscf"          not in self._make(restricted=False).lower()
    def test_mp3_section_when_requested(self):
        assert "@@@@@"      in self._make(do_mp3=True)
        assert "method mp3" in self._make(do_mp3=True)
    def test_no_mp3_by_default(self):      assert "@@@@@" not in self._make()
    def test_closed_shell_non_singlet_raises(self):
        with pytest.raises(ValueError, match="multiplicity 1"):
            self._make(multiplicity=2, closed_shell=True)
    def test_closed_shell_singlet_unrestricted_false(self):
        assert "unrestricted false" in self._make(closed_shell=True)
    def test_field_block_present(self):
        f = parse_electric_field(FIELD_Z)
        assert "$multipole_field" in self._make(electric_field=f)
    def test_no_field_block_without_field(self):
        assert "$multipole_field" not in self._make()


class TestValidateConfig:

    def _xyz(self, tmp_path):
        f = tmp_path / "mol.xyz"; f.touch(); return f

    def _mdn(self, tmp_path):
        f = tmp_path / "mol.molden"; f.touch(); return f

    def test_valid_pyscf(self, tmp_path):
        validate_config(make_config(self._xyz(tmp_path), source="pyscf"))

    def test_no_basis_xyz_source_raises(self, tmp_path):
        # Both qchem and pyscf require --basis; test with pyscf explicitly.
        with pytest.raises(ValueError, match="--basis"):
            validate_config(make_config(self._xyz(tmp_path), source="pyscf", basis=None))

    def test_qcmp3_with_molden_raises(self, tmp_path):
        cfg = make_config(self._mdn(tmp_path), source="molden")
        with pytest.raises(ValueError, match="--qc-mp3"):
            validate_config(replace(cfg, qc_mp3=True))

    def test_zero_threads_raises(self, tmp_path):
        with pytest.raises(ValueError, match="--threads"):
            validate_config(make_config(self._xyz(tmp_path), source="pyscf", threads=0))

    def test_unsupported_source_raises(self, tmp_path):
        cfg = make_config(self._xyz(tmp_path), source="pyscf")
        with pytest.raises(ValueError, match="Unsupported"):
            validate_config(replace(cfg, source="unknown"))

    def test_xyz_required_for_pyscf(self, tmp_path):
        with pytest.raises(ValueError, match=r"\.xyz"):
            validate_config(make_config(self._mdn(tmp_path), source="pyscf"))

    def test_molden_required_for_molden_source(self, tmp_path):
        cfg = make_config(self._xyz(tmp_path), source="pyscf")
        with pytest.raises(ValueError, match=r"\.molden"):
            validate_config(replace(cfg, source="molden"))

    def test_df_without_auxbasis_raises(self, tmp_path):
        # make_config always supplies AUXBASIS when df=True; construct the
        # invalid config directly to test the validation rule.
        cfg = make_config(self._xyz(tmp_path), source="pyscf")
        with pytest.raises(ValueError, match="--auxbasis"):
            validate_config(replace(cfg, df=True, auxbasis=None))

    def test_df_with_auxbasis_valid(self, tmp_path):
        cfg = make_config(self._xyz(tmp_path), source="pyscf", df=True, auxbasis=AUXBASIS)
        validate_config(cfg)  # must not raise

    def test_missing_input_raises(self, tmp_path):
        cfg = make_config(tmp_path / "ghost.xyz", source="pyscf")
        with pytest.raises(ValueError, match="does not exist"):
            validate_config(cfg)

    def test_no_functional_scf_source_raises(self, tmp_path):
        cfg = make_config(self._xyz(tmp_path), source="pyscf")
        with pytest.raises(ValueError, match="--functional is required"):
            validate_config(replace(cfg, functional=None))

    def test_molden_direct_valid_without_functional_or_basis(self, tmp_path):
        cfg = make_config(self._mdn(tmp_path), source="molden-direct")
        assert cfg.functional is None and cfg.basis is None
        validate_config(cfg)  # must not raise

    def test_molden_direct_with_functional_raises(self, tmp_path):
        cfg = make_config(self._mdn(tmp_path), source="molden-direct")
        with pytest.raises(ValueError, match="--functional has no effect"):
            validate_config(replace(cfg, functional="pbe0"))

    def test_molden_direct_with_restricted_valid(self, tmp_path):
        cfg = make_config(self._mdn(tmp_path), source="molden-direct", restricted=True)
        validate_config(cfg)  # must not raise: the file is checked at load time

    @pytest.mark.parametrize("source", ["pyscf", "qchem"])
    def test_closed_shell_non_singlet_xyz_raises(self, source):
        cfg = make_config(DATA_DIR / "cn.xyz", source=source, closed_shell=True)
        with pytest.raises(ValueError, match="spin-singlet"):
            validate_config(cfg)

    @pytest.mark.parametrize("source", ["pyscf", "qchem"])
    def test_closed_shell_singlet_xyz_valid(self, source):
        validate_config(make_config(DATA_DIR / "h2o.xyz", source=source, closed_shell=True))

    def test_molden_direct_requires_molden_file(self, tmp_path):
        cfg = make_config(self._xyz(tmp_path), source="molden-direct")
        with pytest.raises(ValueError, match=r"\.molden"):
            validate_config(cfg)


class TestBuildChildCommand:
    """Batch mode must pass every setting on to its cli.py children."""

    def _cmd(self, tmp_path, **kw):
        xyz = tmp_path / "mol.xyz"; xyz.touch()
        cfg = make_config(xyz, source="pyscf", **kw)
        return _build_child_command(cfg, xyz)

    def test_closed_shell_forwarded(self, tmp_path):
        assert "--closed-shell" in self._cmd(tmp_path, closed_shell=True)

    def test_closed_shell_absent_by_default(self, tmp_path):
        assert "--closed-shell" not in self._cmd(tmp_path)

    def test_restricted_forwarded(self, tmp_path):
        assert "--restricted" in self._cmd(tmp_path, restricted=True)

    def test_df_and_auxbasis_forwarded(self, tmp_path):
        cmd = self._cmd(tmp_path, df=True)
        assert "--df" in cmd
        assert cmd[cmd.index("--auxbasis") + 1] == AUXBASIS

    def test_frozen_core_forwarded(self, tmp_path):
        cmd = self._cmd(tmp_path, frozen_core="fc")
        assert cmd[cmd.index("--frozen-core") + 1] == "fc"

    def test_basis_forwarded(self, tmp_path):
        cmd = self._cmd(tmp_path)
        assert cmd[cmd.index("--basis") + 1] == BASIS

    def test_electric_field_forwarded(self, tmp_path):
        cmd = self._cmd(tmp_path, electric_field=FIELD_Z)
        i = cmd.index("--electric-field")
        assert [float(x) for x in cmd[i + 1:i + 4]] == FIELD_Z

    def test_functional_and_source_forwarded(self, tmp_path):
        cmd = self._cmd(tmp_path, functional="b3lyp")
        assert cmd[cmd.index("--functional") + 1] == "b3lyp"
        assert cmd[cmd.index("--source") + 1] == "pyscf"

    def test_molden_direct_child_has_no_functional(self, tmp_path):
        """A molden-direct batch child must not receive --functional, which
        the child's own validate_config would reject."""
        mdn = tmp_path / "mol.molden"; mdn.touch()
        cfg = make_config(mdn, source="molden-direct")
        cmd = _build_child_command(cfg, mdn)
        assert "--functional" not in cmd
        assert cmd[cmd.index("--source") + 1] == "molden-direct"

    @pytest.mark.parametrize("field", [
        "closed_shell", "restricted", "df", "frozen_core", "basis",
        "functional", "source", "electric_field", "auxbasis",
    ])
    def test_every_physics_field_is_represented(self, tmp_path, field):
        """Every set field appears as its own flag token ('--basis' is a
        substring of '--auxbasis', so tokens, not the joined string)."""
        xyz = tmp_path / "mol.xyz"; xyz.touch()
        cfg = make_config(xyz, source="pyscf", df=True, closed_shell=True,
                          restricted=True, frozen_core="fc",
                          electric_field=FIELD_Z)
        cmd  = _build_child_command(cfg, xyz)
        flag = "--" + field.replace("_", "-")
        assert not getattr(cfg, field) or flag in cmd, \
            f"{field} is set on the config but {flag} is not in {cmd}"


class TestListInputs:
    """Batch mode must pick the input extension that matches the source."""

    @pytest.mark.parametrize("source,ext", [
        ("qchem", ".xyz"), ("pyscf", ".xyz"),
        ("molden", ".molden"), ("molden-direct", ".molden"),
        ("chk", ".chk"),
    ])
    def test_pattern_follows_source(self, tmp_path, source, ext):
        from workflow_common import list_inputs
        for e in (".xyz", ".molden", ".chk"):
            (tmp_path / f"a{e}").touch()
        found = list_inputs(tmp_path, source)
        assert [p.suffix for p in found] == [ext]


class TestEcpByAtomLabel:
    """ECP and -pp basis entries must also be keyed by the indexed atom
    labels (Ag1, I2) of a Molden-built molecule."""

    @staticmethod
    def _mol(labels):
        from pyscf import gto
        mol = gto.Mole()
        mol.atom = [(lab, (0.0, 0.0, 2.7 * i)) for i, lab in enumerate(labels)]
        mol.basis = "def2-svp"
        mol.verbose = 0
        mol.build()
        return mol

    @pytest.mark.parametrize("labels", [["Ag", "H"], ["Ag1", "H2"]])
    def test_heavy_atom_gets_the_family_of_the_basis(self, labels):
        from pyscf.gto import basis as B
        from pyscf_dft import get_ecp
        mol = self._mol(labels)
        ecp = get_ecp("aug-cc-pwCVTZ", mol)
        assert ecp[labels[0]] == "cc-pvdz-pp" and ecp["Ag"] == "cc-pvdz-pp"
        mol.ecp = ecp
        mol.build()
        assert mol._ecp[labels[0]] == B.load_ecp("cc-pvdz-pp", "Ag")
        assert mol._ecp[labels[0]] != B.load_ecp("def2-svp", "Ag")

    def test_def2_family_unchanged(self):
        from pyscf_dft import get_ecp
        ecp = get_ecp("def2-svp", self._mol(["Ag1", "H2"]))
        assert ecp["Ag1"] == "def2-svp"

    def test_pp_basis_keyed_by_label(self):
        from pyscf_dft import handle_basis_core
        b = handle_basis_core("cc-pvdz-pp", self._mol(["I1", "I2"]))
        assert b["I1"] == b["I2"] == b["I"] == "cc-pvdz-pp"
        assert b["default"] == "cc-pvdz" and b["H"] == "cc-pvdz"


def _all_output_paths(input_path):
    """Every path derive_output_paths / derive_batch_paths give for one input."""
    p = derive_output_paths(input_path)
    b = derive_batch_paths(input_path)
    return ([getattr(p, f.name) for f in fields(p) if f.name != "base"]
            + [p.w_matrix("W2alpha_ss"), p.semicanonical_array("coeff_a"), b.stdout, b.stderr])


class TestOutputPaths:
    """Outputs keep the input's whole stem, dots included (h2o.opt.xyz once
    wrote h2o.final.out beside h2o.opt.W*.npy)."""

    @pytest.mark.parametrize("name", ["h2o.xyz", "h2o.opt.xyz", "mol.a.b.molden", "x.chk"])
    def test_every_path_keeps_the_stem(self, tmp_path, name):
        stem = name.rsplit(".", 1)[0]
        for path in _all_output_paths(tmp_path / name):
            assert path.parent == tmp_path.resolve(), path
            assert path.name.startswith(stem + "."), path.name

    def test_inputs_differing_after_a_dot_do_not_collide(self, tmp_path):
        a = set(_all_output_paths(tmp_path / "mol.a.xyz"))
        b = set(_all_output_paths(tmp_path / "mol.b.xyz"))
        assert not a & b, sorted(str(p) for p in a & b)


class TestSsdXss:
    """x_ss of the summary's SSD-OPMP3: the paper's value for the functional."""

    @pytest.mark.parametrize("functional,x_ss", [
        ("hf", 0.0), ("HF", 0.0), ("spw92", 0.75), ("pbe", 0.70),
        ("r2scan", 0.60), ("r2SCAN", 0.60), ("pbe0", 0.50), ("PBE0", 0.50),
        ("wb97x-v", 0.50), ("wB97M-V", 0.50), ("wb97m_v", 0.50), ("ωB97X-V", 0.50),
    ])
    def test_paper_values(self, functional, x_ss):
        value, origin = ssd_x_ss(functional)
        assert value == x_ss and "paper" in origin and functional in origin

    @pytest.mark.parametrize("functional", ["b3lyp", "tpss", "wb97x"])
    def test_other_functional_gets_the_default(self, functional):
        value, origin = ssd_x_ss(functional)
        assert value == DEFAULT_X_SS and origin.startswith("default") and functional in origin

    @pytest.mark.parametrize("functional", [None, ""])
    def test_no_functional_gets_the_default(self, functional):
        value, origin = ssd_x_ss(functional)
        assert value == DEFAULT_X_SS and origin.startswith("default")
