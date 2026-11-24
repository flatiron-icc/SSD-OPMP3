# W matrices

For a reference determinant, computes and writes the eight W matrices of K. Bystrom, T. C. Berkelbach and D. Hait, arXiv:2609.25330 (Eqs. 17–18: W⁽²⁾ and W⁽³⁾, same-spin and opposite-spin, alpha and beta blocks) and the MP2 and MP3 energies given by their traces. The paper's resummation schemes act on these matrices and require another codebase, which is in the paper's Zenodo repository (https://doi.org/10.5281/zenodo.23482348) together with the W-matrix code used for the paper (based on PySCF's ADC(3) module); the OP-MP3, SOP-MP3 and SSD-OPMP3 energies (Eqs. 14, 21 and 22; SSD-OPMP3 with x_ss = 0.5) are printed here by default.

## 1. What is computed

1. **Reference determinant.** A PySCF HF or DFT solution, or the orbitals of a Molden file used as written, without an SCF (`--source molden-direct`; this is how κOOMP2 orbitals from Q-Chem can enter).
2. **Semicanonicalization.** The HF Fock matrix of the determinant's density (UHF, or RHF with `--closed-shell`) is diagonalized separately in the occupied and in the virtual space. The determinant is unchanged, and the orbital energies ε used in MP are the diagonal of this Fock matrix. Its occupied–virtual block gives the non-Brillouin singles energy E_S⁽²⁾ = −Σ_ai |F_ai|² / (ε_a − ε_i), over both spins.
3. **Amplitudes.** c⁽¹⁾ and c⁽²⁾ (Eqs. 7 and 9), from one CCSD iteration with only the terms linear in the amplitudes, starting from c⁽¹⁾ and no singles (`mp3.py`). This returns c⁽¹⁾ + c⁽²⁾ and costs less than one CCSD iteration. The denominators use the semicanonical ε; F_ai enters only E_S⁽²⁾.
4. **W matrices.** Same-spin: c contracted with ⟨jk‖ab⟩, prefactor 1/8; opposite-spin: c contracted with ⟨jk|ab⟩, prefactor 1/4; both symmetrized in (i, j).
5. **Energies.** Tr W⁽²⁾ is the MP2 doubles energy and Tr W⁽³⁾ the third-order energy E_c⁽³⁾; E_MP3 = E_HF + E_S⁽²⁾ + Tr W⁽²⁾ + Tr W⁽³⁾, where E_HF is the HF energy of the reference density.

## 2. Requirements

`numpy`, `scipy`, `pyscf`, `pyyaml`; Q-Chem only for `--source qchem`. Nothing to install: run `python /path/to/cli.py`. Tested with:

| Python | pyscf | numpy | scipy | pyyaml | pytest | Q-Chem |
|---|---|---|---|---|---|---|
| 3.12.12 | 2.15.1 | 2.4.4 | 1.17.1 | 6.0.3 | 9.1.1 | 6.4 |

## 3. Running

Outputs are written next to the input. With `tests/data/h2o.xyz` and `tests/data/cn.xyz` (the CN doublet) copied to a work directory:

```
python cli.py h2o.xyz --source pyscf --functional pbe0 --basis cc-pvdz
python cli.py cn.xyz  --source pyscf --functional pbe0 --basis cc-pvdz --restricted       # RO-DFT
python cli.py h2o.xyz --source pyscf --functional pbe0 --basis cc-pvdz \
    --frozen-core fc --df --auxbasis cc-pvdz-ri --electric-field 0 0 0.001
python cli.py koomp2.molden --source molden-direct                                    # orbitals as written
python cli.py inputs/ --source pyscf --functional pbe0 --basis cc-pvdz --workers 3   # every .xyz in inputs/
```

`python cli.py --help` lists every option.

## 4. Inputs

| `--source` | input | reference determinant |
|---|---|---|
| `qchem` (default) | `.xyz` | Q-Chem SCF, then reconverged by PySCF from Q-Chem's orbitals |
| `pyscf` | `.xyz` | PySCF SCF |
| `molden` | `.molden` | PySCF SCF restarted from the file's orbitals |
| `chk` | `.chk` | PySCF SCF restarted from a PySCF checkpoint, which it overwrites |
| `molden-direct` | `.molden` | the file's orbitals as written; no SCF |

**XYZ**: atom count; charge and multiplicity (further text on the line is ignored; `charge=0 multiplicity=1` also works); one `element x y z` line per atom, in Å. `tests/data/h2o.xyz`:

```
3
0 1
O 0.0000000000 0.0000000000 0.0000000000
H 0.0000000000 0.0000000000 0.9579000000
H 0.9289588892 0.0000000000 -0.2336831018
```

**Molden**: alpha and beta orbital sections; with `--closed-shell`, a single section is also accepted (occupations 0/2, or 0/1, which are doubled). Two sections are checked against the flags: with `--closed-shell` the alpha and beta occupied spaces must coincide, and with `--restricted` (`molden-direct` only) the beta occupied space must lie inside the alpha one. Both checks require every singular value of C_a,occᵀ S C_b,occ to be 1 (to 1e-6), which does not depend on how the occupied orbitals are rotated: semicanonical orbitals pass, spin-polarized ones are rejected. `molden-direct` also needs the complete orbital set (as many orbitals per spin as basis functions; in Q-Chem set `print_orbitals` to a large number).

## 5. Options

| flag | meaning |
|---|---|
| `--functional` | XC functional or `hf`; required except with `molden-direct`, where it is rejected |
| `--basis` | required for `qchem`, `pyscf`, `chk`; the Molden modes take the AO basis from the file and use `--basis` only to choose the ECP |
| `--restricted` | restricted open-shell (RO-DFT/ROHF) reference; with `molden-direct`, the check of §4 |
| `--closed-shell` | RKS/RHF reference and the closed-shell amplitude code; singlets only; overrides `--restricted` |
| `--frozen-core N` / `fc` | frozen orbitals per spin, or PySCF's chemical core count (§7.1) |
| `--df --auxbasis NAME` | density fitting in the amplitude and W stage (§7.2) |
| `--electric-field EX EY EZ` | uniform field, a.u. (§7.3) |
| `--threads`, `--memory` | defaults 8 and 16000 MB |
| `--workers` | batch mode: inputs processed at once |
| `--qc-mp3` | `qchem` mode: also run a Q-Chem MP3 job, for comparison by hand only |

Basis substitutions, applied per element: H and He get the valence basis (`cc-pcvXz` and `cc-pwcvXz` become `cc-pvXz`); elements beyond Kr get an ECP from the basis family (`cc-pvdz-pp` for `cc-*`, `def2-svp` for `def2-*`).

## 6. Outputs

| file | content |
|---|---|
| `<stem>.W{2,3}{alpha,beta}_{ss,os}.npy` | the eight W matrices |
| `<stem>.semicanonical.out` | E_HF, E_S⁽²⁾, reference class, orbital energies, F_ai |
| `<stem>.semicanonical.molden` | the semicanonical orbitals |
| `<stem>.semicanonical.{coeff,occ,fock}_{a,b}.npy` | MO coefficients, 0/1 occupations, MO Fock matrix; with `--closed-shell` one set `{coeff,occ,fock}.npy`, occupations 0/2 |
| `<stem>.final.out`, `.final.yaml` | settings, PySCF version, file paths, energies |
| `<stem>.pyscf.{out,chk,molden}` | PySCF SCF log, checkpoint, orbitals (with `molden-direct`, only an empty `.out`) |
| `<stem>.{in,out,molden}` | Q-Chem input, output and orbitals (`qchem` mode) |
| `<stem>.batch.{stdout,stderr}` | batch mode only |

Each W matrix is real, symmetric and `(nact, nact)` over the active (not frozen) occupied orbitals of its spin. The `OP-MP3`, `SOP-MP3` and `SSD-OPMP3 (x_ss = 0.5)` total energies in `.final.out` are E_HF + E_S⁽²⁾ + the resummed correlation of Eqs. 14, 21 and 22. `MP2 doubles correlation` is Tr W⁽²⁾, `MP2 correlation` adds E_S⁽²⁾, and `MP3 correlation` is Tr W⁽³⁾; the same-spin and opposite-spin parts of both are listed too. Tr W⁽²⁾ from the saved matrices:

```python
import numpy as np
tr = lambda lbl: np.trace(np.load(f"h2o.{lbl}.npy"))
e2 = sum(tr(f"W2{s}_{t}") for s in ("alpha", "beta") for t in ("ss", "os"))
```

## 7. Details

**7.1 Frozen core.** The frozen orbitals are the lowest `N` semicanonical occupied orbitals of each spin. They are removed from the amplitudes, the W matrices (which are not padded with zeros) and E_S⁽²⁾, not from the SCF, the Fock matrix or the semicanonicalization.

**7.2 Density fitting.** `--df` fits the integrals of the amplitude and W stage with `--auxbasis` (a PySCF basis name, substituted per element as in §5). The SCF, the Fock matrix and E_S⁽²⁾ use exact integrals.

**7.3 Electric field.** `--electric-field` adds E·r to the one-electron Hamiltonian and −Σ_A Z_A E·R_A to the nuclear repulsion, both about the origin, so E(F) = E₀ − μ·F − ½ F·α·F. The field enters the SCF and the Fock matrix of step 2, and so the W matrices only through the semicanonical orbitals and orbital energies. `molden-direct` does not reoptimize the orbitals, so its file must come from a calculation in the same field. Q-Chem's `$multipole_field` has the opposite sign; `qchem` mode writes the sign-flipped components, so give the field as it should act.

**7.4 PySCF internals.** The CC object is the one `pyscf.cc.CCSD` builds for the reference (restricted with `--closed-shell`, unrestricted otherwise, density-fitted with `--df`). Its integrals, c⁽¹⁾ and the ⟨ab‖cd⟩ term are computed by PySCF, with PySCF's Fock matrix replaced by diag(ε). The other terms are adapted from `update_amps` of PySCF 2.15.1 (`pyscf/cc/ccsd.py`, `uccsd.py`). `mp3.py` uses non-public parts of `pyscf.cc`: rerun the tests after upgrading PySCF.

**7.5 Memory.** `--memory` is PySCF's `max_memory`. The W stage stores its integrals as a PySCF CCSD run does: in memory when they fit, otherwise in scratch HDF5 files (in PySCF's `TMPDIR`). The closed-shell update and W assembly work in blocks of virtual orbitals sized from `--memory`, as PySCF's RCCSD does; the unrestricted update, like PySCF's UCCSD, holds whole integral blocks.

## 8. Tests

`pytest tests/` from this directory: 2334 tests; those marked `qchem` are skipped when Q-Chem is not on `PATH`. The suite runs H, He, H₂O, CN, CH₂, CH₃ and HCO in cc-pVDZ, and triplet He (two α electrons, no β; cc-pVTZ, since in cc-pVDZ its MP2 and MP3 energies vanish), with HF and PBE0 references, unrestricted, restricted open-shell and closed-shell, with frozen core, density fitting and a field, and checks:

- E_HF, E_S⁽²⁾ and Tr W⁽²⁾ to 1e-5 Eh, and Tr W⁽³⁾ to 1e-4 Eh, against Q-Chem MP2 and MP3 (`tests/data/references/`);
- each closed-shell run against its unrestricted counterpart;
- every source mode against the run whose orbitals it reads;
- batch mode against single runs.

Runs are written to `tests/_runs/`, which each session recreates.

## License

MIT (`LICENSE`), except `mp3.py`, which adapts PySCF code under the Apache License 2.0 (`LICENSE-PySCF`, with PySCF's NOTICE).

## Citation

K. Bystrom, T. C. Berkelbach, D. Hait, *Near Chemical Accuracy From Size-Consistent Padé Resummed Møller-Plesset Perturbation Theory*, arXiv:2609.25330 (2026).
