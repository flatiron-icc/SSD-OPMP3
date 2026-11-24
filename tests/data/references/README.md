# Q-Chem reference data

The Q-Chem 6.4 values that `test_reference.py` compares the workflow with, one file per `<mol>_<functional>_<basis>_<tag>.json` (`<functional>` is `hf` or `pbe0`; `<basis>` is `cc-pvdz`, or `cc-pvtz` for triplet He, `he-triplet`):

| `<tag>` | reference determinant and options |
|---|---|
| `base` | unrestricted |
| `restricted` | restricted open shell |
| `df`, `df-restricted` | density fitting with aug-cc-pVQZ-RI (`rimp2-aug-cc-pvqz` in Q-Chem, `aug-cc-pvqz-ri` in PySCF) |
| `fc-base` | frozen core, Q-Chem's `n_frozen_core fc`: the same orbitals as PySCF's chemical core (1 for H₂O, CH₂, CH₃; 2 for CN, HCO) |
| `field-z`, `field-z-restricted` | field of 0.001 a.u. along z (written `Z -0.001` in Q-Chem's opposite sign convention) |

A closed-shell run is compared with the file of its unrestricted counterpart, and a missing file skips its tests. There is no restricted frozen-core file: Q-Chem fails on that case.

| key | value (Hartree) |
|---|---|
| `e_hf` | HF energy of the reference density |
| `e_nbs` | non-Brillouin singles energy E_S⁽²⁾ |
| `tr_w2_ss`, `tr_w2_os`, `tr_w2` | MP2 doubles correlation: same-spin, opposite-spin, total (without `e_nbs`) |
| `tr_w3` | MP3 total − MP2 total |

`qchem_results/` holds the Q-Chem inputs and outputs: an SCF job, then MP2 and MP3 jobs on its orbitals. Every `he-triplet` job sets a zero `$multipole_field`, which switches off the quadrupole field that Q-Chem otherwise applies. `qchem_results/generate_references.py <outputs>` writes the JSON files from the outputs and checks that `tr_w2 + e_nbs` equals Q-Chem's MP2 correlation energy.
