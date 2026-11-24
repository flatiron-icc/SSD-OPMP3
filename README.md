# Size-Consistent Orbital Padé Resummed Møller-Plesset Perturbation Theory

Provides code for Size-Consistent Padé Resummation of Møller-Plesset Perturbation Theory, as described in K. Bystrom, T. C. Berkelbach and D. Hait, [arXiv:2609.25330](https://arxiv.org/abs/2609.25330).

The paper splits the MP2 and MP3 correlation energies over the occupied orbitals into the matrices W⁽²⁾ and W⁽³⁾, and resums them as matrices, which keeps the resulting energy size-consistent. Damping the same-spin third-order part gives SSD-OPMP3, at the cost of MP3.

| directory | content |
|---|---|
| [`Wmatrix_code/`](Wmatrix_code/) | reference determinant → semicanonical orbitals → the eight W matrices of a molecule, the MP2/MP3 energies given by their traces, and the OP-MP3, SOP-MP3 and SSD-OPMP3 energies (SSD-OPMP3 at the paper's x_ss for the functional, else 0.5) |
| [`analysis_code/`](analysis_code/) | every resummed model of the paper for one molecule from its W matrices, at the scaling parameters given |
| [`interactive_htmls/`](interactive_htmls/) | interactive versions of the paper's figures, one HTML file each with its data embedded; GitHub shows them as source, so download one and open it in a browser |

Each directory has its own README.

The W matrices of the paper were computed with the ADC(3)-based code of the paper's Zenodo repository (https://doi.org/10.5281/zenodo.23482348), which also holds the W matrices of every structure and the supporting information. `Wmatrix_code/` computes the same matrices from a more efficient, CCSD-derived approach.

## Requirements

Python 3.12 with `numpy` and `scipy`; `Wmatrix_code/` also needs `pyscf` and `pyyaml`, and Q-Chem only for `--source qchem`. Nothing to install: run the scripts with Python. Tested versions are in each README.

## Running

    python Wmatrix_code/cli.py h2o.xyz --source pyscf --functional pbe0 --basis cc-pvdz

writes the W matrices and `h2o.final.out`, which gives the MP2, MP3, OP-MP3, SOP-MP3 and SSD-OPMP3 energies, SSD-OPMP3 at the paper's x_ss for the functional, else 0.5 (`Wmatrix_code/README.md` §6). `Wmatrix_code/` writes a molecule's files next to its input, with the input's name as prefix.

`analysis_code/` reads them by that prefix and evaluates every model of the paper at any parameters, e.g. SSD-OPMP3 at x_ss = 0.45 for κOOMP2 orbitals, which enter as a Molden file without a functional:

    python Wmatrix_code/cli.py koomp2.molden --source molden-direct
    python analysis_code/compute_resummed_energies.py koomp2 --x-ss 0.45

The second writes `koomp2.analysis.out` with every model of `analysis_code/README.md` §2.

Run on a directory (`python Wmatrix_code/cli.py DIR/ ...`), `Wmatrix_code/` writes every molecule's files side by side; `compute_resummed_energies.py` then takes one molecule at a time by its prefix, `DIR/<name>`.

## Tests

`cd Wmatrix_code && pytest tests/` (needs `pytest`); the tests that need Q-Chem are skipped when it is not on `PATH`.

## License

MIT (`LICENSE`, with copies in `Wmatrix_code/` and `analysis_code/`), except `Wmatrix_code/mp3.py`, which adapts PySCF code under the Apache License 2.0 (`Wmatrix_code/LICENSE-PySCF`).

## Citation

K. Bystrom, T. C. Berkelbach, D. Hait, *Near Chemical Accuracy From Size-Consistent Padé Resummed Møller-Plesset Perturbation Theory*, arXiv:2609.25330 (2026).

```bibtex
@misc{bystrom2026pade,
  title         = {Near Chemical Accuracy From Size-Consistent Pad{\'e} Resummed M{\o}ller-Plesset Perturbation Theory},
  author        = {Bystrom, Kyle and Berkelbach, Timothy C. and Hait, Diptarka},
  year          = {2026},
  eprint        = {2609.25330},
  archivePrefix = {arXiv}
}
```
