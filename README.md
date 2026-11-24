# Size-Consistent Orbital Padé Resummed Møller-Plesset Perturbation Theory

Provides code for Size-Consistent Padé Resummation of Møller-Plesset Perturbation Theory, as described in K. Bystrom, T. C. Berkelbach and D. Hait, [arXiv:2609.25330](https://arxiv.org/abs/2609.25330).

The paper splits the MP2 and MP3 correlation energies over the occupied orbitals into the matrices W⁽²⁾ and W⁽³⁾, and resums them as matrices, which keeps the resulting energy size-consistent. Damping the same-spin third-order part gives SSD-OPMP3, at the cost of MP3.

| directory | content |
|---|---|
| [`Wmatrix_code/`](Wmatrix_code/) | reference determinant → semicanonical orbitals → the eight W matrices of a molecule, the MP2/MP3 energies given by their traces, and the OP-MP3, SOP-MP3 and SSD-OPMP3 (x_ss = 0.5) energies |
| [`analysis_code/`](analysis_code/) | the resummed models and their fits from W matrices, and their reaction-energy errors against CCSD(T) |
| [`interactive_htmls/`](interactive_htmls/) | interactive versions of the paper's figures, one HTML file each with its data embedded; GitHub shows them as source, so download one and open it in a browser |

Each directory has its own README.

The W matrices of the paper were computed with the ADC(3)-based code of the paper's Zenodo repository (https://doi.org/10.5281/zenodo.23482348), which also holds the W matrices of every structure and the supporting information. `Wmatrix_code/` computes the same matrices from a more efficient, CCSD-derived approach.

## Requirements

Python 3.12 with `numpy`, `scipy` and `pyyaml`; `Wmatrix_code/` also needs `pyscf`, and Q-Chem only for `--source qchem`. Nothing to install: run the scripts with Python. Tested versions are in each README.

## From W matrices to resummed energies

`Wmatrix_code/` writes a molecule's files next to its input, with the input's name as prefix; `analysis_code/` reads them by that prefix:

    python Wmatrix_code/cli.py h2o.xyz --source pyscf --functional pbe0 --basis cc-pvdz
    python analysis_code/compute_resummed_energies.py h2o --x-ss 0.5

The first writes the W matrices and `h2o.final.out`, the second `h2o.analysis.out` with every model of `analysis_code/README.md` §2.

For a dataset, `Wmatrix_code/` run on a directory (`python Wmatrix_code/cli.py TREE/PBE0/ ...`) writes every molecule's files side by side, which is the `--flat` layout of `analysis_code/` (`--flat TREE --xc PBE0`).

## Tests

`cd Wmatrix_code && pytest tests/` (needs `pytest`); the tests that need Q-Chem are skipped when it is not on `PATH`.

## License

MIT (`Wmatrix_code/LICENSE`, `analysis_code/LICENSE`), except `Wmatrix_code/mp3.py`, which adapts PySCF code under the Apache License 2.0 (`Wmatrix_code/LICENSE-PySCF`).

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
