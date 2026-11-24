# Resummed MP3 energies from W matrices

Evaluates the correlation-energy models of K. Bystrom, T. C. Berkelbach and D. Hait, arXiv:2609.25330, for one molecule from its W matrices, at the scaling parameters given. No quantum chemistry is run. Energies are in Hartree.

## 1. Requirements

Python ≥ 3.12, `numpy`, `scipy`. Nothing to install: run `python /path/to/compute_resummed_energies.py`. Tested with Python 3.12.12, numpy 2.4.4, scipy 1.17.1.

## 2. Models

The eight arrays `W{2,3}{alpha,beta}_{ss,os}` (the same-spin and opposite-spin parts of the alpha and beta blocks of W⁽²⁾ and W⁽³⁾, Eqs. 17–18) are assembled (`blocks.py`) into three block-diagonal pairs: `W2`, `W3` (same-spin plus opposite-spin: the full W⁽²⁾, W⁽³⁾), `W2_ss`, `W3_ss` and `W2_os`, `W3_os`. A spin block in which both matrices vanish (e.g. the beta same-spin block of a system with one beta electron) is dropped. With P(A, B) = A (A − B)⁻¹ A (Eq. 13), E2 = Tr W2 and E3 = Tr W3:

| model | E_c | paper |
|---|---|---|
| MP2, MP3 | E2, E2 + E3 | |
| MP2.X | E2 + x E3 | |
| OP-MP3(x) | Tr P(W2, x W3); OP-MP3 is x = 1 | Eqs. 14–15 |
| SOP-MP3(x_ss, x_os) | Tr P(W2_ss, x_ss W3_ss) + Tr P(W2_os, x_os W3_os); SOP-MP3 is (1, 1) | Eqs. 19–21 |
| SSD-OPMP3(x_ss) | SOP-MP3(x_ss, 1) | Eq. 22 |

The total energy of every model is E_HF + E_S⁽²⁾ + E_c: E_HF is the HF energy of the reference density and E_S⁽²⁾ the non-Brillouin singles energy, both read from the molecule's files (§3). The x_ss of SSD-OPMP3 (`--x-ss`) adopted in the paper: HF 0.00, SPW92 0.75, PBE 0.70, r²SCAN 0.60, PBE0 0.50, ωB97X-V 0.50, ωB97M-V 0.50, κOOMP2 0.45.

**Evaluation of Tr P** (`resummation.py`). With Z = W2⁺W3 = V diag(ζ) V⁻¹ and cᵢ = (V⁻¹ W2 V)ᵢᵢ,

    Tr P(W2, x W3) = Σᵢ cᵢ / (1 − x ζᵢ),

so one decomposition per pair serves every x; expanding the fraction gives E2 + x E3 at first order and the higher orders as a geometric series. W2⁺ is the pseudo-inverse of W2, which is its inverse unless W2 has a zero row, as a frozen orbital has in the paper's W matrices; such an orbital drops out, with cᵢ = ζᵢ = 0.

**Also evaluated, not in the paper.** SCS-MP2, SOS-MP2 and SCS-MP3 with their literature coefficients (`config.py`). The scalar variant of each Padé model, written without the O (P-MP3, SP-MP3, SSD-PMP3), in which every Tr P(A, B) becomes (Tr A)² / (Tr A − Tr B); P-MP3 is the paper's E_c^[2,3] (Eq. 1), which is not size-consistent. With `--alt-resummations`, four other resummations, Padé–Borel, exponential, Landau–Zener and logarithmic (the paper mentions the first and the third), each replacing 1/(1 − z) by a function with the same value and slope at z = 0.

Every run gives MP2, MP3, SCS-MP2, SOS-MP2, SCS-MP3, OP-MP3, SOP-MP3 and SSD-OPMP3(x_ss), with the scalar P-MP3, SP-MP3 and SSD-PMP3(x_ss). `--x` adds the models scaled by x: MP2.X, OP-MP3(x) and P-MP3(x); `--x-os` adds those scaled by (x_ss, x_os): MP2.X(x_ss, x_os), SOP-MP3(x_ss, x_os) and SP-MP3(x_ss, x_os).

## 3. Inputs

For a molecule `<name>`, the eight files `<name>.W{2,3}{alpha,beta}_{ss,os}.npy` and `<name>.semicanonical.out`, whose lines `HF energy from density : E_HF` and `Non-Brillouin singles E2 : E_S⁽²⁾` are read (or `<name>.final.out`, from its `Reference energy (HF on ... density)` and `Non-Brillouin singles correction` lines), all in one directory. `../Wmatrix_code/` writes these files.

## 4. Running

    python compute_resummed_energies.py DIR --x-ss 0.45
    python compute_resummed_energies.py DIR/h2o --x 0.7890 --x-ss 0.3809 --x-os 1.0253

`DIR` is a directory holding one molecule's files, or `DIR/<name>` the prefix of one molecule among several. The second example uses the paper's κOOMP2 parameters of OP-MP3(x) and SOP-MP3(x_ss, x_os); the SSD-OPMP3 row always uses `--x-ss`, so there it is at x_ss = 0.3809. Other options: `--alt-resummations`, `--json`, `-o FILE`, `-v`/`-q`.

## 5. Output

`<name>.analysis.out` beside the inputs (or `-o FILE`): the parameters; the base energies `E_HF`, `E_S^(2)` and the traces of the six blocks; then E_c and E_total of every model, the scalar models of each resummation after its orbital ones, under `[Scalar Models]`. `--json` writes the same content to `<name>.analysis.json`.

## 6. Modules

| file | role |
|---|---|
| `compute_resummed_energies.py` | command line, model list, report |
| `config.py` | default (x_ss, x_os), SCS coefficients |
| `blocks.py` | the eight arrays → the three block pairs |
| `resummation.py` | spectral decomposition and the resummation formulas |
| `models.py` | model definitions, names, evaluation |
| `data.py` | reading the W matrices, E_HF and E_S⁽²⁾ |

With this directory on `PYTHONPATH`:

```python
from data import load_system
from models import ModelKey, PADE, SPIN_SEPARATED, precompute, evaluate
sd = load_system("DIR/h2o")                                   # file prefix
key = ModelKey(PADE, SPIN_SEPARATED, x_fixed=(0.45, 1.0))      # SSD-OPMP3(x_ss=0.45)
e_total = sd.e_hf + sd.e_nbs + float(evaluate(precompute(key, {sd.name: sd.blocks}), {})[0])
```
