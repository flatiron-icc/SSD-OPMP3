# Resummed MP3 energies from W matrices

Evaluates the correlation-energy models of K. Bystrom, T. C. Berkelbach and D. Hait, arXiv:2609.25330, from per-molecule W matrices, forms reaction energies from a stoichiometry table, and benchmarks them against CCSD(T) in the same basis. No quantum chemistry is run. Energies are in Hartree internally and in kcal/mol in every printed statistic.

## 1. Requirements

Python ≥ 3.12, `numpy`, `scipy`, `pyyaml`. Nothing to install: run `python /path/to/script.py`. Tested with Python 3.12.12, numpy 2.4.4, scipy 1.17.1, pyyaml 6.0.3.

## 2. Models

Per molecule, the eight arrays `W{2,3}{alpha,beta}_{ss,os}` (the same-spin and opposite-spin parts of the alpha and beta blocks of W⁽²⁾ and W⁽³⁾, Eqs. 17–18) are assembled (`blocks.py`) into three block-diagonal pairs: `W2`, `W3` (same-spin plus opposite-spin: the full W⁽²⁾, W⁽³⁾), `W2_ss`, `W3_ss` and `W2_os`, `W3_os`. A spin block in which both matrices vanish (e.g. the beta same-spin block of a system with one beta electron) is dropped. With P(A, B) = A (A − B)⁻¹ A (Eq. 13), E2 = Tr W2 and E3 = Tr W3:

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

so one decomposition per pair serves every x of a fit; expanding the fraction gives E2 + x E3 at first order and the higher orders as a geometric series. W2⁺ is the pseudo-inverse of W2, which is its inverse unless orbitals are frozen: a frozen orbital has zero rows in W2 and W3 and drops out, with cᵢ = ζᵢ = 0.

**Fits** (`evaluation.py`). A fitted model minimizes the RMSE over one dataset (`--fit-dataset`, default TAE_W4-17nonMR), each parameter restricted to [−1, 2]: one parameter by bounded Brent, two by a 0.25 spacing grid followed by bounded Nelder–Mead. A fit ending on a bound is flagged. Twelve models are fitted (`models.fit_model_keys`): the MP series, the orbital Padé and the scalar Padé (below), each with the third order scaled by x, by (x_ss, x_os), by x_ss at x_os = 1, and by x_os at x_ss = 0. The paper reports MP2.X, MP2.X(x_ss, x_os), OP-MP3(x), SOP-MP3(x_ss, x_os) and SSD-OPMP3(x_ss).

**Also evaluated, not in the paper.** SCS-MP2, SOS-MP2 and SCS-MP3 with their literature coefficients (`config.py`). The scalar variant of each Padé model, written without the O (P-MP3, SP-MP3, SSD-PMP3), in which every Tr P(A, B) becomes (Tr A)² / (Tr A − Tr B); P-MP3 is the paper's E_c^[2,3] (Eq. 1), which is not size-consistent. With `--alt-resummations`, four other resummations, Padé–Borel, exponential, Landau–Zener and logarithmic (the paper mentions the first and the third), each replacing 1/(1 − z) by a function with the same value and slope at z = 0.

## 3. Inputs

**W-matrix trees.** Per molecule `<name>`, the eight files `<name>.W{2,3}{alpha,beta}_{ss,os}.npy` and `<name>.semicanonical.out`, whose lines `HF energy from density : E_HF` and `Non-Brillouin singles E2 : E_S⁽²⁾` are read (or `<name>.final.out`, from its `Reference energy (HF on ... density)` and `Non-Brillouin singles correction` lines). A tree holds one directory per reference determinant (`<xc>`, e.g. `KOOMP2`), with one directory per molecule or all molecules side by side:

    <tree>/<xc>/<name>.molden/<name>.*   --molden
    <tree>/<xc>/<name>.xyz/<name>.*      --xyz
    <tree>/<xc>/<name>.*                 --flat

With several trees, `--molden` trees are searched first, then `--xyz`, then `--flat`, each in command-line order; the first tree holding a molecule supplies it.

**Reaction table** (`--rxn-file`, YAML): `<reaction>: {Dataset: ..., structs: [...], counts: [...]}`. A reaction energy is Σ counts × E_total(structs).

**Reference energies** (`--refs-file`, `--ccsd-file`, YAML): `<structure>: <total energy, Hartree>`. Every structure needs a CCSD(T) total; CCSD totals are optional and add a CCSD row.

## 4. Running

    python do_analysis.py --xc KOOMP2 --x-ss 0.45 --molden TREE --xyz TREE \
        --rxn-file reactions.yaml --refs-file ccsdt.yaml --ccsd-file ccsd.yaml \
        --datasets TAE_W4-17nonMR BH76RC A24 --output-dir OUT

fits the parametric models on `--fit-dataset` (added to the run if absent) and evaluates every model on every dataset. Other options: `--no-fit`, `--x-os` (default 1), `--flat`, `--alt-resummations`, `-v`/`-q`. A molecule missing from every tree is skipped with a warning, and the reactions that need it are dropped.

One molecule, every model, energies in Hartree, written to `<name>.analysis.out` beside the inputs (or `-o FILE`):

    python compute_resummed_energies.py TREE/KOOMP2/W4-17_h2o.xyz --x-ss 0.45
    python compute_resummed_energies.py TREE/KOOMP2/W4-17_h2o.xyz --x 0.7890 --x-ss 0.3809 --x-os 1.0253

`--x` adds the one-parameter fitted models at that x, `--x-os` the two-parameter ones at (`--x-ss`, `--x-os`), `--json` a JSON copy. The SSD-OPMP3 row always uses `--x-ss`, so in the second example it is at x_ss = 0.3809.

## 5. Outputs of `do_analysis.py`

Per dataset:

- `<dataset>.default.csv`: per reaction, the reference energy and the signed error (predicted − reference) of every parameter-free model, CCSD first.
- `<dataset>.fit_eval.csv`: the same for the fitted models.
- `<dataset>.report.txt`: MAE, RMSE, MSE and maximum error per model, with the worst reaction and the W-matrix trees used. A reaction on which a model hits a Padé pole is excluded from that model's statistics and tagged `[!! n non-finite]`.

Once per run, `fits.csv`: the fitted parameters, in-sample statistics, and convergence and bound flags.

## 6. Modules

| file | role |
|---|---|
| `config.py` | unit conversion, SCS coefficients, fit bounds and tolerances |
| `blocks.py` | the eight arrays → the three block pairs |
| `resummation.py` | spectral decomposition and the resummation formulas |
| `models.py` | model definitions, names, evaluation over many molecules |
| `data.py` | reading W matrices, reports, reaction and reference tables |
| `metrics.py`, `evaluation.py` | reaction matrix, statistics, fitting |
| `output.py`, `cli.py` | writers, shared command-line options |

With this directory on `PYTHONPATH`:

```python
from data import load_system
from models import ModelKey, PADE, SPIN_SEPARATED, precompute, evaluate
sd = load_system("TREE/KOOMP2/W4-17_h2o.xyz/W4-17_h2o")        # file prefix
key = ModelKey(PADE, SPIN_SEPARATED, x_fixed=(0.45, 1.0))      # SSD-OPMP3(x_ss=0.45)
e_total = sd.e_hf + sd.e_nbs + float(evaluate(precompute(key, {sd.name: sd.blocks}), {})[0])
```
