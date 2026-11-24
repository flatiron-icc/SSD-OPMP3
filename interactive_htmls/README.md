# Interactive figures

Four interactive HTMLs showing the performance of SSD-OPMP3 and other resummed methods.

| page | shows |
|---|---|
| `errors_by_model.html` | Fig. 2 for any reference determinant and dataset: a violin per model (CCSD, MP3, OP-MP3, SOP-MP3, MP2.X, OP-MP3(x), SSD-OPMP3) with RMSE, MAE and MSE above it; a table of these statistics and of each model's largest error; the error of every model on every reaction |
| `errors_by_reference.html` | Fig. 3 for any model and dataset: the same, with a violin per reference determinant the dataset was computed with |
| `statistics_by_dataset.html` | Figs. 4–6 for any dataset: the RMSE, MAE, MSE or largest \|error\| of every model with κOOMP2 and PBE0, as bars, and the reaction of each largest error |
| `x_ss_x_os_landscape.html` | for any reference determinant and dataset, the RMSE, MAE, MSE or largest \|error\| of SOP-MP3 (Eq. 21), its scalar counterpart SP-MP3 and MP2.X over x_ss, x_os = 0, 0.01, …, 2, with SSD-OPMP3 (x_os = 1 and the paper's x_ss) marked; a table gives the statistic on every dataset at that point and at each dataset's minimum |

Coverage: κOOMP2 and PBE0, all 32 datasets; HF, SPW92, PBE, r²SCAN, ωB97X-V and ωB97M-V, W4-17 TAE nonMR.

- Errors are predicted minus CCSD(T) in the same basis, in kcal/mol, or in percent for Dip146 and Pol130 (defined on the pages). MP2.X and OP-MP3(x) are at the reference's x fitted on W4-17 TAE nonMR, SSD-OPMP3 at the paper's x_ss and x_os = 1.
- A reaction is shown with its ID and as reactants → products, structures with negative coefficients on the left. Dip146 and Pol130 items are named by molecule.
- Violins are Gaussian kernel densities with Scott's bandwidth, on one scale, from 5% of the range below the smallest error to 5% above the largest; the white bar is the mean. Errors beyond the axis are counted at its edge and included in the statistics. Datasets of at most 600 items also show each error as a point.
- The error axis is that of Fig. 2 for W4-17 TAE nonMR on the page by model, and that of Fig. 3 for SSD-OPMP3 on W4-17 TAE nonMR on the page by reference; otherwise it spans the pooled 1st to 99th percentiles of the violins, and zero.
