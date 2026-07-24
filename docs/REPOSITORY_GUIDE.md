# Repository guide

This guide describes the role of each source file at the level of filenames,
comments, entry points, and input/output operations. It is not an API reference
or a line-by-line description of the numerical implementation.

## End-to-end map

```text
profile design
    -> equilibrium input generation and checking
    -> CLT case generation and batch execution
    -> raw-output diagnostics and archiving
    -> scalar and 2D-field extraction
    -> sampled/tabulated datasets
    -> Tasks I, II, and III
    -> inference, evaluation, and parameter-space visualization
```

## `data/`: simulation workflow and post-processing

### Top-level Python programs

| File | Purpose | Principal inputs | Principal outputs |
| --- | --- | --- | --- |
| `data/data_extract-2.3.py` | Current scalar extractor for double-tearing-mode cases. Finds two `q = 2` rational surfaces, local shear, a linear growth-rate interval, maximum inner/outer island widths, and saturated kinetic energy. | Per-case `energy.dat`, `q_p_g.dat`, and `*_Wt_data.csv` files | `Double_Tearing_Train_Database_Bisland.csv` and diagnostic plots |
| `data/data_extract-2.2.py` | Earlier extraction path that also exports a selected two-dimensional `B_y` field and applies the coordinate/normalization conventions used by the `by_p0` workflow. | `energy.dat`, `q_p_g.dat`, `x12d*`, `gridxx.dat`, `gridzz.dat` | Scalar CSV, `data_frame_by_p0/*.csv`, and diagnostic plots |
| `data/pressure_crush_extract-2.py` | Builds the Task II target by tracking the mean central pressure and finding the largest pressure drop in a fixed time window. | `energy.dat`, `q_p_g.dat`, `pt0.dat`, `x12d*`, `gridxx.dat`, `gridzz.dat`, `nstt.dat` | `pressure_crash_cls.csv` and per-case pressure plots |
| `data/compare_plot.py` | Compares central-pressure time histories for two CLT cases. | Two raw CLT case directories | `pressure_comparison_*.png` |
| `data/2Dfield_select-2.py` | Computes grid-gradient magnitudes and performs stratified sampling that emphasizes high-value and high-gradient regions. | Full field CSVs with `X`, `Z`, and `Value` | Sampled field CSVs with an added `Gradient` column |

The `if __name__ == "__main__"` blocks contain study-specific folder names.
Adjust those paths when processing a new campaign.

### `data/CLT i-o server/`

These files support the equilibrium-to-CLT automation shown in
`Program Diagram.pptx`.

| File | Role |
| --- | --- |
| `q_profile_design.m` | Searches a parameter grid for safety-factor profiles matching requested equilibrium characteristics and exports parameter tables. |
| `generate_ineq.py` | Reads parameter CSVs, creates consistently named case workspaces, and modifies equilibrium/transport inputs for batch calculation. |
| `generate_ineq_EAST_all.py` | Batch workspace generator for the EAST-oriented equilibrium/case setup. Its configuration block lists the site-specific files and directories to copy. |
| `inequ` | Template/configuration input used by the equilibrium workflow. |
| `equilibrium.sh` | Shell driver for the equilibrium solver and conversion stage. |
| `auto_equ.py` and `auto_equ.sh` | Automation helpers for running and organizing repeated equilibrium calculations. |
| `show_q.py` | Plots/checks generated safety-factor profiles against the parameter CSVs. |
| `extract csv.py` | Extracts equilibrium information from generated workspaces and produces summary plots/tables. |
| `result sorting.py` | Collects completed results and copies diagnostic scripts into case directories. |
| `run2080.sh`, `run4090.sh` | Server launch scripts for the two GPU environments used during the study. Treat scheduler paths and executable names as site-specific. |

The workflow refers to external solver/executable files such as `eq_transp.F`
and to a configured CLT calculation directory. Those components are not
included in this repository.

### `data/CLT visualization/`

| File | Diagnostic role |
| --- | --- |
| `contour_pt.m` | Pressure/field contour visualization for a CLT case. |
| `paper_fft_gr_figure.m` | Frequency/growth-rate figure preparation. |
| `plot_growth.m` | Growth-rate evolution plot. |
| `plot_ke_fft.m` | Kinetic-energy/frequency diagnostic. |
| `plot_wt.m` | Inner and outer magnetic-island-width evolution. |
| `plot_x12d.m` | Two-dimensional CLT output-field visualization. |

These MATLAB programs operate on the native CLT output layout and are copied
or invoked by the surrounding workflow rather than imported by the Python
models.

## `models/`: surrogate training

| File | Role |
| --- | --- |
| `MLP_train-5.2.py` | Task I MLP training with 10-fold splitting, standardization, early stopping, scalar-response evaluation, and best-fold checkpoint export. The kinetic-energy target is transformed with `log10` during preprocessing. |
| `MLP_transfer_learning_EAST-1.0.py` | Loads the Task I checkpoint, freezes most network layers, fine-tunes the final layer on the 24-case target-domain table, and studies performance versus target-domain sample count. |
| `classification-2.2.py` | Task II random-forest regression, Bayesian hyperparameter search, threshold evaluation, model serialization, two-dimensional pair scans, and optional three-dimensional decision-boundary visualization. |
| `TMONet-3.2.py` | Task III branch-trunk neural-operator training on sampled field points. Splits by physical case before point expansion, standardizes parameters/coordinates, applies asymmetric field scaling, and combines value and spatial-gradient losses. |

## `visualization/`: analysis and inference

| File | Role |
| --- | --- |
| `TaskI_correlation analysis-1.1.py` | Pearson correlations and p-values between the Task I design variables and scalar outputs; produces a publication-style heatmap. |
| `TaskII_correlation analysis-1.2.py` | Pearson correlations and p-values for the pressure-crash dataset. |
| `TaskIII_TMOpredict-2.3.py` | Loads the Task III model/preprocessor, reconstructs all test fields, compares predictions with CLT targets, records inference time, and summarizes PSNR/SSIM and global error metrics. |
| `r1-WBi~Cp.py` | Relates the first rational-surface location and inner island width to pressure-crash percentage; also produces a confusion matrix. |
| `check_select_filed-1.py` | Interpolates and plots sampled field CSVs to visually check the sampling result. The filename preserves the original project spelling. |
| `performance vs data.py` | Plots model error against data quantity from a user-supplied `performance vs data.xlsx`. That workbook is not included. |

## Versioned filenames

Suffixes such as `-2.3`, `-5.2`, and `-1.0` are research-script revision
labels, not Python package versions. The files are standalone programs and are
not imported as conventional Python modules. Renaming them is possible, but any
paper, workflow diagram, or archived command that refers to the original names
should be updated at the same time.

## Recommended working-directory discipline

For reproducibility, create a separate run directory and copy or link only the
required manifest, field directory, and checkpoint into it. Record:

1. the source commit or release;
2. every filename alias or edited path constant;
3. the Python/PyTorch/CUDA environment;
4. the random seed and train/validation/test split; and
5. the generated outputs kept for the paper.

This avoids mixing newly generated checkpoints and figures with the immutable
reference artifacts under `weights/` and the downloaded Hugging Face dataset
under `data/database/`.
