# Data and pretrained models

## Data provenance and scope

The tables and fields are derived from CLT tokamak MHD simulations and are
distributed separately through a Hugging Face dataset. They are numerical
outputs and post-processed targets, not experimental plasma measurements. The
code repository does not distribute the CLT solver or the raw output of every
simulation campaign.

The separately distributed database is source-available but is not open data.
Copyright (c) 2026 Zhejiang University. All rights reserved. Use requires prior
written permission from Zhejiang University or its duly authorized
representative under the terms published with the Hugging Face dataset.

The extraction scripts should be treated as the authoritative executable
record of the transformations applied to the raw CLT layout. Coordinate shifts,
axis swaps, field normalization, logarithmic transforms, and target-selection
rules differ between workflows; do not combine tables solely by column name
without checking the relevant script.

After downloading the dataset, place its contents under `data/database/` while
preserving the published directory layout. This directory is intentionally
ignored by Git in the code repository.

## Primary scalar tables

| File                                                       | Rows | Columns                                                                                                           | Role                                  |
| ---------------------------------------------------------- | ---: | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| `Double_Tearing_Train_Database_Bisland_Ek.csv`           |  195 | `p0`, `r1`, `r2`, `s1`, `s2`, `Wt_Inner_max`, `Wt_Outer_max`, `gamma`, `Ekmax`, `folder_name` | Main Task I scalar-response table     |
| `Transferlearning_Database_Bisland_Database_Bisland.csv` |   24 | Same schema as the Task I table                                                                                   | Target-domain transfer-learning table |
| `pressure_crash_cls.csv`                                 |  246 | `r1`, `r2`, `s1`, `s2`, `crash_percentage`, `folder_name`                                             | Task II training/evaluation table     |
| `TMONet-test_by_p0.csv`                                  |   22 | `p0`, `r1`, `r2`, `s1`, `s2`, `E_kmax`, `gamma`, `folder_name`                                    | Task III test-case manifest           |

Row counts exclude the CSV header.

## Field directories

- `data/database/data_frame_by_p0/` contains dense field CSVs and rendered
  reference images. Dense CSVs normally use `X`, `Z`, and `Value`.
- `data/database/selected_B9_633/` contains sampled training-field CSVs. The
  Task III loader expects `X`, `Z`, `Value`, and `Gradient`.
- `data/database/TMON-test_by_p0/` contains the 22 dense test fields referenced
  by `TMONet-test_by_p0.csv`, together with reference images for selected cases.

`folder_name` is converted to a field filename by replacing `/` and `\` with
`__` and appending `.csv`. All 22 Task III test-manifest rows have a matching
CSV in `TMON-test_by_p0/` in this snapshot.

The Task III training script expects a separate
`Double_Tearing_Train_Database_by_p0.csv` manifest. The Hugging Face release
contains the exact manifest used for training; the 195-row scalar Task I table
is not a substitute for it. Each manifest entry has a corresponding field in
`selected_B9_633/`.

## Variable glossary

The following descriptions summarize how the variables are used by the code.
The associated paper and CLT setup define the definitive physical units and
normalizations.

| Name                               | Code-level meaning                                                                                            |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `p0`                             | Pressure-related equilibrium parameter read from the equilibrium profile table.                               |
| `r1`, `r2`                     | Radial locations of the two crossings of the selected rational surface (`q = 2` in the extraction scripts). |
| `s1`, `s2`                     | Local magnetic-shear values evaluated at `r1` and `r2`.                                                   |
| `Wt_Inner_max`, `Wt_Outer_max` | Maximum inner and outer magnetic-island widths extracted from `*_Wt_data.csv`.                             |
| `gamma`                          | Average linear growth rate identified from a stable interval of the CLT growth-rate history.                  |
| `Ekmax` / `E_kmax`             | Saturated or maximum kinetic-energy feature. Task I applies `log10(Ekmax)` internally.                      |
| `crash_percentage`               | Largest central-pressure drop found within the configured time window.                                        |
| `X`, `Z` or `R`, `Z`       | Spatial coordinates. Some extraction and inference paths shift and swap axes before use.                      |
| `Value`                          | Field value at a spatial coordinate.                                                                          |
| `Gradient`                       | Grid-gradient magnitude used by the sampled-field loss/weighting workflow.                                    |

## Pretrained artifacts

| Artifact                             | Approx. size | Format and role                                                                                           | SHA-256                                                              |
| ------------------------------------ | -----------: | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `weights/TaskI_10folds.pth`        |      0.04 MB | PyTorch Task I MLP state dictionary                                                                       | `0927C2CBD5F65AC7ACAE28CE321EB8D76279FB6B63C9C4610A3894EB33A3BC60` |
| `weights/TaskII_PCcls.pkl`         |      2.04 MB | Joblib/pickle bundle containing the Task II random forest, scaler, feature names, and threshold           | `A94FB2BB2C4F90FEA5ECB55E2E88FFED457A3C2D089FA7A8FB56C0176FAA6919` |
| `weights/TaskIII_TMOCE1.pth`       |      1.56 MB | PyTorch Task III checkpoint with model/optimizer/scheduler state, configuration, losses, and preprocessor | `45FB66D3275381C16FEED57D3FEB6A33234F9AD2D01AF987FE4C7E22D893F431` |
| `weights/TMO_preprocessor_CE1.pth` |     <0.01 MB | Serialized Task III preprocessing object                                                                  | `3EB30AE52F676CCE2979C2EE119732950750EBD3E0530460B48E37CD5308C178` |

Verify an artifact after download with a platform-appropriate SHA-256 utility,
for example `sha256sum` on Linux or `Get-FileHash -Algorithm SHA256` in
PowerShell.

## Serialization warning

PyTorch and joblib artifacts may contain pickle data. Deserialization is not a
safe operation on untrusted input. Do not call `torch.load` or `joblib.load` on
artifacts from an unknown or unverified source. The checksums above identify
the reference artifacts distributed with this release. Checksums are
release-specific and may differ for artifacts distributed in later versions.

## Reproducible archival record

A reproducible study record identifies the exact Hugging Face dataset revision,
tables, manifests, sampled fields, weights, source commit, and environment
description. A versioned immutable archive can be cited independently of later
changes to the code and dataset repositories.
