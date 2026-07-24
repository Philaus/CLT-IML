# CLT-IML

Research code and data for building machine-learning surrogate models from
CLT tokamak magnetohydrodynamic (MHD) simulations.

This repository accompanies a forthcoming paper. It contains utilities for
preparing and post-processing CLT simulation campaigns, three surrogate-modeling
tasks, pretrained model artifacts, evaluation and visualization scripts, and
an overview of the end-to-end workflow. The simulation databases are
distributed separately through Hugging Face.

> **Scope.** This is a research-code snapshot rather than a packaged Python
> library. The CLT solver, the equilibrium solver, and site-specific cluster
> infrastructure are not distributed here. Several scripts use
> working-directory-relative paths and preserve the filenames used during the
> study; read [Running the research scripts](#running-the-research-scripts)
> before execution.

## Software registration and ownership

This repository contains the registered software:

- **Registered title:** 基于托卡马克磁流体数值模拟的深度学习代理模型软件
  [简称：CLT-IML] V1.0
- **English descriptive title:** Deep-learning surrogate model software based
  on tokamak magnetohydrodynamic numerical simulations (CLT-IML), Version 1.0
- **Chinese Computer Software Copyright Registration No.:** `2026SR0627163`
- **Copyright owner:** Zhejiang University
- **Software developer and principal code author:** Y. J. Ma

The source code is publicly viewable for academic communication and
reproducibility assessment. It is **not released under an open-source license**:
use requires prior written permission from Zhejiang University or its duly
authorized representative. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

![CLT-IML workflow](<CLT-IML%20Workflow.png>)

## Research tasks

| Task | Surrogate problem                                        | Inputs                                                         | Targets                                                  | Main implementation                                             | Included artifact                                                                                                                       |
| ---- | -------------------------------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| I    | Scalar response regression                               | `p0`, `r1`, `r2`, `s1`, `s2`                         | `Wt_Inner_max`, `Wt_Outer_max`, `gamma`, `Ekmax` | [`models/MLP_train-5.2.py`](models/MLP_train-5.2.py)           | [`weights/TaskI_10folds.pth`](weights/TaskI_10folds.pth)                                                                               |
| II   | Pressure-crash regression and thresholded classification | `r1`, `r2`, `s1`, `s2`                                 | `crash_percentage`                                     | [`models/classification-2.2.py`](models/classification-2.2.py) | [`weights/TaskII_PCcls.pkl`](weights/TaskII_PCcls.pkl)                                                                                 |
| III  | Two-dimensional field reconstruction                     | `p0`, `r1`, `r2`, `s1`, `s2` and spatial coordinates | Perturbation-field value at each coordinate              | [`models/TMONet-3.2.py`](models/TMONet-3.2.py)                 | [`weights/TaskIII_TMOCE1.pth`](weights/TaskIII_TMOCE1.pth) and [`weights/TMO_preprocessor_CE1.pth`](weights/TMO_preprocessor_CE1.pth) |

Task I uses a multilayer perceptron and 10-fold evaluation to learn four
scalar simulation outcomes. Task II uses a random-forest regressor; the code
uses a pressure-drop threshold of `0.085` when reporting crash/no-crash
classification metrics. Task III is a branch-trunk neural operator that maps
equilibrium parameters and spatial coordinates to a two-dimensional field and
includes a gradient-aware structural loss during training.

The transfer-learning experiment in
[`models/MLP_transfer_learning_EAST-1.0.py`](models/MLP_transfer_learning_EAST-1.0.py)
adapts the Task I MLP to the included target-domain dataset.

## Workflow

The intended research workflow has two stages:

1. **Numerical computation.** Design parameter combinations and equilibrium
   safety-factor profiles, prepare solver inputs, execute equilibrium and CLT
   simulations on the target Linux/GPU environment, then archive the raw
   outputs.
2. **Surrogate modeling.** Extract scalar responses and two-dimensional
   fields, sample the field data, train the three surrogate models, evaluate
   predictions against CLT results, and reconstruct maps over the design
   parameter space.

The editable overview is provided in
[`Program Diagram.pptx`](<Program%20Diagram.pptx>). A detailed file-by-file map
is available in [`docs/REPOSITORY_GUIDE.md`](docs/REPOSITORY_GUIDE.md).

## Repository layout

```text
CLT-IML/
├── data/
│   ├── CLT i-o server/      # equilibrium/case generation and batch helpers
│   ├── CLT visualization/   # MATLAB diagnostics for raw CLT outputs
│   ├── database/            # local location for the Hugging Face dataset
│   └── *.py                 # extraction, comparison, and field sampling
├── models/                  # training and transfer-learning programs
├── visualization/           # analysis, inference, and publication plots
├── weights/                 # pretrained Task I-III artifacts
├── CLT-IML Workflow.png     # workflow figure shown above
├── Program Diagram.pptx     # editable program-flow introduction
├── HUGGINGFACE_DATASET_CARD.md
├── CITATION.cff
├── LICENSE
└── NOTICE
```

The `data/database/` directory is ignored by Git so that the database is not
accidentally committed to GitHub. See
[`docs/DATA_AND_MODELS.md`](docs/DATA_AND_MODELS.md) for schemas, artifact
checksums, coordinate conventions, and reproducibility notes.

## Data availability

The complete simulation database will be hosted as a separate Hugging Face
dataset. After downloading it, preserve its directory structure and place its
contents under `data/database/` in this repository. The public dataset URL will
be added here after upload.

[`HUGGINGFACE_DATASET_CARD.md`](HUGGINGFACE_DATASET_CARD.md) is a prepared
dataset-card template. Copy it to the root of the Hugging Face dataset
repository as `README.md`, then add the final dataset identifier, license, DOI,
and verified file inventory.

## Installation

The exact package versions from the original computing environments were not
recorded in this snapshot. A practical starting point is Python 3.10 or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with the local CPU/CUDA environment if the
generic `torch` package in `requirements.txt` is not appropriate. Task I and
Task III training are designed to benefit from a GPU; the remaining analyses
can run on a CPU.

Additional software is required only for the numerical-computation side:

- MATLAB for the scripts in `data/CLT visualization/` and
  `q_profile_design.m`;
- a Linux/Bash environment for the shell workflows;
- authorized installations of CLT and the equilibrium-solver toolchain used
  by the research group.

## Running the research scripts

### Important path convention

Most programs read and write files relative to the **current working
directory**, not relative to the script itself. Paths are not exposed through a
command-line interface. Run a script from a directory containing the filenames
it expects, or edit the path constants in its `if __name__ == "__main__"`
block. Output figures and checkpoints are also written to that directory.

Filenames containing spaces should be quoted at the shell.

### Analyses that use the included tables

The two correlation analyses can be run directly from the database directory:

```bash
cd data/database
python "../../visualization/TaskI_correlation analysis-1.1.py"
python "../../visualization/TaskII_correlation analysis-1.2.py"
```

They read `Double_Tearing_Train_Database_Bisland_Ek.csv` and
`pressure_crash_cls.csv`, respectively, and create correlation figures in the
current directory.

### Task I training

```bash
cd data/database
python ../../models/MLP_train-5.2.py
```

This launches the full 10-fold training/evaluation workflow and writes a new
`TaskI_10folds.pth` checkpoint plus figures to the current directory. It does
not overwrite the copy under `weights/` unless it is run there with the
corresponding data staged alongside it.

For the transfer-learning program, place
`weights/TaskI_10folds.pth` and
`data/database/Transferlearning_Database_Bisland_Database_Bisland.csv` in the
working directory expected by the script, then run
`models/MLP_transfer_learning_EAST-1.0.py`.

### Task II training and parameter-space plots

The Task II entry point expects the training table to be named
`TaskII_PCcls.csv`; the included equivalent table is
`data/database/pressure_crash_cls.csv`. Create a working copy with the expected
name (or change line 915 of the script) before running
`models/classification-2.2.py`. The script trains the model, saves
`TaskII_PCcls.pkl`, evaluates decision thresholds, and contains optional calls
for two- and three-dimensional parameter-space scans.

### Task III inference

The included test manifest and full test fields are
`data/database/TMONet-test_by_p0.csv` and
`data/database/TMON-test_by_p0/`. The inference program currently expects the
following working-directory names:

| Included file                        | Name expected by the inference script |
| ------------------------------------ | ------------------------------------- |
| `weights/TaskIII_TMOCE1.pth`       | `TMONet_model_CE1.pth`              |
| `weights/TMO_preprocessor_CE1.pth` | `preprocessor_CE1.pth`              |

Stage copies or links with those names in `data/database/`, then run:

```bash
cd data/database
python "../../visualization/TaskIII_TMOpredict-2.3.py"
```

The program reconstructs each test field, saves per-case comparison figures,
and reports aggregate PSNR, SSIM, and timing statistics.

Task III retraining requires a training manifest named
`Double_Tearing_Train_Database_by_p0.csv` and the sampled fields under
`selected_B9_633/`. Preserve those names in the Hugging Face dataset (or update
the path constants in `models/TMONet-3.2.py`) and verify that every manifest row
maps to a sampled field before publication.

## Data and artifact safety

The `.pth` and `.pkl` files use Python/PyTorch serialization. Loading a
serialized artifact can execute Python code. Only load artifacts obtained from
this repository or another trusted source, and verify the SHA-256 checksums in
[`docs/DATA_AND_MODELS.md`](docs/DATA_AND_MODELS.md) when provenance matters.

The data are derived from numerical simulations and are not experimental
measurements. Units and normalization follow the associated CLT setup and the
transformations implemented in the extraction scripts; consult the paper for
the definitive physical conventions.

## Reproducibility

Random seeds are fixed in the principal training programs where implemented,
but exact numerical reproduction can still depend on the PyTorch version,
CUDA/cuDNN backend, device type, and nondeterministic GPU kernels. Report the
software environment, hardware, data split, and any edited path constants when
publishing derived results.

## Citation

The associated manuscript is currently titled:

> W. Zhang, Y. J. Ma, S. Z. Cai, Z. W. Ma, Z. M. Sheng, and Y. Zhang,
> “Millisecond prediction of tokamak disruptions using a first-principles
> surrogate model,” manuscript in preparation.

W. Zhang and Y. J. Ma contributed equally. W. Zhang and Z. W. Ma are the
corresponding authors.

Affiliations:

1. Institute for Fusion Theory and Simulation, School of Physics, Zhejiang
   University, Hangzhou 310027, China.
2. College of Control Science and Engineering, Zhejiang University, Hangzhou
   310027, China.

Machine-readable metadata are provided in [`CITATION.cff`](CITATION.cff). Add
the final venue, year, DOI/arXiv identifier, ORCIDs, repository URL, and release
version when available.

## License

Copyright (c) 2026 Zhejiang University. All rights reserved.

The source code is publicly accessible, but public access does not grant a
right to use, execute, copy, modify, distribute, deploy, or commercialize it.
Except as expressly permitted by applicable law, prior written permission must
be obtained from Zhejiang University or its duly authorized representative.
See the complete [`LICENSE`](LICENSE) and registration [`NOTICE`](NOTICE).

The separately hosted Hugging Face database requires its own dataset license;
the software terms in this repository do not automatically apply to it.
Authorized academic users are also requested to cite the associated paper and
the versioned software/data release.

## Contributing

Bug reports and reproducibility improvements are welcome. Please read
[`CONTRIBUTING.md`](CONTRIBUTING.md) before opening an issue or pull request.
