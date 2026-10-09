# tml-smlm

Topological machine learning for single-molecule localization microscopy (SMLM) data: a multi-scale, confound-audited pipeline for finding and characterizing punctate subcellular structures (here, DNA-damage-response foci) from raw localization point clouds.

This repository is the companion code for the manuscript *"Multi-scale topological machine learning disentangles chromatin damage nano-architecture in irradiated cells"* (Bolo, Schäfer, Bagunu, Hildenbrand, and Hausmann; currently under review — see `CITATION.cff`). It restructures that paper's own analysis pipeline into an installable package organized directly around the paper's own conceptual framework, so a reader who has not seen the original development history can find "the code that made Figure X" by following the paper's own vocabulary through the file tree.

## What's here

The pipeline runs one persistence diagram per nucleus through three structurally different readings ("Views"), each pushed through the same five-tier ladder of increasingly strict questions ("Tiers"). See **[`docs/view_tier_overview.md`](docs/view_tier_overview.md)** for the full plain-language map — that's the right starting point if you're new to this repository.

```
src/tml_smlm/
├── io/                  — data loading
│   ├── kip_archive_loader.py   (the real archive loader; included for transparency, not runnable on the demo data)
│   └── demo_loader.py          (loads this repo's own public demo dataset)
├── features/
│   └── persistence_features.py (builds the shared persistence diagram every View reads)
├── view1_scalar/        — View 1: scalar / distributional summaries
│   └── descriptor_selection.py   (picks each stratum's leading descriptor and checks it survives dropping any one timepoint)
├── view2_functional/    — View 2: functional / local (persistence landscapes, smooth-lasso localization)
│   └── penalty_comparison.py     (checks the smooth-lasso penalty against a true L1 + L1 fused lasso)
├── view3_metric/        — View 3: metric / geometric (Wasserstein / Fréchet)
├── diagnostics/         — non-topological baseline + confound audits
│   ├── size_confound_audit.py
│   └── count_residualization_diagonal_check.py  (follow-up: does existing count-residualization also fix the size confound above? — no)
├── utils/                — shared Wasserstein/Fréchet geometry helpers
└── figures/              — plotting functions behind the manuscript's figures

data/                     — the public demo dataset (see data/README.md)
notebooks/                — a runnable tutorial, notebooks/01_getting_started.ipynb
docs/                     — docs/view_tier_overview.md
```

## Quick start

With conda:

```bash
# create the environment
conda env create -f environment.yml
conda activate tml-smlm

# install this package in editable mode
pip install -e .

# run the tutorial
jupyter lab notebooks/01_getting_started.ipynb
```

Without conda, a plain virtual environment works just as well — `pyproject.toml` declares every dependency the package itself needs, so `pip install -e .` pulls them all in:

```bash
python3 -m venv tml-smlm-env
source tml-smlm-env/bin/activate      # on Windows: tml-smlm-env\Scripts\activate

pip install -e .
pip install jupyterlab ipykernel

# make this environment selectable as a notebook kernel
python -m ipykernel install --user --name tml-smlm --display-name "Python (tml-smlm)"

jupyter lab notebooks/01_getting_started.ipynb
```
In Jupyter, select Kernel → Change Kernel → "Python (tml-smlm)" before running cells, since Jupyter does not automatically pick up a newly created environment's kernel.

The tutorial notebook loads the bundled public demo dataset, computes real persistence diagrams, and runs a worked (if intentionally limited) example of each of the three Views at Tier 1. It takes a few minutes to run end to end — almost all of that time is genuine persistent-homology computation (`ripser`) and permutation testing, not overhead. See the notebook's own opening cells for what it can and can't show on this particular dataset.

## Data

This repository ships one small public demo dataset for the tutorial. It does **not** ship the localization archives the paper's own results are computed from — those cannot be redistributed under the data-sharing agreement with the originating lab. See **[`data/README.md`](data/README.md)** for exactly what's here, where it came from, and why the rest isn't.

`src/tml_smlm/io/kip_archive_loader.py` is included so a reader with access to a structurally similar archive can see exactly how the paper's own numbers were produced; it will not run against the bundled demo data (different file convention).

## Citing this work

See **[`CITATION.cff`](CITATION.cff)**. In short: cite the manuscript above if you use this pipeline, and separately cite Weidner et al. (2023) if you use the bundled demo dataset.

## License

GPLv3 — see **[`LICENSE`](LICENSE)**. The bundled demo dataset is redistributed under the same license from its original source (Weidner et al., 2023); see `data/README.md` for details.
