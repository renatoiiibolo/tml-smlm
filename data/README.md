# Data

This repository ships one small dataset for demonstration purposes, and cannot ship the datasets the accompanying paper's actual results are based on. This split is deliberate, not an oversight.

## What's here: `weidner_sample/`

21 "cancer cell" nuclei and 21 "skin cell" nuclei, single marker, single snapshot (no repair timecourse, no co-staining), taken verbatim from the sample data accompanying:

> J. Weidner, C. Neitzel, M. Gote, J. Deck, K. A. Küntzelmann, G. Pilarczyk, M. Falk, and M. Hausmann. Advanced image-free analysis of the nano-organization of chromatin and other biomolecules by single molecule localization microscopy (SMLM). *Computational and Structural Biotechnology Journal*, 21:2018–2034, 2023. doi:10.1016/j.csbj.2023.03.009.

Repository: https://github.com/jonasw247/TDA_applied_on_SMLM (see `README-weidner-2023.md` for their own description). Weidner et al. (2023) is also the prior work this group's own SMLM localization software is described against in earlier papers, so this choice keeps continuity with a precedent readers of this repository may already be familiar with.

The files are copied unmodified, including their original `LICENSE` (GPLv3, preserved here as `LICENSE-weidner-2023`) and `README.md` (as `README-weidner-2023.md`). We did not copy Weidner et al.'s precomputed persistence-image outputs (the `PersistenceImages .../` folder in their repository) — only the raw per-nucleus localization CSVs, since this repository recomputes everything from raw data rather than reusing someone else's derived intermediate files.

**Why this data can be redistributed here:** Weidner et al. (2023)'s repository is licensed GPLv3, which permits verbatim redistribution provided copyright and license notices are preserved. This repository is licensed GPLv3 too (see the top-level `LICENSE`), so the terms are compatible by construction.

**What this dataset can and can't demonstrate.** It has no DBSCAN cluster assignments, no repair timecourse, and no co-staining (a single marker per nucleus, no second channel to pair against). That means the tutorial notebook (`notebooks/01_getting_started.ipynb`) can only run Scale C of the feature-extraction pipeline (the whole-point-cloud diagram, which needs no cluster labels) and only Tier 1 (does a topological signature separate the two groups at all) for each of the three Views. It cannot demonstrate Scale A/B (per-cluster, per-focus structure — needs DBSCAN labels this dataset doesn't have), Tier 3 (needs multiple timepoints), Tier 4 (needs a matched control trajectory), or Tier 5 (needs two co-stained markers). Those are fully implemented in `src/tml_smlm/`, just not runnable on this particular demo dataset — see the module docstrings under `view1_scalar/`, `view2_functional/`, and `view3_metric/` for what each one needs.

## What's not here: the Küntzelmann and Hahn archives

The two archives the paper's own findings are based on — Küntzelmann et al. (2026)'s heavy-ion dataset and Hahn et al. (2021)'s X-ray dataset — are not included, per the data-sharing agreement with the originating lab (Kirchhoff-Institute for Physics, Heidelberg University). This matches the manuscript's own Data Availability Statement: the underlying localization data are available within the frame of collaboration, or upon reasonable request to the corresponding authors. `src/tml_smlm/io/kip_archive_loader.py` is included so a reader with access to those archives (or a structurally similar one) can see exactly how the paper's numbers were produced, but it will not run against the sample data in this folder — the two datasets use related but not identical file conventions (see that module's own docstring).
