# The View × Tier framework

The paper this repository accompanies runs the same persistence diagram through three structurally different readings (the "Views"), each pushed through the same ladder of five increasingly strict questions (the "Tiers"). This page is a plain-language map of that framework, for a reader who wants to know what a given module does before opening it. It doesn't replace the manuscript's Methods section — it orients you to it.

Every analysis starts the same way: `features/persistence_features.py` builds one persistence diagram per nucleus (a Vietoris–Rips filtration, computed once), then three separate modules read that diagram three different ways.

## The three Views

**View 1 — scalar / distributional** (`view1_scalar/`)
Reduces the diagram to a handful of summary numbers — Betti-curve descriptors, persistent entropy, landscape scalars — and asks the most direct question: is a difference detectable at all, and does a shape-based measure carry that signal rather than raw localization count? This is the topological generalization of simply counting foci, so it inherits that method's oldest weakness as its own built-in check (Tier 2, below): a scalar summary can be dominated by nucleus size or point count instead of genuine shape.

**View 2 — functional / local** (`view2_functional/`)
Treats the diagram as a function of the birth-radius axis (a persistence landscape) rather than a handful of numbers, then uses sparse fused-lasso regression to ask *where along that axis* a group difference concentrates. This answers a question View 1 structurally cannot: at what spatial scale — focus-level clustering, or finer sub-focus structure — does a detected difference actually live?

**View 3 — metric / geometric** (`view3_metric/`)
Treats each nucleus's entire diagram as a single point in a metric space (Wasserstein distance) and studies how a population of these points is arranged, moves over time, and whether two markers' population-level trajectories track each other. This view alone can ask how a population's damage state evolves geometrically, and needs its own statistical caution (multimodality / variance checks in `view3_metric/population_geometry.py`) because Wasserstein-Fréchet means aren't guaranteed unique the way View 2's landscape means are.

A shared utility, `utils/wasserstein_geometry.py`, provides the pairwise-distance and Fréchet mean/variance machinery View 3 depends on.

## The five Tiers

Each View is pushed through the same five tiers, in order of how strict a question they ask:

- **Tier 1 — existence.** Is a difference detectable at all? (Random-forest classification for View 1, sparse fused-lasso for View 2, permutation testing on the Wasserstein distance matrix for View 3.) `view1_scalar/classification.py`, `view2_functional/localization.py`, `view3_metric/population_geometry.py`.
- **Tier 2 — robustness.** Does a Tier 1 result survive scrutiny — degenerate low-count cases, cross-validation seed choice, and, most importantly, residualization against raw localization count? `view1_scalar/robustness.py`, `view2_functional/robustness.py`, `diagnostics/size_confound_audit.py`, `diagnostics/count_nonlinearity_check.py`, `diagnostics/count_residualization_diagonal_check.py` (a follow-up check on `size_confound_audit.py`'s own finding: does the pipeline's existing count-residualization also happen to fix the size correlation it flagged? Confirmed no — it isn't addressing the same thing).
- **Tier 3 — structure.** How does a detected difference factor across dose, timepoint, cell type, and marker? `view1_scalar/temporal_structure.py`, `view2_functional/temporal_trajectory.py`.
- **Tier 4 — trajectory.** Two related questions: does a population's distance to its own matched control return toward zero over time (baseline-return, Tier 4a), and do two trajectories move together or apart (cross-trajectory, Tier 4b)? `view3_metric/population_geometry.py`, `view3_metric/multiple_comparisons.py` (Holm-corrected multiple comparisons for the baseline-return arm).
- **Tier 5 — coupling.** The strictest question: within a single, verified co-stained nucleus, does one marker's local signature correlate with the other's? `view3_metric/coupling.py`, `view2_functional/coupling.py`.

Not every View × Tier combination is meaningful or buildable for a given dataset — the manuscript gates each cell on whether the data actually supports the test (e.g. Tier 3/4 need multiple real timepoints; Tier 5 needs verified co-staining). `diagnostics/cluster_presence_baseline.py` sits outside this Tier ladder entirely: it's a non-topological baseline (focus presence/absence) that the topological results are compared against, not a View.

## What this repository's demo can and can't show

The bundled demo dataset (`data/weidner_sample/`, see `data/README.md`) has no DBSCAN cluster labels, one marker, and one snapshot per nucleus. That supports View 1/2/3 at Tier 1 only. It cannot demonstrate Tiers 2 through 5, or the per-cluster/per-focus scales the real archives carry — those need data this public sample set was never meant to have. `notebooks/01_getting_started.ipynb` walks through what *is* runnable, and says plainly where it stops; the modules above are the fully worked versions, ready to run against data with the right structure (e.g. the paper's own archives, accessed via `io/kip_archive_loader.py`, not redistributable here — see `data/README.md`).

## Replicate, Resolve, Reveal

Separately from View/Tier — which describes what a given analysis *can test* — the manuscript classifies each of its findings by how it relates to prior published work on the same or related archives:

- **Replicate** — reproduces a previously published statistic.
- **Resolve** — gives a previously descriptive or qualitative claim a formal, testable form.
- **Reveal** — reports something no prior analysis of either source archive addressed.

Several findings are a mix of two of these. This classification isn't implemented as code anywhere in this repository — it's an interpretive label the manuscript applies to a result after the fact, not a property of any module here.
