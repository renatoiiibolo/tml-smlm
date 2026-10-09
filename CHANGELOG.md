# Changelog

Each entry says what changed and which part of the paper it keeps in step with.

## 0.2.0

Released alongside the revised manuscript.

### Added
- `view1_scalar/descriptor_selection.py`: the rule that picks each stratum's leading descriptor (largest signal-to-noise ratio of the between-group gap, computed on count-residualized trajectories) as a callable function, with a leave-one-timepoint-out check. The winner is unchanged in 6/6, 6/6, 7/7 and 7/7 drops for the four panels of Figure 4. `--basis raw` reproduces the original raw-trajectory basis for comparison. *Paper: Results, "Leading discriminative descriptor"; Supplementary Methods S1.*
- `view2_functional/penalty_comparison.py`: refits the headline panels with a true L1 + L1 fused lasso (ADMM, checked against an independent solver) and sweeps its regularization strength. No strength gives a sparsity level near the smooth-lasso fit. It also holds the shape diagnostic for the fitted coefficient profile. *Paper: Supplementary Methods S1, "Selection-stability checks".*

### Changed
- View 2 is named **smooth-lasso** throughout the prose and logs. The penalty (L1 plus an L2-squared penalty on adjacent differences, fitted by an augmented Lasso) is the Smooth-Lasso of Hebiri and van de Geer (2011). Localizing a discriminative region along an ordered predictor follows Tibshirani et al. (2005), whose fused lasso uses an L1 difference penalty instead. Identifiers (`fl_cv`, `fl_` prefixes, the `fused_lasso` result key) are unchanged so existing code keeps working. *Paper: Methods, View 2.*
- `utils/wasserstein_geometry.py`: the Fréchet-mean credit now reads Turner et al. (2014) for the iterative update, notes that this implementation starts from the medoid (reproducible run to run) instead of a random draw, and cites Munch et al. (2015) only for the contrast with a probabilistic mean.
- Coupling p-values (`view3_metric/coupling.py`, `view2_functional/coupling.py`) are stored to four significant figures when below 1e-3, instead of six decimals, so values of order 1e-13 to 1e-7 are no longer stored as 0.0. Values of 1e-3 and above are stored as before. *Paper: Figure 3 annotations.*
- `figures/paper_figures.py`: coupling annotations recompute very small p-values from the correlation and sample size (for example 3.0e-13 for the strongest stratum), and a stored zero is reported as "p < 5e-7". The Figure 5 y-axis reads "Discriminative spatial scale (nm)".
- The tutorial notebook reports permutation p-values with the standard (B+1) floor: a permutation p-value is never exactly 0, so View 1 (100 permutations) reads 0.0099 and View 3 (1,000 permutations) reads 0.0010 where it previously printed 0.0000. The library functions are unchanged and still return the raw proportion.
- `CITATION.cff` moves to 0.2.0 and lists Michael Hausmann's Heidelberg affiliation only.
- `docs/view_tier_overview.md`, `README.md` and the tutorial notebook use the smooth-lasso name and list the two new modules. The notebook's balanced accuracy, Fréchet variances and observed statistics are unchanged.

### Not changed
- Other modules still round stored p-values to six decimals. None of their p-values is printed to more than two significant figures in the paper, so nothing quoted is affected.
- The demo dataset limits the tutorial to Scale C and Tier 1, as before.
