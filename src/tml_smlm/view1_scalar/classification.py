#!/usr/bin/env python3
"""
Classification analysis for View 1 (scalar / distributional topological summaries):
does a persistent-homology-derived signature discriminate between the two compared
groups in a stratum (e.g. cell line, within one marker and radiation source), and
does that discrimination survive fold-safe residualization against raw localization
or cluster count? This implements Tier 1 (does a classifiable signal exist) and
Tier 2 (is it robust to the count confound) of the paper's validation framework.

Four feature sets are evaluated per marker/source stratum:
  - Scale B        cluster-level Betti-curve / landscape descriptors
  - Scale C        point-cloud descriptors from the full localization cloud
  - Scale C+noise  Scale C plus descriptors of the background/noise cloud
  - All-scales     the above plus mean-only Scale A summaries, pooled

For each feature set, a random forest is cross-validated with and without
residualizing every feature against the count covariate it is expected to
correlate with (cluster-derived features against n_clusters, whole/noise-cloud
features against n_localisations). Raw-vs-residualized accuracy is compared with
a paired Wilcoxon test, and the residualized accuracy is checked against a
timepoint-stratified permutation null. A residualization diagnostic (r² of each
feature against its assigned covariate) confirms the correction is doing real
work rather than residualizing against the wrong count. Runs are repeated on the
full population and on an irradiated-only subset, to separate a genuine cell-line
effect from one driven by control-vs-treated separation, and a cross-source
comparison checks whether a topological signal transfers across radiation quality
on the one marker both sources share.

INPUTS
------
  data/feature_matrix.csv   (produced by the feature-extraction step of the pipeline)

OUTPUTS
-------
  results/classification_results.json
  results/classification_feature_importance.csv
  results/classification_summary.txt

USAGE
-----
  # Smoke test first -- full-scale permutation runs take real time
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  python -u classification.py --n-permutations 200 --n-jobs -1

  # Full run
  python -u classification.py --n-permutations 10000 --n-jobs -1

DEPENDENCIES
------------
  numpy  pandas  scipy  scikit-learn  joblib
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

N_TREES = 500
N_SPLITS = 5
N_REPEATS = 10
CV_SEED = 42
N_PERMUTATIONS_DEFAULT = 10_000

SOURCE_CONFIG: Dict[str, Dict] = {
    "kuentzelmann": {
        "target_col": "cell_type", "pos_label": "U87", "ref_label": "NHDF",
        "radiation_type": "heavy-ion",
    },
    "hahn": {
        "target_col": "cell_type", "pos_label": "MCF7", "ref_label": "HGF",
        "radiation_type": "photon",
    },
}
SOURCES = ["kuentzelmann", "hahn"]
RESID_COVARIATE = {"scaleA": "n_clusters", "scaleB": "n_clusters", "scaleC": "n_localisations", "noise": "n_localisations"}


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE COLUMN GROUPS
# ══════════════════════════════════════════════════════════════════════════════

def _cols(df: pd.DataFrame, prefix: str, exclude_suffix: Optional[str] = None) -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if exclude_suffix:
        cols = [c for c in cols if not c.endswith(exclude_suffix)]
    return cols


def scale_a_mean_cols(df: pd.DataFrame) -> List[str]:
    """Scale A columns, `_sd` excluded. Within-nucleus spread computed over fewer
    than 3 cluster values is sampling noise, not shape information, and every
    cell-type stratum in this data has a substantial fraction of nuclei below
    that threshold -- so the `_sd` columns are dropped everywhere, and only
    `_mean` is kept."""
    return _cols(df, "scaleA", exclude_suffix="_sd")


def scale_b_cols(df: pd.DataFrame) -> List[str]:
    return _cols(df, "scaleB")


def scale_c_cols(df: pd.DataFrame) -> List[str]:
    return _cols(df, "scaleC")


def noise_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in _cols(df, "noise_") if c != "noise_fraction"]


# ══════════════════════════════════════════════════════════════════════════════
# FOLD-SAFE RESIDUALISER
# ══════════════════════════════════════════════════════════════════════════════

class FoldSafeResidualiser(BaseEstimator, TransformerMixin):
    """
    Fit OLS(feature ~ its own covariate(s)) on the training fold only, apply to the
    test fold. Fitting on the training fold alone and applying the fitted coefficients
    to the held-out fold keeps the covariate adjustment from leaking test-set
    information into the model, the same discipline as any other fold-scoped
    preprocessing step in a cross-validated pipeline.

    Two calling conventions, exactly one required:
      - count_cols=<str or list>: every feature column uses the same covariate(s).
        Correct whenever a run's whole feature set shares one covariate -- Scale B
        alone (n_clusters) or Scale C(+noise) alone (n_localisations).
      - covariate_map=<dict {feature_col: covariate_col_or_list}>: each feature
        column is residualized against its own designated covariate, in one
        fit/transform call. This is what a run mixes feature groups that don't
        share a covariate needs -- Scale A/B (n_clusters) together with Scale
        C/noise (n_localisations) in the "all-scales" run. A feature with no
        entry in covariate_map raises KeyError rather than silently falling back
        to a default, so an unassigned feature fails loudly instead of getting
        residualized against the wrong count.
    """

    def __init__(self, count_cols: Union[str, List[str], None] = None, covariate_map: Optional[Dict[str, Union[str, List[str]]]] = None):
        if (count_cols is None) == (covariate_map is None):
            raise ValueError("FoldSafeResidualiser needs exactly one of count_cols or covariate_map, not both or neither.")
        self.count_cols = count_cols
        self.covariate_map = covariate_map

    def _covariates_for(self, col: str) -> List[str]:
        if self.covariate_map is not None:
            cov = self.covariate_map[col]  # KeyError on a missing assignment is deliberate -- see class docstring
            return [cov] if isinstance(cov, str) else list(cov)
        cc = self.count_cols
        return [cc] if isinstance(cc, str) else list(cc)

    def _all_covariate_cols(self) -> List[str]:
        if self.covariate_map is not None:
            out = set()
            for v in self.covariate_map.values():
                out.update([v] if isinstance(v, str) else v)
            return sorted(out)
        cc = self.count_cols
        return [cc] if isinstance(cc, str) else list(cc)

    def _design_for(self, X: pd.DataFrame, col: str) -> np.ndarray:
        covs = self._covariates_for(col)
        cols = [np.ones(len(X))] + [X[c].to_numpy(dtype=float) for c in covs]
        return np.column_stack(cols)

    def fit(self, X: pd.DataFrame, y=None):
        all_cov_cols = self._all_covariate_cols()
        self.feature_cols_ = [c for c in X.columns if c not in all_cov_cols]
        self.coefs_: Dict[str, np.ndarray] = {}
        self.fit_failed_cols_: List[str] = []
        for col in self.feature_cols_:
            design = self._design_for(X, col)
            y_col = X[col].to_numpy(dtype=float)
            finite = np.all(np.isfinite(design), axis=1) & np.isfinite(y_col)
            min_n = design.shape[1] + 1
            ok = finite.sum() >= min_n and all(
                np.std(design[finite, j]) > 1e-9 for j in range(1, design.shape[1])
            )
            if not ok:
                self.coefs_[col] = np.zeros(design.shape[1])
                self.fit_failed_cols_.append(col)
                continue
            try:
                beta, *_ = np.linalg.lstsq(design[finite], y_col[finite], rcond=None)
                self.coefs_[col] = beta
            except np.linalg.LinAlgError:
                self.coefs_[col] = np.zeros(design.shape[1])
                self.fit_failed_cols_.append(col)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in self.feature_cols_:
            design = self._design_for(X, col)
            out[col] = X[col].to_numpy(dtype=float) - design @ self.coefs_[col]
        return pd.DataFrame(out, index=X.index)


# ══════════════════════════════════════════════════════════════════════════════
# RF / CV CORE
# ══════════════════════════════════════════════════════════════════════════════

def _make_pipeline(impute: bool, n_jobs: int = -1) -> Pipeline:
    steps = []
    if impute:
        steps.append(("impute", SimpleImputer(strategy="median")))
    steps.append(("rf", RandomForestClassifier(
        n_estimators=N_TREES, class_weight="balanced", random_state=CV_SEED, n_jobs=n_jobs,
    )))
    return Pipeline(steps)


def cv_balanced_accuracy(
    X: pd.DataFrame, y: np.ndarray, impute: bool = False,
    resid_count_col: Optional[str] = None,
    resid_covariate_map: Optional[Dict[str, str]] = None,
    return_importances: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, pd.DataFrame]]:
    """
    RepeatedStratifiedKFold(5, 10) = 50 folds. CV_SEED is shared across every call
    site in this module, which is what makes paired raw-vs-residualized comparisons
    valid -- both runs see identical folds.
    resid_count_col and resid_covariate_map are mutually exclusive -- pass at most one.
    resid_covariate_map is what a mixed-feature-group run (e.g. all-scales) needs;
    see FoldSafeResidualiser's own docstring for why a single shared covariate is wrong there.
    """
    if resid_count_col is not None and resid_covariate_map is not None:
        raise ValueError("Pass at most one of resid_count_col / resid_covariate_map")
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=CV_SEED)
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    scores, importances = [], []
    feature_names: Optional[List[str]] = None

    for train_idx, test_idx in rskf.split(X, y):
        Xtr, Xte = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        ytr, yte = y[train_idx], y[test_idx]
        if resid_count_col is not None:
            resid = FoldSafeResidualiser(count_cols=resid_count_col).fit(Xtr)
            Xtr, Xte = resid.transform(Xtr), resid.transform(Xte)
        elif resid_covariate_map is not None:
            resid = FoldSafeResidualiser(covariate_map=resid_covariate_map).fit(Xtr)
            Xtr, Xte = resid.transform(Xtr), resid.transform(Xte)
        pipe = _make_pipeline(impute=impute)
        pipe.fit(Xtr, ytr)
        scores.append(balanced_accuracy_score(yte, pipe.predict(Xte)))
        if return_importances:
            importances.append(pipe.named_steps["rf"].feature_importances_)
            feature_names = feature_names or list(Xtr.columns)

    scores_arr = np.array(scores)
    if not return_importances:
        return scores_arr
    imp_df = pd.DataFrame(np.array(importances), columns=feature_names)
    return scores_arr, imp_df


def paired_significance(raw: np.ndarray, resid: np.ndarray) -> Dict:
    """Paired Wilcoxon: did accuracy change after residualizing out the count covariate?"""
    diff = raw - resid
    if np.allclose(diff, 0):
        stat, p = float("nan"), 1.0
    else:
        stat, p = wilcoxon(raw, resid)
    delta = float(raw.mean() - resid.mean())
    if abs(delta) < 1e-6:
        direction = "no meaningful change after residualizing"
    elif delta > 0:
        direction = "accuracy FELL after residualizing (count-mediated signal was real)"
    else:
        direction = "accuracy ROSE after residualizing (raw features were muddied by count)"
    return {
        "raw_mean_ba": round(float(raw.mean()), 4), "raw_sd_ba": round(float(raw.std(ddof=1)), 4),
        "resid_mean_ba": round(float(resid.mean()), 4), "resid_sd_ba": round(float(resid.std(ddof=1)), 4),
        "raw_minus_resid": round(delta, 4), "direction": direction,
        "wilcoxon_stat": float(stat), "p_value": float(p),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PERMUTATION TEST — stratified by timepoint
# ══════════════════════════════════════════════════════════════════════════════

def _one_permutation(
    X_vals: np.ndarray, feat_cols: List[str], y: np.ndarray, strata: np.ndarray,
    impute: bool, resid_col: Optional[str], resid_covariate_map: Optional[Dict[str, str]], seed: int,
) -> float:
    """Single permutation worker. RF n_jobs=1 inside -- the outer Parallel pool is where the parallelism lives."""
    rng = np.random.default_rng(seed)
    y_perm = y.copy()
    for sv in np.unique(strata):
        mask = (strata == sv)
        y_perm[mask] = rng.permutation(y[mask])

    X_df = pd.DataFrame(X_vals, columns=feat_cols)
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=1, random_state=0)
    fold_bas = []
    for tr, te in rskf.split(X_df, y_perm):
        Xtr, Xte = X_df.iloc[tr].copy(), X_df.iloc[te].copy()
        ytr, yte = y_perm[tr], y_perm[te]
        if resid_col is not None and resid_col in X_df.columns:
            r = FoldSafeResidualiser(count_cols=resid_col).fit(Xtr)
            Xtr, Xte = r.transform(Xtr), r.transform(Xte)
        elif resid_covariate_map is not None:
            r = FoldSafeResidualiser(covariate_map=resid_covariate_map).fit(Xtr)
            Xtr, Xte = r.transform(Xtr), r.transform(Xte)
        pipe = _make_pipeline(impute=impute, n_jobs=1)
        pipe.fit(Xtr, ytr)
        fold_bas.append(balanced_accuracy_score(yte, pipe.predict(Xte)))
    return float(np.mean(fold_bas))


def permutation_test(
    X: pd.DataFrame, y: np.ndarray, observed_ba: float, n_permutations: int,
    impute: bool, resid_col: Optional[str], strata: np.ndarray,
    rng_seed: int, n_jobs: int,
    resid_covariate_map: Optional[Dict[str, str]] = None,
) -> Dict:
    """Permutes labels within each timepoint stratum, not globally, so that a
    permuted dataset still has the same timepoint composition as the real one --
    otherwise the null distribution would pick up timepoint imbalance rather than
    testing only the group-label association."""
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    feat_cols = list(X.columns)
    X_vals = X.to_numpy(dtype=float)
    base_rng = np.random.default_rng(rng_seed)
    seeds = base_rng.integers(0, 2**31, size=n_permutations).tolist()

    perm_bas = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_permutation)(X_vals, feat_cols, y, strata, impute, resid_col, resid_covariate_map, s)
        for s in seeds
    )
    perm_bas_arr = np.array(perm_bas)
    p_val = float((perm_bas_arr >= observed_ba).mean())
    return {
        "observed_ba": round(observed_ba, 4), "n_permutations": n_permutations,
        "perm_mean_ba": round(float(perm_bas_arr.mean()), 4),
        "perm_sd_ba": round(float(perm_bas_arr.std()), 4),
        "p_value": round(p_val, 6), "permutation_mode": "stratified_by_timepoint",
    }


# ══════════════════════════════════════════════════════════════════════════════
# STRATUM PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def make_timepoint_label(tp_h) -> str:
    if pd.isna(tp_h):
        return "control"
    v = float(tp_h)
    return f"{int(v)}h" if v == int(v) else f"{v}h"


def prepare_stratum(df: pd.DataFrame, source: str, marker: str, irradiated_only: bool = False) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict]:
    sub = df[(df["source"] == source) & (df["marker"] == marker)].copy()
    if irradiated_only:
        sub = sub[~sub["is_control"]].copy()
    cfg = SOURCE_CONFIG[source]
    y = (sub[cfg["target_col"]] == cfg["pos_label"]).astype(int).to_numpy()
    tp_labels = sub["timepoint_h"].apply(make_timepoint_label).to_numpy()
    meta = {
        "source": source, "marker": marker, "irradiated_only": irradiated_only,
        "n_total": len(sub), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
        "pos_label": str(cfg["pos_label"]), "ref_label": str(cfg["ref_label"]),
        "n_zero_cluster": int((sub["n_clusters"] == 0).sum()),
    }
    return sub, y, tp_labels, meta


# ══════════════════════════════════════════════════════════════════════════════
# ONE RF RUN
# ══════════════════════════════════════════════════════════════════════════════

def run_one(
    sub: pd.DataFrame, y: np.ndarray, tp_labels: np.ndarray, feat_cols: List[str],
    impute: bool, n_permutations: int, label: str, n_jobs: int,
    resid_col: Optional[str] = None, resid_covariate_map: Optional[Dict[str, str]] = None,
) -> Dict:
    if (resid_col is None) == (resid_covariate_map is None):
        raise ValueError(f"run_one({label}): pass exactly one of resid_col / resid_covariate_map")
    if len(sub) < 20 or y.sum() < 5 or (len(y) - y.sum()) < 5:
        return {"skipped": True, "reason": f"Insufficient data: n={len(sub)}, n_pos={int(y.sum())}", "label": label}

    feat_cols = [c for c in feat_cols if c in sub.columns]
    covariate_cols = [resid_col] if resid_col is not None else sorted(set(resid_covariate_map.values()))

    X_feat_only = sub[feat_cols].copy()
    raw_scores, imp_df = cv_balanced_accuracy(X_feat_only, y, impute=impute, return_importances=True)
    raw_ba = float(raw_scores.mean())

    X_with_resid = sub[feat_cols + covariate_cols].copy()
    resid_scores, resid_imp_df = cv_balanced_accuracy(
        X_with_resid, y, impute=impute,
        resid_count_col=resid_col, resid_covariate_map=resid_covariate_map,
        return_importances=True,
    )
    resid_ba = float(resid_scores.mean())

    sig = paired_significance(raw_scores, resid_scores)

    perm_result = permutation_test(
        X_with_resid, y, observed_ba=resid_ba, n_permutations=n_permutations,
        impute=impute, resid_col=resid_col, resid_covariate_map=resid_covariate_map,
        strata=tp_labels, rng_seed=abs(hash(label)) % (2**31), n_jobs=n_jobs,
    )

    # OOB, on the residualized full-population fit
    if resid_col is not None:
        resid_oob = FoldSafeResidualiser(count_cols=resid_col).fit(X_with_resid)
    else:
        resid_oob = FoldSafeResidualiser(covariate_map=resid_covariate_map).fit(X_with_resid)
    X_oob = resid_oob.transform(X_with_resid)
    if impute:
        X_oob_arr = SimpleImputer(strategy="median").fit_transform(X_oob)
    else:
        X_oob_arr = X_oob.to_numpy()
    rf_oob = RandomForestClassifier(n_estimators=N_TREES, class_weight="balanced", random_state=CV_SEED, oob_score=True, n_jobs=-1)
    rf_oob.fit(X_oob_arr, y)
    oob_ba = float(balanced_accuracy_score(y, rf_oob.oob_decision_function_.argmax(axis=1)))

    imp_summary = {
        feat: {"mean_importance": round(float(resid_imp_df[feat].mean()), 6), "sd_importance": round(float(resid_imp_df[feat].std(ddof=1)), 6)}
        for feat in resid_imp_df.columns
    }

    return {
        "label": label, "n": len(sub), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
        "feat_cols": feat_cols,
        "resid_col": resid_col if resid_col is not None else f"covariate_map({covariate_cols})",
        "raw_ba": sig["raw_mean_ba"], "raw_sd": sig["raw_sd_ba"],
        "resid_ba": sig["resid_mean_ba"], "resid_sd": sig["resid_sd_ba"],
        "raw_minus_resid": sig["raw_minus_resid"], "direction": sig["direction"],
        "wilcoxon_p": sig["p_value"], "permutation": perm_result,
        "oob_ba": round(oob_ba, 4), "oob_cv_discordance": round(abs(oob_ba - resid_ba), 4),
        "oob_cv_flagged": bool(abs(oob_ba - resid_ba) > 0.05),
        "feature_importances": imp_summary,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — marker-stratified RF, four feature sets, one population
# ══════════════════════════════════════════════════════════════════════════════

def residualization_diagnostic(sub: pd.DataFrame, df: pd.DataFrame) -> Dict:
    """
    r² between each feature and the covariate it's about to be residualized
    against, computed on the full stratum (not fold-restricted -- this is a
    diagnostic about the data, not a CV estimate). Confirms residualization is
    non-vacuous before trusting it, and confirms every feature is paired with
    its intended covariate (Scale A/B -> n_clusters, Scale C/noise ->
    n_localisations): a near-zero r² for a pairing that should be strong is a
    sign the wrong covariate is being used, not that the covariate has no effect.
    """
    groups = {
        "scaleA_mean": (scale_a_mean_cols(df), "n_clusters"),
        "scaleB": (scale_b_cols(df), "n_clusters"),
        "scaleC": (scale_c_cols(df), "n_localisations"),
        "noise": (noise_cols(df), "n_localisations"),
    }
    diag: Dict = {}
    for group_name, (fcols, covar) in groups.items():
        fcols = [c for c in fcols if c in sub.columns]
        cov_vals = sub[covar].to_numpy(dtype=float)
        r2_vals = {}
        for fc in fcols:
            feat_vals = sub[fc].to_numpy(dtype=float)
            mask = np.isfinite(feat_vals) & np.isfinite(cov_vals)
            if mask.sum() > 3 and np.std(cov_vals[mask]) > 1e-9:
                corr = np.corrcoef(feat_vals[mask], cov_vals[mask])[0, 1]
                r2_vals[fc] = round(float(corr ** 2), 4)
            else:
                r2_vals[fc] = float("nan")
        finite_r2 = [v for v in r2_vals.values() if np.isfinite(v)]
        diag[group_name] = {
            "covariate": covar, "r2_per_feature": r2_vals,
            "mean_r2": round(float(np.mean(finite_r2)), 4) if finite_r2 else float("nan"),
            "max_r2": round(float(np.max(finite_r2)), 4) if finite_r2 else float("nan"),
        }
    return diag


def run_section(df: pd.DataFrame, source: str, marker: str, n_permutations: int, n_jobs: int, irradiated_only: bool = False) -> Dict:
    sub, y, tp_labels, meta = prepare_stratum(df, source, marker, irradiated_only=irradiated_only)
    result: Dict = {"meta": meta, "residualization_diagnostics": residualization_diagnostic(sub, df)}

    b_cols = scale_b_cols(df)
    result["scale_B"] = run_one(
        sub, y, tp_labels, b_cols, resid_col="n_clusters", impute=True,
        n_permutations=n_permutations, label=f"{source}/{marker}/scaleB/{'irr' if irradiated_only else 'full'}", n_jobs=n_jobs,
    )

    c_cols = scale_c_cols(df)
    result["scale_C"] = run_one(
        sub, y, tp_labels, c_cols, resid_col="n_localisations", impute=False,
        n_permutations=n_permutations, label=f"{source}/{marker}/scaleC/{'irr' if irradiated_only else 'full'}", n_jobs=n_jobs,
    )

    cn_cols = c_cols + noise_cols(df)
    result["scale_C_noise"] = run_one(
        sub, y, tp_labels, cn_cols, resid_col="n_localisations", impute=False,
        n_permutations=n_permutations, label=f"{source}/{marker}/scaleCnoise/{'irr' if irradiated_only else 'full'}", n_jobs=n_jobs,
    )

    a_mean_cols = scale_a_mean_cols(df)
    all_cols = a_mean_cols + b_cols + cn_cols
    all_scales_covariate_map = {c: "n_clusters" for c in a_mean_cols + b_cols}
    all_scales_covariate_map.update({c: "n_localisations" for c in cn_cols})
    result["all_scales"] = run_one(
        sub, y, tp_labels, all_cols, impute=True, resid_covariate_map=all_scales_covariate_map,
        n_permutations=n_permutations, label=f"{source}/{marker}/allscales/{'irr' if irradiated_only else 'full'}", n_jobs=n_jobs,
    )

    for run_key in ["scale_B", "scale_C", "scale_C_noise", "all_scales"]:
        r = result[run_key]
        if r.get("skipped"):
            logger.info(f"    [{run_key}] SKIPPED: {r.get('reason')}")
        else:
            logger.info(
                f"    [{run_key}] raw={r['raw_ba']:.4f} resid={r['resid_ba']:.4f} "
                f"Wilcoxon-p={r['wilcoxon_p']:.3g} perm-p={r['permutation']['p_value']:.3g} "
                f"OOB={r['oob_ba']:.4f} flag={r['oob_cv_flagged']}"
            )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION E — radiation-geometry comparison, shared marker yH2AX
# ══════════════════════════════════════════════════════════════════════════════

def run_section_e(all_section_a: Dict[str, Dict]) -> Dict:
    """
    Does topological cell-line discriminability transfer between heavy-ion
    (kuentzelmann) and photon (hahn) damage, on the one marker both sources share?
    """
    shared_marker = "yH2AX"
    rows = []
    for source in SOURCES:
        mk_res = all_section_a.get(source, {}).get(shared_marker, {})
        for run_key in ["scale_B", "scale_C", "scale_C_noise", "all_scales"]:
            run = mk_res.get(run_key, {})
            if run.get("skipped"):
                continue
            rows.append({
                "source": source, "radiation_type": SOURCE_CONFIG[source]["radiation_type"],
                "scale_subset": run_key, "resid_ba": run.get("resid_ba"),
                "perm_p": run.get("permutation", {}).get("p_value"),
            })
    summary: Dict[str, Dict] = {}
    for run_key in ["scale_B", "scale_C", "scale_C_noise", "all_scales"]:
        by_source = {r["source"]: r for r in rows if r["scale_subset"] == run_key}
        heavy_ion = by_source.get("kuentzelmann", {})
        photon = by_source.get("hahn", {})
        heavy_sig = bool(heavy_ion.get("perm_p") is not None and heavy_ion["perm_p"] < 0.05)
        photon_sig = bool(photon.get("perm_p") is not None and photon["perm_p"] < 0.05)
        summary[run_key] = {
            "kuentzelmann_heavy_ion": heavy_ion, "hahn_photon": photon,
            "sig_in_heavy_ion": heavy_sig, "sig_in_photon": photon_sig,
            "sig_in_both": heavy_sig and photon_sig,
        }
    return {
        "shared_marker": shared_marker,
        "comparison_table": rows,
        "summary_by_scale": summary,
        "_note": (
            "sig_in_both=True on the full-population run does not by itself mean "
            "the effect is radiation-quality-independent -- check the matching "
            "irradiated-only row for hahn specifically before treating a "
            "full-population 'both significant' result as a clean transfer "
            "between radiation types, since a pooled cell-line effect can be "
            "partly a control-vs-treated baseline difference rather than a "
            "true topology signal."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=Path("data/feature_matrix.csv"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers for the permutation test. -1 = all cores.")
    ap.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS_DEFAULT,
                     help=f"Default {N_PERMUTATIONS_DEFAULT}. Use a few hundred for a smoke test.")
    args = ap.parse_args()

    if not args.features.exists():
        logger.error(f"Feature matrix not found: {args.features}")
        return 1

    df = pd.read_csv(args.features, dtype={"replicate": str})
    logger.info("=" * 70)
    logger.info("View 1, Tier 1-2 classification analysis")
    logger.info("=" * 70)
    logger.info(f"Loaded {len(df)} nuclei from {args.features}")
    logger.info(f"n_permutations: {args.n_permutations}   n_jobs: {args.n_jobs}")
    logger.info(f"Scale A columns used (mean only, _sd excluded): {scale_a_mean_cols(df)}")

    args.results_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict = {}
    all_section_a: Dict[str, Dict] = {}
    all_section_c: Dict[str, Dict] = {}

    for source in SOURCES:
        markers = sorted(df[df["source"] == source]["marker"].unique())
        logger.info(f"\n{'='*70}\nSOURCE: {source}  ({SOURCE_CONFIG[source]['radiation_type']})  markers={markers}\n{'='*70}")
        src_results: Dict = {"source": source, "section_A": {}, "section_C_irradiated_only": {}}

        for mk in markers:
            logger.info(f"\n[Section A] {source}/{mk}  (full population)")
            src_results["section_A"][mk] = run_section(df, source, mk, args.n_permutations, args.n_jobs, irradiated_only=False)

            logger.info(f"[Section C] {source}/{mk}  (irradiated only — sham-exclusion sensitivity)")
            src_results["section_C_irradiated_only"][mk] = run_section(df, source, mk, args.n_permutations, args.n_jobs, irradiated_only=True)

        all_section_a[source] = src_results["section_A"]
        all_section_c[source] = src_results["section_C_irradiated_only"]
        all_results[source] = src_results

    logger.info(f"\n{'='*70}\n[Section E] Radiation-geometry comparison (shared marker: yH2AX)\n{'='*70}")
    sec_e = run_section_e(all_section_a)
    for rk, smry in sec_e["summary_by_scale"].items():
        logger.info(f"  {rk}: heavy_ion_sig={smry['sig_in_heavy_ion']}  photon_sig={smry['sig_in_photon']}  both={smry['sig_in_both']}")
    all_results["section_E_radiation_geometry"] = sec_e

    json_path = args.results_dir / "classification_results.json"
    with open(json_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    logger.info(f"\nJSON results -> {json_path}")

    imp_rows = []
    for source in SOURCES:
        for section_name, section in [("full", all_section_a[source]), ("irradiated_only", all_section_c[source])]:
            for mk, mk_res in section.items():
                for run_key in ["scale_B", "scale_C", "scale_C_noise", "all_scales"]:
                    run = mk_res.get(run_key, {})
                    for feat, imp in run.get("feature_importances", {}).items():
                        imp_rows.append({
                            "source": source, "population": section_name, "marker": mk, "scale_subset": run_key,
                            "feature": feat, "mean_importance": imp.get("mean_importance"), "sd_importance": imp.get("sd_importance"),
                        })
    if imp_rows:
        imp_df = pd.DataFrame(imp_rows).sort_values(["source", "population", "marker", "scale_subset", "mean_importance"], ascending=[True, True, True, True, False])
        imp_path = args.results_dir / "classification_feature_importance.csv"
        imp_df.to_csv(imp_path, index=False)
        logger.info(f"Feature importance CSV -> {imp_path}  ({len(imp_df)} rows)")

    txt_path = args.results_dir / "classification_summary.txt"
    with open(txt_path, "w") as fh:
        fh.write("View 1 classification — summary\n" + "=" * 70 + "\n\n")
        for source in SOURCES:
            fh.write(f"SOURCE: {source}  ({SOURCE_CONFIG[source]['radiation_type']})\n" + "-" * 60 + "\n")
            for section_label, section in [("FULL POPULATION", all_section_a[source]), ("IRRADIATED ONLY", all_section_c[source])]:
                fh.write(f"  [{section_label}]\n")
                for mk, mk_res in section.items():
                    fh.write(f"    Marker: {mk}\n")
                    for run_key in ["scale_B", "scale_C", "scale_C_noise", "all_scales"]:
                        run = mk_res.get(run_key, {})
                        if run.get("skipped"):
                            fh.write(f"      [{run_key}] SKIPPED: {run.get('reason')}\n")
                        else:
                            fh.write(
                                f"      [{run_key}] raw={run['raw_ba']:.4f} resid={run['resid_ba']:.4f} "
                                f"Wilcoxon-p={run['wilcoxon_p']:.3g} perm-p={run['permutation']['p_value']:.3g} "
                                f"OOB={run['oob_ba']:.4f} flag={run['oob_cv_flagged']}\n"
                            )
            fh.write("\n")
        fh.write("[Section E] Radiation-geometry comparison (yH2AX)\n" + "-" * 60 + "\n")
        for rk, smry in sec_e["summary_by_scale"].items():
            fh.write(f"  {rk}: heavy_ion_sig={smry['sig_in_heavy_ion']}  photon_sig={smry['sig_in_photon']}  both={smry['sig_in_both']}\n")
    logger.info(f"Summary -> {txt_path}")

    logger.info("=" * 70)
    logger.info("classification analysis complete.")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
