#!/usr/bin/env python3
"""
Population geometry in Wasserstein space (View 3, Tiers 1-4).

View 3 treats each persistence diagram as a single point in a metric space
under the 2-Wasserstein (W2) distance, rather than reducing it to scalar
summaries (View 1) or a landscape function (View 2). It is a third,
independent way of asking whether topological state differs between
groups; agreement with the other two views is strong, method-independent
confirmation, and disagreement is informative in its own right.

Three analyses share one Wasserstein/Fréchet toolkit
(tml_smlm.utils.wasserstein_geometry). Tiers 1-3, population geometry (per
source/marker, pooling all irradiated nuclei into one pairwise W2 distance
matrix): whether cell type separates in Wasserstein space at all
(medoid-distance permutation test on the full population); a
variance/bimodality diagnostic that gates whether a Fréchet-based point
estimate should be trusted; and the same separation test repeated within
each individual timepoint, to localize where the separation actually
lives. Tier 4a, baseline return (per source/cell_type/marker): whether an
irradiated population's Fréchet mean moves back toward its own matched
sham control over the repair timecourse, using that cell type's real
control samples as the reference point rather than an assumed zero or the
first irradiated timepoint. Tier 4b, cross-marker time-shift alignment
(per source/cell_type): whether two damage markers' population-level
trajectories move together, offset by some lag drawn from the dataset's
own observed timepoint spacing, tested by shuffling timepoint labels
within each marker.

Every pairwise distance matrix is also residualized against per-nucleus
localization count (partial-Mantel style) so a claim about shape can be
told apart from one that is really about count. All distance-matrix-based
tests reuse the shared module's medoid-based permutation machinery -- no
diagram-level recomputation inside a permutation loop.

Homology dimension: scaleC_h1, not h0. H1 loops are never exposed to the
H0 infinite-bar degeneracy that affects birth-death pairs anchored at the
point cloud's own global connected-component structure, so H1 is the
safer dimension for a metric built directly on birth-death coordinates.

The Wasserstein/Fréchet machinery itself (distance, pairwise matrix,
Fréchet mean/variance, permutation test) lives in
tml_smlm.utils.wasserstein_geometry and is used unmodified here -- see
that module's docstring for its own design rationale. Its permutation
test returns a raw p-value with no pseudocount floor, so the standard
+1/(n+1) floor correction is applied here instead.

Runtime note: cost is dominated by building the pooled pairwise W2
distance matrix, not by the permutation tests or the Fréchet mean's own
iterative refinement. For a marker with on the order of 300 irradiated
nuclei this can take well over an hour at modest parallelism, and a full
run across both sources and all three sections can take several hours.
More cores speed up the distance-matrix build (embarrassingly parallel)
but not the Fréchet mean's refinement loop (explicitly serial in the
shared utility), so expect sub-linear speedup from added cores, not
proportional.

The baseline-return output includes the actual control Fréchet mean
diagram (not just its variance) and the control sample count, so a
downstream analysis working at the level of individually matched nuclei
can reuse this stratum's already-computed control reference directly
rather than recomputing it.

This dataset carries a single dose level per source, so a cross-dose
alignment arm is not applicable here; only the cross-marker time-shift
arm (Tier 4b) is implemented.

Outputs
-------
  results/population_geometry_results.json
  results/population_geometry_summary.txt

Usage
-----
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  python -m tml_smlm.view3_metric.population_geometry \\
      --diagrams diagrams/ --features data/feature_matrix.csv \\
      --n-perm 1000 --n-jobs -1 2>&1 | tee population_geometry.log
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from tml_smlm.utils.wasserstein_geometry import (
    wasserstein_distance_2,
    pairwise_distance_matrix,
    medoid_index,
    frechet_mean,
    frechet_variance,
    medoid_distance_statistic,
    permutation_group_test_wasserstein,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SCALE = "scaleC_h1"
MIN_PER_GROUP_FRECHET = 3   # minimum diagrams to compute a Fréchet mean/variance at all
MIN_PER_GROUP_TEST = 5      # minimum diagrams per group for a permutation test to run
SOURCES = ["kuentzelmann", "hahn"]


# ══════════════════════════════════════════════════════════════════════════
# P-VALUE FLOOR — applied here, not in the shared utility (see module docstring)
# ══════════════════════════════════════════════════════════════════════════

def correct_pvalue_floor(raw_p: float, n_permutations: int) -> float:
    count_ge = int(round(raw_p * n_permutations))
    return float((count_ge + 1) / (n_permutations + 1))


def holm_bonferroni(p_values: Dict[str, float]) -> Dict[str, Dict]:
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, Dict] = {}
    running_max = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        running_max = max(running_max, adj)
        out[name] = {"raw_p": round(p, 6), "holm_adjusted_p": round(running_max, 6), "significant_at_0.05": bool(running_max < 0.05)}
    return out


# ══════════════════════════════════════════════════════════════════════════
# DIAGRAM LOADING — same stem/key convention the ingestion step writes
# ══════════════════════════════════════════════════════════════════════════

def _npz_stem(row: pd.Series) -> str:
    raw = f"{row['source']}__{row['cell_type']}__{row['marker']}__{row['condition']}__{row['replicate']}"
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", raw)


def load_diagram(diagrams_dir: Path, row: pd.Series, arr_key: str = SCALE) -> Optional[np.ndarray]:
    npz_file = row.get("npz_file", _npz_stem(row) + ".npz")
    path = diagrams_dir / str(npz_file)
    if not path.exists():
        path = diagrams_dir / (_npz_stem(row) + ".npz")
    try:
        with np.load(path, allow_pickle=False) as d:
            if arr_key not in d:
                return None
            arr = d[arr_key]
            return arr.astype(float).reshape(-1, 2) if len(arr) else np.empty((0, 2))
    except (FileNotFoundError, OSError, ValueError):
        return None


def _tp_label(tp) -> str:
    if pd.isna(tp):
        return "control"
    v = float(tp)
    return f"{int(v)}h" if v == int(v) else f"{v}h"


# ══════════════════════════════════════════════════════════════════════════
# VARIANCE / BIMODALITY DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════

def variance_bimodality_diagnostic(D_full: np.ndarray, idx: np.ndarray, frechet_var: float, all_group_vars: List[float]) -> Dict:
    """
    Fréchet means of persistence diagrams are not unique in general, and a
    fixed-cardinality gradient-descent implementation can converge to a
    locally-optimal but unrepresentative point when a group is actually
    multimodal (e.g. two distinct damage-response subpopulations pooled
    into one nominal group). Before trusting a Fréchet-based estimate or a
    bootstrap CI built on it, check for that failure mode. Two simple,
    honestly-labeled heuristics, not a formal test:
      - relative variance elevation: this group's Fréchet variance vs. the
        smallest variance among the other groups in the same comparison
      - a medoid-distance bimodality ratio: within this group, the ratio of
        (max medoid-distance) to (median medoid-distance) -- a single very
        distant member relative to the rest is a crude multimodality signal,
        not a real dip-test, and is labeled as such in the output.
    """
    med = medoid_index(D_full, idx)
    dists = D_full[med, idx]
    dists = dists[dists > 0]
    bimodality_ratio = float(dists.max() / np.median(dists)) if len(dists) > 0 and np.median(dists) > 0 else float("nan")
    other_vars = [v for v in all_group_vars if v != frechet_var and v > 0]
    rel_elevation = float(frechet_var / min(other_vars)) if other_vars else float("nan")
    flagged = bool((np.isfinite(rel_elevation) and rel_elevation > 2.0) or (np.isfinite(bimodality_ratio) and bimodality_ratio > 3.0))
    return {
        "frechet_variance": round(float(frechet_var), 6),
        "relative_variance_elevation_vs_min_other_group": round(rel_elevation, 4) if np.isfinite(rel_elevation) else None,
        "medoid_distance_bimodality_ratio": round(bimodality_ratio, 4) if np.isfinite(bimodality_ratio) else None,
        "flagged": flagged,
        "_note": "Heuristic screen, not a formal multimodality test. If flagged, treat this group's Fréchet mean/variance and any test built on it with extra caution -- inspect the diagram set directly before citing.",
    }


# ══════════════════════════════════════════════════════════════════════════
# DISTANCE-MATRIX RESIDUALIZATION — partial-Mantel style (Smouse et al. 1986)
# ══════════════════════════════════════════════════════════════════════════

def residualize_distance_matrix(D: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """
    Regress the observed pairwise W2 distance matrix's upper triangle
    against a covariate-difference matrix (|covariate_i - covariate_j|),
    return the residual matrix -- a partial-Mantel-style test (Smouse et
    al. 1986), the standard way to ask "does a distance-matrix-based group
    difference survive controlling for a third variable" when the objects
    being compared are diagrams, not scalars. Other analyses in this
    pipeline residualize against count (n_clusters/n_localisations) at the
    level of a scalar feature or landscape vector; a raw persistence
    diagram has no equivalent "residualized diagram" operation, so this
    residualizes the PAIRWISE DISTANCES directly instead, which is the
    object every test in this module actually operates on
    (medoid_distance_statistic never touches a diagram directly, only D).

    Residual entries are not guaranteed non-negative and D_resid does not
    satisfy the triangle inequality in general -- expected, not a defect.
    medoid_index/medoid_distance_statistic only need a numeric
    dissimilarity to compare (medoid_index sums squared entries, so a
    negative residual contributes the same as its positive counterpart;
    medoid_distance_statistic is a plain lookup) -- neither requires a
    formal metric. Diagonal is exactly 0 by construction (a nucleus's
    distance to itself carries no covariate difference to regress out).
    """
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    if len(iu[0]) < 3:
        return D.copy()
    d_vals = D[iu]
    c_vals = np.abs(covariate[iu[0]] - covariate[iu[1]])
    if np.std(c_vals) < 1e-9:
        return D.copy()  # covariate doesn't vary enough in this stratum to regress against
    X = np.column_stack([np.ones(len(d_vals)), c_vals])
    beta, *_ = np.linalg.lstsq(X, d_vals, rcond=None)
    resid_vals = d_vals - X @ beta
    D_resid = np.zeros_like(D)
    D_resid[iu] = resid_vals
    D_resid[(iu[1], iu[0])] = resid_vals
    return D_resid


# ══════════════════════════════════════════════════════════════════════════
# SECTION A — TIERS 1-3: POPULATION GEOMETRY
# ══════════════════════════════════════════════════════════════════════════

def section_a_population_geometry(df: pd.DataFrame, source: str, marker: str, diagrams_dir: Path, n_perm: int, n_jobs: int) -> Optional[Dict]:
    sub = df[(df["source"] == source) & (df["marker"] == marker) & (~df["is_control"])].reset_index(drop=True)
    diagrams, keep = [], []
    for i, r in sub.iterrows():
        dgm = load_diagram(diagrams_dir, r)
        if dgm is not None:
            diagrams.append(dgm); keep.append(i)
    if len(diagrams) < 2 * MIN_PER_GROUP_TEST:
        return None
    sub = sub.iloc[keep].reset_index(drop=True)
    cell_types = sorted(sub["cell_type"].unique())
    if len(cell_types) != 2:
        return None
    ct_a, ct_b = cell_types

    logger.info(f"  [A] {source}/{marker}: building pooled distance matrix, n={len(diagrams)}")
    D = pairwise_distance_matrix(diagrams, n_jobs=n_jobs)
    covar = sub["n_localisations"].to_numpy(dtype=float)
    D_resid = residualize_distance_matrix(D, covar)
    labels = sub["cell_type"].to_numpy()

    # Tier 1: full-population existence test
    idx_a = np.where(labels == ct_a)[0]; idx_b = np.where(labels == ct_b)[0]
    group_vars: Dict[str, float] = {}
    frechet_by_ct: Dict[str, Dict] = {}
    for ct, idx in [(ct_a, idx_a), (ct_b, idx_b)]:
        if len(idx) < MIN_PER_GROUP_FRECHET:
            frechet_by_ct[ct] = {"skipped": f"n={len(idx)} < {MIN_PER_GROUP_FRECHET}"}
            continue
        sub_dgms = [diagrams[i] for i in idx]
        D_sub = D[np.ix_(idx, idx)]
        mean_dgm, info = frechet_mean(sub_dgms, D=D_sub, n_jobs=n_jobs)
        var = frechet_variance(sub_dgms, mean_dgm)
        group_vars[ct] = var
        frechet_by_ct[ct] = {"n": int(len(idx)), "frechet_mean_n_points": int(len(mean_dgm)), "frechet_variance": round(float(var), 6), "frechet_mean_info": info}

    for ct, idx in [(ct_a, idx_a), (ct_b, idx_b)]:
        if ct in group_vars:
            frechet_by_ct[ct]["variance_bimodality_diagnostic"] = variance_bimodality_diagnostic(D, idx, group_vars[ct], list(group_vars.values()))

    t1_result = None
    if len(idx_a) >= MIN_PER_GROUP_TEST and len(idx_b) >= MIN_PER_GROUP_TEST:
        raw = permutation_group_test_wasserstein(D, labels, ct_a, ct_b, "medoid_distance", n_perm, rng_seed=0)
        raw["p_value_floor_corrected"] = correct_pvalue_floor(raw["p_value"], n_perm)
        resid = permutation_group_test_wasserstein(D_resid, labels, ct_a, ct_b, "medoid_distance", n_perm, rng_seed=0)
        resid["p_value_floor_corrected"] = correct_pvalue_floor(resid["p_value"], n_perm)
        t1_result = {"raw": raw, "residualized_against_n_localisations": resid}
        logger.info(
            f"    T1 existence: raw observed={raw['observed']:.2f} p={raw['p_value_floor_corrected']:.4g}  |  "
            f"resid observed={resid['observed']:.2f} p={resid['p_value_floor_corrected']:.4g}"
        )

    # Tier 3: same test, per individual timepoint, raw and residualized
    t3_by_timepoint: Dict[str, Dict] = {}
    tp_labels = sub["timepoint_h"].apply(_tp_label).to_numpy()
    for tp in sorted(set(tp_labels)):
        tmask = tp_labels == tp
        idx_a_t = np.where(tmask & (labels == ct_a))[0]
        idx_b_t = np.where(tmask & (labels == ct_b))[0]
        if len(idx_a_t) < MIN_PER_GROUP_TEST or len(idx_b_t) < MIN_PER_GROUP_TEST:
            t3_by_timepoint[tp] = {"skipped": f"n_a={len(idx_a_t)}, n_b={len(idx_b_t)} < {MIN_PER_GROUP_TEST}"}
            continue
        combined_idx = np.concatenate([idx_a_t, idx_b_t])

        def _t3_test(Dmat: np.ndarray) -> Dict:
            observed = medoid_distance_statistic(Dmat, idx_a_t, idx_b_t)
            rng = np.random.default_rng(0)
            null_vals = np.empty(n_perm)
            for p in range(n_perm):
                perm = rng.permutation(combined_idx)
                null_vals[p] = medoid_distance_statistic(Dmat, perm[: len(idx_a_t)], perm[len(idx_a_t):])
            p_raw = float((null_vals >= observed).mean())
            return {"observed_w2_medoid_to_medoid": round(float(observed), 6), "p_value_floor_corrected": correct_pvalue_floor(p_raw, n_perm)}

        raw_t3 = _t3_test(D)
        resid_t3 = _t3_test(D_resid)
        t3_by_timepoint[tp] = {
            "n_a": int(len(idx_a_t)), "n_b": int(len(idx_b_t)),
            "raw": raw_t3, "residualized_against_n_localisations": resid_t3,
        }
        logger.info(
            f"    T3 @ {tp}: raw observed={raw_t3['observed_w2_medoid_to_medoid']:.2f} p={raw_t3['p_value_floor_corrected']:.4g}  |  "
            f"resid observed={resid_t3['observed_w2_medoid_to_medoid']:.2f} p={resid_t3['p_value_floor_corrected']:.4g}"
        )

    return {
        "source": source, "marker": marker, "cell_type_ref": ct_a, "cell_type_other": ct_b,
        "n_total": len(diagrams), "frechet_by_cell_type": frechet_by_ct,
        "t1_existence_full_population": t1_result, "t3_structure_by_timepoint": t3_by_timepoint,
        "_residualization_note": "raw = unresidualized W2 distance matrix (as originally reported). residualized_against_n_localisations = partial-Mantel-style residualization (see residualize_distance_matrix docstring), controlling for the same count covariate other views in this pipeline use for this scale. Both reported; neither supersedes the other -- a real signal that survives residualization is stronger evidence than either alone.",
    }


# ══════════════════════════════════════════════════════════════════════════
# SECTION B — TIER 4a: BASELINE RETURN
# ══════════════════════════════════════════════════════════════════════════

def section_b_baseline_return(df: pd.DataFrame, source: str, cell_type: str, marker: str, diagrams_dir: Path, n_perm: int, n_jobs: int) -> Optional[Dict]:
    sub = df[(df["source"] == source) & (df["cell_type"] == cell_type) & (df["marker"] == marker)].reset_index(drop=True)
    diagrams, keep = [], []
    for i, r in sub.iterrows():
        dgm = load_diagram(diagrams_dir, r)
        if dgm is not None:
            diagrams.append(dgm); keep.append(i)
    if len(diagrams) < 2 * MIN_PER_GROUP_TEST:
        return None
    sub = sub.iloc[keep].reset_index(drop=True)

    is_ctrl = sub["is_control"].to_numpy()
    tp_labels = sub["timepoint_h"].apply(_tp_label).to_numpy()
    n_ctrl = int(is_ctrl.sum())
    if n_ctrl < MIN_PER_GROUP_TEST:
        return {"source": source, "cell_type": cell_type, "marker": marker, "skipped": f"n_control={n_ctrl} < {MIN_PER_GROUP_TEST}"}

    logger.info(f"  [B] {source}/{cell_type}/{marker}: building pooled distance matrix, n={len(diagrams)}")
    D = pairwise_distance_matrix(diagrams, n_jobs=n_jobs)
    covar = sub["n_localisations"].to_numpy(dtype=float)
    D_resid = residualize_distance_matrix(D, covar)
    idx_ctrl = np.where(is_ctrl)[0]
    ctrl_dgms = [diagrams[i] for i in idx_ctrl]
    ctrl_mean, ctrl_info = frechet_mean(ctrl_dgms, D=D[np.ix_(idx_ctrl, idx_ctrl)], n_jobs=n_jobs)
    ctrl_var = frechet_variance(ctrl_dgms, ctrl_mean)

    trajectory: List[Dict] = []
    tests_vs_control: Dict[str, Dict] = {}
    for tp in sorted(set(tp_labels[~is_ctrl])):
        idx_tp = np.where((tp_labels == tp) & (~is_ctrl))[0]
        if len(idx_tp) < MIN_PER_GROUP_FRECHET:
            trajectory.append({"timepoint": tp, "skipped": f"n={len(idx_tp)} < {MIN_PER_GROUP_FRECHET}"})
            continue
        tp_dgms = [diagrams[i] for i in idx_tp]
        tp_mean, tp_info = frechet_mean(tp_dgms, D=D[np.ix_(idx_tp, idx_tp)], n_jobs=n_jobs)
        w2_to_ctrl = wasserstein_distance_2(tp_mean, ctrl_mean)
        trajectory.append({
            "timepoint": tp, "n": int(len(idx_tp)), "w2_distance_to_own_control": round(float(w2_to_ctrl), 4),
            "frechet_mean_n_points": int(len(tp_mean)), "frechet_mean_total_persistence": round(float((tp_mean[:, 1] - tp_mean[:, 0]).sum()), 4) if len(tp_mean) else 0.0,
        })
        if len(idx_tp) >= MIN_PER_GROUP_TEST:
            combined = np.concatenate([idx_tp, idx_ctrl])

            def _b_test(Dmat: np.ndarray) -> Dict:
                observed = medoid_distance_statistic(Dmat, idx_tp, idx_ctrl)
                rng = np.random.default_rng(0)
                null_vals = np.empty(n_perm)
                for p in range(n_perm):
                    perm = rng.permutation(combined)
                    null_vals[p] = medoid_distance_statistic(Dmat, perm[: len(idx_tp)], perm[len(idx_tp):])
                p_raw = float((null_vals >= observed).mean())
                return {"observed_w2_medoid_to_medoid": round(float(observed), 6), "p_value_floor_corrected": correct_pvalue_floor(p_raw, n_perm)}

            raw_b = _b_test(D)
            resid_b = _b_test(D_resid)
            tests_vs_control[tp] = {
                "n_tp": int(len(idx_tp)), "n_control": n_ctrl,
                "raw": raw_b, "residualized_against_n_localisations": resid_b,
                "_note": "medoid-to-medoid W2, the statistic the permutation test actually uses -- distinct from the trajectory entry's mean-to-mean w2_distance_to_own_control above, which is descriptive. residualized_* controls for whether count alone (not shape) explains a 'not yet returned to baseline' result -- see module docstring.",
            }
            logger.info(
                f"    {tp} vs control: raw W2={raw_b['observed_w2_medoid_to_medoid']:.2f} p={raw_b['p_value_floor_corrected']:.4g}  |  "
                f"resid W2={resid_b['observed_w2_medoid_to_medoid']:.2f} p={resid_b['p_value_floor_corrected']:.4g}"
            )

    return {
        "source": source, "cell_type": cell_type, "marker": marker,
        "control_frechet_mean": ctrl_mean.tolist(), "control_n": n_ctrl,
        "control_frechet_variance": round(float(ctrl_var), 6),
        "control_variance_bimodality_diagnostic": variance_bimodality_diagnostic(D, idx_ctrl, ctrl_var, [ctrl_var]),
        "trajectory": sorted(trajectory, key=lambda r: r["timepoint"]),
        "tests_vs_control": tests_vs_control,
        "_interpretation": "Baseline = this cell type's own is_control==True sham samples, never an assumed zero or the first irradiated timepoint. A trajectory that shrinks toward the control's own variance scale, with the latest timepoint's test non-significant, is evidence of return to baseline; a persistently significant latest timepoint, with the trajectory report above showing what the residual Fréchet mean's own point count/total persistence look like, is evidence it has not returned. Check the residualized test alongside the raw one before citing persistence as a SHAPE claim -- a raw-significant, residualized-non-significant result means count hasn't returned to baseline, which is a real but different finding from shape persisting.",
    }


# ══════════════════════════════════════════════════════════════════════════
# SECTION C — TIER 4b: CROSS-MARKER TIME-SHIFT ALIGNMENT
# ══════════════════════════════════════════════════════════════════════════

def section_c_cross_marker_shift(df: pd.DataFrame, source: str, cell_type: str, marker_a: str, marker_b: str, diagrams_dir: Path, n_perm: int, n_jobs: int) -> Optional[Dict]:
    sub = df[(df["source"] == source) & (df["cell_type"] == cell_type) & (df["marker"].isin([marker_a, marker_b])) & (~df["is_control"])].reset_index(drop=True)
    diagrams, keep = [], []
    for i, r in sub.iterrows():
        dgm = load_diagram(diagrams_dir, r)
        if dgm is not None:
            diagrams.append(dgm); keep.append(i)
    if len(diagrams) < 2 * MIN_PER_GROUP_TEST:
        return None
    sub = sub.iloc[keep].reset_index(drop=True)
    marker_lab = sub["marker"].to_numpy()
    tp_vals = sub["timepoint_h"].to_numpy(dtype=float)
    tp_lab = sub["timepoint_h"].apply(_tp_label).to_numpy()

    logger.info(f"  [C] {source}/{cell_type}, {marker_a} vs {marker_b}: building pooled distance matrix, n={len(diagrams)}")
    D = pairwise_distance_matrix(diagrams, n_jobs=n_jobs)
    covar = sub["n_localisations"].to_numpy(dtype=float)
    D_resid = residualize_distance_matrix(D, covar)

    tps_a = sorted(set(tp_vals[marker_lab == marker_a]))
    tps_b = sorted(set(tp_vals[marker_lab == marker_b]))
    if not tps_a or not tps_b:
        return None
    # Candidate lags: differences actually present between this source's own timepoints, not an arbitrary continuous grid.
    all_tps = sorted(set(tps_a) | set(tps_b))
    candidate_lags = sorted(set(round(t2 - t1, 4) for t1 in all_tps for t2 in all_tps))
    tol = (min(np.diff(all_tps)) / 2.0) if len(all_tps) > 1 else 0.5

    def _alignment_cost(lag: float, marker_labels_local: np.ndarray, Dmat: np.ndarray) -> Tuple[float, int]:
        total, n_matched = 0.0, 0
        for ta in tps_a:
            idx_a = np.where((marker_labels_local == marker_a) & (np.isclose(tp_vals, ta)))[0]
            if len(idx_a) == 0:
                continue
            target = ta + lag
            tb_matches = [tb for tb in tps_b if abs(tb - target) <= tol]
            if not tb_matches:
                continue
            tb = min(tb_matches, key=lambda x: abs(x - target))
            idx_b = np.where((marker_labels_local == marker_b) & (np.isclose(tp_vals, tb)))[0]
            if len(idx_b) == 0:
                continue
            total += medoid_distance_statistic(Dmat, idx_a, idx_b)
            n_matched += 1
        return (total / n_matched if n_matched > 0 else float("inf")), n_matched

    def _run_section_c_test(Dmat: np.ndarray) -> Optional[Dict]:
        lag_costs = {}
        for lag in candidate_lags:
            cost, n_matched = _alignment_cost(lag, marker_lab, Dmat)
            if n_matched > 0:
                lag_costs[lag] = cost
        if not lag_costs:
            return None

        best_lag = min(lag_costs, key=lag_costs.get)
        best_cost = lag_costs[best_lag]

        rng = np.random.default_rng(0)
        null_best_costs = np.empty(n_perm)
        null_best_lags = np.empty(n_perm)
        for p in range(n_perm):
            perm_marker_lab = marker_lab.copy()
            perm_tp_vals = tp_vals.copy()
            for m in (marker_a, marker_b):
                m_idx = np.where(marker_lab == m)[0]
                perm_tp_vals[m_idx] = rng.permutation(tp_vals[m_idx])
            costs = {}
            for lag in candidate_lags:
                total, n_matched = 0.0, 0
                for ta in tps_a:
                    idx_a = np.where((perm_marker_lab == marker_a) & (np.isclose(perm_tp_vals, ta)))[0]
                    if len(idx_a) == 0:
                        continue
                    target = ta + lag
                    tb_matches = [tb for tb in tps_b if abs(tb - target) <= tol]
                    if not tb_matches:
                        continue
                    tb = min(tb_matches, key=lambda x: abs(x - target))
                    idx_b = np.where((perm_marker_lab == marker_b) & (np.isclose(perm_tp_vals, tb)))[0]
                    if len(idx_b) == 0:
                        continue
                    total += medoid_distance_statistic(Dmat, idx_a, idx_b)
                    n_matched += 1
                if n_matched > 0:
                    costs[lag] = total / n_matched
            if costs:
                bl = min(costs, key=costs.get)
                null_best_costs[p] = costs[bl]
                null_best_lags[p] = bl
            else:
                null_best_costs[p] = float("inf")
                null_best_lags[p] = float("nan")

        finite = np.isfinite(null_best_costs)
        p_raw = float((null_best_costs[finite] <= best_cost).mean()) if finite.any() else float("nan")
        p_corr = correct_pvalue_floor(p_raw, int(finite.sum())) if finite.any() else None
        return {
            "lag_cost_curve": {str(k): round(v, 4) for k, v in lag_costs.items()},
            "best_lag_h": best_lag, "best_aligned_cost": round(float(best_cost), 4),
            "existence_test": {"p_value_floor_corrected": p_corr, "n_permutations_finite": int(finite.sum()), "n_permutations_total": n_perm},
            "magnitude_uncertainty": {
                "best_lag_h": best_lag,
                "null_best_lag_distribution_summary": {
                    "median": float(np.nanmedian(null_best_lags)) if np.isfinite(null_best_lags).any() else None,
                    "p10": float(np.nanpercentile(null_best_lags, 10)) if np.isfinite(null_best_lags).any() else None,
                    "p90": float(np.nanpercentile(null_best_lags, 90)) if np.isfinite(null_best_lags).any() else None,
                },
            },
        }

    raw_c = _run_section_c_test(D)
    resid_c = _run_section_c_test(D_resid)
    if raw_c is None or resid_c is None:
        return {"source": source, "cell_type": cell_type, "marker_a": marker_a, "marker_b": marker_b, "skipped": "no timepoint pairs matched within tolerance for any candidate lag"}

    logger.info(
        f"    raw:   best_lag={raw_c['best_lag_h']:+.2f}h  cost={raw_c['best_aligned_cost']:.2f}  p={raw_c['existence_test']['p_value_floor_corrected']}  "
        f"(checked {len(candidate_lags)} candidate lags -- the lag value is only meaningful if p is significant)"
    )
    logger.info(
        f"    resid: best_lag={resid_c['best_lag_h']:+.2f}h  cost={resid_c['best_aligned_cost']:.2f}  p={resid_c['existence_test']['p_value_floor_corrected']}"
    )

    return {
        "source": source, "cell_type": cell_type, "marker_a": marker_a, "marker_b": marker_b,
        "candidate_lags_h": candidate_lags,
        "raw": raw_c, "residualized_against_n_localisations": resid_c,
        "_note": "Uncertainty in magnitude_uncertainty is reported via the permutation null's own spread of best-fit lags, not a bootstrap CI on a Fréchet-based ratio statistic -- deliberately avoiding the exact construction the variance/bimodality diagnostic above is built to catch as unreliable for this kind of estimator. residualized_against_n_localisations controls for whether an apparent lag reflects a shared count trend rather than genuine cross-marker shape alignment.",
    }


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diagrams", type=Path, default=Path("diagrams"))
    ap.add_argument("--features", type=Path, default=Path("data/feature_matrix.csv"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--sources", nargs="+", default=SOURCES)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    for p in [args.features, args.diagrams]:
        if not p.exists():
            logger.error(f"Not found: {p}")
            return 1
    args.results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features, dtype={"replicate": str})
    logger.info("=" * 70)
    logger.info("Population geometry — View 3, Tiers 1-4")
    logger.info("=" * 70)
    logger.info(f"Sources: {args.sources}   n_perm: {args.n_perm}   n_jobs: {args.n_jobs}")

    results: Dict = {"section_A_population_geometry": {}, "section_B_baseline_return": {}, "section_C_cross_marker_shift": {}}

    for source in args.sources:
        markers = sorted(df[df["source"] == source]["marker"].unique())
        cell_types = sorted(df[df["source"] == source]["cell_type"].unique())
        logger.info(f"\n{'='*60}\nSOURCE: {source}  markers={markers}  cell_types={cell_types}\n{'='*60}")

        results["section_A_population_geometry"][source] = {}
        for mk in markers:
            r = section_a_population_geometry(df, source, mk, args.diagrams, args.n_perm, args.n_jobs)
            if r is not None:
                results["section_A_population_geometry"][source][mk] = r

        results["section_B_baseline_return"][source] = {}
        for ct in cell_types:
            for mk in markers:
                r = section_b_baseline_return(df, source, ct, mk, args.diagrams, args.n_perm, args.n_jobs)
                if r is not None:
                    results["section_B_baseline_return"][source][f"{ct}/{mk}"] = r

        results["section_C_cross_marker_shift"][source] = {}
        if len(markers) == 2:
            for ct in cell_types:
                r = section_c_cross_marker_shift(df, source, ct, markers[0], markers[1], args.diagrams, args.n_perm, args.n_jobs)
                if r is not None:
                    results["section_C_cross_marker_shift"][source][ct] = r

    json_path = args.results_dir / "population_geometry_results.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"\nResults -> {json_path}")

    txt_path = args.results_dir / "population_geometry_summary.txt"
    with open(txt_path, "w") as fh:
        fh.write("Population geometry — summary\n" + "=" * 70 + "\n\n")
        fh.write("[Section A: T1 existence, full population -- raw | residualized against n_localisations]\n")
        for source, by_marker in results["section_A_population_geometry"].items():
            for mk, r in by_marker.items():
                t1 = r.get("t1_existence_full_population")
                if t1:
                    raw, resid = t1["raw"], t1["residualized_against_n_localisations"]
                    fh.write(
                        f"  {source}/{mk}: {r['cell_type_other']} vs {r['cell_type_ref']}  "
                        f"raw observed={raw['observed']:.2f} p={raw['p_value_floor_corrected']:.4g}  |  "
                        f"resid observed={resid['observed']:.2f} p={resid['p_value_floor_corrected']:.4g}\n"
                    )
        fh.write("\n[Section B: T4a latest-timepoint test vs. own control -- raw | residualized]\n")
        for source, by_stratum in results["section_B_baseline_return"].items():
            for key, r in by_stratum.items():
                tests = r.get("tests_vs_control", {})
                if tests:
                    latest = sorted(tests.keys(), key=lambda k: float(re.sub("[^0-9.]", "", k) or 0))[-1]
                    t = tests[latest]
                    raw, resid = t["raw"], t["residualized_against_n_localisations"]
                    fh.write(
                        f"  {source}/{key} @ {latest} (latest tested): "
                        f"raw W2={raw['observed_w2_medoid_to_medoid']:.2f} p={raw['p_value_floor_corrected']:.4g}  |  "
                        f"resid W2={resid['observed_w2_medoid_to_medoid']:.2f} p={resid['p_value_floor_corrected']:.4g}\n"
                    )
        fh.write("\n[Section C: T4b best-lag alignment -- raw | residualized]\n")
        for source, by_ct in results["section_C_cross_marker_shift"].items():
            for ct, r in by_ct.items():
                if r.get("skipped"):
                    continue
                raw, resid = r["raw"], r["residualized_against_n_localisations"]
                fh.write(
                    f"  {source}/{ct} ({r['marker_a']} vs {r['marker_b']}): "
                    f"raw best_lag={raw['best_lag_h']:+.2f}h p={raw['existence_test'].get('p_value_floor_corrected')}  |  "
                    f"resid best_lag={resid['best_lag_h']:+.2f}h p={resid['existence_test'].get('p_value_floor_corrected')}\n"
                )
    logger.info(f"Summary -> {txt_path}")

    logger.info("=" * 70)
    logger.info("Done.")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
