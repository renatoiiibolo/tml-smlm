#!/usr/bin/env python3
"""
Temporal structure of the cell-type effect on leading topological features.

View 1 (scalar/distributional), Tier 3: having identified which topological
features discriminate cell type within each source/marker stratum, this
module asks whether that discrimination is constant across the DNA-damage
repair timecourse or interacts with timepoint -- i.e., does the cell-type
gap in a feature's value change shape over time.

The dataset carries a single dose level in every stratum, so timepoint is
the only temporal axis available to probe.

Two passes:
  1. leading-feature trajectory -- for each source/marker stratum, a
     cell_type x timepoint interaction F-test on that stratum's own top
     features from the classification analysis, residualised against the
     appropriate covariate for each feature's scale group.
  2. a focused pass on the kuentzelmann/yH2AX stratum, where several
     independent checks in this pipeline converge on the same 1h/4h
     window, reporting the per-timepoint trajectory explicitly.

Each test is a nested-model F-test (full: feature ~ C(cell_type)*C(timepoint)
vs. reduced: feature ~ C(cell_type)+C(timepoint)), fit once with OLS rather
than cross-validated -- this is a trajectory-shape question, not an accuracy
estimate. Residualisation and testing use irradiated rows only
(is_control == False), since the cell-type gap during repair is what's
being asked about. Holm correction is applied within each stratum's own
tested set of features.

Usage
-----
  python -m tml_smlm.view1_scalar.temporal_structure --features data/feature_matrix.csv --results-dir results/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SOURCES = ["kuentzelmann", "hahn"]

# Probe 1 feature lists -- each source/marker stratum's own top-5 features
# by mean importance from the classification analysis. Deliberately not a
# feature list shared across strata: which feature actually carries the
# classification signal differs by source and marker.
PROBE1_FEATURES = {
    ("kuentzelmann", "53BP1"): [
        "scaleC_h0_betti_var", "scaleC_h0_betti_mean", "noise_h1_betti_var",
        "scaleC_h1_betti_mean", "noise_h1_betti_mean",
    ],
    ("kuentzelmann", "yH2AX"): [
        "noise_h1_betti_var", "scaleC_h1_landscape_peak_t", "scaleC_h1_betti_var",
        "scaleC_h1_landscape_integral", "noise_h1_landscape_integral",
    ],
    ("hahn", "Mre11"): [
        "scaleC_h0_landscape_integral", "scaleC_h0_betti_var", "scaleC_h0_persistent_entropy",
        "scaleC_h0_landscape_peak_t", "scaleC_h1_landscape_peak_t",
    ],
    ("hahn", "yH2AX"): [
        "scaleC_h0_betti_var", "noise_h0_landscape_integral", "scaleC_h0_landscape_integral",
        "scaleA_h1_landscape_peak_t_mean", "scaleA_h0_betti_mean_mean",
    ],
}

# Probe 2: kuentzelmann/yH2AX's own top features (same list as above --
# Probe 2 restricts the population and reporting, not the feature set).
PROBE2_FEATURES = PROBE1_FEATURES[("kuentzelmann", "yH2AX")]
PROBE2_FLAGGED_TIMEPOINTS = ["1h", "4h"]  # flagged by the cluster-presence baseline as quasi-complete separation


def _resid_covar_for(feat: str) -> str:
    """Same covariate convention as the classification analysis: Scale A/B
    features residualise against n_clusters, Scale C/noise features against
    n_localisations."""
    if feat.startswith("scaleA") or feat.startswith("scaleB"):
        return "n_clusters"
    return "n_localisations"  # scaleC, noise


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def holm_bonferroni(p_values: Dict[str, float]) -> Dict[str, Dict]:
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, Dict] = {}
    running_max = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        running_max = max(running_max, adj)
        out[name] = {
            "raw_p": round(p, 6),
            "holm_adjusted_p": round(running_max, 6),
            "significant_at_0.05": bool(running_max < 0.05),
        }
    return out


def _make_timepoint_label(tp_h: float) -> str:
    if tp_h == int(tp_h):
        return f"{int(tp_h)}h"
    return f"{tp_h}h"


def _residualise_series(df: pd.DataFrame, feature: str, covar: str) -> pd.Series:
    """Global OLS residualisation of feature against covar within df. NaN where data is missing."""
    sub = df[[feature, covar]].dropna()
    if len(sub) < 5:
        return pd.Series(float("nan"), index=df.index)
    X = np.column_stack([np.ones(len(sub)), sub[covar].to_numpy(dtype=float)])
    y = sub[feature].to_numpy(dtype=float)
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return pd.Series(float("nan"), index=df.index)
    resid = pd.Series(float("nan"), index=df.index)
    resid.loc[sub.index] = y - (X @ beta)
    return resid


def _interaction_f_test(data: pd.DataFrame, feature_col: str, group_col: str, time_col: str) -> Dict:
    """Nested-model F-test: full = feature ~ C(group)*C(time), reduced = feature ~ C(group)+C(time)."""
    if not HAS_STATSMODELS:
        return {"error": "statsmodels not installed"}
    sub = data[[feature_col, group_col, time_col]].dropna().copy()
    sub = sub.rename(columns={feature_col: "_y", group_col: "_g", time_col: "_t"})
    if sub["_g"].nunique() < 2 or sub["_t"].nunique() < 2 or len(sub) < 20:
        return {"skipped": True, "reason": f"n={len(sub)}, groups={sub['_g'].nunique()}, timepoints={sub['_t'].nunique()}"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_full = smf.ols("_y ~ C(_g) * C(_t)", data=sub).fit()
            m_red = smf.ols("_y ~ C(_g) + C(_t)", data=sub).fit()
            table = anova_lm(m_red, m_full)
        f_stat = float(table["F"].iloc[1])
        p_value = float(table["Pr(>F)"].iloc[1])
        df_diff = float(table.index[1]) if "df_diff" not in table.columns else float(table["df_diff"].iloc[1])
        return {
            "n": int(len(sub)), "n_groups": int(sub["_g"].nunique()), "n_timepoints": int(sub["_t"].nunique()),
            "f_statistic": round(f_stat, 4), "df_diff": round(df_diff, 1), "p_value": round(p_value, 6),
        }
    except Exception as e:
        return {"error": str(e)}


def _trajectory_stats(df: pd.DataFrame, feature: str, group_col: str, time_col: str) -> List[Dict]:
    """Per-(group x timepoint) mean +/- SE for feature. Used to build trajectory plots."""
    rows = []
    for (grp, tp), sub in df.groupby([group_col, time_col]):
        vals = sub[feature].dropna()
        if len(vals) < 3:
            continue
        rows.append({
            group_col: grp, time_col: float(tp), "timepoint_label": _make_timepoint_label(float(tp)),
            "n": int(len(vals)), "mean": round(float(vals.mean()), 6),
            "se": round(float(vals.sem()), 6), "median": round(float(vals.median()), 6),
        })
    return sorted(rows, key=lambda r: (str(r[group_col]), r[time_col]))


# ══════════════════════════════════════════════════════════════════════════════
# PROBE 1 — leading-feature temporal trajectory, per source × marker
# ══════════════════════════════════════════════════════════════════════════════

def run_probe1(df: pd.DataFrame) -> Dict:
    logger.info("\n" + "=" * 70)
    logger.info("PROBE 1 — leading-feature temporal trajectory (source-specific feature lists)")
    logger.info("=" * 70)
    results: Dict = {}

    for src in SOURCES:
        results[src] = {}
        markers = sorted(df[df["source"] == src]["marker"].unique())
        for mk in markers:
            feat_list = PROBE1_FEATURES.get((src, mk), [])
            mk_df = df[(df["source"] == src) & (df["marker"] == mk)].copy()
            mk_irr = mk_df[~mk_df["is_control"]].copy()

            for feat in feat_list:
                if feat not in mk_irr.columns:
                    continue
                mk_irr[f"_resid_{feat}"] = _residualise_series(mk_irr, feat, _resid_covar_for(feat))

            feat_results: Dict = {}
            p_vals: Dict[str, float] = {}
            for feat in feat_list:
                if feat not in mk_irr.columns:
                    continue
                resid_col = f"_resid_{feat}"
                traj_raw = _trajectory_stats(mk_irr, feat, "cell_type", "timepoint_h")
                traj_resid = _trajectory_stats(mk_irr, resid_col, "cell_type", "timepoint_h")
                f_result = _interaction_f_test(mk_irr, resid_col, "cell_type", "timepoint_h")
                if not f_result.get("skipped") and not f_result.get("error"):
                    p_vals[feat] = f_result.get("p_value", 1.0)
                    logger.info(f"  {src}/{mk}/{feat}: F={f_result.get('f_statistic','?')} p={f_result.get('p_value','?')}")
                feat_results[feat] = {
                    "residualised_against": _resid_covar_for(feat),
                    "trajectory_raw": traj_raw, "trajectory_residualised": traj_resid,
                    "interaction_f_test": f_result,
                }

            holm = holm_bonferroni(p_vals)
            for feat in feat_results:
                if feat in holm:
                    feat_results[feat]["interaction_f_test"]["holm_adjusted_p"] = holm[feat]["holm_adjusted_p"]
                    feat_results[feat]["interaction_f_test"]["significant_at_0.05_holm"] = holm[feat]["significant_at_0.05"]

            results[src][mk] = {
                "features_tested": feat_list, "results": feat_results,
                "_note": (
                    f"Feature list is {src}/{mk}'s own top-5 by mean importance from the "
                    f"classification analysis, not a list shared across strata. "
                    f"Holm correction over {len(p_vals)} features tested for this stratum. "
                    f"Irradiated rows only (is_control == False)."
                ),
            }
    results["_interpretation"] = (
        "Tests whether the cell-type gap in each stratum's own leading classification "
        "feature(s) is constant across the repair timecourse or changes shape. A "
        "significant interaction after Holm correction means the classification "
        "signal for that feature is not a flat offset between cell types -- see "
        "trajectory_residualised for the shape of the change."
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PROBE 2 — targeted: kuentzelmann/yH2AX at the flagged timepoints
# ══════════════════════════════════════════════════════════════════════════════

def run_probe2(df: pd.DataFrame) -> Dict:
    logger.info("\n" + "=" * 70)
    logger.info("PROBE 2 — kuentzelmann/yH2AX at the flagged timepoints (1h, 4h)")
    logger.info("=" * 70)

    mk_df = df[(df["source"] == "kuentzelmann") & (df["marker"] == "yH2AX")].copy()
    mk_irr = mk_df[~mk_df["is_control"]].copy()

    for feat in PROBE2_FEATURES:
        if feat in mk_irr.columns:
            mk_irr[f"_resid_{feat}"] = _residualise_series(mk_irr, feat, _resid_covar_for(feat))

    feat_results: Dict = {}
    p_vals: Dict[str, float] = {}
    for feat in PROBE2_FEATURES:
        if feat not in mk_irr.columns:
            continue
        resid_col = f"_resid_{feat}"
        traj_full = _trajectory_stats(mk_irr, resid_col, "cell_type", "timepoint_h")
        flagged_only = [r for r in traj_full if r["timepoint_label"] in PROBE2_FLAGGED_TIMEPOINTS]
        f_result = _interaction_f_test(mk_irr, resid_col, "cell_type", "timepoint_h")
        if not f_result.get("skipped") and not f_result.get("error"):
            p_vals[feat] = f_result.get("p_value", 1.0)
            logger.info(f"  kuentzelmann/yH2AX/{feat}: F={f_result.get('f_statistic','?')} p={f_result.get('p_value','?')}")
        feat_results[feat] = {
            "residualised_against": _resid_covar_for(feat),
            "trajectory_all_timepoints": traj_full,
            "trajectory_at_flagged_timepoints": flagged_only,
            "interaction_f_test": f_result,
        }

    holm = holm_bonferroni(p_vals)
    for feat in feat_results:
        if feat in holm:
            feat_results[feat]["interaction_f_test"]["holm_adjusted_p"] = holm[feat]["holm_adjusted_p"]
            feat_results[feat]["interaction_f_test"]["significant_at_0.05_holm"] = holm[feat]["significant_at_0.05"]

    return {
        "features_tested": PROBE2_FEATURES,
        "flagged_timepoints": PROBE2_FLAGGED_TIMEPOINTS,
        "results": feat_results,
        "_note": (
            f"Holm correction over {len(p_vals)} features tested within kuentzelmann/yH2AX. "
            "Direct follow-up to three independent findings already on record about this "
            "stratum: the cluster-presence baseline flagged 1h and 4h as quasi-complete "
            "separation; the classification analysis found yH2AX (not 53BP1) carries a "
            "real, count-mediated component; and the classification robustness check "
            "found that dropping the 4h timepoint costs more accuracy than dropping any "
            "other timepoint tested."
        ),
        "_interpretation": (
            "If the interaction is significant and the flagged-timepoint trajectory "
            "shows the cell-type gap widening or reversing specifically at 4h, that "
            "ties the count-mediated signal found in the classification analysis to a "
            "specific point in the repair timecourse, rather than leaving it as a pooled "
            "classification result. If not, the convergence of independent checks on "
            "this stratum more likely reflects something about its data structure "
            "(sample size, variance) than a specific event at 4h -- worth stating "
            "either way, not just when it confirms the hypothesis."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    if not HAS_STATSMODELS:
        logger.error("statsmodels is required: pip install statsmodels")
        return 1

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=Path("data/feature_matrix.csv"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    if not args.features.exists():
        logger.error(f"Feature matrix not found: {args.features}")
        return 1

    df = pd.read_csv(args.features, dtype={"replicate": str})
    logger.info("=" * 70)
    logger.info("Temporal structure probes (View 1, Tier 3)")
    logger.info("=" * 70)
    logger.info(f"Loaded {len(df)} nuclei from {args.features}")

    args.results_dir.mkdir(parents=True, exist_ok=True)

    probe1 = run_probe1(df)
    probe2 = run_probe2(df)

    results = {"probe1_leading_feature_temporal": probe1, "probe2_kuentzelmann_yH2AX_flagged_timepoints": probe2}

    json_path = args.results_dir / "temporal_structure_results.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"\nResults -> {json_path}")

    txt_path = args.results_dir / "temporal_structure_summary.txt"
    with open(txt_path, "w") as fh:
        fh.write("Temporal structure — summary\n" + "=" * 70 + "\n\n")
        fh.write("[PROBE 1 — leading-feature temporal trajectory]\n")
        for src, src_res in probe1.items():
            if src.startswith("_"):
                continue
            for mk, mk_res in src_res.items():
                for feat, fr in mk_res.get("results", {}).items():
                    ft = fr.get("interaction_f_test", {})
                    if ft.get("skipped") or ft.get("error"):
                        continue
                    fh.write(
                        f"  {src}/{mk}/{feat}: F={ft.get('f_statistic','?')} p={ft.get('p_value','?')} "
                        f"holm_adj={ft.get('holm_adjusted_p','?')} sig={ft.get('significant_at_0.05_holm','?')}\n"
                    )
        fh.write("\n[PROBE 2 — kuentzelmann/yH2AX at flagged timepoints (1h, 4h)]\n")
        for feat, fr in probe2.get("results", {}).items():
            ft = fr.get("interaction_f_test", {})
            if ft.get("skipped") or ft.get("error"):
                continue
            fh.write(
                f"  {feat}: F={ft.get('f_statistic','?')} p={ft.get('p_value','?')} "
                f"holm_adj={ft.get('holm_adjusted_p','?')} sig={ft.get('significant_at_0.05_holm','?')}\n"
            )
            for row in fr.get("trajectory_at_flagged_timepoints", []):
                fh.write(f"      {row['cell_type']} @ {row['timepoint_label']}: mean={row['mean']:.4g} (n={row['n']})\n")
    logger.info(f"Summary -> {txt_path}")

    logger.info("=" * 70)
    logger.info("Done.")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
