#!/usr/bin/env python3
"""
Checks whether the pipeline's existing count-residualization leaves nonlinear
structure behind.

Every count-residualization step used elsewhere in the pipeline (the
classification module's FoldSafeResidualiser, the population-geometry
module's distance-matrix residualization, the coupling modules'
timepoint+count residualization) removes a purely LINEAR term in the count
covariate (n_clusters / n_localisations). Persistence-based summary
statistics can plausibly grow with point count in a saturating,
diminishing-returns way -- a real, TDA-specific reason a linear term might
not fully capture the relationship, distinct from a generic "maybe it's
nonlinear somewhere" worry. This is a closeable question: does
monotonic-but-nonlinear structure survive the existing linear removal, and
if so, does one interpretable quadratic term close it.

Why Spearman, not Pearson, is the right check
----------------------------------------------
OLS residuals are orthogonal to the regressor they were fit against by
construction (the normal equations guarantee the Pearson correlation
between an OLS residual and its own regressor is exactly zero), so testing
Pearson correlation here would trivially return ~0 regardless of whether
nonlinear structure remains. Spearman rank correlation carries no such
guarantee: it is sensitive to any remaining monotonic relationship, linear
or not, which is exactly what a missed quadratic/saturating term would
leave behind. A near-zero Spearman confirms the linear removal genuinely
worked; a nonzero one is real, informative signal.

Why this doesn't force a linear-vs-blackbox tradeoff
------------------------------------------------------
A quadratic term added to the same OLS design (feature ~ 1 + count +
count^2) is still a closed-form, fully interpretable regression, not a step
toward a blackbox model. This module tests directly whether that small
addition is even needed anywhere before deciding whether to write it into
any pipeline stage. If no flagged feature needs it, nothing changes.

Scope
-----
Reuses the size-confound audit's own feature scoping exactly: the same
above-baseline-importance features from the classification module's
all-scales run, both sources, all markers -- the natural, already
established scope for a second confound-adequacy check on the same feature
set, not a new one invented here. The flagging threshold (CAVEAT_THRESHOLD
= 0.3) is reused directly for the same reason it was chosen there: it
isn't tuned to this particular check.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

from tml_smlm.view1_scalar.classification import FoldSafeResidualiser, scale_a_mean_cols, scale_b_cols, scale_c_cols, noise_cols
from tml_smlm.diagnostics.size_confound_audit import all_scales_feature_list, MARKER_PAIRS, CAVEAT_THRESHOLD


def covariate_for(feat: str) -> str:
    """Same scale-prefix -> covariate assignment used throughout the pipeline."""
    if feat.startswith("scaleA") or feat.startswith("scaleB"):
        return "n_clusters"
    return "n_localisations"  # scaleC_*, noise_*


def quadratic_residual(x: pd.Series, covar: pd.Series) -> np.ndarray:
    """feature ~ 1 + covar + covar^2 -- still closed-form OLS, still fully
    interpretable, just not a mode FoldSafeResidualiser offers."""
    X = np.column_stack([np.ones(len(covar)), covar.to_numpy(dtype=float), covar.to_numpy(dtype=float) ** 2])
    y = x.to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def audit_stratum(source: str, marker: str, features_df: pd.DataFrame, classification_importances: Dict) -> Dict:
    all_feats = all_scales_feature_list(features_df)
    n_features = len(all_feats)
    baseline = 1.0 / n_features
    meaningful = [f for f in all_feats if f in classification_importances and classification_importances[f]["mean_importance"] > baseline]

    sub = features_df[(features_df["source"] == source) & (features_df["marker"] == marker)].copy()
    per_feature: Dict[str, Dict] = {}
    n_flagged = n_closed_by_quadratic = n_still_open = 0

    for feat in meaningful:
        covar_col = covariate_for(feat)
        d = sub[[feat, covar_col]].dropna()
        if len(d) < 20:
            per_feature[feat] = {"skipped": f"n={len(d)} < 20 after dropping NaN"}
            continue

        resid_linear = FoldSafeResidualiser(count_cols=covar_col).fit(d).transform(d)[feat]
        rho_sanity, _ = spearmanr(d[feat], d[covar_col])  # raw feature vs covariate, before any residualization -- context, not the diagnostic itself
        rho_linear, p_linear = spearmanr(resid_linear, d[covar_col])

        entry = {
            "covariate": covar_col, "n": len(d),
            "raw_spearman_vs_covariate": round(float(rho_sanity), 4),
            "linear_residual_spearman_vs_covariate": round(float(rho_linear), 4),
            "linear_residual_p": float(p_linear),
        }

        if abs(rho_linear) > CAVEAT_THRESHOLD:
            n_flagged += 1
            resid_quad = quadratic_residual(d[feat], d[covar_col])
            rho_quad, p_quad = spearmanr(resid_quad, d[covar_col])
            entry["flagged"] = True
            entry["quadratic_residual_spearman_vs_covariate"] = round(float(rho_quad), 4)
            entry["quadratic_residual_p"] = float(p_quad)
            entry["closed_by_quadratic_term"] = bool(abs(rho_quad) <= CAVEAT_THRESHOLD)
            if entry["closed_by_quadratic_term"]:
                n_closed_by_quadratic += 1
            else:
                n_still_open += 1
        else:
            entry["flagged"] = False

        per_feature[feat] = entry

    logger.info(f"  {source}/{marker}: {len(meaningful)} meaningful features, {n_flagged} flagged (|rho|>{CAVEAT_THRESHOLD}), {n_closed_by_quadratic} closed by adding a quadratic term, {n_still_open} still open after it")
    return {
        "source": source, "marker": marker, "n_meaningful": len(meaningful),
        "n_flagged": n_flagged, "n_closed_by_quadratic": n_closed_by_quadratic, "n_still_open_after_quadratic": n_still_open,
        "per_feature": per_feature,
        "_interpretation": (
            "linear_residual_spearman_vs_covariate near zero confirms the existing linear removal already "
            "works -- no nonlinear structure to find. A flagged, nonzero value means monotonic structure "
            "the linear term missed; closed_by_quadratic_term=true means one interpretable quadratic term "
            "(still closed-form OLS, not a blackbox model) resolves it. still_open after the quadratic term "
            "would mean the remaining structure isn't well described by a low-order polynomial at all."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=Path("data/feature_matrix.csv"))
    ap.add_argument("--classification-json", type=Path, default=Path("results/classification_results.json"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    for p in [args.features, args.classification_json]:
        if not p.exists():
            logger.error(f"Not found: {p}")
            return 1
    args.results_dir.mkdir(parents=True, exist_ok=True)

    fm = pd.read_csv(args.features, dtype={"replicate": str})
    with open(args.classification_json) as f:
        classification = json.load(f)

    logger.info("=" * 70)
    logger.info("count_nonlinearity_check")
    logger.info("=" * 70)
    logger.info(f"Loaded features ({len(fm)} rows), classification results")

    results: Dict = {}
    for source, markers in MARKER_PAIRS.items():
        if source not in fm["source"].unique():
            continue
        results[source] = {}
        for marker in markers:
            importances = classification.get(source, {}).get("section_A", {}).get(marker, {}).get("all_scales", {}).get("feature_importances")
            if not importances:
                logger.warning(f"  {source}/{marker}: no all_scales feature_importances found in classification JSON -- skipping")
                results[source][marker] = {"skipped": "no classification all_scales feature_importances found"}
                continue
            results[source][marker] = audit_stratum(source, marker, fm, importances)

    json_path = args.results_dir / "count_nonlinearity.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"\nJSON results -> {json_path}")

    total_flagged = sum(r.get("n_flagged", 0) for by_mk in results.values() for r in by_mk.values())
    total_closed = sum(r.get("n_closed_by_quadratic", 0) for by_mk in results.values() for r in by_mk.values())
    total_open = sum(r.get("n_still_open_after_quadratic", 0) for by_mk in results.values() for r in by_mk.values())

    txt_path = args.results_dir / "count_nonlinearity_summary.txt"
    with open(txt_path, "w") as fh:
        fh.write("count_nonlinearity_check — summary\n" + "=" * 70 + "\n\n")
        fh.write(f"Archive-wide: {total_flagged} features flagged, {total_closed} closed by a quadratic term, {total_open} still open after it\n\n")
        for source, by_mk in results.items():
            for marker, r in by_mk.items():
                if r.get("skipped"):
                    fh.write(f"{source}/{marker}: SKIPPED ({r['skipped']})\n")
                    continue
                fh.write(f"{source}/{marker}: {r['n_meaningful']} meaningful, {r['n_flagged']} flagged, {r['n_closed_by_quadratic']} closed, {r['n_still_open_after_quadratic']} still open\n")
                for feat, v in r["per_feature"].items():
                    if v.get("flagged"):
                        fh.write(f"    [flagged] {feat}: linear_resid_rho={v['linear_residual_spearman_vs_covariate']} quad_resid_rho={v.get('quadratic_residual_spearman_vs_covariate')} closed={v.get('closed_by_quadratic_term')}\n")
                fh.write("\n")
    logger.info(f"Summary -> {txt_path}")
    logger.info(f"\nArchive-wide: {total_flagged} flagged, {total_closed} closed by a quadratic term, {total_open} still open after it")

    logger.info("=" * 70)
    logger.info("count_nonlinearity_check complete.")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
