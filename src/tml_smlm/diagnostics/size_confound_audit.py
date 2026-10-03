#!/usr/bin/env python3
"""
Hypothesis-free screen of every above-baseline-importance classification
feature against each nucleus's own physical size.

The classification analysis already residualizes cluster- and localization-
count features against the count covariates they're expected to track
(n_clusters, n_localisations) -- that correction is targeted: it fixes the
confounds we already had a name for. This module asks a different question:
did anything else -- a feature nobody specifically suspected -- end up
carrying raw nucleus size rather than real topology, just because it was
never checked? An automatic screen over every meaningful feature catches
that class of confound; a targeted check by construction cannot, since it
only ever looks where someone already thought to look.

METHOD
------
Per (source, marker), using the classification module's own full-feature
"all_scales" run:
  1. "Meaningful" features = importance strictly above the uniform baseline
     (1/n_features for that run) -- an objective cutoff, not hand-picked.
  2. Each meaningful feature's per-nucleus values are Spearman-correlated
     against that nucleus's own diagonal, sqrt(size_x_nm^2 + size_y_nm^2),
     recovered by joining the feature matrix back to the manifest (the size
     columns don't survive into feature_matrix.csv).
  3. Pre-registered thresholds, decided before looking at this data:
     rho > 0.9 -> excluded outright, treated as untested; 0.3 < rho <= 0.9
     -> tested but flagged with a caveat; rho <= 0.3 -> clean. Holm
     correction is applied within the tested-and-flagged tier per
     (source, marker) group.
  4. A fixed pair of features known from an earlier stage to have a size-
     tracking degeneracy (since fixed at the source) is cross-referenced
     explicitly, to confirm the fix actually eliminated their size-tracking
     behavior post-correction, rather than assuming it did.

This is disclosure infrastructure, not a biological finding: an excluded or
caveated feature here is not withdrawn from the classification results --
those already stand on their own count-residualization. This is an
independent, additional check for size specifically.

OUTPUTS
-------
  results/size_confound_audit.json
  results/size_confound_table.csv
  results/size_confound_summary.txt

USAGE
-----
  python -u size_confound_audit.py --manifest data/manifest.csv --features data/feature_matrix.csv \\
      --classification-json results/classification_results.json --results-dir results/ 2>&1 | tee size_confound_audit.log

RUNTIME NOTE
------------
  No diagrams loaded, no CV, no permutation -- Spearman correlations on
  already-computed columns. Expect seconds.

DEPENDENCIES
------------
  numpy  pandas  scipy
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
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from tml_smlm.view1_scalar.classification import scale_a_mean_cols, scale_b_cols, scale_c_cols, noise_cols

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

KEY_COLS = ["source", "cell_type", "marker", "condition", "replicate"]
EXCLUDE_THRESHOLD = 0.9   # rho > this -> excluded outright, untested
CAVEAT_THRESHOLD = 0.3    # this < rho <= EXCLUDE_THRESHOLD -> tested, flagged
KNOWN_DEGENERACY_FEATURES = [
    "scaleC_h0_landscape_integral", "scaleC_h0_landscape_peak_t",
    "noise_h0_landscape_integral", "noise_h0_landscape_peak_t",
]

MARKER_PAIRS = {
    "kuentzelmann": ["53BP1", "yH2AX"],
    "hahn": ["Mre11", "yH2AX"],
}


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


def all_scales_feature_list(df: pd.DataFrame) -> List[str]:
    """Same feature-group union the classification module's own all_scales run uses."""
    return scale_a_mean_cols(df) + scale_b_cols(df) + scale_c_cols(df) + noise_cols(df)


def audit_one_stratum(
    source: str, marker: str, features_df: pd.DataFrame, manifest_df: pd.DataFrame,
    classification_importances: Dict[str, Dict],
) -> Dict:
    all_feats = all_scales_feature_list(features_df)
    n_features = len(all_feats)
    baseline = 1.0 / n_features

    meaningful = [f for f in all_feats if f in classification_importances and classification_importances[f]["mean_importance"] > baseline]
    logger.info(f"  {source}/{marker}: {len(meaningful)} of {n_features} all_scales features above baseline importance ({baseline:.4f})")
    if not meaningful:
        return {"source": source, "marker": marker, "skipped": "no above-baseline features found in classification importances"}

    sub = features_df[(features_df["source"] == source) & (features_df["marker"] == marker)].copy()
    m = manifest_df[(manifest_df["source"] == source) & (manifest_df["marker"] == marker)].copy()
    joined = sub.merge(m[KEY_COLS + ["size_x_nm", "size_y_nm"]], on=KEY_COLS, how="left")
    n_missing_size = int(joined["size_x_nm"].isna().sum())
    joined["diagonal_nm"] = np.sqrt(joined["size_x_nm"] ** 2 + joined["size_y_nm"] ** 2)

    per_feature: Dict[str, Dict] = {}
    tested_pvalues: Dict[str, float] = {}
    for feat in meaningful:
        vals = joined[[feat, "diagonal_nm"]].dropna()
        if len(vals) < 10:
            per_feature[feat] = {"tier": "skipped", "reason": f"n={len(vals)} < 10 after dropping NaN"}
            continue
        rho, p = spearmanr(vals[feat], vals["diagonal_nm"])
        rho = float(rho)
        entry = {
            "importance": round(classification_importances[feat]["mean_importance"], 6),
            "spearman_rho_vs_diagonal": round(rho, 4), "p_value": float(p), "n": int(len(vals)),
        }
        if abs(rho) > EXCLUDE_THRESHOLD:
            entry["tier"] = "excluded_untested"
        elif abs(rho) > CAVEAT_THRESHOLD:
            entry["tier"] = "tested_flagged"
            tested_pvalues[feat] = float(p)
        else:
            entry["tier"] = "clean"
        per_feature[feat] = entry

    holm = holm_bonferroni(tested_pvalues) if tested_pvalues else {}
    for feat, h in holm.items():
        per_feature[feat]["holm_adjusted_p"] = h["holm_adjusted_p"]
        per_feature[feat]["survives_holm_0.05"] = h["significant_at_0.05"]

    n_excluded = sum(1 for v in per_feature.values() if v.get("tier") == "excluded_untested")
    n_flagged = sum(1 for v in per_feature.values() if v.get("tier") == "tested_flagged")
    n_clean = sum(1 for v in per_feature.values() if v.get("tier") == "clean")
    logger.info(f"    excluded={n_excluded}  flagged={n_flagged}  clean={n_clean}  (n_missing_size={n_missing_size})")

    known_degeneracy_check = {
        f: per_feature[f] for f in KNOWN_DEGENERACY_FEATURES if f in per_feature
    }
    if known_degeneracy_check:
        logger.info(f"    known-degeneracy features present in this run's meaningful set: {list(known_degeneracy_check.keys())}")
        for f, v in known_degeneracy_check.items():
            logger.info(f"      {f}: rho={v.get('spearman_rho_vs_diagonal')} tier={v.get('tier')}")

    return {
        "source": source, "marker": marker, "n_features_total": n_features, "baseline_importance": round(baseline, 6),
        "n_meaningful": len(meaningful), "n_missing_size_data": n_missing_size,
        "n_excluded_untested": n_excluded, "n_tested_flagged": n_flagged, "n_clean": n_clean,
        "per_feature": per_feature,
        "known_degeneracy_cross_check": known_degeneracy_check,
        "_interpretation": (
            "excluded_untested features should not be cited as topological findings without independent "
            "confirmation the value isn't just nucleus size. tested_flagged features can be cited with the "
            "caveat stated explicitly. clean features are unaffected by this screen. This does not withdraw "
            "any classification result -- those already residualize against count; this is an independent, "
            "additional check for size specifically, not previously performed for any feature outside "
            "the two known-degeneracy features."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--features", type=Path, default=Path("data/feature_matrix.csv"))
    ap.add_argument("--classification-json", type=Path, default=Path("results/classification_results.json"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    for p in [args.manifest, args.features, args.classification_json]:
        if not p.exists():
            logger.error(f"Not found: {p}")
            return 1
    args.results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("size_confound_audit — automatic, hypothesis-free size screen")
    logger.info("=" * 70)

    manifest_df = pd.read_csv(args.manifest, dtype={"replicate": str})
    features_df = pd.read_csv(args.features, dtype={"replicate": str})
    with open(args.classification_json) as f:
        classification = json.load(f)
    logger.info(f"Loaded manifest ({len(manifest_df)} rows), features ({len(features_df)} rows), classification results")

    if "size_x_nm" not in manifest_df.columns or "size_y_nm" not in manifest_df.columns:
        logger.error("data/manifest.csv is missing size_x_nm/size_y_nm -- this screen cannot run without recorded nucleus size.")
        return 1

    results: Dict = {}
    for source, markers in MARKER_PAIRS.items():
        if source not in features_df["source"].unique():
            continue
        results[source] = {}
        for marker in markers:
            classification_run = classification.get(source, {}).get("section_A", {}).get(marker, {}).get("all_scales", {})
            importances = classification_run.get("feature_importances")
            if not importances:
                logger.warning(f"  {source}/{marker}: no all_scales feature_importances found in classification JSON -- skipping")
                results[source][marker] = {"skipped": "no classification all_scales feature_importances found"}
                continue
            results[source][marker] = audit_one_stratum(source, marker, features_df, manifest_df, importances)

    json_path = args.results_dir / "size_confound_audit.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"\nJSON results -> {json_path}")

    rows = []
    for source, by_marker in results.items():
        for marker, r in by_marker.items():
            if r.get("skipped"):
                continue
            for feat, v in r.get("per_feature", {}).items():
                if v.get("tier") == "skipped":
                    continue
                rows.append({
                    "source": source, "marker": marker, "feature": feat,
                    "importance": v.get("importance"), "spearman_rho_vs_diagonal": v.get("spearman_rho_vs_diagonal"),
                    "p_value": v.get("p_value"), "holm_adjusted_p": v.get("holm_adjusted_p"), "tier": v.get("tier"),
                })
    csv_path = args.results_dir / "size_confound_table.csv"
    if rows:
        pd.DataFrame(rows).sort_values(["source", "marker", "tier", "spearman_rho_vs_diagonal"], ascending=[True, True, True, False]).to_csv(csv_path, index=False)
        logger.info(f"CSV table -> {csv_path}  ({len(rows)} rows)")

    txt_path = args.results_dir / "size_confound_summary.txt"
    with open(txt_path, "w") as fh:
        fh.write("size_confound_audit — summary\n" + "=" * 70 + "\n\n")
        for source, by_marker in results.items():
            for marker, r in by_marker.items():
                if r.get("skipped"):
                    fh.write(f"{source}/{marker}: SKIPPED ({r['skipped']})\n")
                    continue
                fh.write(
                    f"{source}/{marker}: {r['n_meaningful']} meaningful of {r['n_features_total']} features "
                    f"(baseline={r['baseline_importance']:.4f}) -- "
                    f"excluded={r['n_excluded_untested']} flagged={r['n_tested_flagged']} clean={r['n_clean']}\n"
                )
                for feat, v in sorted(r["per_feature"].items(), key=lambda kv: kv[1].get("spearman_rho_vs_diagonal", 0) if isinstance(kv[1].get("spearman_rho_vs_diagonal"), float) else 0, reverse=True):
                    if v.get("tier") in ("excluded_untested", "tested_flagged"):
                        fh.write(f"    [{v['tier']}] {feat}: rho={v['spearman_rho_vs_diagonal']} imp={v['importance']}\n")
                if r.get("known_degeneracy_cross_check"):
                    fh.write("  Known-degeneracy cross-check:\n")
                    for feat, v in r["known_degeneracy_cross_check"].items():
                        fh.write(f"    {feat}: rho={v.get('spearman_rho_vs_diagonal')} tier={v.get('tier')}\n")
                fh.write("\n")
    logger.info(f"Summary -> {txt_path}")

    logger.info("=" * 70)
    logger.info("size_confound_audit complete.")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
