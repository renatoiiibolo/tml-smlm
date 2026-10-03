#!/usr/bin/env python3
"""
Robustness battery for View 1's classification result (View 1, Tier 2).

Two questions, both asked on top of the classification module rather than by
rederiving anything:

  A. CV-seed sensitivity -- is the reported balanced accuracy a property of
     the data, or an artifact of one particular 5x10 cross-validation split?
     Reruns the headline stratum/feature-set combinations under five extra
     seeds alongside the classification module's own default (42).

  B. Leave-one-timepoint-out -- for source/marker pairs where the cluster-
     presence analysis flagged a timepoint as quasi-complete separation
     (100% of nuclei with clusters), does dropping that single timepoint
     move the all-scales classification result disproportionately? RF on
     continuous shape features doesn't share logistic regression's
     separation failure mode, but a result carried by one saturated
     timepoint would still be worth knowing before treating the pooled
     number as representative of the full repair timecourse.

Both sections reuse `cv_balanced_accuracy`, `FoldSafeResidualiser`,
`prepare_stratum`, and the feature-column-group helpers from the
classification module directly, so the statistical logic here is
byte-identical to the classification module's own runs -- nothing is
reimplemented.

No permutation tests are rerun here: that answers a different, much more
expensive question (does the observed accuracy beat a null), already
answered once per run by the classification module. This battery only asks
whether the accuracy estimate itself is stable.

DEFERRED, NOT DROPPED
----------------------
Scale C's point-cloud subsampling seed (fixed in the feature-extraction
step) is not re-examined here -- checking it would require rerunning the
full feature-extraction pipeline, well outside the scope of a CV/refit
check like this one.

OUTPUTS
-------
  results/robustness_seed_sensitivity.json
  results/robustness_leave_one_timepoint_out.json
  results/robustness_summary.txt

USAGE
-----
  python -u robustness.py --features data/feature_matrix.csv --results-dir results/

RUNTIME NOTE
------------
No permutation tests, so this is CV refits only (raw + residualized, 50
folds each): 7 headline runs x 5 extra seeds for Section A, 4 flagged
timepoints x 1 rerun each for Section B. Well under an hour on 8 cores.

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
from typing import Dict, List

import numpy as np
import pandas as pd

from tml_smlm.view1_scalar import classification

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EXTRA_SEEDS = [11, 22, 33, 44, 55]  # alongside the classification module's own default (42)

# The stratum/feature-set combinations the paper's classification claims
# actually rest on -- not a full sweep of every source x marker x scale x
# population combination the classification module can produce.
HEADLINE_RUNS = [
    # (source, marker, scale_subset, impute, irradiated_only) -- resid
    # covariate(s) derived via _resid_kwargs_for, not hardcoded here.
    ("hahn",         "Mre11", "scale_C_noise", False, False),
    ("hahn",         "Mre11", "all_scales",    True,  False),
    ("hahn",         "yH2AX", "all_scales",    True,  False),
    ("kuentzelmann", "53BP1", "all_scales",    True,  False),
    ("kuentzelmann", "yH2AX", "all_scales",    True,  False),
    # Sham-exclusion versions of the hahn/Mre11 stratum above. A separate
    # presence-vs-shape comparison rests specifically on these two
    # irradiated-only scale_C_noise numbers (hahn/Mre11 and hahn/yH2AX,
    # sham rows dropped), so they get the same seed check. Not extended to
    # every irradiated-only combination the classification module
    # produces -- the other sham-excluded numbers aren't load-bearing for
    # any current claim.
    ("hahn",         "Mre11", "scale_C_noise", False, True),
    ("hahn",         "yH2AX", "scale_C_noise", False, True),
]

# Timepoints the cluster-presence analysis flagged as quasi-complete
# separation (100% of nuclei with at least one cluster) for that
# source/marker pair.
FLAGGED_TIMEPOINTS = [
    ("kuentzelmann", "yH2AX", "1h"),
    ("kuentzelmann", "yH2AX", "4h"),
    ("hahn",         "yH2AX", "1h"),
    ("hahn",         "yH2AX", "3h"),
]


def _feat_cols_for(df: pd.DataFrame, scale_subset: str) -> List[str]:
    """Same feature-set definitions the classification module uses, looked up by name rather than duplicated."""
    if scale_subset == "scale_B":
        return classification.scale_b_cols(df)
    if scale_subset == "scale_C":
        return classification.scale_c_cols(df)
    if scale_subset == "scale_C_noise":
        return classification.scale_c_cols(df) + classification.noise_cols(df)
    if scale_subset == "all_scales":
        return (
            classification.scale_a_mean_cols(df)
            + classification.scale_b_cols(df)
            + classification.scale_c_cols(df)
            + classification.noise_cols(df)
        )
    raise ValueError(f"Unknown scale_subset: {scale_subset}")


def _resid_kwargs_for(df: pd.DataFrame, scale_subset: str) -> Dict:
    """
    Which residualization kwargs `cv_balanced_accuracy` needs for a given
    scale_subset. all_scales mixes two covariates (Scale A/B -> n_clusters,
    Scale C/noise -> n_localisations) and needs the per-feature
    covariate_map rather than one shared covariate for the whole set --
    passing a single covariate here would residualize some features
    against the wrong count.
    """
    if scale_subset in ("scale_B",):
        return {"resid_count_col": "n_clusters"}
    if scale_subset in ("scale_C", "scale_C_noise"):
        return {"resid_count_col": "n_localisations"}
    if scale_subset == "all_scales":
        a_mean = classification.scale_a_mean_cols(df)
        b = classification.scale_b_cols(df)
        c = classification.scale_c_cols(df)
        noise = classification.noise_cols(df)
        cov_map = {col: "n_clusters" for col in a_mean + b}
        cov_map.update({col: "n_localisations" for col in c + noise})
        return {"resid_covariate_map": cov_map}
    raise ValueError(f"Unknown scale_subset: {scale_subset}")


def _covariate_cols_for(df: pd.DataFrame, scale_subset: str) -> List[str]:
    """Which raw covariate column(s) must be pulled into X alongside the features, for this scale_subset."""
    kwargs = _resid_kwargs_for(df, scale_subset)
    if "resid_count_col" in kwargs:
        return [kwargs["resid_count_col"]]
    return sorted(set(kwargs["resid_covariate_map"].values()))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — CV-SEED SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════

def run_seed_sensitivity(df: pd.DataFrame) -> Dict:
    results: Dict = {}
    # cv_balanced_accuracy reads CV_SEED from the classification module's
    # own global namespace at call time, so setting it here before each
    # call is a real, scoped way to force a different seed without
    # touching the module's actual CV logic. Restored right after.
    original_seed = classification.CV_SEED

    for source, marker, scale_subset, impute, irradiated_only in HEADLINE_RUNS:
        pop_tag = "irr" if irradiated_only else "full"
        label = f"{source}/{marker}/{scale_subset}/{pop_tag}"
        logger.info(f"\n[Seed sensitivity] {label}")
        sub, y, tp_labels, meta = classification.prepare_stratum(df, source, marker, irradiated_only=irradiated_only)
        feat_cols = [c for c in _feat_cols_for(df, scale_subset) if c in sub.columns]
        resid_kwargs = _resid_kwargs_for(df, scale_subset)
        covariate_cols = _covariate_cols_for(df, scale_subset)
        X_feat_only = sub[feat_cols].copy()
        X_with_resid = sub[feat_cols + covariate_cols].copy()

        raw_bas, resid_bas = [], []
        for seed in [42] + EXTRA_SEEDS:  # 42 first, matching the classification module's own default as the anchor point
            classification.CV_SEED = seed
            raw_scores = classification.cv_balanced_accuracy(X_feat_only, y, impute=impute)
            resid_scores = classification.cv_balanced_accuracy(X_with_resid, y, impute=impute, **resid_kwargs)
            raw_bas.append(float(raw_scores.mean()))
            resid_bas.append(float(resid_scores.mean()))
            logger.info(f"    seed={seed}: raw={raw_bas[-1]:.4f}  resid={resid_bas[-1]:.4f}")

        classification.CV_SEED = original_seed
        raw_arr, resid_arr = np.array(raw_bas), np.array(resid_bas)
        results[label] = {
            "meta": meta, "seeds": [42] + EXTRA_SEEDS,
            "raw_ba_per_seed": raw_bas, "resid_ba_per_seed": resid_bas,
            "raw_ba_mean": round(float(raw_arr.mean()), 4), "raw_ba_range": round(float(raw_arr.max() - raw_arr.min()), 4),
            "resid_ba_mean": round(float(resid_arr.mean()), 4), "resid_ba_range": round(float(resid_arr.max() - resid_arr.min()), 4),
            "anchor_resid_ba": round(resid_bas[0], 4),  # seed=42, i.e. the classification module's own reported number
            "flagged_unstable": bool(resid_arr.max() - resid_arr.min() > 0.05),  # same 0.05 threshold used for OOB/CV discordance
        }
        logger.info(
            f"    resid_ba across seeds: mean={results[label]['resid_ba_mean']:.4f} "
            f"range={results[label]['resid_ba_range']:.4f} flagged={results[label]['flagged_unstable']}"
        )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — LEAVE-ONE-TIMEPOINT-OUT
# ══════════════════════════════════════════════════════════════════════════════

def run_leave_one_timepoint_out(df: pd.DataFrame) -> Dict:
    results: Dict = {}
    for source, marker, drop_tp in FLAGGED_TIMEPOINTS:
        label = f"{source}/{marker}/drop_{drop_tp}"
        logger.info(f"\n[Leave-one-timepoint-out] {label}")
        sub, y, tp_labels, meta = classification.prepare_stratum(df, source, marker, irradiated_only=False)
        keep_mask = np.asarray([classification.make_timepoint_label(v) != drop_tp for v in sub["timepoint_h"]])
        n_dropped = int((~keep_mask).sum())

        feat_cols = [c for c in _feat_cols_for(df, "all_scales") if c in sub.columns]
        resid_kwargs = _resid_kwargs_for(df, "all_scales")
        covariate_cols = _covariate_cols_for(df, "all_scales")
        X_full = sub[feat_cols + covariate_cols].copy()
        y_full = y

        full_resid_ba = float(classification.cv_balanced_accuracy(X_full, y_full, impute=True, **resid_kwargs).mean())

        sub_dropped = sub.loc[keep_mask].reset_index(drop=True)
        y_dropped = y_full[keep_mask]
        X_dropped = sub_dropped[feat_cols + covariate_cols].copy()
        if len(sub_dropped) < 20 or y_dropped.sum() < 5 or (len(y_dropped) - y_dropped.sum()) < 5:
            results[label] = {"skipped": True, "reason": "insufficient data after drop", "n_dropped": n_dropped}
            logger.info(f"    SKIPPED: insufficient data after dropping {n_dropped} rows at {drop_tp}")
            continue
        dropped_resid_ba = float(classification.cv_balanced_accuracy(X_dropped, y_dropped, impute=True, **resid_kwargs).mean())

        delta = round(dropped_resid_ba - full_resid_ba, 4)
        results[label] = {
            "n_dropped": n_dropped, "n_remaining": int(len(sub_dropped)),
            "full_population_resid_ba": round(full_resid_ba, 4),
            "timepoint_dropped_resid_ba": round(dropped_resid_ba, 4),
            "delta": delta,
            "flagged_disproportionate": bool(abs(delta) > 0.05),
            "_note": (
                f"The cluster-presence analysis flagged {drop_tp} as 100% cluster presence for "
                "this source/marker (quasi-complete separation on the binary presence measure). "
                "This checks whether the continuous, shape-based classification result is "
                "similarly carried by that one timepoint."
            ),
        }
        logger.info(f"    full={full_resid_ba:.4f}  dropped={dropped_resid_ba:.4f}  delta={delta:+.4f}  flagged={results[label]['flagged_disproportionate']}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=Path("data/feature_matrix.csv"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    if not args.features.exists():
        logger.error(f"Feature matrix not found: {args.features}")
        return 1

    df = pd.read_csv(args.features, dtype={"replicate": str})
    logger.info("=" * 70)
    logger.info("View 1, Tier 2 robustness — CV-seed and leave-one-timepoint-out sensitivity")
    logger.info("=" * 70)
    logger.info(f"Loaded {len(df)} nuclei from {args.features}")

    args.results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*70}\n[Section A] CV-seed sensitivity\n{'='*70}")
    seed_results = run_seed_sensitivity(df)
    with open(args.results_dir / "robustness_seed_sensitivity.json", "w") as fh:
        json.dump(seed_results, fh, indent=2, default=str)
    logger.info(f"\nSeed sensitivity -> {args.results_dir / 'robustness_seed_sensitivity.json'}")

    logger.info(f"\n{'='*70}\n[Section B] Leave-one-timepoint-out\n{'='*70}")
    ltpo_results = run_leave_one_timepoint_out(df)
    with open(args.results_dir / "robustness_leave_one_timepoint_out.json", "w") as fh:
        json.dump(ltpo_results, fh, indent=2, default=str)
    logger.info(f"Leave-one-timepoint-out -> {args.results_dir / 'robustness_leave_one_timepoint_out.json'}")

    txt_path = args.results_dir / "robustness_summary.txt"
    with open(txt_path, "w") as fh:
        fh.write("View 1, Tier 2 robustness — summary\n" + "=" * 70 + "\n\n")
        fh.write("[Section A] CV-seed sensitivity (anchor seed=42 + 5 more)\n" + "-" * 60 + "\n")
        for label, r in seed_results.items():
            fh.write(
                f"  {label}: anchor resid_ba={r['anchor_resid_ba']:.4f}  "
                f"mean across 6 seeds={r['resid_ba_mean']:.4f}  range={r['resid_ba_range']:.4f}  "
                f"flagged={r['flagged_unstable']}\n"
            )
        fh.write("\n[Section B] Leave-one-timepoint-out (flagged quasi-separation timepoints)\n" + "-" * 60 + "\n")
        for label, r in ltpo_results.items():
            if r.get("skipped"):
                fh.write(f"  {label}: SKIPPED — {r['reason']}\n")
            else:
                fh.write(
                    f"  {label}: full={r['full_population_resid_ba']:.4f}  "
                    f"dropped={r['timepoint_dropped_resid_ba']:.4f}  delta={r['delta']:+.4f}  "
                    f"flagged={r['flagged_disproportionate']}\n"
                )
        fh.write(
            "\nDeferred, not dropped: Scale C's point-cloud subsampling seed (fixed in the "
            "feature-extraction step) is not re-examined here -- would require a full rerun of "
            "the feature-extraction pipeline.\n"
        )
    logger.info(f"Summary -> {txt_path}")

    logger.info("=" * 70)
    logger.info("Robustness battery complete.")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
