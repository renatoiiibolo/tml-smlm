"""
View 2 (functional), Tier 2 — robustness of the sparse-localization result.

The localization module's smooth-lasso fit picks out particular grid
positions along the birth-radius axis as discriminative. This module asks
whether that selection is stable, via two checks against a set of
"headline" marker/source/scale panels — the load-bearing findings the
downstream temporal-trajectory and coupling analyses build on:

  - CV-seed sensitivity: refit at several additional cross-validation
    seeds and compare both the CV balanced accuracy and the set of
    selected grid positions.
  - Leave-one-timepoint-out: refit with each timepoint's nuclei dropped
    in turn, again comparing accuracy and selected positions against the
    full-population fit.

Why the grid positions matter as much as the accuracy: with 300
densely-spaced grid points, neighboring tent-function columns are nearly
collinear. Collinear predictors are the textbook case where an L1-type
sparse selector picks a different member of a correlated cluster on a
different CV split or subsample while the underlying signal is unchanged.
A smooth-lasso fit's balanced accuracy and even its number of nonzero
coefficients can look stable across seeds while the actual selected
position moves — so this module tracks Jaccard overlap of the selected
index sets, not just accuracy, to catch that failure mode directly.

vectorise, residualise, fl_cv, _load_dgm, and SCALE_COVAR are imported
from the localization module rather than redefined.
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

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

import tml_smlm.view2_functional.localization as localization

EXTRA_SEEDS = [11, 22, 33, 44, 55]  # alongside the localization module's own default seed (42)

# The three panels a spatial-scale claim was actually made about, not every
# panel the localization module produced:
#   - hahn/Mre11/scaleC_h0/lambda1 (n_nz=1) — a single-grid-point selection,
#     the most specific and therefore most fragile claim to re-check.
#   - kuentzelmann/yH2AX/scaleC_h1/lambda1 (n_nz=2) — the one significant
#     panel for this marker, tied to a temporal divergence seen at 4h.
#   - hahn/yH2AX/noise_h1/lambda2 (n_nz=25) — the most distributed
#     significant panel, included as a contrast case: does stability
#     behave differently for a diffuse selection than a sparse one?
HEADLINE_PANELS = [
    {"source": "hahn", "marker": "Mre11", "scale": "scaleC", "hom": "h0", "key": "scaleC_h0", "k": 1},
    {"source": "kuentzelmann", "marker": "yH2AX", "scale": "scaleC", "hom": "h1", "key": "scaleC_h1", "k": 1},
    {"source": "hahn", "marker": "yH2AX", "scale": "noise", "hom": "h1", "key": "noise_h1", "k": 2},
]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _prepare_panel(panel: Dict, fm: pd.DataFrame, diagrams_dir: Path):
    """Vectorize + residualize once -- every seed/drop refit reuses this, never re-loads diagrams."""
    source, marker, scale, hom, arr_key, k = panel["source"], panel["marker"], panel["scale"], panel["hom"], panel["key"], panel["k"]
    sub = fm[(fm["source"] == source) & (fm["marker"] == marker)].copy()
    dgms = [localization._load_dgm(diagrams_dir, row, arr_key) for _, row in sub.iterrows()]
    L, grid = localization.vectorise(dgms, k)
    covar_col = localization.SCALE_COVAR.get(scale, "n_localisations")
    covar = sub[covar_col].to_numpy(dtype=float)
    L_r = localization.residualise(L, covar)
    y = (sub["cell_type"] == localization.SOURCE_CFG[source]["pos"]).astype(int).to_numpy()
    tp = sub["timepoint_h"].apply(localization._tp_label).to_numpy()
    return L_r, y, tp, sub


def cv_seed_sensitivity(panel: Dict, L_r: np.ndarray, y: np.ndarray) -> Dict:
    lbl = f"{panel['source']}/{panel['marker']}/{panel['key']}/lambda{panel['k']}"
    original_seed = localization.CV_SEED
    results_per_seed = []
    try:
        for seed in [original_seed] + EXTRA_SEEDS:
            localization.CV_SEED = seed
            fl = localization.fl_cv(L_r, y)
            beta = np.asarray(fl["beta_lam_min"])
            idx = set(np.where(np.abs(beta) > 1e-6)[0].tolist())
            results_per_seed.append({"seed": seed, "resid_ba": fl["ba_at_lam_min"], "n_nz": fl["n_nonzero_min"], "selected_indices": sorted(idx)})
            logger.info(f"  [CV-seed] {lbl} seed={seed}: resid_ba={fl['ba_at_lam_min']:.4f} n_nz={fl['n_nonzero_min']} indices={sorted(idx)}")
    finally:
        localization.CV_SEED = original_seed

    bas = [r["resid_ba"] for r in results_per_seed]
    index_sets = [set(r["selected_indices"]) for r in results_per_seed]
    pairwise_jaccard = [_jaccard(index_sets[i], index_sets[j]) for i in range(len(index_sets)) for j in range(i + 1, len(index_sets))]
    return {
        "label": lbl, "per_seed": results_per_seed,
        "resid_ba_mean": round(float(np.mean(bas)), 4), "resid_ba_range": round(float(np.max(bas) - np.min(bas)), 4),
        "mean_pairwise_jaccard": round(float(np.mean(pairwise_jaccard)), 4) if pairwise_jaccard else None,
        "min_pairwise_jaccard": round(float(np.min(pairwise_jaccard)), 4) if pairwise_jaccard else None,
        "_interpretation": "mean_pairwise_jaccard=1.0 means the exact same grid positions were selected every seed -- a stable spatial claim. Lower values mean the position itself moves across seeds even if resid_ba/n_nz look stable, which resid_ba/n_nz alone would not catch.",
    }


def leave_one_timepoint_out(panel: Dict, L_r: np.ndarray, y: np.ndarray, tp: np.ndarray) -> Dict:
    lbl = f"{panel['source']}/{panel['marker']}/{panel['key']}/lambda{panel['k']}"
    full_fl = localization.fl_cv(L_r, y)
    full_idx = set(np.where(np.abs(np.asarray(full_fl["beta_lam_min"])) > 1e-6)[0].tolist())
    full_ba = full_fl["ba_at_lam_min"]

    per_timepoint = []
    for t in sorted(set(tp)):
        mask = tp != t
        if mask.sum() < 20 or len(set(y[mask])) < 2:
            per_timepoint.append({"timepoint": t, "skipped": f"n_remaining={int(mask.sum())} or single class after drop"})
            continue
        fl = localization.fl_cv(L_r[mask], y[mask])
        idx = set(np.where(np.abs(np.asarray(fl["beta_lam_min"])) > 1e-6)[0].tolist())
        jac = _jaccard(full_idx, idx)
        delta = round(fl["ba_at_lam_min"] - full_ba, 4)
        per_timepoint.append({
            "timepoint": t, "n_dropped": int((~mask).sum()), "n_remaining": int(mask.sum()),
            "resid_ba": fl["ba_at_lam_min"], "delta_ba": delta, "n_nz": fl["n_nonzero_min"],
            "selected_indices": sorted(idx), "jaccard_vs_full": round(jac, 4),
        })
        logger.info(f"  [drop_{t}] {lbl}: resid_ba={fl['ba_at_lam_min']:.4f} (delta={delta:+.4f}) jaccard_vs_full={jac:.4f}")

    return {"label": lbl, "full_population_indices": sorted(full_idx), "full_population_ba": full_ba, "per_timepoint": per_timepoint}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diagrams", type=Path, default=Path("diagrams"))
    ap.add_argument("--features", type=Path, default=Path("data/feature_matrix.csv"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    for p in [args.features, args.diagrams]:
        if not p.exists():
            logger.error(f"Not found: {p}")
            return 1
    args.results_dir.mkdir(parents=True, exist_ok=True)

    fm = pd.read_csv(args.features, dtype={"replicate": str})
    logger.info("=" * 70)
    logger.info("View 2 / Tier 2 — sparse-localization robustness")
    logger.info("=" * 70)
    logger.info(f"Panels: {[p['key'] + '/' + p['source'] + '/' + p['marker'] for p in HEADLINE_PANELS]}")

    results: Dict = {"cv_seed_sensitivity": {}, "leave_one_timepoint_out": {}}
    for panel in HEADLINE_PANELS:
        lbl = f"{panel['source']}/{panel['marker']}/{panel['key']}/lambda{panel['k']}"
        logger.info(f"\n{'='*60}\n{lbl}\n{'='*60}")
        L_r, y, tp, sub = _prepare_panel(panel, fm, args.diagrams)
        results["cv_seed_sensitivity"][lbl] = cv_seed_sensitivity(panel, L_r, y)
        results["leave_one_timepoint_out"][lbl] = leave_one_timepoint_out(panel, L_r, y, tp)

    json_path = args.results_dir / "view2_robustness_fl_results.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"\nResults -> {json_path}")

    txt_path = args.results_dir / "view2_robustness_summary.txt"
    with open(txt_path, "w") as fh:
        fh.write("View 2 / Tier 2 — sparse-localization robustness summary\n" + "=" * 70 + "\n\n")
        fh.write("[CV-seed sensitivity]\n")
        for lbl, r in results["cv_seed_sensitivity"].items():
            fh.write(f"  {lbl}: resid_ba_mean={r['resid_ba_mean']} range={r['resid_ba_range']} mean_jaccard={r['mean_pairwise_jaccard']} min_jaccard={r['min_pairwise_jaccard']}\n")
        fh.write("\n[Leave-one-timepoint-out]\n")
        for lbl, r in results["leave_one_timepoint_out"].items():
            fh.write(f"  {lbl}: full_ba={r['full_population_ba']}\n")
            for row in r["per_timepoint"]:
                if row.get("skipped"):
                    fh.write(f"    {row['timepoint']}: skipped ({row['skipped']})\n")
                else:
                    fh.write(f"    {row['timepoint']}: delta_ba={row['delta_ba']:+.4f} jaccard_vs_full={row['jaccard_vs_full']}\n")
    logger.info(f"Summary -> {txt_path}")

    logger.info("=" * 70)
    logger.info("View 2 / Tier 2 robustness complete.")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
