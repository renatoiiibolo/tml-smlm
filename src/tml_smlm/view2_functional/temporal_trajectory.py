"""
View 2 (functional), Tier 3 — temporal trajectory of the discriminative
spatial scale.

Does the birth-radius position the fused-lasso fit selects as
discriminative shift across the repair timecourse, or stay fixed? The
birth-radius axis is a real physical length scale, so whether the selected
position sits at a small scale (individual nascent foci) or a large one
(coalesced/matured foci, domain-level reorganization) is a direct
topological readout of focus maturation as repair proceeds.

Per-timepoint strata are much smaller than the pooled panels the
robustness module fits, so a single fit per timepoint would risk reporting
CV-split noise as if it were a real trajectory. Instead this module reruns
the robustness module's own cv_seed_sensitivity check, unmodified,
separately on each timepoint's subset, and reports each timepoint's
seed-stable core (the intersection of selected indices across seeds that
found any signal at all) rather than a single seed's selection.

Scope: the same three headline panels the robustness module tests, no
others.

Aggregation note: a strict intersection used to claim "consensus across
units" (seeds, timepoints, or anything else) is fragile -- one
non-informative unit silently zeroes an intersection even when every other
unit agrees. This module classifies each seed and each timepoint as
signal-bearing or not before intersecting, and excludes only the
genuinely non-informative ones (zero seeds selecting anything) from the
aggregate; units that disagree on which nonzero position to pick are real
disagreement and are kept in, not treated the same as no signal at all.
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

import tml_smlm.view2_functional.robustness as robustness

MIN_PER_TIMEPOINT = 20  # matching the localization/robustness modules' own gating convention


def per_timepoint_trajectory(panel: Dict, L_r: np.ndarray, y: np.ndarray, tp: np.ndarray) -> Dict:
    lbl = f"{panel['source']}/{panel['marker']}/{panel['key']}/lambda{panel['k']}"
    trajectory: List[Dict] = []
    for t in sorted(set(tp)):
        mask = tp == t
        n = int(mask.sum())
        if n < MIN_PER_TIMEPOINT or len(set(y[mask])) < 2:
            trajectory.append({"timepoint": t, "skipped": f"n={n} < {MIN_PER_TIMEPOINT} or single class present"})
            logger.info(f"  {lbl} @ {t}: skipped (n={n})")
            continue
        seed_result = robustness.cv_seed_sensitivity(panel, L_r[mask], y[mask])
        # Exclude non-informative seeds (n_nz==0, found nothing) before intersecting. A single dead
        # seed among several otherwise-unanimous seeds would otherwise mask a genuine majority
        # consensus under a naive full intersection. Seeds that disagree on WHICH nonzero indices to
        # select (both informative, different answers) are not excluded -- only genuinely
        # uninformative seeds are.
        informative = [s for s in seed_result["per_seed"] if s["n_nz"] > 0]
        core_sets = [set(s["selected_indices"]) for s in informative]
        core = sorted(set.intersection(*core_sets)) if core_sets else []
        n_informative_seeds = len(informative)
        # All seeds selecting zero coefficients means genuinely no signal at this timepoint --
        # distinct from seeds selecting nonzero but disagreeing coefficients (real signal, contested
        # exact position). Both can produce an empty core, but only the former is excluded from the
        # cross-timepoint aggregate below.
        all_seeds_zero = n_informative_seeds == 0
        trajectory.append({
            "timepoint": t, "n": n,
            "seed_stable_core": core, "n_informative_seeds": n_informative_seeds, "all_seeds_selected_zero": all_seeds_zero,
            "resid_ba_mean": seed_result["resid_ba_mean"], "resid_ba_range": seed_result["resid_ba_range"],
            "mean_pairwise_jaccard": seed_result["mean_pairwise_jaccard"],
            "per_seed": seed_result["per_seed"],  # retained in full -- needed to distinguish "no signal" from "signal, contested position" after the fact
        })
        tag = "NO SIGNAL" if all_seeds_zero else ("contested" if not core else "stable")
        logger.info(f"  {lbl} @ {t}: n={n} core={core} [{tag}, {n_informative_seeds}/6 informative] resid_ba_mean={seed_result['resid_ba_mean']:.4f} (jaccard={seed_result['mean_pairwise_jaccard']})")

    valid = [r for r in trajectory if not r.get("skipped")]
    signal_bearing = [r for r in valid if not r["all_seeds_selected_zero"]]
    no_signal = [r for r in valid if r["all_seeds_selected_zero"]]
    cores_over_time = [set(r["seed_stable_core"]) for r in signal_bearing]
    stable_across_signal_bearing_timepoints = sorted(set.intersection(*cores_over_time)) if cores_over_time else []
    union_across_signal_bearing_timepoints = sorted(set.union(*cores_over_time)) if cores_over_time else []

    return {
        "label": lbl, "trajectory": trajectory,
        "n_timepoints_tested": len(valid), "n_timepoints_skipped": len(trajectory) - len(valid),
        "n_no_signal_timepoints": len(no_signal), "no_signal_timepoint_labels": [r["timepoint"] for r in no_signal],
        "n_signal_bearing_timepoints": len(signal_bearing),
        "position_stable_across_signal_bearing_timepoints": stable_across_signal_bearing_timepoints,
        "union_of_signal_bearing_timepoint_cores": union_across_signal_bearing_timepoints,
        "_interpretation": (
            "Computed only over signal-bearing timepoints (at least one of 6 seeds selected a nonzero "
            "coefficient) -- no_signal_timepoint_labels lists which timepoints were excluded and why "
            "(all 6 seeds selected zero coefficients, i.e. chance-level accuracy, not a different position). "
            "position_stable_across_signal_bearing_timepoints nonempty means that index was seed-stably "
            "selected at every timepoint that had any signal at all -- the position does not shift. Empty "
            "with a nonempty union means the position moves across signal-bearing timepoints -- a real "
            "trajectory. A signal-bearing timepoint can still have an empty OWN seed_stable_core (check "
            "all_seeds_selected_zero==False with seed_stable_core==[] in the trajectory) -- that means real "
            "signal was present (resid_ba above chance) but seeds disagreed on the exact position, itself "
            "worth reporting, not the same as no signal at all."
        ),
    }


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
    logger.info("View 2 / Tier 3 — temporal trajectory of discriminative spatial scale")
    logger.info("=" * 70)
    logger.info(f"Panels: {[p['key'] + '/' + p['source'] + '/' + p['marker'] for p in robustness.HEADLINE_PANELS]}")

    results: Dict = {}
    for panel in robustness.HEADLINE_PANELS:
        lbl = f"{panel['source']}/{panel['marker']}/{panel['key']}/lambda{panel['k']}"
        logger.info(f"\n{'='*60}\n{lbl}\n{'='*60}")
        L_r, y, tp, sub = robustness._prepare_panel(panel, fm, args.diagrams)
        results[lbl] = per_timepoint_trajectory(panel, L_r, y, tp)

    json_path = args.results_dir / "temporal_trajectory_position.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"\nResults -> {json_path}")

    txt_path = args.results_dir / "temporal_trajectory_summary.txt"
    with open(txt_path, "w") as fh:
        fh.write("View 2 / Tier 3 — temporal trajectory summary\n" + "=" * 70 + "\n\n")
        for lbl, r in results.items():
            fh.write(f"{lbl}: {r['n_timepoints_tested']} tested, {r['n_timepoints_skipped']} skipped, {r['n_no_signal_timepoints']} no-signal ({r['no_signal_timepoint_labels']})\n")
            fh.write(f"  stable across signal-bearing timepoints: {r['position_stable_across_signal_bearing_timepoints']}\n")
            fh.write(f"  union across signal-bearing timepoints: {r['union_of_signal_bearing_timepoint_cores']}\n")
            for row in r["trajectory"]:
                if row.get("skipped"):
                    fh.write(f"    {row['timepoint']}: skipped ({row['skipped']})\n")
                else:
                    tag = "NO SIGNAL" if row["all_seeds_selected_zero"] else ("contested" if not row["seed_stable_core"] else "stable")
                    fh.write(f"    {row['timepoint']}: n={row['n']} core={row['seed_stable_core']} [{tag}, {row['n_informative_seeds']}/6 informative] resid_ba_mean={row['resid_ba_mean']}\n")
            fh.write("\n")
    logger.info(f"Summary -> {txt_path}")

    logger.info("=" * 70)
    logger.info("Temporal trajectory analysis complete.")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
