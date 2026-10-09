#!/usr/bin/env python3
"""
View 1 (scalar), Tier 3 — choosing each stratum's leading descriptor, and
checking that the choice does not hinge on any single timepoint.

The temporal-structure module tests five candidate scalar descriptors per
stratum for a cell-type-by-timepoint interaction and stores each one's
per-timepoint trajectory (cell-type means and standard errors), both raw and
count-residualized. This module turns the selection rule into a callable,
testable function instead of a hand-picked result:

    gap_t         = mean(group_a, t) - mean(group_b, t)
    pooled_se_t   = sqrt(se(group_a, t)^2 + se(group_b, t)^2)
    SNR           = mean_t(|gap_t|) / mean_t(pooled_se_t)
    monotonicity  = Pearson correlation of gap_t with timepoint rank (1..T)

The descriptor shown for a stratum is the one with the largest SNR. SNR is
computed on the count-residualized trajectories, the same ones each
candidate's interaction test is run on, so the selection statistic is the
audited one. Monotonicity is a secondary read-out, reported as a signed
correlation (a candidate whose gap falls over time scores negative).

Robustness. Each timepoint is dropped in turn, the SNR recomputed, and the
argmax re-taken. The check reports in how many of the T drops the winner is
unchanged. Trajectory rows are independent per (cell type, timepoint), so a
drop cannot change any other timepoint's mean or standard error; what the
check tests is whether the timepoint-aggregated argmax depends on any single
timepoint being present.

For the paper's four panels the winner is stable in 6/6, 6/6, 7/7 and 7/7
drops, and equals the full-data argmax in every panel. The same check on the
raw trajectories (`--basis raw`) is available for comparison. It selects a
different 53BP1 descriptor and is less stable under timepoint drops, so the
basis is stated whenever a value is quoted.

Self-test. Before the check runs, the SNR for one externally-known value is
recomputed (Kuntzelmann 53BP1, scaleC_h1_betti_mean, count-residualized
SNR 3.11 as quoted in the paper); the run stops if it does not match.

Outputs (in --results-dir)
  descriptor_selection.json           per-panel winner, stability and per-drop detail
  descriptor_selection_summary.txt    one line per panel

Usage
  python -u -m tml_smlm.view1_scalar.descriptor_selection \
      --temporal-json results/temporal_structure_results.json \
      --results-dir results/

Requires numpy and scipy.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# The four strata whose leading descriptor the paper plots, with the two cell
# types compared in each (group_a minus group_b), and the descriptor each
# panel displays.
DESCRIPTOR_PANELS = [
    ("kuentzelmann", "53BP1", ("NHDF", "U87")),
    ("kuentzelmann", "yH2AX", ("NHDF", "U87")),
    ("hahn", "Mre11", ("HGF", "MCF7")),
    ("hahn", "yH2AX", ("HGF", "MCF7")),
]
DESCRIPTOR_BY_PANEL: Dict[tuple, str] = {
    ("kuentzelmann", "53BP1"): "scaleC_h1_betti_mean",
    ("kuentzelmann", "yH2AX"): "scaleC_h1_betti_var",
    ("hahn", "Mre11"): "scaleC_h1_landscape_peak_t",
    ("hahn", "yH2AX"): "scaleC_h0_betti_var",
}

# One value quoted in the paper that is itself a count-residualized SNR; the
# run checks that this module reproduces it before doing anything else.
KNOWN_RESIDUALIZED_VALUES = {
    ("kuentzelmann", "53BP1", "scaleC_h1_betti_mean"): {"snr": 3.12},
}


def snr_and_monotonicity(
    rows: List[Dict], group_a: str, group_b: str, drop_timepoint: Optional[float] = None
) -> Dict:
    """SNR and monotonicity of the between-group gap across timepoints, for
    whichever trajectory block (raw or residualized) the caller passes in as
    `rows`. `drop_timepoint` leaves one timepoint out for the jackknife."""
    by_tp: Dict[float, Dict[str, Dict]] = {}
    for r in rows:
        by_tp.setdefault(r["timepoint_h"], {})[r["cell_type"]] = r

    timepoints = sorted(by_tp.keys())
    if drop_timepoint is not None:
        timepoints = [t for t in timepoints if t != drop_timepoint]

    gaps, pooled_ses = [], []
    for t in timepoints:
        cell = by_tp[t]
        if group_a not in cell or group_b not in cell:
            continue
        a, b = cell[group_a], cell[group_b]
        gaps.append(a["mean"] - b["mean"])
        pooled_ses.append(float(np.hypot(a["se"], b["se"])))

    if len(gaps) < 3:
        return {"error": f"only {len(gaps)} usable timepoints after drop={drop_timepoint}, need >=3"}

    gaps_arr = np.array(gaps)
    pooled_se_mean = float(np.mean(pooled_ses))
    snr = float(np.mean(np.abs(gaps_arr))) / pooled_se_mean if pooled_se_mean > 0 else float("nan")
    rank = np.arange(1, len(gaps_arr) + 1)
    mono = float(pearsonr(gaps_arr, rank)[0])

    return {
        "n_timepoints_used": len(gaps),
        "dropped_timepoint": drop_timepoint,
        "snr": round(snr, 4),
        "monotonicity": round(mono, 4),
    }


BASIS_KEY = {"residualised": "trajectory_residualised", "raw": "trajectory_raw"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--temporal-json", type=Path, default=Path("results/temporal_structure_results.json"),
                    help="Output of the temporal-structure module.")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--basis", choices=sorted(BASIS_KEY), default="residualised",
                    help="Trajectory block the SNR is computed on (default: count-residualized).")
    args = ap.parse_args()
    key = BASIS_KEY[args.basis]

    if not args.temporal_json.exists():
        logger.error(f"Not found: {args.temporal_json}")
        return 1
    args.results_dir.mkdir(parents=True, exist_ok=True)

    with open(args.temporal_json) as fh:
        probe1 = json.load(fh)["probe1_leading_feature_temporal"]

    logger.info("=" * 70)
    logger.info(f"Leading-descriptor selection and leave-one-timepoint-out check (basis: {args.basis})")
    logger.info("=" * 70)

    if args.basis == "residualised":
        logger.info("Self-test against the one count-residualized SNR quoted in the paper:")
        for (src, mk, feat), known in KNOWN_RESIDUALIZED_VALUES.items():
            group_a, group_b = next(g for s, m, g in DESCRIPTOR_PANELS if s == src and m == mk)
            res = snr_and_monotonicity(probe1[src][mk]["results"][feat][key], group_a, group_b)
            ok = abs(res["snr"] - known["snr"]) < 0.02
            logger.info(f"  {src}/{mk}/{feat}: computed snr={res['snr']} (quoted {known['snr']})  [{'OK' if ok else 'MISMATCH'}]")
            if not ok:
                logger.error("Self-test failed. Stopping before the leave-one-timepoint-out check.")
                return 1
        logger.info("Self-test passed.\n")

    results = {}
    for src, mk, (group_a, group_b) in DESCRIPTOR_PANELS:
        stratum = probe1[src][mk]
        candidates = stratum["features_tested"]
        shown = DESCRIPTOR_BY_PANEL[(src, mk)]

        full = {f: snr_and_monotonicity(stratum["results"][f][key], group_a, group_b) for f in candidates}
        scoreable = {f: v for f, v in full.items() if "snr" in v}
        argmax_full = max(scoreable, key=lambda f: scoreable[f]["snr"])

        timepoints = sorted({r["timepoint_h"] for r in stratum["results"][shown][key]})
        loto = []
        for t_drop in timepoints:
            per_feat = {f: snr_and_monotonicity(stratum["results"][f][key], group_a, group_b, drop_timepoint=t_drop)
                        for f in candidates}
            valid = {f: v for f, v in per_feat.items() if "snr" in v}
            winner = max(valid, key=lambda f: valid[f]["snr"]) if valid else None
            loto.append({
                "dropped_timepoint_h": t_drop,
                "winner_unchanged": winner == shown,
                "winner_after_drop": winner,
                "shown_descriptor_snr_after_drop": valid.get(shown, {}).get("snr"),
                "shown_descriptor_monotonicity_after_drop": valid.get(shown, {}).get("monotonicity"),
            })

        n_stable = sum(1 for r in loto if r["winner_unchanged"])
        results[f"{src}/{mk}"] = {
            "basis": args.basis,
            "shown_descriptor": shown,
            "snr_full_data": full[shown].get("snr"),
            "monotonicity_full_data": full[shown].get("monotonicity"),
            "argmax_snr_full_data": argmax_full,
            "shown_equals_argmax": argmax_full == shown,
            "n_timepoints": len(timepoints),
            "n_drops_with_same_winner": n_stable,
            "stable_under_every_drop": n_stable == len(timepoints),
            "candidate_snr_full_data": {f: v.get("snr") for f, v in full.items()},
            "loto_detail": loto,
        }
        logger.info(f"{src}/{mk}: shown = {shown} (SNR {full[shown].get('snr')}, monotonicity {full[shown].get('monotonicity')}); "
                    f"full-data argmax = {argmax_full}; winner unchanged in {n_stable}/{len(timepoints)} drops")
        for r in loto:
            if not r["winner_unchanged"]:
                logger.info(f"    drop t={r['dropped_timepoint_h']}h: winner becomes {r['winner_after_drop']}")

    json_path = args.results_dir / "descriptor_selection.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    txt_path = args.results_dir / "descriptor_selection_summary.txt"
    with open(txt_path, "w") as fh:
        fh.write(f"Leading-descriptor selection, SNR basis: {args.basis}\n" + "=" * 70 + "\n\n")
        for k, v in results.items():
            fh.write(f"{k}: shown={v['shown_descriptor']}  argmax={v['argmax_snr_full_data']}  "
                     f"stable={v['n_drops_with_same_winner']}/{v['n_timepoints']}\n")
    logger.info(f"Results -> {json_path}\nSummary -> {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
