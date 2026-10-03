#!/usr/bin/env python3
"""
Within-stratum Holm step-down correction for the baseline-return
persistence test (View 3, Tier 4a).

The population-geometry module's `section_B_baseline_return` output
reports, for each (source, cell_type, marker) stratum, one permutation
p-value per irradiated timepoint testing whether that timepoint's
Fréchet mean still differs from its own matched control. Those
per-timepoint tests are each individually valid, but a stratum with
several timepoints is running several tests against the same null, and
the unadjusted p-values don't account for that. Holm's (1979) step-down
procedure is applied here, after the fact, to correct for it.

This is a Holm correction on an already-computed, already-reported set
of p-values -- it needs nothing from the population-geometry module's
own machinery (the pairwise Wasserstein distance matrices, the Fréchet
means, the permutation tests themselves), only the p-values it already
wrote to disk. Holm's procedure requires no assumption about how the
p-values were produced, so doing this as a separate pass over an
existing JSON result, rather than folding it into the original
computation, changes nothing about the correction's validity and keeps
the audited object -- the originally reported significance calls --
untouched.

The family being corrected is each stratum's own set of per-timepoint
tests against its own control, not every test in the pipeline pooled
together. A stratum's timepoints are naturally one family: they share a
control, a cell type, and a marker, and answer one question ("has this
population returned to baseline, and when"). Pooling across strata would
correct for a comparison that was never actually being made.

Applies Holm correction separately to the raw and the
count-residualized p-value sets already present in each timepoint's
test record, so the corrected table sits alongside both, rather than
picking one.

Output
------
  results/multiple_comparisons_holm_t4a.json
  results/multiple_comparisons_summary.txt

Usage
-----
  python -m tml_smlm.view3_metric.multiple_comparisons \\
      --population-geometry-json results/population_geometry_results.json \\
      --results-dir results/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def holm(pvals: Dict[str, float]) -> Dict[str, Dict]:
    """Holm (1979) step-down procedure. Returns raw p, Holm-adjusted p, and significance at alpha=0.05 per test."""
    items: List[Tuple[str, float]] = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, Dict] = {}
    running_max = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        running_max = max(running_max, adj)
        out[name] = {"raw_p": round(p, 6), "holm_adjusted_p": round(running_max, 6), "significant_at_0.05": bool(running_max < 0.05)}
    return out


def process_stratum(source: str, stratum: str, tests: Dict) -> Dict:
    raw_p = {tp: v["raw"]["p_value_floor_corrected"] for tp, v in tests.items()}
    resid_p = {tp: v["residualized_against_n_localisations"]["p_value_floor_corrected"] for tp, v in tests.items()}
    raw_holm = holm(raw_p)
    resid_holm = holm(resid_p)

    surviving_raw = sorted([tp for tp, v in raw_holm.items() if v["significant_at_0.05"]])
    surviving_resid = sorted([tp for tp, v in resid_holm.items() if v["significant_at_0.05"]])

    logger.info(f"  {source}/{stratum}: {len(tests)} timepoints -- raw survives {len(surviving_raw)}: {surviving_raw}  |  residualized survives {len(surviving_resid)}: {surviving_resid}")

    return {
        "source": source, "stratum": stratum, "n_timepoints": len(tests),
        "raw_holm": raw_holm, "residualized_holm": resid_holm,
        "timepoints_surviving_raw": surviving_raw,
        "timepoints_surviving_residualized": surviving_resid,
        "_interpretation": (
            "Family = this stratum's own set of per-timepoint tests against its own control. "
            "timepoints_surviving_residualized is the number to cite -- residualization against count "
            "and multiple-comparisons correction address two different concerns and both have now been "
            "applied; timepoints_surviving_raw is reported for completeness, not as an alternative "
            "headline figure."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--population-geometry-json", type=Path, default=Path("results/population_geometry_results.json"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    if not args.population_geometry_json.exists():
        logger.error(f"Not found: {args.population_geometry_json}")
        return 1
    args.results_dir.mkdir(parents=True, exist_ok=True)

    with open(args.population_geometry_json) as f:
        pop_geom = json.load(f)

    logger.info("=" * 70)
    logger.info("Holm correction, T4a baseline-return, within-stratum")
    logger.info("=" * 70)

    b = pop_geom.get("section_B_baseline_return")
    if not b:
        logger.error("No section_B_baseline_return in this JSON -- wrong file, or an older schema")
        return 1

    results: Dict = {}
    for source, by_stratum in b.items():
        results[source] = {}
        for stratum, r in by_stratum.items():
            tests = r.get("tests_vs_control")
            if not tests:
                logger.warning(f"  {source}/{stratum}: no tests_vs_control -- skipping")
                continue
            results[source][stratum] = process_stratum(source, stratum, tests)

    json_path = args.results_dir / "multiple_comparisons_holm_t4a.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"\nJSON results -> {json_path}")

    txt_path = args.results_dir / "multiple_comparisons_summary.txt"
    with open(txt_path, "w") as fh:
        fh.write("Holm-corrected T4a (baseline-return) results — summary\n" + "=" * 70 + "\n\n")
        fh.write("Within-stratum Holm-corrected T4a (baseline-return) results, residualized column is the one to cite:\n\n")
        for source, by_stratum in results.items():
            for stratum, r in by_stratum.items():
                fh.write(f"{source}/{stratum}: {r['n_timepoints']} timepoints tested\n")
                fh.write(f"  survives raw+Holm: {r['timepoints_surviving_raw']}\n")
                fh.write(f"  survives residualized+Holm: {r['timepoints_surviving_residualized']}\n\n")
    logger.info(f"Summary -> {txt_path}")

    logger.info("=" * 70)
    logger.info("Done.")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
