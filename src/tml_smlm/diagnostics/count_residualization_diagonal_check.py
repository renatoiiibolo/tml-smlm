#!/usr/bin/env python3
"""
Does the pipeline's existing count-residualization also happen to fix the
size confound `size_confound_audit.py` found -- or is it addressing a
different thing entirely?

`size_confound_audit.py` flags two features (`scaleC_h0_landscape_integral`,
`scaleC_h0_landscape_peak_t`) as still correlated with nucleus diagonal
(physical size) for one archive's Mre11/yH2AX strata, even after an earlier
degeneracy in those same features was fixed at the source. Every
classification result already residualizes against localization count
(`n_localisations`) for an unrelated reason -- so it's a fair question
whether that existing correction is quietly absorbing the size correlation
too, which would make the size-confound finding above moot. This module
answers that question directly rather than leaving it as an assumption.

WHAT THIS DOES NOT DO
----------------------------------------------------------------------
Does not revisit `size_confound_audit.py`'s own scope or conclusion -- that
screen tests raw feature values against diagonal, correctly, and that
result stands regardless of this module's answer. This is a narrower,
second check: for the two features that screen makes a specific
before/after claim about, does residualizing against count first change
the correlation with diagonal. Scoped to exactly those two features and
one archive's two marker strata, not a general audit -- the general audit
is `size_confound_audit.py` itself.

METHOD
------
For each (source, marker, feature) triple: join the feature matrix back to
the manifest to recover each nucleus's own diagonal
(`sqrt(size_x_nm^2 + size_y_nm^2)`); compute the raw Spearman correlation
between the feature and diagonal; residualize the feature against
localization count using the same fold-safe residualizer the classification
module uses (`FoldSafeResidualiser`, an identical mechanism, not a new one);
recompute the Spearman correlation between the residual and diagonal; and
report whether count-residualization increased or decreased that
correlation.

SCOPE NOTE
----------
The four (source, marker, feature) combinations checked here are fixed
to the specific strata and features `size_confound_audit.py` flagged in
the real archives this pipeline was built for. This module reads
`data/feature_matrix.csv` and `data/manifest.csv` directly and expects the
column names and marker identities those archives produce; it will not
find anything to check against the small public demo dataset bundled with
this repository (different, unnamed marker, no matching source label).

OUTPUTS
-------
  results/count_residualization_diagonal_check.json
  results/count_residualization_diagonal_check_summary.txt

USAGE
-----
  python -u count_residualization_diagonal_check.py --features data/feature_matrix.csv \\
      --manifest data/manifest.csv --results-dir results/ 2>&1 | tee count_residualization_diagonal_check.log

RUNTIME NOTE
------------
  Two features, two strata -- Spearman correlations on already-loaded
  columns. Expect a sub-second run.

DEPENDENCIES
------------
  numpy  pandas  scipy
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

from tml_smlm.view1_scalar.classification import FoldSafeResidualiser

# The two known-degeneracy features size_confound_audit.py makes a specific
# before/after claim about, for the one archive/marker pair where they're flagged.
TARGET_FEATURES: List[Dict] = [
    {"source": "hahn", "marker": "Mre11", "feature": "scaleC_h0_landscape_integral", "covariate": "n_localisations"},
    {"source": "hahn", "marker": "Mre11", "feature": "scaleC_h0_landscape_peak_t", "covariate": "n_localisations"},
    {"source": "hahn", "marker": "yH2AX", "feature": "scaleC_h0_landscape_integral", "covariate": "n_localisations"},
    {"source": "hahn", "marker": "yH2AX", "feature": "scaleC_h0_landscape_peak_t", "covariate": "n_localisations"},
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=Path("data/feature_matrix.csv"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    for p in [args.features, args.manifest]:
        if not p.exists():
            logger.error(f"Not found: {p}")
            return 1
    args.results_dir.mkdir(parents=True, exist_ok=True)

    fm = pd.read_csv(args.features, dtype={"replicate": str})
    mf = pd.read_csv(args.manifest, dtype={"replicate": str})
    KEY = ["source", "cell_type", "marker", "condition", "replicate"]

    logger.info("=" * 70)
    logger.info("count_residualization_diagonal_check")
    logger.info("=" * 70)

    results = []
    for t in TARGET_FEATURES:
        src, mk, feat, covar = t["source"], t["marker"], t["feature"], t["covariate"]
        sub = fm[(fm["source"] == src) & (fm["marker"] == mk)].copy()
        m = mf[(mf["source"] == src) & (mf["marker"] == mk)].copy()
        j = sub.merge(m[KEY + ["size_x_nm", "size_y_nm"]], on=KEY, how="left")
        j["diagonal_nm"] = np.sqrt(j["size_x_nm"] ** 2 + j["size_y_nm"] ** 2)
        d = j[[feat, covar, "diagonal_nm"]].dropna()

        rho_raw, p_raw = spearmanr(d[feat], d["diagonal_nm"])
        resid = FoldSafeResidualiser(count_cols=covar).fit(d).transform(d)[feat]
        rho_resid, p_resid = spearmanr(resid, d["diagonal_nm"])

        entry = {
            "source": src, "marker": mk, "feature": feat, "covariate": covar, "n": len(d),
            "raw_spearman_vs_diagonal": round(float(rho_raw), 4),
            "count_residualized_spearman_vs_diagonal": round(float(rho_resid), 4),
            "increased_after_residualization": bool(abs(rho_resid) > abs(rho_raw)),
        }
        results.append(entry)
        logger.info(f"  {src}/{mk}/{feat}: raw_rho={entry['raw_spearman_vs_diagonal']}  count_resid_rho={entry['count_residualized_spearman_vs_diagonal']}  increased={entry['increased_after_residualization']}")

    json_path = args.results_dir / "count_residualization_diagonal_check.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"\nJSON results -> {json_path}")

    txt_path = args.results_dir / "count_residualization_diagonal_check_summary.txt"
    with open(txt_path, "w") as fh:
        fh.write("count_residualization_diagonal_check — summary\n" + "=" * 70 + "\n\n")
        fh.write("Does count-residualization (already applied throughout the pipeline) also reduce\n")
        fh.write("the diagonal correlation size_confound_audit.py found, for the two known-degeneracy features specifically?\n\n")
        for e in results:
            fh.write(f"{e['source']}/{e['marker']}/{e['feature']}: raw={e['raw_spearman_vs_diagonal']} -> count-resid={e['count_residualized_spearman_vs_diagonal']}  (increased={e['increased_after_residualization']})\n")
    logger.info(f"Summary -> {txt_path}")

    logger.info("=" * 70)
    logger.info("count_residualization_diagonal_check complete.")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
