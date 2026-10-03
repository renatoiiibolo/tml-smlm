#!/usr/bin/env python3
"""
View 2 (functional), Tier 5 — position-specific within-nucleus cross-marker
coupling.

View 3's coupling test asks a holistic question: within a verified
co-stained nucleus, do two markers' overall Wasserstein displacement from
their own control reference move together? That's a single number per
marker summarizing an entire diagram's difference from baseline. This
module asks a narrower, complementary question: at the SPECIFIC filtration
position the robustness module confirmed is stable for a given
marker/scale, do the two co-stained markers' persistence-landscape VALUES
at that position move together within one nucleus? The two tests can
disagree — a nucleus could show typical overall displacement while its
value at one specific stable position is unusually high or low, which the
holistic Wasserstein metric cannot see, and vice versa. Put differently:
this asks whether the SPATIAL SCALE of reorganization is coupled between
markers, not just the aggregate distance from control.

Why a shared grid is needed, and why this isn't a re-run of localization
or robustness
--------------------------------------------------------------------------
The robustness module confirmed each panel's stable core index on that
panel's own, marker-specific grid (e.g. the hahn/Mre11/scaleC_h0 grid was
built only from Mre11 diagrams). Correlating "the value at index 299"
between Mre11 and yH2AX only refers to the same physical birth-radius for
both markers if the SAME grid is used for both — the localization and
robustness modules' own per-marker grid construction does not guarantee
that. This module pools both markers' diagrams for a panel's cell
type/scale, builds ONE shared grid from that pool (reusing localization's
own `_build_grid`), and vectorizes each marker separately on that shared
grid (reusing `_clean`/`_landscape` directly). Neither the localization
nor the robustness module is modified or rerun; their primitives are
reused at a finer grain than robustness itself needed.

Which position
--------------------------------------------------------------------------
The robustness module's POOLED-population stable core is used here, not
a per-timepoint trajectory — this is a population-level coupling question,
the same level View 3's test operates at, not a temporal one. The panel's
own per-nucleus value is the mean landscape value across its full
confirmed core, not an arbitrarily chosen single index from a multi-index
result.

Pairing and residualization — reused from View 3, not rederived
--------------------------------------------------------------------------
Same `co_stain_partner`-based pairing View 3 uses (the manifest's own
validated column, not re-inferred), and the same three-tier
residualization (raw pooled, timepoint, timepoint + each marker's own
count covariate). `_residualise_against_timepoint` is imported directly
from `tml_smlm.view3_metric.coupling` rather than reimplemented here —
deliberately: the same statistical logic should behave identically
regardless of which View's coupling test calls it, so this module treats
View 3's helper as the single source of truth for that computation
instead of maintaining a second copy that could drift.

Scope: the co-stained partner of each of the robustness module's three
headline panels
--------------------------------------------------------------------------
hahn/Mre11/scaleC_h0 pairs with hahn/yH2AX (same scale/homology,
scaleC_h0); kuentzelmann/yH2AX/scaleC_h1 pairs with kuentzelmann/53BP1
(same scale/homology, scaleC_h1); hahn/yH2AX/noise_h1 pairs with
hahn/Mre11 (same scale/homology, noise_h1). Not extended to every possible
cross-marker combination — only the pairing that shares the panel's own
confirmed stable position.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

import tml_smlm.view2_functional.localization as localization
import tml_smlm.view2_functional.robustness as robustness
from tml_smlm.view3_metric.coupling import _residualise_against_timepoint

MIN_PAIRS_FOR_CORR = 10
KEY_COLS = ["source", "cell_type", "marker", "condition", "replicate"]

# The robustness module's own confirmed pooled stable cores -- not re-derived here.
PANEL_COUPLING = [
    {"source": "hahn", "marker_a": "Mre11", "marker_b": "yH2AX", "scale": "scaleC", "hom": "h0", "key": "scaleC_h0", "k": 1, "core_a": [299]},
    {"source": "kuentzelmann", "marker_a": "yH2AX", "marker_b": "53BP1", "scale": "scaleC", "hom": "h1", "key": "scaleC_h1", "k": 1, "core_a": [218]},
    {"source": "hahn", "marker_a": "yH2AX", "marker_b": "Mre11", "scale": "noise", "hom": "h1", "key": "noise_h1", "k": 2, "core_a": [101, 140, 157, 299]},
]


def _npz_stem(row) -> str:
    raw = f"{row['source']}__{row['cell_type']}__{row['marker']}__{row['condition']}__{row['replicate']}"
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", raw)


def _load_dgm(diagrams_dir: Path, row, arr_key: str) -> Optional[np.ndarray]:
    npz_file = row.get("npz_file", _npz_stem(row) + ".npz") if hasattr(row, "get") else _npz_stem(row) + ".npz"
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
    if pd.isna(tp) or tp == "":
        return "control"
    v = float(tp)
    return f"{int(v)}h" if v == int(v) else f"{v}h"


def _vectorise_on_grid(diagrams: List[np.ndarray], grid: np.ndarray, k: int) -> np.ndarray:
    """Same math as localization's vectorise, but against a PROVIDED (shared) grid instead of building a new one per call."""
    cleaned = [localization._clean(d) for d in diagrams]
    return np.vstack([localization._landscape(d, grid, k) for d in cleaned])


def build_marker_landscape(fm: pd.DataFrame, diagrams_dir: Path, source: str, marker: str, arr_key: str, k: int, grid: np.ndarray, covar_col: str):
    sub = fm[(fm["source"] == source) & (fm["marker"] == marker)].reset_index(drop=True)
    dgms = [_load_dgm(diagrams_dir, row, arr_key) for _, row in sub.iterrows()]
    keep = [i for i, d in enumerate(dgms) if d is not None]
    sub = sub.iloc[keep].reset_index(drop=True)
    dgms = [dgms[i] for i in keep]
    L = _vectorise_on_grid(dgms, grid, k)
    covar = sub[covar_col].to_numpy(dtype=float)
    L_r = localization.residualise(L, covar)
    return sub, L_r


def process_panel(fm: pd.DataFrame, manifest: pd.DataFrame, diagrams_dir: Path, panel: Dict) -> Optional[Dict]:
    source, marker_a, marker_b = panel["source"], panel["marker_a"], panel["marker_b"]
    scale, arr_key, k, core_a = panel["scale"], panel["key"], panel["k"], panel["core_a"]
    lbl = f"{source}/{marker_a}-{marker_b}/{arr_key}/lambda{k}"
    logger.info(f"\n{'='*60}\n{lbl}\n{'='*60}")

    both = fm[(fm["source"] == source) & (fm["marker"].isin([marker_a, marker_b]))].reset_index(drop=True)
    dgms_pool = [d for d in (_load_dgm(diagrams_dir, row, arr_key) for _, row in both.iterrows()) if d is not None]
    if len(dgms_pool) < 20:
        return {"label": lbl, "skipped": f"pooled diagram count {len(dgms_pool)} < 20"}
    grid = localization._build_grid([localization._clean(d) for d in dgms_pool])
    covar_col = localization.SCALE_COVAR.get(scale, "n_localisations")

    sub_a, L_r_a = build_marker_landscape(fm, diagrams_dir, source, marker_a, arr_key, k, grid, covar_col)
    sub_b, L_r_b = build_marker_landscape(fm, diagrams_dir, source, marker_b, arr_key, k, grid, covar_col)
    val_a = L_r_a[:, core_a].mean(axis=1)
    # marker_b's value is extracted at the SAME shared-grid index positions -- the whole point of the shared grid.
    val_b = L_r_b[:, core_a].mean(axis=1)
    sub_a = sub_a.copy(); sub_a["_value"] = val_a
    sub_b = sub_b.copy(); sub_b["_value"] = val_b

    m = manifest[(manifest["source"] == source) & (manifest["marker"].isin([marker_a, marker_b]))].copy()
    cluster_to_key = {row["cluster_file"]: tuple(row[c] for c in KEY_COLS) for _, row in m.iterrows()}
    a_lookup = {tuple(row[c] for c in KEY_COLS): row for _, row in sub_a.iterrows()}
    b_lookup = {tuple(row[c] for c in KEY_COLS): row for _, row in sub_b.iterrows()}

    rows = []
    n_no_partner = n_marker_mismatch = n_qc_disagree = n_missing = 0
    for _, a_row in m[m["marker"] == marker_a].iterrows():
        a_key = tuple(a_row[c] for c in KEY_COLS)
        partner_cluster = a_row.get("co_stain_partner")
        if not isinstance(partner_cluster, str) or partner_cluster not in cluster_to_key:
            n_no_partner += 1
            continue
        b_key = cluster_to_key[partner_cluster]
        if b_key[2] != marker_b:
            n_marker_mismatch += 1
            continue
        if a_row.get("co_stain_size_check") not in (None, "agree"):
            n_qc_disagree += 1
            continue
        a_feat = a_lookup.get(a_key); b_feat = b_lookup.get(b_key)
        if a_feat is None or b_feat is None:
            n_missing += 1
            continue
        rows.append({
            "source": source, "cell_type": a_key[1],
            "timepoint_h": a_feat["timepoint_h"], "timepoint_label": _tp_label(a_feat["timepoint_h"]),
            "is_control": bool(a_feat["is_control"]),
            f"{marker_a}_value": a_feat["_value"], f"{marker_b}_value": b_feat["_value"],
            f"{marker_a}_n_localisations": float(a_feat[covar_col]), f"{marker_b}_n_localisations": float(b_feat[covar_col]),
        })
    logger.info(f"  {len(rows)} pairs built. Excluded: {n_no_partner} no partner, {n_marker_mismatch} marker mismatch, {n_qc_disagree} QC disagree, {n_missing} feature lookup failure")

    paired = pd.DataFrame(rows)
    if len(paired) < MIN_PAIRS_FOR_CORR:
        return {"label": lbl, "skipped": f"n_pairs={len(paired)} < {MIN_PAIRS_FOR_CORR}"}

    col_a, col_b = f"{marker_a}_value", f"{marker_b}_value"
    count_a, count_b = f"{marker_a}_n_localisations", f"{marker_b}_n_localisations"

    if paired[col_a].std() == 0 or paired[col_b].std() == 0:
        reason = f"zero variance in {marker_a if paired[col_a].std() == 0 else marker_b}'s landscape value at the confirmed core position(s) -- nothing to correlate"
        logger.info(f"  SKIPPED: {reason}")
        return {"label": lbl, "skipped": reason}

    rho_raw, p_raw = spearmanr(paired[col_a], paired[col_b])
    resid_a = _residualise_against_timepoint(paired, col_a)
    resid_b = _residualise_against_timepoint(paired, col_b)
    valid_t = paired.assign(resid_a=resid_a, resid_b=resid_b).dropna(subset=["resid_a", "resid_b"])
    rho_t, p_t = (spearmanr(valid_t["resid_a"], valid_t["resid_b"]) if len(valid_t) >= MIN_PAIRS_FOR_CORR else (None, None))

    resid_a_c = _residualise_against_timepoint(paired, col_a, count_col=count_a)
    resid_b_c = _residualise_against_timepoint(paired, col_b, count_col=count_b)
    valid_c = paired.assign(resid_a=resid_a_c, resid_b=resid_b_c).dropna(subset=["resid_a", "resid_b"])
    rho_c, p_c = (spearmanr(valid_c["resid_a"], valid_c["resid_b"]) if len(valid_c) >= MIN_PAIRS_FOR_CORR else (None, None))

    logger.info(f"  raw rho={rho_raw:.4f} p={p_raw:.4g}  |  timepoint-resid rho={rho_t} p={p_t}  |  +count-resid rho={rho_c} p={p_c}")

    return {
        "label": lbl, "source": source, "marker_a": marker_a, "marker_b": marker_b,
        "scale": scale, "core_indices_shared_grid": core_a, "n_pairs": len(paired),
        "grid_min_nm": round(float(grid[0]), 2), "grid_max_nm": round(float(grid[-1]), 2),
        "raw_pooled_spearman_rho": round(float(rho_raw), 4), "raw_pooled_spearman_p": round(float(p_raw), 6),
        "timepoint_resid_spearman_rho": round(float(rho_t), 4) if rho_t is not None else None,
        "timepoint_resid_spearman_p": round(float(p_t), 6) if p_t is not None else None,
        "timepoint_count_resid_spearman_rho": round(float(rho_c), 4) if rho_c is not None else None,
        "timepoint_count_resid_spearman_p": round(float(p_c), 6) if p_c is not None else None,
        "_interpretation": (
            "This asks whether the two markers' LANDSCAPE VALUES at the panel's own confirmed stable "
            "position(s) move together within one nucleus -- a position-specific, complementary question to "
            "View 3's holistic Wasserstein-displacement coupling test, not a re-derivation of it. raw is confounded "
            "by shared timepoint response; timepoint_resid is the primary test; timepoint_count_resid is the "
            "strongest version, controlling for both timepoint and each marker's own count."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diagrams", type=Path, default=Path("diagrams"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--features", type=Path, default=Path("data/feature_matrix.csv"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    for p in [args.diagrams, args.manifest, args.features]:
        if not p.exists():
            logger.error(f"Not found: {p}")
            return 1
    args.results_dir.mkdir(parents=True, exist_ok=True)

    fm = pd.read_csv(args.features, dtype={"replicate": str})
    manifest = pd.read_csv(args.manifest, dtype={"replicate": str})
    logger.info("=" * 70)
    logger.info("View 2, Tier 5 — position-specific within-nucleus cross-marker coupling")
    logger.info("=" * 70)
    logger.info(f"Loaded features ({len(fm)} rows), manifest ({len(manifest)} rows)")

    results = []
    for panel in PANEL_COUPLING:
        r = process_panel(fm, manifest, args.diagrams, panel)
        if r is not None:
            results.append(r)

    json_path = args.results_dir / "view2_position_coupling.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"\nResults -> {json_path}")

    txt_path = args.results_dir / "view2_position_coupling_summary.txt"
    with open(txt_path, "w") as fh:
        fh.write("View 2, Tier 5 — position-specific within-nucleus cross-marker coupling — summary\n" + "=" * 70 + "\n\n")
        for r in results:
            if r.get("skipped"):
                fh.write(f"{r['label']}: SKIPPED ({r['skipped']})\n\n")
                continue
            fh.write(f"{r['label']}: n_pairs={r['n_pairs']} core={r['core_indices_shared_grid']}\n")
            fh.write(f"  raw pooled: rho={r['raw_pooled_spearman_rho']} p={r['raw_pooled_spearman_p']}\n")
            fh.write(f"  timepoint-resid: rho={r['timepoint_resid_spearman_rho']} p={r['timepoint_resid_spearman_p']}\n")
            fh.write(f"  timepoint+count-resid: rho={r['timepoint_count_resid_spearman_rho']} p={r['timepoint_count_resid_spearman_p']}\n\n")
    logger.info(f"Summary -> {txt_path}")

    logger.info("=" * 70)
    logger.info("View 2, Tier 5 complete.")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
