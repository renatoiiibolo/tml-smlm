#!/usr/bin/env python3
"""
Within-nucleus cross-marker coupling (View 3, Tier 5).

Every verified co-stained nucleus contributes two numbers: each marker's own
standardized Wasserstein displacement from its (source, cell type, marker)
control reference. "Coupling" here means a Spearman correlation between
those two per-marker displacements, computed across nuclei paired by the
dataset's own co-staining record rather than pooled across nuclei that
happen to share a condition. The pairing itself is verified before use (the
co-stain size-agreement flag and marker identity of the partner are both
checked, not assumed).

Three tiers are reported side by side rather than one replacing another:
raw pooled Spearman (descriptive, confounded by shared timepoint and count),
Spearman on residuals after regressing each marker's displacement on
timepoint alone, and Spearman on residuals after regressing on timepoint
plus that marker's own localization count. A correlation that survives the
third tier is not explained by shared elapsed time or shared physical
density/size; a correlation present at the second tier but not the third
means count was doing real explanatory work, which is itself informative
rather than a failed result. A per-timepoint stratified raw correlation is
reported as a secondary, descriptive check alongside the primary tiers.

Important scope limitation: a positive result here is a statistical
association between two markers' displacement-from-control within the same
physical nucleus. It does not by itself establish a biological mechanism,
a causal or temporal ordering between the two markers' damage-response
states, or that the association would hold outside the strata tested.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from tml_smlm.utils.wasserstein_geometry import wasserstein_distance_2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SCALE = "scaleC_h1"  # must match the population-geometry step's scale -- the Frechet references are only meaningful on the same array
MIN_PAIRS_FOR_CORR = 10
KEY_COLS = ["source", "cell_type", "marker", "condition", "replicate"]


# ══════════════════════════════════════════════════════════════════════════
# DIAGRAM LOADING
# ══════════════════════════════════════════════════════════════════════════

def _npz_stem(row) -> str:
    raw = f"{row['source']}__{row['cell_type']}__{row['marker']}__{row['condition']}__{row['replicate']}"
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", raw)


def load_diagram(diagrams_dir: Path, row, arr_key: str = SCALE) -> Optional[np.ndarray]:
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


# ══════════════════════════════════════════════════════════════════════════
# CONTROL REFERENCE — reused directly from the population-geometry step,
# required, not recomputed here
# ══════════════════════════════════════════════════════════════════════════

def load_control_references(population_geometry_json_path: Path) -> Dict[Tuple[str, str, str], Dict]:
    """
    (source, cell_type, marker) -> {frechet_mean, frechet_sigma}, read from
    the population-geometry step's baseline-return section. That step is
    the only place the (expensive) Frechet mean is computed; this step
    requires the field to already be present and does not fall back to
    recomputing it. A stratum missing it is a problem with the upstream
    run being pointed at -- surface it as an error rather than silently
    working around it.
    """
    refs: Dict[Tuple[str, str, str], Dict] = {}
    with open(population_geometry_json_path) as f:
        pop_geom = json.load(f)
    section_b = pop_geom.get("section_B_baseline_return", {})

    missing: List[str] = []
    for source, by_stratum in section_b.items():
        for key, r in by_stratum.items():
            if r.get("skipped"):
                continue
            cell_type, marker = key.split("/", 1)
            mean_field = r.get("control_frechet_mean")
            var_field = r.get("control_frechet_variance")
            if mean_field is None or var_field is None or var_field <= 0:
                missing.append(f"{source}/{key}")
                continue
            refs[(source, cell_type, marker)] = {
                "frechet_mean": np.asarray(mean_field, dtype=float),
                "frechet_sigma": float(np.sqrt(var_field)),
            }
    if missing:
        logger.error(
            f"Population-geometry JSON is missing control_frechet_mean for: {missing}. "
            "Rerun that step so it saves control_frechet_mean in Section B before running this step."
        )
    return refs


# ══════════════════════════════════════════════════════════════════════════
# PAIRED DATASET
# ══════════════════════════════════════════════════════════════════════════

def build_paired_dataset(
    manifest: pd.DataFrame, features: pd.DataFrame, diagrams_dir: Path,
    refs: Dict[Tuple[str, str, str], Dict], source: str, cell_type: str, marker_a: str, marker_b: str,
) -> pd.DataFrame:
    m = manifest[(manifest["source"] == source) & (manifest["cell_type"] == cell_type)].copy()
    cluster_to_key = {row["cluster_file"]: tuple(row[c] for c in KEY_COLS) for _, row in m.iterrows()}

    feat_lookup = {tuple(row[c] for c in KEY_COLS): row for _, row in features.iterrows()}

    rows = []
    n_no_partner = n_qc_disagree = n_missing_ref = n_missing_diagram = n_marker_mismatch = 0
    for _, a_row in m[m["marker"] == marker_a].iterrows():
        a_key = tuple(a_row[c] for c in KEY_COLS)
        partner_cluster = a_row.get("co_stain_partner")
        if not isinstance(partner_cluster, str) or partner_cluster not in cluster_to_key:
            n_no_partner += 1
            continue
        b_key = cluster_to_key[partner_cluster]
        if b_key[2] != marker_b:  # KEY_COLS[2] == "marker"
            n_marker_mismatch += 1
            continue
        size_check = a_row.get("co_stain_size_check")
        if size_check is not None and size_check != "agree":
            n_qc_disagree += 1
            continue

        a_feat = feat_lookup.get(a_key)
        b_feat = feat_lookup.get(b_key)
        if a_feat is None or b_feat is None:
            n_missing_diagram += 1
            continue

        ref_a = refs.get((source, cell_type, marker_a))
        ref_b = refs.get((source, cell_type, marker_b))
        if ref_a is None or ref_b is None:
            n_missing_ref += 1
            continue

        dgm_a = load_diagram(diagrams_dir, a_feat)
        dgm_b = load_diagram(diagrams_dir, b_feat)
        if dgm_a is None or dgm_b is None:
            n_missing_diagram += 1
            continue

        disp_a = float(wasserstein_distance_2(dgm_a, ref_a["frechet_mean"])) / ref_a["frechet_sigma"]
        disp_b = float(wasserstein_distance_2(dgm_b, ref_b["frechet_mean"])) / ref_b["frechet_sigma"]

        rows.append({
            "source": source, "cell_type": cell_type,
            "timepoint_h": a_feat["timepoint_h"], "timepoint_label": _tp_label(a_feat["timepoint_h"]),
            "is_control": bool(a_feat["is_control"]),
            f"{marker_a}_displacement": disp_a, f"{marker_b}_displacement": disp_b,
            f"{marker_a}_replicate": a_key[4], f"{marker_b}_replicate": b_key[4],
            f"{marker_a}_n_localisations": float(a_feat["n_localisations"]), f"{marker_b}_n_localisations": float(b_feat["n_localisations"]),
        })

    logger.info(
        f"  {source}/{cell_type} ({marker_a} vs {marker_b}): {len(rows)} pairs built. "
        f"Excluded: {n_no_partner} no co_stain_partner, {n_marker_mismatch} partner marker != {marker_b}, "
        f"{n_qc_disagree} co_stain_size_check != agree, {n_missing_ref} no control reference, {n_missing_diagram} diagram/feature-row load failure"
    )
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# RESIDUALIZATION + CORRELATION
# ══════════════════════════════════════════════════════════════════════════

def _round_p(p: float) -> float:
    """Round a p-value for JSON storage without losing small ones.

    A plain six-decimal rounding would store any p below 5e-7 as exactly 0.0,
    and the coupling p-values in the strongest strata are of order 1e-13 to
    1e-7. p >= 1e-3 is rounded to six decimals; p < 1e-3 keeps four
    significant figures."""
    p = float(p)
    if p >= 1e-3 or p == 0.0:
        return round(p, 6)
    return float(f"{p:.4g}")


def _residualise_against_timepoint(df: pd.DataFrame, col: str, count_col: Optional[str] = None) -> pd.Series:
    """
    OLS(col ~ C(timepoint_label) [+ count_col]) -- descriptive/correlational,
    not fold-safe cross-validation. count_col, when given, adds a continuous
    covariate (n_localisations) alongside the timepoint dummies, separating
    "coupled because both markers respond to overall damage load/nucleus
    size" from "coupled beyond what count and timepoint together explain."

    NOTE: this exact name is imported directly by the corresponding
    functional-association module in view2_functional/coupling.py --
    do not rename without updating that import.
    """
    cols = [col, "timepoint_label"] + ([count_col] if count_col else [])
    sub = df[cols].dropna()
    if len(sub) < 5:
        return pd.Series(float("nan"), index=df.index)
    dummies = pd.get_dummies(sub["timepoint_label"], drop_first=True)
    design_cols = [np.ones(len(sub))] + [dummies[c].to_numpy(dtype=float) for c in dummies.columns]
    if count_col:
        design_cols.append(sub[count_col].to_numpy(dtype=float))
    X = np.column_stack(design_cols)
    y = sub[col].to_numpy(dtype=float)
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return pd.Series(float("nan"), index=df.index)
    resid = pd.Series(float("nan"), index=df.index)
    resid.loc[sub.index] = y - (X @ beta)
    return resid


def primary_test(paired: pd.DataFrame, marker_a: str, marker_b: str) -> Dict:
    d = paired.copy()
    col_a, col_b = f"{marker_a}_displacement", f"{marker_b}_displacement"
    count_a, count_b = f"{marker_a}_n_localisations", f"{marker_b}_n_localisations"
    d["resid_a"] = _residualise_against_timepoint(d, col_a)
    d["resid_b"] = _residualise_against_timepoint(d, col_b)
    valid = d.dropna(subset=["resid_a", "resid_b"])

    result: Dict = {"n_pairs": int(len(valid)), "marker_a": marker_a, "marker_b": marker_b}
    if len(valid) < MIN_PAIRS_FOR_CORR:
        result["skipped"] = f"n={len(valid)} < {MIN_PAIRS_FOR_CORR}"
        return result

    rho, p = spearmanr(valid["resid_a"], valid["resid_b"])
    result["residual_spearman_rho"] = round(float(rho), 4)
    result["residual_spearman_p"] = _round_p(p)

    rho_raw, p_raw = spearmanr(d[col_a], d[col_b])
    result["raw_pooled_spearman_rho"] = round(float(rho_raw), 4)
    result["raw_pooled_spearman_p"] = _round_p(p_raw)
    result["raw_pooled_note"] = "Descriptive only -- confounded by shared timepoint response, see module docstring. residual_* (timepoint only) is the primary test; residual_with_count_* below adds the count covariate."

    # Third tier: residualize against timepoint AND each marker's own n_localisations.
    d["resid_a_count"] = _residualise_against_timepoint(d, col_a, count_col=count_a)
    d["resid_b_count"] = _residualise_against_timepoint(d, col_b, count_col=count_b)
    valid_count = d.dropna(subset=["resid_a_count", "resid_b_count"])
    if len(valid_count) >= MIN_PAIRS_FOR_CORR:
        rho_c, p_c = spearmanr(valid_count["resid_a_count"], valid_count["resid_b_count"])
        result["residual_with_count_spearman_rho"] = round(float(rho_c), 4)
        result["residual_with_count_spearman_p"] = _round_p(p_c)
        result["residual_with_count_n_pairs"] = int(len(valid_count))
        result["residual_with_count_note"] = (
            "Residualized against timepoint AND each marker's own n_localisations. A coupling that survives here "
            "is not explained by either shared timepoint response or shared physical count/size -- the strongest "
            "version of the within-nucleus coupling claim. A coupling present in residual_* but not here means "
            "count was doing real work explaining it, not that the coupling is absent."
        )
    else:
        result["residual_with_count_skipped"] = f"n={len(valid_count)} < {MIN_PAIRS_FOR_CORR}"
    return result


def secondary_stratified_test(paired: pd.DataFrame, marker_a: str, marker_b: str) -> Dict:
    col_a, col_b = f"{marker_a}_displacement", f"{marker_b}_displacement"
    results: Dict = {}
    for tp, grp in paired.groupby("timepoint_label"):
        if len(grp) < MIN_PAIRS_FOR_CORR:
            results[tp] = {"n": int(len(grp)), "skipped": f"n < {MIN_PAIRS_FOR_CORR}"}
            continue
        rho, p = spearmanr(grp[col_a], grp[col_b])
        results[tp] = {"n": int(len(grp)), "spearman_rho": round(float(rho), 4), "spearman_p": _round_p(p)}
    return results


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diagrams", type=Path, default=Path("diagrams"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--features", type=Path, default=Path("data/feature_matrix.csv"))
    ap.add_argument("--population-geometry-json", type=Path, default=Path("results/population_geometry.json"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    for p in [args.diagrams, args.manifest, args.features, args.population_geometry_json]:
        if not p.exists():
            logger.error(f"Not found: {p}")
            return 1
    args.results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("View 3, Tier 5 — within-nucleus cross-marker coupling")
    logger.info("=" * 70)

    manifest = pd.read_csv(args.manifest, dtype={"replicate": str})
    features = pd.read_csv(args.features, dtype={"replicate": str})
    logger.info(f"Loaded manifest ({len(manifest)} rows) and feature matrix ({len(features)} rows)")

    refs = load_control_references(args.population_geometry_json)
    if not refs:
        logger.error("No usable control references found -- cannot proceed. See the error above for which strata are missing control_frechet_mean.")
        return 1
    logger.info(f"Control references available for: {sorted(refs.keys())}")

    marker_pairs = {
        "kuentzelmann": ("53BP1", "yH2AX"),
        "hahn": ("Mre11", "yH2AX"),
    }

    all_results: Dict = {}
    all_paired_rows: List[pd.DataFrame] = []
    for source, (marker_a, marker_b) in marker_pairs.items():
        if source not in manifest["source"].unique():
            continue
        all_results[source] = {}
        for cell_type in sorted(manifest[manifest["source"] == source]["cell_type"].unique()):
            logger.info(f"\n[{source}/{cell_type}] {marker_a} vs {marker_b}")
            paired = build_paired_dataset(manifest, features, args.diagrams, refs, source, cell_type, marker_a, marker_b)
            if len(paired) == 0:
                all_results[source][cell_type] = {"skipped": "no pairs built"}
                continue
            all_paired_rows.append(paired)

            primary = primary_test(paired, marker_a, marker_b)
            secondary = secondary_stratified_test(paired, marker_a, marker_b)
            if not primary.get("skipped"):
                logger.info(
                    f"    Primary (timepoint-resid): n={primary['n_pairs']} rho={primary['residual_spearman_rho']} p={primary['residual_spearman_p']}  "
                    f"(raw pooled rho={primary['raw_pooled_spearman_rho']}, confounded)"
                )
                if "residual_with_count_spearman_rho" in primary:
                    logger.info(
                        f"    Primary (timepoint+count-resid): n={primary['residual_with_count_n_pairs']} "
                        f"rho={primary['residual_with_count_spearman_rho']} p={primary['residual_with_count_spearman_p']}"
                    )
            for tp, r in secondary.items():
                if r.get("skipped"):
                    logger.info(f"    {tp}: skipped ({r['skipped']})")
                else:
                    logger.info(f"    {tp}: n={r['n']} rho={r['spearman_rho']} p={r['spearman_p']}")

            all_results[source][cell_type] = {
                "marker_a": marker_a, "marker_b": marker_b, "n_pairs": len(paired),
                "primary_residualized_test": primary, "secondary_stratified_test": secondary,
            }

    if all_paired_rows:
        paired_path = args.results_dir / "view3_paired_dataset.csv"
        pd.concat(all_paired_rows, ignore_index=True).to_csv(paired_path, index=False)
        logger.info(f"\nPaired dataset -> {paired_path}")

    json_path = args.results_dir / "view3_cross_marker_coupling.json"
    with open(json_path, "w") as fh:
        json.dump({
            "results": all_results,
            "interpretive_framing": (
                "A weak/null residual correlation is consistent with real within-nucleus decoupling between "
                "the two markers' damage-response states, rather than noise around a shared population trend. "
                "A strong positive correlation is a statistical association between the two markers' "
                "displacement-from-control within the same nucleus; it does not by itself establish a shared "
                "biological mechanism or a causal/temporal order between the markers, and should be reported "
                "as an association, not over-interpreted as mechanistic evidence."
            ),
        }, fh, indent=2, default=str)
    logger.info(f"JSON results -> {json_path}")

    txt_path = args.results_dir / "view3_coupling_summary.txt"
    with open(txt_path, "w") as fh:
        fh.write("View 3, Tier 5 — within-nucleus cross-marker coupling — summary\n" + "=" * 70 + "\n\n")
        for source, by_ct in all_results.items():
            for ct, r in by_ct.items():
                if r.get("skipped"):
                    fh.write(f"{source}/{ct}: SKIPPED ({r['skipped']})\n")
                    continue
                p = r["primary_residualized_test"]
                fh.write(f"{source}/{ct} ({r['marker_a']} vs {r['marker_b']}), n={r['n_pairs']}:\n")
                if p.get("skipped"):
                    fh.write(f"  Primary: skipped ({p['skipped']})\n")
                else:
                    fh.write(f"  Primary (timepoint-resid): rho={p['residual_spearman_rho']} p={p['residual_spearman_p']}\n")
                    if "residual_with_count_spearman_rho" in p:
                        fh.write(f"  Primary (timepoint+count-resid): rho={p['residual_with_count_spearman_rho']} p={p['residual_with_count_spearman_p']}\n")
                    fh.write(f"  Raw pooled (confounded, descriptive only): rho={p['raw_pooled_spearman_rho']} p={p['raw_pooled_spearman_p']}\n")
                for tp, sr in r["secondary_stratified_test"].items():
                    if sr.get("skipped"):
                        fh.write(f"  {tp}: skipped ({sr['skipped']})\n")
                    else:
                        fh.write(f"  {tp}: n={sr['n']} rho={sr['spearman_rho']} p={sr['spearman_p']}\n")
                fh.write("\n")
    logger.info(f"Summary -> {txt_path}")

    logger.info("=" * 70)
    logger.info("Done.")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
