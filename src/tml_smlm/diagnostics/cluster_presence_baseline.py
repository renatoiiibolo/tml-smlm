#!/usr/bin/env python3
"""
Cluster-presence baseline: does a nucleus have any DBSCAN-detected damage
focus cluster at all?

This is the non-topological comparison point for the persistent-homology
results elsewhere in this package (View 1 / View 2 / View 3): the classic
"focus count" style test, which asks only whether clustering happened, not
what shape or internal structure it had. Two two-cell-line datasets are
analysed the same way: a pooled Fisher exact test of cell type vs. cluster
presence, an adjusted logistic regression controlling for marker and
timepoint, a cell-type x marker interaction likelihood-ratio test, and a
sham-exclusion sensitivity refit that drops control rows to check the
cell-type effect isn't just an artifact of the control stratum.

Both datasets fit the same model:
    has_clusters ~ C(cell_type) + C(marker) + C(timepoint_label)
"kuentzelmann" compares NHDF vs. U87 (markers 53BP1 and yH2AX, single
pooled control). "hahn" compares HGF vs. MCF7 (markers Mre11 and yH2AX,
timepoint-matched sham).

Some cells in the resulting contingency table sit at or near 0%/100%
cluster presence (e.g. one cell type/marker/control combination with zero
clusters across every nucleus). This is a real, checked feature of the
data, not a modelling artifact -- statsmodels' logit() will produce large,
poorly determined coefficients and standard errors for terms dominated by
such a cell. That's expected; the joint likelihood-ratio p-value is the
number to trust in that situation, not the individual near-separated
coefficient.

Reference levels: cell_type is NHDF for kuentzelmann and HGF for hahn (the
non-cancer line in each pair); marker is yH2AX (present in both datasets).

Inputs
------
  data/feature_matrix.csv                              per-nucleus feature table
  results/persistence_features_degeneracy_report.json  optional; cross-referenced if present

Outputs
-------
  results/cluster_presence_contingency.csv
  results/cluster_presence_model_summary.txt
  results/cluster_presence_results.json

Usage
-----
  python -u cluster_presence_baseline.py --features data/feature_matrix.csv --results-dir results/

Dependencies: numpy, pandas, scipy, statsmodels
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2 as _chi2
from scipy.stats import fisher_exact

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SOURCES = ["kuentzelmann", "hahn"]

# The non-cancer cell line in each pair.
CELL_TYPE_REF = {
    "kuentzelmann": "NHDF",
    "hahn":         "HGF",
}
MARKER_REF = "yH2AX"   # present in both datasets


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY — Wilson confidence interval
# ══════════════════════════════════════════════════════════════════════════════

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score interval for a proportion k/n. Appropriate for small n or
    proportions near 0 or 1 where the normal approximation breaks down --
    exactly the regime the near-separated cells described in the module
    docstring sit in. Returns (lower, upper); both NaN if n == 0.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return ((centre - half) / denom, (centre + half) / denom)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY — Timepoint label
# ══════════════════════════════════════════════════════════════════════════════

def make_timepoint_label(tp_h) -> str:
    """
    timepoint_h (float or NaN) -> a string safe for use as a categorical
    covariate. NaN (single-aliquot control in kuentzelmann) becomes
    "control", so it reads as a real level in the model rather than
    silently dropping those rows. Numeric timepoints keep their natural
    label ("0.08h", "4h") rather than a prefixed condition string, which
    avoids collinearity with cell_type.
    """
    if pd.isna(tp_h):
        return "control"
    v = float(tp_h)
    if v == int(v):
        return f"{int(v)}h"
    return f"{v}h"


# ══════════════════════════════════════════════════════════════════════════════
# CONTINGENCY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_contingency(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Zero-cluster and has-cluster counts, with Wilson 95% CIs, over group_cols."""
    rows = []
    for key, grp in df.groupby(group_cols):
        if not isinstance(key, tuple):
            key = (key,)
        n = len(grp)
        k_zero = int((grp["n_clusters"] == 0).sum())
        rate = k_zero / n if n > 0 else float("nan")
        lo, hi = wilson_ci(k_zero, n)
        row = dict(zip(group_cols, key))
        row.update({
            "n": n,
            "n_zero_cluster": k_zero,
            "n_has_cluster": n - k_zero,
            "zero_cluster_rate": round(rate, 4),
            "wilson_ci_lo": round(lo, 4),
            "wilson_ci_hi": round(hi, 4),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# FISHER EXACT TEST — pooled 2×2
# ══════════════════════════════════════════════════════════════════════════════

def pooled_fisher(df: pd.DataFrame, group_col: str, ref_level: str) -> Dict:
    """
    Unconditional Fisher exact, pooled 2x2: group_col level x has_clusters.
    OR reported as P(has_clusters | ref_level) / P(has_clusters | other) --
    OR > 1 means the reference level has higher odds of having clusters.
    """
    df = df.copy()
    df["has_clusters"] = (df["n_clusters"] > 0)
    levels = sorted(df[group_col].unique())
    if len(levels) != 2:
        return {"error": f"Expected 2 levels in {group_col}, got {levels}"}

    other = [lv for lv in levels if lv != ref_level][0]
    a = int(df[(df[group_col] == ref_level) & ~df["has_clusters"]].shape[0])
    b = int(df[(df[group_col] == ref_level) &  df["has_clusters"]].shape[0])
    c = int(df[(df[group_col] == other)     & ~df["has_clusters"]].shape[0])
    d = int(df[(df[group_col] == other)     &  df["has_clusters"]].shape[0])
    table = [[a, b], [c, d]]
    raw_or, p = fisher_exact(table)
    or_has_clusters = float(1.0 / raw_or) if raw_or != 0 else float("inf")
    return {
        "table_2x2": {
            ref_level: {"no_clusters": a, "has_clusters": b},
            other:     {"no_clusters": c, "has_clusters": d},
        },
        f"OR_{ref_level}_vs_{other}_has_clusters": round(or_has_clusters, 4),
        "p_value": float(p),
        "interpretation": (
            f"OR > 1 means {ref_level} has higher odds of having >=1 cluster "
            f"than {other}, unconditional (no covariate adjustment)."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# LOGISTIC REGRESSION — two-cell-line sources
# ══════════════════════════════════════════════════════════════════════════════

def _lrt(m_full, m_null) -> Dict:
    """Likelihood-ratio test comparing m_full against m_null."""
    stat = float(2 * (m_full.llf - m_null.llf))
    df_diff = int(m_full.df_model - m_null.df_model)
    p = float(1 - _chi2.cdf(stat, df_diff)) if df_diff > 0 else float("nan")
    return {"llr_stat": round(stat, 4), "df_diff": df_diff, "p_value": round(p, 6)}


def _extract_ct_coef(model, ct_param: str) -> Dict:
    """OR, 95% CI, and p-value for one named coefficient."""
    if ct_param not in model.params:
        return {"error": f"Parameter '{ct_param}' not found in model"}
    coef = float(model.params[ct_param])
    se   = float(model.bse[ct_param])
    pval = float(model.pvalues[ct_param])
    or_  = float(np.exp(coef))
    ci_lo = float(np.exp(coef - 1.96 * se))
    ci_hi = float(np.exp(coef + 1.96 * se))
    return {
        "coef": round(coef, 4),
        "odds_ratio": round(or_, 4),
        "ci_95_lo": round(ci_lo, 4),
        "ci_95_hi": round(ci_hi, 4),
        "p_value": round(pval, 6),
    }


def fit_two_cellline_models(
    df: pd.DataFrame, source: str, markers: List[str], cell_type_ref: str,
) -> Dict:
    """
    Fits four logistic models for one dataset: main effects, a cell-type x
    marker interaction, a descriptive cell-type x timepoint interaction, and
    a sham-exclusion refit of the main-effects model.
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required: pip install statsmodels")

    df = df.copy()
    df["has_clusters"] = (df["n_clusters"] > 0).astype(int)
    df["timepoint_label"] = df["timepoint_h"].apply(make_timepoint_label)

    other_ct = [c for c in df["cell_type"].unique() if c != cell_type_ref][0]
    df["cell_type"] = pd.Categorical(df["cell_type"], categories=[cell_type_ref, other_ct])
    ct_param = f"C(cell_type)[T.{other_ct}]"

    has_multi_marker = len(markers) > 1
    if has_multi_marker:
        other_mk = [m for m in markers if m != MARKER_REF][0]
        df["marker"] = pd.Categorical(df["marker"], categories=[MARKER_REF, other_mk])
        mk_param = f"C(marker)[T.{other_mk}]"

    results: Dict = {}

    def _fit_model(formula: str, label: str, data: pd.DataFrame):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                m = smf.logit(formula, data=data).fit(disp=False, maxiter=200)
                for w in caught:
                    logger.warning(f"  [{label}] {w.category.__name__}: {w.message}")
                return m
            except Exception as e:
                logger.error(f"  [{label}] Model failed: {e}")
                return None

    # ── Main-effects model ───────────────────────────────────────────────────
    formula_m1 = (
        "has_clusters ~ C(cell_type) + C(marker) + C(timepoint_label)"
        if has_multi_marker else
        "has_clusters ~ C(cell_type) + C(timepoint_label)"
    )
    m1 = _fit_model(formula_m1, f"{source}/main_effects", df)
    if m1 is None:
        results["main_effects"] = {"error": "Model failed to converge"}
    else:
        ct_result = _extract_ct_coef(m1, ct_param)
        ct_result["reference_level"] = cell_type_ref
        ct_result["comparison_level"] = other_ct
        ct_result["interpretation"] = (
            f"OR for {other_ct} vs {cell_type_ref} having >=1 cluster, "
            f"adjusted for {'marker and ' if has_multi_marker else ''}timepoint. "
            f"OR < 1 means {other_ct} has lower odds of having >=1 cluster than {cell_type_ref}."
        )
        results["main_effects"] = {
            "formula": formula_m1, "n_obs": int(m1.nobs),
            "pseudo_r2": round(float(m1.prsquared), 4),
            "cell_type_coef": ct_result,
            "summary_text": m1.summary().as_text(),
        }

    # ── Interaction: cell_type × marker ──────────────────────────────────────
    if has_multi_marker and m1 is not None:
        formula_m2 = "has_clusters ~ C(cell_type) * C(marker) + C(timepoint_label)"
        m2 = _fit_model(formula_m2, f"{source}/interaction_marker", df)
        if m2 is None:
            results["interaction_marker"] = {"error": "Model failed"}
        else:
            inter_param = f"C(cell_type)[T.{other_ct}]:C(marker)[T.{other_mk}]"
            inter_coef = (
                _extract_ct_coef(m2, inter_param) if inter_param in m2.params
                else {"error": f"Interaction parameter '{inter_param}' not found"}
            )
            results["interaction_marker"] = {
                "formula": formula_m2, "interaction_term": inter_param,
                "interaction_coef": inter_coef,
                "lrt_vs_main_effects": _lrt(m2, m1),
                "interpretation": (
                    f"LRT p tests whether the {other_ct}-vs-{cell_type_ref} effect on "
                    f"cluster presence significantly differs between {other_mk} and "
                    f"{MARKER_REF}. p < 0.05 = the cell-line effect is marker-dependent, "
                    f"not a single number pooling across markers would honestly represent."
                ),
                "summary_text": m2.summary().as_text(),
            }

    # ── Interaction: cell_type × timepoint (descriptive) ─────────────────────
    if m1 is not None:
        formula_m3 = (
            "has_clusters ~ C(cell_type) * C(timepoint_label) + C(marker)"
            if has_multi_marker else
            "has_clusters ~ C(cell_type) * C(timepoint_label)"
        )
        m3 = _fit_model(formula_m3, f"{source}/interaction_timepoint_descriptive", df)
        if m3 is None:
            results["interaction_timepoint_descriptive"] = {"error": "Model failed"}
        else:
            inter_terms_tp = [p for p in m3.params.index if "C(cell_type)" in p and "C(timepoint_label)" in p]
            results["interaction_timepoint_descriptive"] = {
                "formula": formula_m3,
                "lrt_vs_main_effects": _lrt(m3, m1),
                "n_interaction_terms": len(inter_terms_tp),
                "_note": (
                    "DESCRIPTIVE, not confirmatory. Tests whether the cell-line effect "
                    "varies across the repair timecourse rather than being one "
                    "condition-averaged OR. Cells at or near 0%/100% (see module "
                    "docstring) will produce large, unstable individual coefficients "
                    "here -- read the joint LRT p-value, not the per-term coefficients."
                ),
                "summary_text": m3.summary().as_text(),
            }

    # ── Sham-exclusion sensitivity ────────────────────────────────────────────
    # Both datasets drop is_control rows here. For hahn's timepoint-matched
    # design this also drops the paired sham row at each timepoint -- fine,
    # because timepoint_label is already a covariate in the refit model, so
    # the matched-pair structure isn't what's carrying the timepoint signal.
    df_nosham = df[~df["is_control"]].copy()
    nosham_label = (
        "irradiated only (matched-sham rows dropped)" if source == "hahn"
        else "irradiated only (single-aliquot control dropped)"
    )
    m_nosham = _fit_model(formula_m1, f"{source}/sham_exclusion", df_nosham)
    if m_nosham is None:
        results["sham_exclusion"] = {"error": "Model failed", "label": nosham_label}
    else:
        ct_nosham = _extract_ct_coef(m_nosham, ct_param)
        ct_nosham["reference_level"] = cell_type_ref
        ct_nosham["comparison_level"] = other_ct
        results["sham_exclusion"] = {
            "label": nosham_label, "n_obs": int(m_nosham.nobs),
            "cell_type_coef": ct_nosham,
            "interpretation": (
                "Compare OR and direction with main_effects. Discordance (different "
                "direction, or p crossing 0.05) would suggest the cluster-presence "
                "asymmetry is driven by the control condition rather than holding "
                "across the repair window."
            ),
            "summary_text": m_nosham.summary().as_text(),
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PER-SOURCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_source(df: pd.DataFrame, source: str) -> Dict:
    grp = df[df["source"] == source].copy()
    markers = sorted(grp["marker"].unique())
    ct_ref = CELL_TYPE_REF[source]
    logger.info(f"\n{'='*60}")
    logger.info(f"source={source}  cell_types={sorted(grp['cell_type'].unique())}  markers={markers}  n={len(grp)}")
    logger.info(f"    zero_cluster_rate: {(grp['n_clusters']==0).mean():.3f}")

    result: Dict = {
        "source": source, "n_total": int(len(grp)),
        "n_zero_cluster": int((grp["n_clusters"] == 0).sum()),
        "zero_cluster_rate_overall": round(float((grp["n_clusters"] == 0).mean()), 4),
        "cell_type_ref": ct_ref, "markers": markers,
    }

    grp["timepoint_label"] = grp["timepoint_h"].apply(make_timepoint_label)
    result["contingency_table"] = build_contingency(
        grp, ["cell_type", "marker", "timepoint_label"]
    ).to_dict(orient="records")

    result["fisher_exact_pooled"] = pooled_fisher(grp, "cell_type", ct_ref)
    other_ct = [c for c in grp["cell_type"].unique() if c != ct_ref][0]
    fisher_or = result["fisher_exact_pooled"].get(f"OR_{ct_ref}_vs_{other_ct}_has_clusters", "?")
    logger.info(f"    Fisher exact (pooled): OR={fisher_or}  p={result['fisher_exact_pooled']['p_value']:.2e}")

    if HAS_STATSMODELS:
        result["logistic_models"] = fit_two_cellline_models(grp, source, markers, ct_ref)
        ct_coef = result["logistic_models"].get("main_effects", {}).get("cell_type_coef", {})
        if "odds_ratio" in ct_coef:
            logger.info(
                f"    Logistic main-effects OR: {ct_coef['odds_ratio']:.4f} "
                f"[{ct_coef['ci_95_lo']:.4f}, {ct_coef['ci_95_hi']:.4f}] p={ct_coef['p_value']:.2e}"
            )
        se_coef = result["logistic_models"].get("sham_exclusion", {}).get("cell_type_coef", {})
        if "odds_ratio" in se_coef:
            logger.info(f"    Sham-exclusion OR: {se_coef['odds_ratio']:.4f}  p={se_coef['p_value']:.2e}")
    else:
        result["logistic_models"] = {"error": "statsmodels not available"}

    return result


def cross_source_summary(results: Dict[str, Dict]) -> List[Dict]:
    radiation_type = {"kuentzelmann": "heavy-ion", "hahn": "photon"}
    rows = []
    for src, res in results.items():
        fisher = res.get("fisher_exact_pooled", {})
        ct_ref = res.get("cell_type_ref", "")
        me = res.get("logistic_models", {}).get("main_effects", {}).get("cell_type_coef", {})
        se = res.get("logistic_models", {}).get("sham_exclusion", {}).get("cell_type_coef", {})
        fisher_or = next((v for k, v in fisher.items() if k.startswith("OR_") and k.endswith("_has_clusters")), None)
        rows.append({
            "source": src, "radiation_type": radiation_type.get(src, "unknown"),
            "n_total": res.get("n_total"),
            "zero_cluster_rate_overall": res.get("zero_cluster_rate_overall"),
            "fisher_p": fisher.get("p_value"), "fisher_OR_ref_vs_other": fisher_or,
            "logistic_OR": me.get("odds_ratio"), "logistic_OR_lo": me.get("ci_95_lo"),
            "logistic_OR_hi": me.get("ci_95_hi"), "logistic_p": me.get("p_value"),
            "sham_excl_OR": se.get("odds_ratio"), "sham_excl_p": se.get("p_value"),
            "cell_type_ref": ct_ref,
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# DEGENERACY CROSS-REFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def load_feature_extraction_degeneracy(results_dir: Path, sources: List[str]) -> Dict:
    """
    Optionally loads a degeneracy report produced by the feature-extraction
    step (keyed "{source}__{cell_type}", plus "_any_flag"/"_interpretation"
    meta keys), reporting how often nuclei come out with too few clusters to
    trust downstream features. Filtered here to just the entries belonging
    to this script's two datasets, so a reader sees only what's relevant to
    cluster presence rather than the whole file verbatim.
    """
    path = results_dir / "persistence_features_degeneracy_report.json"
    if not path.exists():
        return {}
    with open(path) as fh:
        full_report = json.load(fh)
    return {
        k: v for k, v in full_report.items()
        if not k.startswith("_") and v.get("source") in sources
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=Path("data/feature_matrix.csv"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    if not HAS_STATSMODELS:
        logger.error("statsmodels not installed: pip install statsmodels")
        return 1
    if not args.features.exists():
        logger.error(f"Feature matrix not found: {args.features}")
        return 1

    df = pd.read_csv(args.features, dtype={"replicate": str})
    logger.info("=" * 70)
    logger.info("Cluster-presence baseline analysis")
    logger.info("=" * 70)
    logger.info(f"Loaded {len(df)} nuclei from {args.features}")
    logger.info(f"Overall zero-cluster rate: {(df['n_clusters']==0).sum()}/{len(df)} = {(df['n_clusters']==0).mean()*100:.1f}%")

    args.results_dir.mkdir(parents=True, exist_ok=True)

    degeneracy = load_feature_extraction_degeneracy(args.results_dir, SOURCES)
    if degeneracy:
        logger.info("\nCross-referencing degeneracy report:")
        for k, v in degeneracy.items():
            logger.info(f"    {k}: frac_degenerate={v.get('frac_degenerate')}  flagged={v.get('flagged')}")
    else:
        logger.info("\nNo degeneracy report found in --results-dir -- proceeding without the cross-reference.")

    results: Dict[str, Dict] = {}
    for src in SOURCES:
        results[src] = analyse_source(df, src)

    logger.info(f"\n{'='*60}")
    logger.info("Cross-source summary")
    comparison_rows = cross_source_summary(results)
    logger.info("    Source              Rad type   zero%  Fisher-p   OR     logistic-p")
    for r in comparison_rows:
        # None-checks here, not `value or float('nan')` -- that pattern
        # silently turns a genuine 0.0 p-value into nan, since 0.0 is
        # falsy in Python. A p-value that rounds to exactly 0.0
        # (astronomically small) would otherwise get logged as nan even
        # though the stored value was correct.
        fisher_p = r["fisher_p"] if r["fisher_p"] is not None else float("nan")
        logistic_or = r["logistic_OR"] if r["logistic_OR"] is not None else float("nan")
        logistic_p = r["logistic_p"] if r["logistic_p"] is not None else float("nan")
        logger.info(
            f"    {r['source']:<20} {r['radiation_type']:<10} "
            f"{(r['zero_cluster_rate_overall'] or 0)*100:>5.1f}%  "
            f"{fisher_p:>9.2e}  "
            f"{logistic_or:>6.4f}  "
            f"{logistic_p:>10.2e}"
        )

    all_cont_rows = []
    for src, res in results.items():
        for row in res.get("contingency_table", []):
            row["source"] = src
            all_cont_rows.append(row)
    cont_df = pd.DataFrame(all_cont_rows)
    cont_path = args.results_dir / "cluster_presence_contingency.csv"
    cont_df.to_csv(cont_path, index=False)
    logger.info(f"\nContingency table -> {cont_path}  ({len(cont_df)} rows)")

    txt_path = args.results_dir / "cluster_presence_model_summary.txt"
    with open(txt_path, "w") as fh:
        fh.write("Cluster-presence baseline — model summaries\n")
        fh.write("=" * 78 + "\n\n")
        if degeneracy:
            fh.write("Cross-reference, feature-extraction degeneracy report:\n")
            for k, v in degeneracy.items():
                fh.write(f"  {k}: frac_degenerate={v.get('frac_degenerate')}  flagged={v.get('flagged')}\n")
            fh.write("\n")
        for src, res in results.items():
            fh.write(f"SOURCE: {src}\n" + "-" * 60 + "\n")
            models = res.get("logistic_models", {})
            for model_key in ["main_effects", "interaction_marker", "interaction_timepoint_descriptive", "sham_exclusion"]:
                if model_key not in models:
                    continue
                m = models[model_key]
                if "error" in m:
                    fh.write(f"\n[{model_key}] ERROR: {m['error']}\n")
                    continue
                fh.write(f"\n{'='*50}\nModel: {model_key}\n")
                fh.write(f"Formula/label: {m.get('formula') or m.get('label', '')}\n{'='*50}\n")
                fh.write(m.get("summary_text", "[no summary text]"))
                fh.write("\n")
                if model_key == "sham_exclusion" and m.get("interpretation"):
                    fh.write(f"\nNote: {m['interpretation']}\n")
            fh.write("\n\n")
    logger.info(f"Model summaries -> {txt_path}")

    def _strip_summary(d):
        if isinstance(d, dict):
            return {k: _strip_summary(v) for k, v in d.items() if k != "summary_text"}
        if isinstance(d, list):
            return [_strip_summary(i) for i in d]
        return d

    json_payload = {
        "n_total": int(len(df)),
        "n_zero_cluster_overall": int((df["n_clusters"] == 0).sum()),
        "zero_cluster_rate_overall": round(float((df["n_clusters"] == 0).mean()), 4),
        "degeneracy_cross_reference": degeneracy,
        "per_source": {src: _strip_summary(res) for src, res in results.items()},
        "cross_source_summary": comparison_rows,
    }
    json_path = args.results_dir / "cluster_presence_results.json"
    with open(json_path, "w") as fh:
        json.dump(json_payload, fh, indent=2, default=str)
    logger.info(f"JSON results -> {json_path}")

    logger.info("=" * 70)
    logger.info("Cluster-presence baseline analysis complete.")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
