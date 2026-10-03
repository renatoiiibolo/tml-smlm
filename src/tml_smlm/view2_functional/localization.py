"""
View 2 (functional), Tier 1 — sparse localization along the birth-radius axis.

A persistence diagram can be treated as a function of the filtration
parameter: its persistence landscape. This module vectorizes landscapes
onto a shared grid and uses a cross-validated fused-lasso fit to ask not
just *whether* two groups differ, but *where* along the birth-radius axis
the difference is concentrated. The fusion penalty favors solutions where
the selected grid positions are contiguous, so a hit reads as a localized
region rather than scattered noise — the "sparse localization" the module
implements.

Downstream modules (robustness, temporal-trajectory and coupling analyses)
import several names from here directly and rely on their exact behavior:
`_load_dgm`, `vectorise`, `SCALE_COVAR`, `residualise`, `CV_SEED`, `fl_cv`,
`_clean`, `_landscape`, `_build_grid`, `SOURCE_CFG`, `_tp_label`.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

N_GRID_PTS = 300
GRID_PCT = 99.0
N_LAMBDA_GRID = 40
CV_SEED = 42  # module-level, intentionally mutable: callers doing repeated-CV
              # sensitivity checks set s6.CV_SEED = seed, call fl_cv, then restore it
N_PERM_DEF = 1000

# Which covariate to residualize each spatial scale against before fitting.
# Scale B's group signal is driven by cluster-count geometry, so residualizing
# on n_localisations there would remove exactly what we want to keep; Scale C
# and noise scales are residualized against localization count instead.
SCALE_COVAR = {"scaleB": "n_clusters", "scaleC": "n_localisations", "noise": "n_localisations"}

SOURCE_CFG: Dict[str, Dict] = {
    "kuentzelmann": {"target": "cell_type", "pos": "U87", "perm": "stratified"},
    "hahn": {"target": "cell_type", "pos": "MCF7", "perm": "stratified"},
}

# Scale B is always pooled across markers (H0 only); Scale C and noise get
# both H0 and H1 panels per marker.
PANEL_DEFS: List[Dict] = [
    {"scale": "scaleB", "hom": "h0", "key": "scaleB_h0"},
    {"scale": "scaleC", "hom": "h0", "key": "scaleC_h0"},
    {"scale": "scaleC", "hom": "h1", "key": "scaleC_h1"},
    {"scale": "noise", "hom": "h0", "key": "noise_h0"},
    {"scale": "noise", "hom": "h1", "key": "noise_h1"},
]


# ══════════════════════════════════════════════════════════════════════════
# LANDSCAPE (pure numpy)
# ══════════════════════════════════════════════════════════════════════════

def _clean(dgm: np.ndarray) -> np.ndarray:
    if len(dgm) == 0:
        return dgm
    dgm = dgm[np.isfinite(dgm[:, 1])]
    if len(dgm) == 0:
        return dgm
    return dgm[(dgm[:, 1] - dgm[:, 0]) > 0]


def _landscape(dgm: np.ndarray, grid: np.ndarray, k: int) -> np.ndarray:
    """k-th persistence landscape lambda_k evaluated at grid (k 1-indexed)."""
    n = len(grid)
    if len(dgm) == 0 or k > len(dgm):
        return np.zeros(n)
    b = dgm[:, 0:1]; d = dgm[:, 1:2]; t = grid[np.newaxis, :]
    tents = np.maximum(0.0, np.minimum(t - b, d - t))
    if k == 1:
        return tents.max(axis=0)
    idx = min(k - 1, tents.shape[0] - 1)
    return -np.partition(-tents, idx, axis=0)[idx]


def _build_grid(diagrams: List[np.ndarray]) -> np.ndarray:
    all_d = np.concatenate([d[:, 1] for d in diagrams if len(d) > 0]) if any(len(d) > 0 for d in diagrams) else np.array([1.0])
    gmax = float(np.percentile(all_d, GRID_PCT))
    if gmax <= 0:
        gmax = float(all_d.max()) if len(all_d) > 0 else 1.0
    return np.linspace(0.0, gmax, N_GRID_PTS)


def vectorise(diagrams: List[np.ndarray], k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorize a population of diagrams onto one shared grid. The grid is
    built from the pooled death-time distribution (99th percentile as the
    cap, to keep a few extreme bars from stretching the whole axis) rather
    than per-diagram: every row of the resulting matrix has to be
    comparable column-by-column for the fused lasso to treat columns as a
    single ordered axis with a meaningful fusion penalty between neighbors.
    """
    cleaned = [_clean(d) for d in diagrams]
    grid = _build_grid(cleaned)
    return np.vstack([_landscape(d, grid, k) for d in cleaned]), grid


# ══════════════════════════════════════════════════════════════════════════
# RESIDUALIZATION — one grid column at a time
# ══════════════════════════════════════════════════════════════════════════

def residualise(L: np.ndarray, covar: np.ndarray) -> np.ndarray:
    fin = np.isfinite(covar)
    if fin.sum() < 5:
        return L.copy()
    X = np.column_stack([np.ones(len(covar)), covar])
    R = L.copy()
    for j in range(L.shape[1]):
        y = L[:, j]; m = fin & np.isfinite(y)
        if m.sum() < 5:
            continue
        try:
            beta = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
            R[:, j] = y - X @ beta
        except np.linalg.LinAlgError:
            pass
    return R


# ══════════════════════════════════════════════════════════════════════════
# FUSED LASSO
# ══════════════════════════════════════════════════════════════════════════

def _D(p: int) -> np.ndarray:
    D = np.zeros((p - 1, p))
    for i in range(p - 1):
        D[i, i] = -1.0; D[i, i + 1] = 1.0
    return D


def _fl_fit(X: np.ndarray, y: np.ndarray, lam: float, warm_start_coef: Optional[np.ndarray] = None, tol: float = 1e-6) -> np.ndarray:
    """
    One fused-lasso fit at a single lambda, via the standard trick of
    augmenting X with lam * D (D the first-difference operator) and
    y with zeros, then handing the whole thing to an ordinary Lasso.

    warm_start_coef, when given, initializes coordinate descent from a
    nearby solution instead of zero. That only helps when the nearby
    solution really is nearby: true for consecutive points on a lambda
    path (see _fl_path_fit), not true when initializing a permutation fit
    from the observed data's solution — a shuffled-label null typically
    has a much sparser optimum than the real data, so starting there can
    leave coordinate descent fighting a bad start rather than refining a
    good one. Permutation fits are therefore always cold-started
    (warm_start_coef=None) with a relaxed tolerance — see _fl_cv_ba_perm.
    """
    n, p = X.shape
    D = _D(p)
    Xaug = np.vstack([X, lam * D])
    yaug = np.concatenate([y.astype(float), np.zeros(p - 1)])
    alpha = lam * len(yaug) / n
    clf = Lasso(alpha=alpha, fit_intercept=False, max_iter=100000, tol=tol, warm_start=True)
    clf.coef_ = warm_start_coef.copy() if warm_start_coef is not None else np.zeros(p)
    clf.fit(Xaug, yaug)
    return clf.coef_


def _fl_path_fit(X: np.ndarray, y: np.ndarray, lam_grid_desc: np.ndarray) -> Dict[float, np.ndarray]:
    """
    Fit the full lambda path once, largest lambda to smallest, each fit
    warm-started from the previous (larger) lambda's solution. Consecutive
    solutions along a lasso path are close, so this converges far more
    reliably, and faster, than fitting each lambda cold from zero.
    """
    coef = None
    betas: Dict[float, np.ndarray] = {}
    for lam in lam_grid_desc:  # already largest -> smallest, see fl_cv
        coef = _fl_fit(X, y, lam, warm_start_coef=coef)
        betas[float(lam)] = coef.copy()
    return betas


def _fl_cv_ba_path(X: np.ndarray, y: np.ndarray, lam_grid_desc: np.ndarray) -> np.ndarray:
    """5-fold CV balanced accuracy at every lambda in the grid, one warm-started path per fold."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    cv_bas = np.zeros(len(lam_grid_desc))
    for tr, te in skf.split(X, y):
        betas = _fl_path_fit(X[tr], y[tr], lam_grid_desc)
        for i, lam in enumerate(lam_grid_desc):
            beta = betas[float(lam)]
            thr = float(np.median(X[tr] @ beta))
            cv_bas[i] += balanced_accuracy_score(y[te], (X[te] @ beta >= thr).astype(int))
    return cv_bas / 5


def _fl_cv_ba(X: np.ndarray, y: np.ndarray, lam: float, warm_start_coef: Optional[np.ndarray] = None) -> float:
    """Single-lambda CV balanced accuracy for the observed (unpermuted) data — tight tolerance, since this feeds the reported coefficients."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    bas = []
    for tr, te in skf.split(X, y):
        beta = _fl_fit(X[tr], y[tr], lam, warm_start_coef=warm_start_coef)
        thr = float(np.median(X[tr] @ beta))
        bas.append(balanced_accuracy_score(y[te], (X[te] @ beta >= thr).astype(int)))
    return float(np.mean(bas))


def _fl_cv_ba_perm(X: np.ndarray, y: np.ndarray, lam: float) -> float:
    """
    Single-lambda CV balanced accuracy for one permutation's shuffled y.
    Cold-started (no warm_start_coef) for the reason given in _fl_fit.
    Tolerance relaxed to 1e-4: a permutation replicate only needs a stable
    classification threshold to contribute one sample to the null
    distribution, not a precisely converged coefficient vector, unlike the
    observed fit whose beta_lam_min is reported directly.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    bas = []
    for tr, te in skf.split(X, y):
        beta = _fl_fit(X[tr], y[tr], lam, warm_start_coef=None, tol=1e-4)
        thr = float(np.median(X[tr] @ beta))
        bas.append(balanced_accuracy_score(y[te], (X[te] @ beta >= thr).astype(int)))
    return float(np.mean(bas))


def fl_cv(X: np.ndarray, y: np.ndarray) -> Dict:
    n, p = X.shape
    lam_max = float(np.abs(X.T @ y.astype(float)).max() / n) or 1.0
    lam_grid = np.geomspace(lam_max, lam_max / 100, N_LAMBDA_GRID)  # largest -> smallest, required for warm-starting
    cv_bas = _fl_cv_ba_path(X, y, lam_grid)

    i_min = int(np.argmax(cv_bas))
    lam_min = float(lam_grid[i_min]); ba_min = float(cv_bas[i_min])
    se = float(cv_bas.std() / np.sqrt(N_LAMBDA_GRID))
    cands = lam_grid[cv_bas >= ba_min - se]
    lam_1se = float(cands.max()) if len(cands) else lam_min
    i_1se = int(np.argmin(np.abs(lam_grid - lam_1se)))
    ba_1se = float(cv_bas[i_1se])

    full_betas = _fl_path_fit(X, y.astype(float), lam_grid)
    beta_min = full_betas[lam_min]
    beta_1se = full_betas[lam_1se]
    return {
        "lam_min": lam_min, "ba_at_lam_min": ba_min, "lam_1se": lam_1se, "ba_at_lam_1se": ba_1se,
        "n_nonzero_min": int((np.abs(beta_min) > 1e-6).sum()), "n_nonzero_1se": int((np.abs(beta_1se) > 1e-6).sum()),
        "beta_lam_min": beta_min.tolist(), "beta_lam_1se": beta_1se.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════
# PERMUTATION TEST — stratified by timepoint. Cold-started, relaxed-tolerance
# single-lambda fits (see _fl_cv_ba_perm), not warm-started from the observed
# solution (see _fl_fit for why that would be the wrong initialization).
# ══════════════════════════════════════════════════════════════════════════

def _one_perm(X: np.ndarray, y: np.ndarray, lam: float, strat: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    yp = y.copy()
    for sv in np.unique(strat):
        m = strat == sv; yp[m] = rng.permutation(y[m])
    return _fl_cv_ba_perm(X, yp, lam)


def perm_test(X: np.ndarray, y: np.ndarray, obs_ba: float, lam: float, n_perm: int, tp: np.ndarray, n_jobs: int) -> Dict:
    seeds = np.random.default_rng(0).integers(0, 2**31, n_perm).tolist()
    pbas = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_one_perm)(X, y, lam, tp, s) for s in seeds)
    pbas = np.array(pbas)
    p_val = float((pbas >= obs_ba).mean())
    return {
        "observed_ba": round(obs_ba, 4), "n_permutations": n_perm,
        "perm_mean_ba": round(float(pbas.mean()), 4), "perm_sd_ba": round(float(pbas.std()), 4),
        "p_value": round(p_val, 6), "perm_mode": "stratified_by_timepoint",
    }


# ══════════════════════════════════════════════════════════════════════════
# DIAGRAM LOADING
# ══════════════════════════════════════════════════════════════════════════

def _npz_stem(row: pd.Series) -> str:
    raw = f"{row['source']}__{row['cell_type']}__{row['marker']}__{row['condition']}__{row['replicate']}"
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", raw)


def _load_dgm(diagrams_dir: Path, row: pd.Series, arr_key: str) -> np.ndarray:
    npz_file = row.get("npz_file", _npz_stem(row) + ".npz")
    path = diagrams_dir / str(npz_file)
    if not path.exists():
        path = diagrams_dir / (_npz_stem(row) + ".npz")
    try:
        with np.load(path, allow_pickle=False) as npz:
            return npz[arr_key].astype(np.float64) if arr_key in npz else np.empty((0, 2))
    except Exception:
        return np.empty((0, 2))


def _tp_label(tp) -> str:
    if pd.isna(tp):
        return "control"
    v = float(tp)
    return f"{int(v)}h" if v == int(v) else f"{v}h"


# ══════════════════════════════════════════════════════════════════════════
# PROCESS ONE PANEL
# ══════════════════════════════════════════════════════════════════════════

def process_panel(
    source: str, marker: Optional[str], pdef: Dict, k: int, fm: pd.DataFrame,
    diagrams_dir: Path, n_perm: int, n_jobs: int, landscapes_dir: Path,
) -> Dict:
    scale = pdef["scale"]; hom = pdef["hom"]; arr_key = pdef["key"]
    lbl = f"{source}/{marker or 'pooled'}/{scale}_{hom}/lambda{k}"
    cfg = SOURCE_CFG[source]

    sub = fm[fm["source"] == source].copy()
    if marker:
        sub = sub[sub["marker"] == marker].copy()
    if len(sub) < 20:
        return {"label": lbl, "skipped": True, "reason": f"n={len(sub)}"}

    dgms = [_load_dgm(diagrams_dir, row, arr_key) for _, row in sub.iterrows()]
    L, grid = vectorise(dgms, k)

    covar_col = SCALE_COVAR.get(scale, "n_localisations")
    covar = sub[covar_col].to_numpy(dtype=float)
    L_r = residualise(L, covar)

    lsc_name = f"{source}_{marker or 'pooled'}_{scale}_{hom}_lambda{k}.npz"
    np.savez_compressed(landscapes_dir / lsc_name, landscape=L_r, grid=grid)

    y = (sub[cfg["target"]] == cfg["pos"]).astype(int).to_numpy()
    tp = sub["timepoint_h"].apply(_tp_label).to_numpy()
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos < 5 or n_neg < 5:
        return {"label": lbl, "skipped": True, "reason": f"n_pos={n_pos}, n_neg={n_neg}"}

    logger.info(f"  FL-CV  {lbl}  n={len(y)}")
    fl = fl_cv(L_r, y)

    logger.info(f"  Perm   {lbl}  ({n_perm})")
    pt = perm_test(L_r, y, fl["ba_at_lam_min"], fl["lam_min"], n_perm, tp, n_jobs)

    logger.info(f"  >> {lbl}: FL_ba={fl['ba_at_lam_min']:.4f} perm_p={pt['p_value']:.4g} n_nz={fl['n_nonzero_min']}")

    return {
        "label": lbl, "source": source, "marker": marker, "scale": scale, "hom": hom, "k_order": k,
        "n_nuclei": len(y), "n_pos": n_pos, "n_neg": n_neg, "covar_col": covar_col,
        "grid_min_nm": round(float(grid[0]), 2), "grid_max_nm": round(float(grid[-1]), 2),
        "fused_lasso": fl, "permutation": pt, "landscape_file": lsc_name,
    }


# ══════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE — maps each panel to its View 1 (scalar-summary) counterpart
# on the same scale/homology group, so the two answers to "does this group
# discriminate the classes" sit side by side.
# ══════════════════════════════════════════════════════════════════════════

def _view1_scale_key(scale: str) -> str:
    if scale == "scaleB":
        return "scale_B"
    if scale == "scaleC":
        return "scale_C"
    return "scale_C_noise"  # noise: no noise-only View 1 arm; compare against the combined one


def comparison_table(fl_results: List[Dict], view1_path: Optional[Path]) -> pd.DataFrame:
    v1: Dict = {}
    if view1_path and view1_path.exists():
        with open(view1_path) as f:
            v1 = json.load(f)
    rows = []
    for r in fl_results:
        if r.get("skipped"):
            continue
        src = r["source"]; mk = r.get("marker") or "pooled"
        scale_key = _view1_scale_key(r["scale"])
        v1_ba = v1_p = None
        try:
            mk_r = v1.get(src, {}).get("section_A", {}).get(mk or "", {})
            run = mk_r.get(scale_key, {})
            if not run.get("skipped"):
                v1_ba = run.get("resid_ba")
                v1_p = run.get("permutation", {}).get("p_value")
        except Exception:
            pass
        rows.append({
            "source": src, "marker": mk, "scale": r["scale"], "hom_dim": r["hom"], "lambda_order": r["k_order"],
            "view2_fl_ba": r["fused_lasso"]["ba_at_lam_min"], "view2_fl_perm_p": r["permutation"]["p_value"],
            "view2_fl_n_nonzero": r["fused_lasso"]["n_nonzero_min"],
            "view1_scale_key": scale_key, "view1_resid_ba": v1_ba, "view1_perm_p": v1_p,
            "grid_min_nm": r["grid_min_nm"], "grid_max_nm": r["grid_max_nm"],
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diagrams", type=Path, default=Path("diagrams"))
    ap.add_argument("--features", type=Path, default=Path("data/feature_matrix.csv"))
    ap.add_argument("--view1-json", type=Path, default=Path("results/view1_primary_analysis.json"))
    ap.add_argument("--landscapes-dir", type=Path, default=Path("landscapes"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--sources", nargs="+", default=list(SOURCE_CFG.keys()))
    ap.add_argument("--n-perm", type=int, default=N_PERM_DEF)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--lambda-orders", nargs="+", type=int, default=[1, 2])
    args = ap.parse_args()

    for p in [args.features, args.diagrams]:
        if not p.exists():
            logger.error(f"Not found: {p}")
            return 1
    args.landscapes_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    fm = pd.read_csv(args.features, dtype={"replicate": str})
    if "npz_file" not in fm.columns:
        fm["npz_file"] = fm.apply(lambda r: _npz_stem(r) + ".npz", axis=1)

    logger.info("=" * 70)
    logger.info("View 2 / Tier 1 — persistence-landscape fused lasso localization")
    logger.info("=" * 70)
    logger.info(f"Diagrams : {args.diagrams.resolve()}")
    logger.info(f"Sources  : {args.sources}")
    logger.info(f"n_perm   : {args.n_perm}   n_jobs: {args.n_jobs}   lambda orders: {args.lambda_orders}")

    all_results: List[Dict] = []
    for source in args.sources:
        if source not in SOURCE_CFG:
            logger.warning(f"Unknown source '{source}' -- skipping.")
            continue
        logger.info(f"\n{'='*60}\nSOURCE: {source}\n{'='*60}")
        src_markers = sorted(fm[fm["source"] == source]["marker"].unique())

        for pdef in PANEL_DEFS:
            markers_to_run = [None] if pdef["scale"] == "scaleB" else src_markers
            for marker in markers_to_run:
                for k in args.lambda_orders:
                    try:
                        r = process_panel(source, marker, pdef, k, fm, args.diagrams, args.n_perm, args.n_jobs, args.landscapes_dir)
                        all_results.append(r)
                    except Exception as e:
                        logger.error(f"FAILED {source}/{marker}/{pdef['scale']}_{pdef['hom']}/lambda{k}: {e}")
                        logger.debug(traceback.format_exc())

    def _strip(d):
        if isinstance(d, dict):
            return {k: _strip(v) for k, v in d.items() if k not in ("beta_lam_min", "beta_lam_1se")}
        if isinstance(d, list):
            return [_strip(i) for i in d]
        return d

    fl_path = args.results_dir / "view2_localization_fl_results.json"
    with open(fl_path, "w") as fh:
        json.dump([_strip(r) for r in all_results], fh, indent=2, default=str)
    logger.info(f"\nFL results -> {fl_path}")

    comp_df = comparison_table(all_results, args.view1_json)
    comp_path = args.results_dir / "view2_localization_comparison_table.csv"
    comp_df.to_csv(comp_path, index=False)
    logger.info(f"Comparison table -> {comp_path}  ({len(comp_df)} rows)")

    logger.info("\n=== View 1 (scalar summary) vs View 2 (fused lasso) SUMMARY (lambda1 only) ===")
    if comp_df.empty:
        logger.warning("No panels produced a result (all skipped) -- nothing to summarize. Check n=20 minimum per panel and that diagrams/ has the expected .npz files.")
    else:
        logger.info(f"{'Label':<45} {'FL_BA':>7} {'FL_p':>8} {'V1_BA':>7} {'V1_p':>8}")
        for _, row in comp_df[comp_df["lambda_order"] == 1].iterrows():
            lbl = f"{row['source']}/{row['marker']}/{row['scale']}_{row['hom_dim']}"
            fl_ba = f"{row['view2_fl_ba']:.4f}" if pd.notna(row.get("view2_fl_ba")) else "   -"
            fl_p = f"{row['view2_fl_perm_p']:.4g}" if pd.notna(row.get("view2_fl_perm_p")) else "  -"
            v1_ba = f"{row['view1_resid_ba']:.4f}" if pd.notna(row.get("view1_resid_ba")) else "   -"
            v1_p = f"{row['view1_perm_p']:.4g}" if pd.notna(row.get("view1_perm_p")) else "  -"
            logger.info(f"  {lbl:<43} {fl_ba:>7} {fl_p:>8} {v1_ba:>7} {v1_p:>8}")

    logger.info("=" * 70)
    logger.info("View 2 / Tier 1 localization complete.")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
