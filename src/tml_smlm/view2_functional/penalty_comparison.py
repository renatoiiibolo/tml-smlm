#!/usr/bin/env python3
"""
View 2 (functional), Tier 1 follow-up: is the smooth-lasso penalty the right
one for this landscape grid?

The localization module fits an L1 sparsity term plus an L2-squared penalty
on the difference between adjacent coefficients (the Smooth-Lasso of Hebiri
and van de Geer, 2011). Tibshirani et al. (2005) use an L1 penalty on those
differences instead (the "fused lasso"), which rewards exactly flat runs of
coefficients. Which penalty suits this data cannot be settled by argument
alone, so this module checks it two ways.

1. Shape diagnostic. Each panel is refit once at its own selected lambda, and
   the largest contiguous block of active grid positions is summarized by two
   descriptive numbers: the block's normalized range (near 0 means flat and
   blocky, larger means smoothly graded) and the Hoyer sparsity of its first
   differences. Both are heuristics, not a formal test, and a block of a
   single position has zero range by construction, which says nothing about
   a plateau. Read the beta profile itself (the plots) before the verdict
   label.

2. True fused-lasso comparison (`--compare-l1-fusion`). The same panel is fit
   with the genuine two-term objective

       (1/(2n)) ||y - X b||^2  +  lam1 ||b||_1  +  lam2 ||D b||_1

   by ADMM with residual-balancing step size. A fixed step size left beta up
   to ~80% off an independent reference solution (cvxpy, used once as an
   oracle and not imported here) at the largest lambda tested; the adaptive
   version agreed to 2e-7 to 3e-9 relative error. Two comparisons are
   reported per panel. The first reuses the smooth-lasso lambda for both
   penalty weights, which is NOT sparsity-matched because an L2-squared and
   an L1 coefficient are different units. The second sweeps one shared
   multiplier on lam1 = lam2 and reports the point whose active count is
   closest to the smooth-lasso fit, together with the full sweep.

On the paper's three headline panels the sweep shows no intermediate regime:
the true fused-lasso path is either empty or selects several dozen to over a
hundred positions, never a count near the smooth-lasso fit's 1, 2 or 25. That
is consistent with the near-collinear, densely sampled landscape grid, and it
is why the smooth-lasso penalty is used. Points adjacent to each cliff can be
flagged `converged: false` (ill-conditioning at a bifurcation), so read the
order of magnitude of a cliff, not its exact multiplier.

`clean_match` in the sweep output only says that some multiplier came within
the tolerance of the target count. A fit with zero active coefficients can
still be reported as the nearest point to a target of one; check
`achieved_n_active` before reading a match as a localized fused-lasso fit.

Outputs (in --results-dir)
  penalty_comparison.json            per-panel beta, grid, shape statistics,
                                     and the fused-lasso comparison block
  penalty_comparison_summary.txt     human-readable verdict table
  penalty_comparison_profiles/*.png  beta-versus-grid plots (skipped with a
                                     warning if matplotlib is missing)

Usage
  python -u -m tml_smlm.view2_functional.penalty_comparison \
      --diagrams diagrams/ --features data/feature_matrix.csv \
      --results-dir results/ --compare-l1-fusion \
      --sweep-hi-mult 2000 --sweep-points 24 \
      --l1-fusion-max-iter 20000 --l1-fusion-tol 1e-5

  Add --all-panels to cover every panel the localization module produces
  instead of the three headline ones, and --no-plots to skip the figures.
  A headline-panel run takes about two minutes with the wide sweep above.

Requires numpy, pandas, scipy and scikit-learn (matplotlib is optional).
Needs the localization and robustness modules in this package.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

import tml_smlm.view2_functional.localization as localization
import tml_smlm.view2_functional.robustness as robustness

ACTIVE_THRESH = 1e-6  # matches the localization module's own n_nonzero threshold exactly

try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe; this script never shows a window
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ══════════════════════════════════════════════════════════════════════════
# PANEL PREPARATION — generalizes the robustness module's own _prepare_panel to also
# handle a pooled (marker=None) panel, which the robustness module's HEADLINE_PANELS never
# needed (see module docstring). Everything else is identical to the robustness module's
# version, reusing the same localization primitives the same way.
# ══════════════════════════════════════════════════════════════════════════

def _prepare_panel(panel: Dict, fm: pd.DataFrame, diagrams_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    source, marker, scale, hom, arr_key, k = (
        panel["source"], panel.get("marker"), panel["scale"], panel["hom"], panel["key"], panel["k"],
    )
    sub = fm[fm["source"] == source].copy()
    if marker:
        sub = sub[sub["marker"] == marker].copy()
    dgms = [localization._load_dgm(diagrams_dir, row, arr_key) for _, row in sub.iterrows()]
    L, grid = localization.vectorise(dgms, k)
    covar_col = localization.SCALE_COVAR.get(scale, "n_localisations")
    covar = sub[covar_col].to_numpy(dtype=float)
    L_r = localization.residualise(L, covar)
    y = (sub["cell_type"] == localization.SOURCE_CFG[source]["pos"]).astype(int).to_numpy()
    return L_r, y, grid, sub


def _all_panels(fm: pd.DataFrame, sources: List[str], lambda_orders: List[int]) -> List[Dict]:
    """Every source x PANEL_DEFS x marker x lambda-order combination the localization module itself would produce."""
    panels: List[Dict] = []
    for source in sources:
        if source not in localization.SOURCE_CFG:
            logger.warning(f"Unknown source '{source}' -- skipping.")
            continue
        src_markers = sorted(fm[fm["source"] == source]["marker"].unique())
        for pdef in localization.PANEL_DEFS:
            markers_to_run = [None] if pdef["scale"] == "scaleB" else src_markers
            for marker in markers_to_run:
                for k in lambda_orders:
                    panels.append({
                        "source": source, "marker": marker, "scale": pdef["scale"], "hom": pdef["hom"],
                        "key": pdef["key"], "k": k,
                    })
    return panels


def _panel_label(panel: Dict) -> str:
    return f"{panel['source']}/{panel.get('marker') or 'pooled'}/{panel['key']}/lambda{panel['k']}"


# ══════════════════════════════════════════════════════════════════════════
# SHAPE DIAGNOSTIC — the actual question this script exists to answer
# ══════════════════════════════════════════════════════════════════════════

def _active_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous (start, end) index pairs, inclusive, where mask is True."""
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if start is not None and (not v or i == len(mask) - 1):
            end = i - 1 if not v else i
            runs.append((start, end))
            start = None
    return runs


def _hoyer_sparsity(d: np.ndarray) -> Optional[float]:
    """Hoyer (2004) sparsity measure of a vector's entries, in [0, 1]. None if undefined (m<2 or all-zero)."""
    m = len(d)
    if m < 2:
        return None
    l2 = float(np.sqrt(np.sum(d ** 2)))
    if l2 == 0.0:
        return None
    l1 = float(np.sum(np.abs(d)))
    return float((np.sqrt(m) - l1 / l2) / (np.sqrt(m) - 1))


def shape_diagnostic(beta: np.ndarray, thresh: float = ACTIVE_THRESH) -> Dict:
    active = np.abs(beta) > thresh
    n_active = int(active.sum())
    if n_active == 0:
        return {"n_active": 0, "n_active_blocks": 0, "verdict": "no active coefficients -- panel selected nothing at lam_min"}

    runs = _active_runs(active)
    run_sizes = [e - s + 1 for s, e in runs]
    largest = runs[int(np.argmax(run_sizes))]
    seg = beta[largest[0]:largest[1] + 1]
    active_vals = beta[active]
    amp = float(np.max(np.abs(active_vals)))
    seg_range = float(seg.max() - seg.min())
    normalized_range = (seg_range / amp) if amp > 0 else 0.0

    hoyer = _hoyer_sparsity(np.diff(seg)) if len(seg) >= 3 else None

    if normalized_range < 0.15 and (hoyer is None or hoyer > 0.4):
        verdict = "flat / piecewise-constant-like within its largest active block (favors a true L1-fusion penalty)"
    elif normalized_range > 0.4:
        verdict = "smoothly graded within its largest active block (consistent with the implemented L2 choice)"
    else:
        verdict = "ambiguous -- neither pattern clearly dominates"

    return {
        "n_active": n_active,
        "n_active_blocks": len(runs),
        "block_sizes": run_sizes,
        "largest_block_grid_span": [int(largest[0]), int(largest[1])],
        "largest_block_n": int(len(seg)),
        "largest_block_beta_range": round(seg_range, 6),
        "active_amplitude": round(amp, 6),
        "normalized_largest_block_range": round(normalized_range, 4),
        "hoyer_sparsity_of_block_diffs": round(hoyer, 4) if hoyer is not None else None,
        "verdict": verdict,
    }


def _plot_profile(label: str, beta: np.ndarray, grid: np.ndarray, out_dir: Path) -> Optional[str]:
    if not _HAS_MPL:
        return None
    try:
        fig, ax = plt.subplots(figsize=(8, 3))
        active = np.abs(beta) > ACTIVE_THRESH
        ax.plot(grid, beta, color="#888888", linewidth=0.8, label="beta (all grid points)")
        ax.scatter(grid[active], beta[active], color="#c0392b", s=14, zorder=3, label="active (|beta|>1e-6)")
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_xlabel("grid position (filtration value, nm)")
        ax.set_ylabel("beta (lam_min)")
        ax.set_title(label)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fname = out_dir / (label.replace("/", "_") + ".png")
        fig.savefig(fname, dpi=130)
        plt.close(fig)
        return str(fname)
    except Exception as e:
        logger.warning(f"  Plot failed for {label}: {e}")
        return None


def _plot_comparison(label: str, beta_l2: np.ndarray, beta_l1: np.ndarray, grid: np.ndarray, out_dir: Path) -> Optional[str]:
    if not _HAS_MPL:
        return None
    try:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(grid, beta_l2, color="#2166ac", linewidth=1.0, label="implemented (L1 + L2-squared)")
        ax.plot(grid, beta_l1, color="#b2182b", linewidth=1.0, label="true fused lasso (L1 + L1), same lambda")
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_xlabel("grid position (filtration value, nm)")
        ax.set_ylabel("beta")
        ax.set_title(f"{label} -- L2 vs. L1-fusion, same lambda")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fname = out_dir / (label.replace("/", "_") + "_l1fusion_compare.png")
        fig.savefig(fname, dpi=130)
        plt.close(fig)
        return str(fname)
    except Exception as e:
        logger.warning(f"  Comparison plot failed for {label}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# TRUE (L1 + L1) FUSED LASSO, VIA ADMM -- added for --compare-l1-fusion.
# Solves, exactly (not approximated the way the localization module's augmented-Lasso
# reformulation approximates its own L2-squared version):
#
#     minimize  (1/(2n)) ||y - X beta||^2  +  lam1 ||beta||_1  +  lam2 ||D beta||_1
#
# the standard penalized ("Lagrangian") form of Tibshirani et al. (2005)'s
# two-constraint fused lasso -- equivalent to their own constrained form up
# to a reparametrization of the two bounds s1, s2 into lam1, lam2, the same
# convention essentially every modern fused-lasso implementation (R's
# genlasso, Python's various generalized-lasso solvers) actually ships, not
# a second approximation layered on top of theirs. D is the localization module's own first-
# difference operator (`localization._D`), reused unchanged.
# ══════════════════════════════════════════════════════════════════════════

def _soft_threshold(v: np.ndarray, k: float) -> np.ndarray:
    return np.sign(v) * np.maximum(np.abs(v) - k, 0.0)


def fused_lasso_admm(
    X: np.ndarray, y: np.ndarray, lam1: float, lam2: float,
    rho0: float = 1.0, max_iter: int = 6000, tol: float = 1e-6,
    adapt_every: int = 25, tau: float = 2.0, mu_bal: float = 10.0,
    warm: Optional[Dict] = None,
) -> Tuple[np.ndarray, bool, int, Dict]:
    """
    ADMM for the generalized-lasso form of the fused lasso (consensus
    splitting on z = [lam1*I; lam2*D] @ beta). The step size is adapted by
    residual balancing (Boyd et al. 2011, Sec. 3.4.1). This is required: a
    fixed step size (rho = max(lam1, lam2, 1)) is badly scaled once the
    penalty weights grow, and at the largest headline-panel lambda it left
    beta up to ~80% off an independent reference solution, while the
    adaptive step agreed to 2e-7 or better at every lambda tested (see the
    module docstring).

    Returns (beta, converged, n_iter, state). `converged=False` after
    max_iter is logged as a warning by the caller, never silently accepted
    -- matching this project's standing convergence-disclosure convention
    (the localization module's own docstring). `state` (a dict of the final
    z, u, rho) can be passed back in as `warm=` to continue from this
    solution at a nearby lambda, the same warm-starting principle the localization module's
    own `_fl_path_fit` already uses for its own lambda path -- used by
    `_sparsity_matched_fused_lasso` below to walk a lambda-multiplier path
    without refitting each point from a cold start.
    """
    n, p = X.shape
    D = localization._D(p)
    I = np.eye(p)
    mu1, mu2 = n * lam1, n * lam2
    Dt = np.vstack([mu1 * I, mu2 * D]) if mu2 > 0 else mu1 * I
    XtX = X.T @ X
    Xty = X.T @ y.astype(float)
    DtD = Dt.T @ Dt

    rho = rho0 if warm is None else warm.get("rho", rho0)
    M = XtX + rho * DtD
    c_and_lower = cho_factor(M)
    m = Dt.shape[0]
    if warm is not None and warm.get("z") is not None and len(warm["z"]) == m:
        z = warm["z"].copy()
        u = warm["u"].copy()
    else:
        z = np.zeros(m)
        u = np.zeros(m)
    beta = np.zeros(p)
    converged = False
    n_iter = max_iter

    for it in range(max_iter):
        rhs = Xty + rho * Dt.T @ (z - u)
        beta = cho_solve(c_and_lower, rhs)
        Dtb = Dt @ beta
        z_new = _soft_threshold(Dtb + u, 1.0 / rho)
        r = Dtb - z_new               # primal residual
        s = rho * (Dt.T @ (z_new - z))  # dual residual
        z = z_new
        u = u + r

        pr, dr = np.linalg.norm(r), np.linalg.norm(s)
        if pr < tol * max(1.0, np.linalg.norm(Dtb), np.linalg.norm(z)) and dr < tol * max(1.0, np.linalg.norm(rho * u)):
            converged = True
            n_iter = it + 1
            break

        if (it + 1) % adapt_every == 0:
            if pr > mu_bal * dr:
                rho *= tau; u /= tau
                M = XtX + rho * DtD; c_and_lower = cho_factor(M)
            elif dr > mu_bal * pr:
                rho /= tau; u *= tau
                M = XtX + rho * DtD; c_and_lower = cho_factor(M)

    return beta, converged, n_iter, {"z": z, "u": u, "rho": rho}


# ══════════════════════════════════════════════════════════════════════════
# SPARSITY-MATCHED L1-FUSION COMPARISON
# --------------------------------------------------------------------------
# Reusing the localization module's own lam_min as lam1 = lam2 for the true
# fused lasso does NOT give it a comparable sparsity level: on a synthetic
# problem of the paper's size it leaves 70-90 of 300 coefficients active where
# the smooth-lasso model selects 2-5, because an L2-squared penalty
# coefficient and an L1 penalty coefficient are different units with no
# exact equivalence (the ADMM solver itself is validated, see module
# docstring). This sweep therefore asks the more useful question: at what fusion
# strength (scaling lam1=lam2 together, the same single-coupled-parameter
# philosophy the localization module itself uses) does a true fused lasso reach a sparsity
# level comparable to our own, and what does it select there? Some panels
# may show a smooth transition; others show a cliff (the active count jumps
# from dozens straight to zero between two adjacent tested multipliers, with
# nothing in between). A cliff is reported as such and not forced into a
# false match.
# ══════════════════════════════════════════════════════════════════════════

def _sparsity_matched_fused_lasso(
    X: np.ndarray, y: np.ndarray, lam_base: float, target_n_active: int,
    n_points: int = 14, lo_mult: float = 0.5, hi_mult: float = 300.0,
    max_iter: int = 3000, tol: float = 1e-5,
) -> Dict:
    """
    Walks a descending (sparsest-first, matching the localization module's own largest-to-
    smallest lambda-path convention) grid of multipliers c in
    [hi_mult, lo_mult] * lam_base, warm-starting each ADMM solve from the
    previous multiplier's solution. Returns the full sweep plus the point
    closest to target_n_active, and flags explicitly whether that point is
    a reasonably close match or only the nearer edge of a cliff.
    """
    mult_grid = np.geomspace(hi_mult, lo_mult, n_points)
    sweep: List[Dict] = []
    warm = None
    best = None
    best_gap = None
    for c in mult_grid:
        lam = c * lam_base
        beta, converged, n_iter, warm = fused_lasso_admm(X, y, lam, lam, max_iter=max_iter, tol=tol, warm=warm)
        na = int((np.abs(beta) > ACTIVE_THRESH).sum())
        entry = {"multiplier": float(c), "lambda": float(lam), "n_active": na, "converged": bool(converged), "beta": beta}
        sweep.append(entry)
        gap = abs(na - target_n_active)
        if best_gap is None or gap < best_gap:
            best, best_gap = entry, gap

    # Bracketing points around the target, for an honest "was this a clean match or a cliff" read.
    # (stripped of their own beta vectors -- only `best` carries its beta onward, the only one the
    # caller actually needs for the shape/distance comparison.)
    def _strip_beta(e: Optional[Dict]) -> Optional[Dict]:
        return None if e is None else {k: v for k, v in e.items() if k != "beta"}

    below = [e for e in sweep if e["n_active"] <= target_n_active]
    above = [e for e in sweep if e["n_active"] > target_n_active]
    nearest_below = _strip_beta(max(below, key=lambda e: e["n_active"]) if below else None)
    nearest_above = _strip_beta(min(above, key=lambda e: e["n_active"]) if above else None)
    clean_match = best_gap <= max(1, int(0.5 * max(target_n_active, 1)))

    return {
        "target_n_active": target_n_active, "best": best, "achieved_gap": best_gap, "clean_match": clean_match,
        "nearest_below": nearest_below, "nearest_above": nearest_above,
        "sweep_summary": [{"multiplier": e["multiplier"], "lambda": e["lambda"], "n_active": e["n_active"], "converged": e["converged"]} for e in sweep],
    }


def _plot_sparsity_sweep(label: str, sweep_result: Dict, out_dir: Path) -> Optional[str]:
    if not _HAS_MPL:
        return None
    try:
        s = sweep_result["sweep_summary"]
        mults = [e["multiplier"] for e in s]
        nas = [e["n_active"] for e in s]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(mults, nas, marker="o", markersize=3, color="#444444", linewidth=1.0)
        ax.axhline(sweep_result["target_n_active"], color="#2166ac", linestyle="--", linewidth=1.0, label=f"L2 target ({sweep_result['target_n_active']})")
        if sweep_result["best"] is not None:
            ax.scatter([sweep_result["best"]["multiplier"]], [sweep_result["best"]["n_active"]], color="#b2182b", s=30, zorder=3, label="closest L1-fusion match")
        ax.set_xscale("log")
        ax.set_xlabel("lambda multiplier c  (lam1 = lam2 = c * the localization module's lam_min)")
        ax.set_ylabel("n_active (L1-fusion)")
        ax.set_title(f"{label} -- L1-fusion sparsity vs. multiplier")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fname = out_dir / (label.replace("/", "_") + "_l1fusion_sweep.png")
        fig.savefig(fname, dpi=130)
        plt.close(fig)
        return str(fname)
    except Exception as e:
        logger.warning(f"  Sweep plot failed for {label}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diagrams", type=Path, default=Path("diagrams"))
    ap.add_argument("--features", type=Path, default=Path("data/feature_matrix.csv"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--sources", nargs="+", default=list(localization.SOURCE_CFG.keys()))
    ap.add_argument("--lambda-orders", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--lambda-which", choices=["min", "1se"], default="min",
                     help="Which of the localization module's selected lambdas to refit at (default: lam_min, the value the paper reports).")
    ap.add_argument("--all-panels", action="store_true",
                     help="Sweep every source x panel-def x marker x lambda-order combination, not just the three headline panels.")
    ap.add_argument("--plots", dest="plots", action="store_true", default=True)
    ap.add_argument("--no-plots", dest="plots", action="store_false")
    ap.add_argument("--compare-l1-fusion", action="store_true",
                     help="Additionally fit a true (L1+L1) fused lasso via ADMM at the same lambda, and report "
                          "its own shape diagnostic plus the L2-vs-L1-fusion beta distance and active-set Jaccard overlap.")
    ap.add_argument("--l1-fusion-max-iter", type=int, default=6000)
    ap.add_argument("--l1-fusion-tol", type=float, default=1e-6)
    ap.add_argument("--sweep-lo-mult", type=float, default=0.5,
                     help="Sparsity-matched sweep: smallest lambda multiplier tested (default 0.5).")
    ap.add_argument("--sweep-hi-mult", type=float, default=300.0,
                     help="Sparsity-matched sweep: largest lambda multiplier tested (default 300). Widen this if "
                          "the real archive's panels report clean_match=False with the sparsest tested point "
                          "(mult=sweep-hi-mult) still less sparse than the L2 target -- that means the true "
                          "transition lies beyond this script's default search range, not that none exists.")
    ap.add_argument("--sweep-points", type=int, default=14,
                     help="Sparsity-matched sweep: number of log-spaced multipliers tested (default 14).")
    args = ap.parse_args()

    for p in [args.features, args.diagrams]:
        if not p.exists():
            logger.error(f"Not found: {p}")
            return 1
    args.results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.results_dir / "penalty_comparison_profiles"
    if args.plots and _HAS_MPL:
        plots_dir.mkdir(parents=True, exist_ok=True)
    elif args.plots and not _HAS_MPL:
        logger.warning("matplotlib not available -- skipping all plots (JSON/summary output is unaffected).")

    fm = pd.read_csv(args.features, dtype={"replicate": str})
    if "npz_file" not in fm.columns:
        fm["npz_file"] = fm.apply(lambda r: localization._npz_stem(r) + ".npz", axis=1)

    logger.info("=" * 70)
    logger.info("View 2 penalty comparison: smooth-lasso vs. true fused lasso")
    logger.info("=" * 70)

    if args.all_panels:
        panels = _all_panels(fm, args.sources, args.lambda_orders)
        logger.info(f"Mode: --all-panels ({len(panels)} panels)")
    else:
        panels = [
            {"source": p["source"], "marker": p["marker"], "scale": p["scale"], "hom": p["hom"], "key": p["key"], "k": p["k"]}
            for p in robustness.HEADLINE_PANELS
        ]
        logger.info(f"Mode: headline panels only ({len(panels)} panels, from the robustness module's HEADLINE_PANELS)")
    logger.info(f"lambda-which: lam_{args.lambda_which}")

    results: List[Dict] = []
    for panel in panels:
        lbl = _panel_label(panel)
        try:
            L_r, y, grid, sub = _prepare_panel(panel, fm, args.diagrams)
            if len(sub) < 20 or y.sum() < 5 or (len(y) - y.sum()) < 5:
                logger.info(f"  SKIP {lbl}: n={len(sub)}, n_pos={int(y.sum())}, n_neg={int(len(y) - y.sum())}")
                results.append({"label": lbl, "skipped": True, "reason": f"n={len(sub)}"})
                continue

            fl = localization.fl_cv(L_r, y)
            lam_key = "lam_min" if args.lambda_which == "min" else "lam_1se"
            beta_key = "beta_lam_min" if args.lambda_which == "min" else "beta_lam_1se"
            beta = np.asarray(fl[beta_key])
            diag = shape_diagnostic(beta)

            plot_path = _plot_profile(lbl, beta, grid, plots_dir) if args.plots else None

            logger.info(f"  {lbl}: n_active={diag['n_active']} n_blocks={diag.get('n_active_blocks')} -> {diag['verdict']}")

            entry = {
                "label": lbl, "source": panel["source"], "marker": panel.get("marker"), "scale": panel["scale"],
                "hom": panel["hom"], "k_order": panel["k"], "lambda_which": args.lambda_which,
                f"{lam_key}": fl[lam_key], "ba_at_this_lambda": fl.get(f"ba_at_{lam_key}"),
                "shape_diagnostic": diag, "plot_path": plot_path,
            }

            if args.compare_l1_fusion:
                active_l2 = set(np.where(np.abs(beta) > ACTIVE_THRESH)[0].tolist())
                lam = fl[lam_key]

                # (1) Same nominal lambda -- fast, exact, but NOT sparsity-matched (see module docstring
                # and the block comment above _sparsity_matched_fused_lasso): report it as what it is,
                # a same-nominal-penalty-coefficient comparison, not a same-sparsity one.
                beta_l1, converged, n_iter, _ = fused_lasso_admm(
                    L_r, y, lam, lam, max_iter=args.l1_fusion_max_iter, tol=args.l1_fusion_tol,
                )
                if not converged:
                    logger.warning(f"  [L1-fusion/same-lambda] {lbl}: ADMM did NOT converge within "
                                    f"{args.l1_fusion_max_iter} iterations (tol={args.l1_fusion_tol}) -- "
                                    f"result reported as-is, not discarded, but treat with more caution.")
                diag_l1 = shape_diagnostic(beta_l1)
                dist = float(np.linalg.norm(beta_l1 - beta))
                dist_rel = dist / (float(np.linalg.norm(beta)) + 1e-12)
                active_l1 = set(np.where(np.abs(beta_l1) > ACTIVE_THRESH)[0].tolist())
                jac = robustness._jaccard(active_l2, active_l1)
                cmp_plot = _plot_comparison(lbl, beta, beta_l1, grid, plots_dir) if args.plots else None

                logger.info(
                    f"  [L1-fusion/same-lambda] {lbl}: converged={converged} iters={n_iter} "
                    f"n_active_l1={diag_l1['n_active']} (vs. L2's {diag['n_active']}) jaccard={jac:.4f} "
                    f"||beta_l1-beta_l2||={dist:.4g} (rel={dist_rel:.4g}) -> {diag_l1['verdict']}"
                )

                # (2) Sparsity-matched -- scale lam1=lam2 together (same single-coupled-parameter
                # philosophy the localization module itself uses) until the L1-fusion active count is as close as this
                # script's tested range gets to the L2 model's own active count. See the block comment
                # above _sparsity_matched_fused_lasso for why this is the more meaningful comparison.
                sweep = _sparsity_matched_fused_lasso(
                    L_r, y, lam, diag["n_active"],
                    n_points=args.sweep_points, lo_mult=args.sweep_lo_mult, hi_mult=args.sweep_hi_mult,
                )
                best = sweep["best"]
                beta_sm = best["beta"]
                diag_sm = shape_diagnostic(beta_sm)
                dist_sm = float(np.linalg.norm(beta_sm - beta))
                dist_sm_rel = dist_sm / (float(np.linalg.norm(beta)) + 1e-12)
                active_sm = set(np.where(np.abs(beta_sm) > ACTIVE_THRESH)[0].tolist())
                jac_sm = robustness._jaccard(active_l2, active_sm)
                sweep_plot = _plot_sparsity_sweep(lbl, sweep, plots_dir) if args.plots else None
                cmp_sm_plot = _plot_comparison(lbl + "_sparsity_matched", beta, beta_sm, grid, plots_dir) if args.plots else None

                logger.info(
                    f"  [L1-fusion/sparsity-matched] {lbl}: best mult={best['multiplier']:.3g} "
                    f"n_active_l1={diag_sm['n_active']} (target {diag['n_active']}, gap={sweep['achieved_gap']}, "
                    f"clean_match={sweep['clean_match']}) jaccard={jac_sm:.4f} "
                    f"||beta_l1-beta_l2||={dist_sm:.4g} (rel={dist_sm_rel:.4g}) -> {diag_sm['verdict']}"
                )
                if not sweep["clean_match"]:
                    logger.warning(f"  [L1-fusion/sparsity-matched] {lbl}: no tested multiplier reached a sparsity "
                                    f"level close to the L2 model's own ({diag['n_active']} active) -- see "
                                    f"sweep_summary in the JSON output for the full multiplier path.")
                    if best["multiplier"] >= args.sweep_hi_mult * 0.999:
                        logger.warning(f"  [L1-fusion/sparsity-matched] {lbl}: the closest point found sits at this "
                                        f"run's own sparsest tested multiplier ({args.sweep_hi_mult}) -- the true "
                                        f"transition may lie beyond the tested range entirely. Rerun with a larger "
                                        f"--sweep-hi-mult before treating 'no clean match' as a real finding rather "
                                        f"than a search-range limitation.")

                entry["l1_fusion_comparison"] = {
                    "same_lambda": {
                        "lambda_used": lam,
                        "note": "lam1=lam2=the same lambda the localization module's own CV already selected for this panel. This is "
                                "a same-nominal-penalty-coefficient comparison, NOT a same-sparsity one -- an L2-"
                                "squared coefficient and an L1 coefficient are different units with no exact "
                                "equivalence, which is why n_active can differ a great deal here even though both "
                                "fits use the same lambda number.",
                        "converged": converged, "n_iter": n_iter, "shape_diagnostic": diag_l1,
                        "beta_distance_l2_norm": round(dist, 6), "beta_distance_relative_to_l2_norm": round(dist_rel, 6),
                        "active_set_jaccard_vs_l2": round(jac, 4),
                        "n_active_l1_fusion": diag_l1["n_active"], "n_active_l2": diag["n_active"],
                        "comparison_plot_path": cmp_plot,
                    },
                    "sparsity_matched": {
                        "note": "lam1=lam2 scaled together by a single multiplier, swept over a wide log-spaced grid "
                                "(see module docstring), to the value whose resulting active-coefficient count is "
                                "closest to the L2 model's own -- the more meaningful of the two comparisons for "
                                "asking what a true fused lasso would actually select at comparable sparsity.",
                        "multiplier_used": best["multiplier"], "lambda_used": best["lambda"],
                        "converged": best["converged"], "target_n_active": diag["n_active"],
                        "achieved_n_active": diag_sm["n_active"], "achieved_gap": sweep["achieved_gap"],
                        "clean_match": sweep["clean_match"],
                        "nearest_below": sweep["nearest_below"], "nearest_above": sweep["nearest_above"],
                        "shape_diagnostic": diag_sm,
                        "beta_distance_l2_norm": round(dist_sm, 6), "beta_distance_relative_to_l2_norm": round(dist_sm_rel, 6),
                        "active_set_jaccard_vs_l2": round(jac_sm, 4),
                        "sweep_summary": sweep["sweep_summary"],
                        "sweep_plot_path": sweep_plot, "comparison_plot_path": cmp_sm_plot,
                    },
                }

            results.append(entry)
        except Exception as e:
            logger.error(f"FAILED {lbl}: {e}")
            results.append({"label": lbl, "failed": True, "reason": str(e)})

    json_path = args.results_dir / "penalty_comparison.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"\nResults -> {json_path}")

    txt_path = args.results_dir / "penalty_comparison_summary.txt"
    with open(txt_path, "w") as fh:
        fh.write("Penalty comparison (smooth-lasso vs. true fused lasso) -- summary\n" + "=" * 70 + "\n\n")
        for r in results:
            if r.get("skipped") or r.get("failed"):
                fh.write(f"{r['label']}: {'skipped' if r.get('skipped') else 'FAILED'} ({r.get('reason')})\n")
                continue
            d = r["shape_diagnostic"]
            fh.write(
                f"{r['label']}: n_active={d.get('n_active')} n_blocks={d.get('n_active_blocks')} "
                f"norm_range={d.get('normalized_largest_block_range')} hoyer={d.get('hoyer_sparsity_of_block_diffs')}\n"
                f"    -> {d.get('verdict')}\n"
            )
            cmp = r.get("l1_fusion_comparison")
            if cmp is not None:
                sl = cmp["same_lambda"]
                dl1 = sl["shape_diagnostic"]
                fh.write(
                    f"    [L1-fusion, same lambda] converged={sl['converged']} n_active={sl['n_active_l1_fusion']} "
                    f"(vs. L2's {sl['n_active_l2']}) jaccard_vs_L2={sl['active_set_jaccard_vs_l2']} "
                    f"||beta_l1-beta_l2||={sl['beta_distance_l2_norm']} (rel={sl['beta_distance_relative_to_l2_norm']})\n"
                    f"      -> {dl1.get('verdict')}\n"
                )
                sm = cmp["sparsity_matched"]
                dsm = sm["shape_diagnostic"]
                fh.write(
                    f"    [L1-fusion, sparsity-matched] mult={sm['multiplier_used']:.3g} converged={sm['converged']} "
                    f"n_active={sm['achieved_n_active']} (target {sm['target_n_active']}, gap={sm['achieved_gap']}, "
                    f"clean_match={sm['clean_match']}) jaccard_vs_L2={sm['active_set_jaccard_vs_l2']} "
                    f"||beta_l1-beta_l2||={sm['beta_distance_l2_norm']} (rel={sm['beta_distance_relative_to_l2_norm']})\n"
                    f"      -> {dsm.get('verdict')}\n"
                )
    logger.info(f"Summary -> {txt_path}")

    logger.info("=" * 70)
    logger.info("Penalty comparison complete.")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
