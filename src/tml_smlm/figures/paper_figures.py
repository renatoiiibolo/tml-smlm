"""
Reproduces the paper's main figures (2-5) from each analysis module's
output: one function per figure, reading the JSON each upstream module
writes to results/ and rendering it with matplotlib. This is a simplified,
illustrative version -- it shows the real data transformations and the
real plotting choices behind each figure, but is not the exact
camera-ready figure submitted to the journal (no EPS export, no
journal-specific font/sizing compliance).

Figure 2  Population-shape separation by genotype (View 3: Frechet mean /
          Wasserstein medoid distance), raw vs. residualized against each
          nucleus's own localization count, per timepoint.
Figure 3  Within-nucleus cross-marker coupling, cancer vs. normal -- the
          paper's central claim. Panel (a)/(b): View 3 (whole-diagram
          Wasserstein displacement from control). Panel (c): the same
          question asked by View 2 (persistence-landscape value at one
          confirmed grid position), as an independent check.
Figure 4  Leading discriminative descriptor per stratum, over the repair
          timecourse -- each panel plots whichever tested feature is that
          stratum's own strongest, most defensible descriptor, rather than
          one statistic forced across all four panels.
Figure 5  Does the spatial scale that discriminates cancer from normal
          shift with time? Selected persistence-landscape grid position
          (converted to nm) plotted per timepoint, for three headline
          stratum/marker combinations.

Expects, under --basedir/results/:
  temporal_structure_results.json     (view1_scalar/temporal_structure.py)
  view2_localization_fl_results.json  (view2_functional/localization.py)
  population_geometry_results.json    (view3_metric/population_geometry.py)
  view3_cross_marker_coupling.json    (view3_metric/coupling.py)
  temporal_trajectory_position.json   (view2_functional/temporal_trajectory.py)
  view2_position_coupling.json        (view2_functional/coupling.py)

Usage:
  python -m tml_smlm.figures.paper_figures --basedir . --only 3,4
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# STYLE
# ==============================================================================

DPI = 300

TICK_SIZE = 12
LEGEND_SIZE = 12
FONT_SIZE = 13
LABEL_SIZE = 14
PANEL_LABEL_SIZE = 19

WIDE_3 = (13.0, 7.5)          # 1x3 layout (Fig. 5)
GRID_2x2 = (12.0, 10.5)       # 2x2 layout (Figs. 2, 4)
FIG3_SIZE = (13.0, 9.5)       # Fig. 3, three panels in a 2x2-ish gridspec

PALETTE = {
    "navy": "#1D4E63", "teal_mid": "#357A83", "cyan": "#0099AF",
    "rust": "#B2500E", "amber": "#B97608", "charcoal": "#323232",
}

# Genotype is the axis that recurs across Figs. 3-5: cool for normal, warm
# for cancer. Marker gets its own independent color channel (cyan/teal/amber)
# so genotype and marker can both be read off the same plot at once.
GENOTYPE_COLOR = {"normal": PALETTE["navy"], "cancer": PALETTE["rust"]}
GENOTYPE_LABEL = {"normal": "Normal", "cancer": "Cancer"}
CELL_TYPE_GENOTYPE = {"NHDF": "normal", "U87": "cancer", "HGF": "normal", "MCF7": "cancer"}
CELL_TYPE_LABEL = {"NHDF": "NHDF", "U87": "U87", "HGF": "CCD-1059SK", "MCF7": "MCF-7"}

MARKER_COLOR = {"yH2AX": PALETTE["cyan"], "53BP1": PALETTE["teal_mid"], "Mre11": PALETTE["amber"]}
MARKER_LABEL = {"yH2AX": "γH2AX", "53BP1": "53BP1", "Mre11": "MRE11"}

# Dataset is distinguished by marker shape, not color, so color stays free
# for genotype/marker: circles for Kuentzelmann (heavy-ion), squares for
# Hahn (photon).
DATASET_MARKER_SHAPE = {"kuentzelmann": "o", "hahn": "s"}
DATASET_LABEL = {
    "kuentzelmann": "Küntzelmann et al. (2026), heavy-ion",
    "hahn": "Hahn et al. (2021), photon",
}

SIG_COLOR = PALETTE["navy"]
NS_COLOR = "#555555"
GRID_COLOR = "#EBEBEB"
SPINE_COLOR = "#BBBBBB"
TICK_COLOR = "#555555"
TEXT_COLOR = PALETTE["charcoal"]

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "text.color": TEXT_COLOR,
    "axes.labelsize": LABEL_SIZE,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "xtick.color": TICK_COLOR,
    "ytick.color": TICK_COLOR,
    "legend.fontsize": LEGEND_SIZE,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "#CCCCCC",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "axes.edgecolor": SPINE_COLOR,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": DPI,
})


# ==============================================================================
# HELPERS
# ==============================================================================

def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(SPINE_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=TICK_COLOR, length=4.0, width=0.8, labelsize=TICK_SIZE)


def _panel_label(ax: plt.Axes, label: str, x: float = -0.20, y: float = 1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE,
             fontweight="bold", color=TEXT_COLOR, va="bottom", ha="right")


def _safe_corner_label(ax: plt.Axes, text: str, xs, ys, log_x: bool = True) -> None:
    """Place a short label in whichever axes corner has no data point near it.

    Works in the axes' actual rendered coordinate system (get_xlim/get_ylim
    called after the data is plotted), not a fixed fraction of the raw data
    range -- a data point can sit near an axis edge in rendered space even
    if it isn't near the edge of the raw value range.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if log_x:
        x0, x1 = np.log10(xlim[0]), np.log10(xlim[1])
        fx = (np.log10(xs) - x0) / (x1 - x0)
    else:
        fx = (xs - xlim[0]) / (xlim[1] - xlim[0])
    fy = (ys - ylim[0]) / (ylim[1] - ylim[0])
    mx, my = 0.40, 0.18  # label footprint: wider than tall
    left, right = fx <= mx, fx >= (1 - mx)
    top, bottom = fy >= (1 - my), fy <= my
    corners = [
        (0.04, 0.96, "left", "top", not np.any(left & top)),
        (0.04, 0.04, "left", "bottom", not np.any(left & bottom)),
        (0.96, 0.96, "right", "top", not np.any(right & top)),
        (0.96, 0.04, "right", "bottom", not np.any(right & bottom)),
    ]
    safe = [c for c in corners if c[-1]]
    if safe:
        lx, ly, ha, va, _ = safe[0]
    else:
        masks = [left & top, left & bottom, right & top, right & bottom]
        counts = [int(np.sum(mm)) for mm in masks]
        best = int(np.argmin(counts))
        lx, ly, ha, va, _ = corners[best]
        logger.warning(f"[_safe_corner_label] no fully clear corner for "
                        f"'{text.splitlines()[0]}...' -- using least-crowded corner.")
    ax.text(lx, ly, text, transform=ax.transAxes, ha=ha, va=va, fontsize=TICK_SIZE,
             color=TEXT_COLOR, bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=3), zorder=6)


def _legend_below(ax: plt.Axes, handles=None, labels=None, ncol: int = 2,
                   title: Optional[str] = None, y_offset: float = -0.30):
    kwargs = dict(loc="upper center", bbox_to_anchor=(0.5, y_offset),
                  borderaxespad=0.0, framealpha=1.0, edgecolor="#CCCCCC",
                  fontsize=LEGEND_SIZE, ncol=ncol)
    if title is not None:
        kwargs["title"] = title
    if handles is not None:
        return ax.legend(handles=handles, labels=labels, **kwargs)
    return ax.legend(**kwargs)


def _savefig(fig: plt.Figure, stem: Path) -> None:
    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(0.8)
    png_path = stem.with_suffix(".png")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    logger.info(f"  PNG -> {png_path.name}")
    plt.close(fig)


def _write_notes(out_dir: Path, stem: str, text: str) -> None:
    path = out_dir / f"{stem}_notes.txt"
    with open(path, "w") as fh:
        fh.write(text.strip() + "\n")
    logger.info(f"  Notes -> {path.name}")


def _fmt_p(p: float) -> str:
    """p-value annotation. Some of these underflow to exactly 0.0 in double
    precision; report those as an explicit upper bound rather than '0.0'."""
    if p <= 0.0:
        return "p < 1e-300"
    if p < 0.001:
        return f"p = {p:.1e}"
    return f"p = {p:.3f}"


def holm_bonferroni(pvals: List[Tuple[str, float]]) -> Dict[str, float]:
    """Standard Holm step-down correction. Returns {label: holm_adjusted_p}."""
    order = sorted(pvals, key=lambda x: x[1])
    m = len(order)
    adjusted: Dict[str, float] = {}
    running_max = 0.0
    for i, (label, p) in enumerate(order):
        val = min((m - i) * p, 1.0)
        running_max = max(running_max, val)
        adjusted[label] = running_max
    return adjusted


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_all(base_dir: Path) -> Dict:
    json_paths = {
        "temporal_structure":     base_dir / "results" / "temporal_structure_results.json",
        "localization_fl":        base_dir / "results" / "view2_localization_fl_results.json",
        "population_geometry":    base_dir / "results" / "population_geometry_results.json",
        "cross_marker_coupling":  base_dir / "results" / "view3_cross_marker_coupling.json",
        "temporal_trajectory":    base_dir / "results" / "temporal_trajectory_position.json",
        "position_coupling":      base_dir / "results" / "view2_position_coupling.json",
    }
    d: Dict = {}
    for key, p in json_paths.items():
        if p.exists():
            with open(p) as fh:
                d[key] = json.load(fh)
            logger.info(f"  loaded {p}")
        else:
            d[key] = None
            logger.warning(f"  MISSING: {p}")
    return d


# ==============================================================================
# FIG. 2 -- population-shape separation by genotype (View 3)
# ==============================================================================
# Asks: is the population-level SHAPE of the persistence-diagram set
# distinguishable between genotypes, and is that separation stable across
# the repair timecourse or an artifact of localization count? Both the raw
# and count-residualized tiers are genuine permutation tests (own p-value
# each), so both are plotted as real lines rather than one test plus a
# reference line.

FIG2_PANELS = [
    ("kuentzelmann", "53BP1"),
    ("kuentzelmann", "yH2AX"),
    ("hahn", "Mre11"),
    ("hahn", "yH2AX"),
]


def make_fig2(d: Dict, out_dir: Path) -> None:
    if d.get("population_geometry") is None:
        logger.warning("[Fig.2] population_geometry_results.json not found -- skipping.")
        return
    secA = d["population_geometry"]["section_A_population_geometry"]

    fig, axes = plt.subplots(2, 2, figsize=GRID_2x2, gridspec_kw={"hspace": 0.55, "wspace": 0.3})
    axes = axes.flatten()

    RAW_COLOR, RESID_COLOR = NS_COLOR, SIG_COLOR
    survive_counts = {}

    for ax, (src, marker) in zip(axes, FIG2_PANELS):
        entry = secA.get(src, {}).get(marker)
        if entry is None:
            ax.text(0.5, 0.5, f"{src}/{marker}\nnot found", ha="center", va="center",
                     transform=ax.transAxes, color=NS_COLOR)
            continue
        t3 = entry["t3_structure_by_timepoint"]
        tps = sorted(t3.keys(), key=lambda t: float(t.rstrip("h")))
        x = [float(tp.rstrip("h")) for tp in tps]

        n_survive = 0
        for tier_key, color, ls in [("raw", RAW_COLOR, "--"),
                                     ("residualized_against_n_localisations", RESID_COLOR, "-")]:
            y = [t3[tp][tier_key]["observed_w2_medoid_to_medoid"] for tp in tps]
            p = [t3[tp][tier_key]["p_value_floor_corrected"] for tp in tps]
            ax.plot(x, y, color=color, lw=1.8, ls=ls, zorder=2, solid_capstyle="round")
            sig = [pi < 0.05 for pi in p]
            ax.scatter([xi for xi, s in zip(x, sig) if s], [yi for yi, s in zip(y, sig) if s],
                       marker=DATASET_MARKER_SHAPE[src], s=80, facecolor=color,
                       edgecolor="white", linewidth=1.0, zorder=5)
            ax.scatter([xi for xi, s in zip(x, sig) if not s], [yi for yi, s in zip(y, sig) if not s],
                       marker=DATASET_MARKER_SHAPE[src], s=80, facecolor="white",
                       edgecolor=color, linewidth=1.8, zorder=5)
            if tier_key != "raw":
                n_survive = sum(sig)
        survive_counts[(src, marker)] = (n_survive, len(tps))

        ax.axhline(0, color=SPINE_COLOR, lw=1.0, zorder=1)
        ax.set_xscale("log")
        ax.set_xlabel("Time post-irradiation (h)")
        _style_ax(ax)
        ax.set_title(f"{DATASET_LABEL[src].split(',')[0]}, {MARKER_LABEL[marker]}",
                     fontsize=TICK_SIZE, color=TEXT_COLOR, pad=8)

    axes[0].set_ylabel("Population-shape separation\n(W2 medoid distance)")
    axes[2].set_ylabel("Population-shape separation\n(W2 medoid distance)")

    for i, ax in enumerate(axes):
        _panel_label(ax, f"({chr(97 + i)})")

    handles = [
        Line2D([0], [0], color=RAW_COLOR, marker="o", ms=9, lw=1.8, ls="--", label="Raw"),
        Line2D([0], [0], color=RESID_COLOR, marker="o", ms=9, lw=1.8, label="Residualized (audited)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TEXT_COLOR, markeredgecolor="white",
               ms=9, label="filled: p < 0.05"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=TEXT_COLOR,
               markeredgewidth=1.8, ms=9, label="open: not significant"),
    ]
    handles += [Line2D([0], [0], marker=DATASET_MARKER_SHAPE[s], color="w", markerfacecolor=TEXT_COLOR,
                       markeredgecolor="white", ms=10, label=DATASET_LABEL[s]) for s in ("kuentzelmann", "hahn")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=3, framealpha=1.0, edgecolor="#CCCCCC", fontsize=LEGEND_SIZE)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.13, hspace=0.55, wspace=0.3)
    _savefig(fig, out_dir / "fig2_population_shape")

    surv_bits = []
    for label, (src, marker) in zip("abcd", FIG2_PANELS):
        sv = survive_counts.get((src, marker))
        if sv:
            surv_bits.append(f"({label}) {sv[0]} of {sv[1]} timepoints")
    surv_sentence = "; ".join(surv_bits) + " survive residualization at p<0.05."

    _write_notes(out_dir, "fig2_population_shape",
        "View 3 (Frechet mean / Wasserstein) population-shape separation between the cancer and normal "
        "cell line, per timepoint: Wasserstein-2 distance between each genotype's own Frechet-medoid "
        "persistence diagram, both raw and residualized against each nucleus's own localization count "
        "(permutation-null p-values, 1000 permutations per test; filled markers p<0.05, open markers "
        "not significant). The residualized tier is what actually resolves the question -- raw "
        "separation is significant at nearly every timepoint in all four strata, but audit reveals this "
        "is substantially count-confounded in both Kuentzelmann strata: " + surv_sentence)


# ==============================================================================
# FIG. 3 -- the hero figure: within-nucleus cross-marker coupling
# ==============================================================================

def make_fig3(d: Dict, out_dir: Path) -> None:
    if d.get("cross_marker_coupling") is None:
        logger.warning("[Fig.3] view3_cross_marker_coupling.json not found -- skipping.")
        return

    fig = plt.figure(figsize=FIG3_SIZE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.85, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # ── (a) within-nucleus coupling, cancer vs. normal, both archives ────────
    # Strictest residualization tier (timepoint + localization count): this
    # is the one claim the paper stakes new ground on, so the audited tier
    # is the point here, not an imposition.
    strata = [("kuentzelmann", "NHDF"), ("kuentzelmann", "U87"), ("hahn", "HGF"), ("hahn", "MCF7")]
    res = d["cross_marker_coupling"]["results"]
    x = np.arange(len(strata))
    any_ns = False
    for xi, (src, ct) in zip(x, strata):
        entry = res[src][ct]["primary_residualized_test"]
        rho = entry["residual_with_count_spearman_rho"]
        p = entry["residual_with_count_spearman_p"]
        genotype = CELL_TYPE_GENOTYPE[ct]
        col = GENOTYPE_COLOR[genotype]
        if p >= 0.05:
            any_ns = True
        ax_a.bar([xi], [rho], width=0.6, color=col, zorder=3,
                 edgecolor="white" if p < 0.05 else NS_COLOR,
                 linewidth=2.2 if p >= 0.05 else 0.0)
        ax_a.annotate(f"ρ = {rho:.3f}\n{_fmt_p(p)}", xy=(xi, rho), xytext=(0, 8),
                      textcoords="offset points", ha="center", va="bottom",
                      fontsize=TICK_SIZE, color=TEXT_COLOR)
    ax_a.axhline(0, color=SPINE_COLOR, lw=1.0, zorder=2)
    ax_a.set_ylim(top=ax_a.get_ylim()[1] * 1.35)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([
        rf"{DATASET_LABEL[s].split(',')[0]}" "\n" rf"{CELL_TYPE_LABEL[c]}" "\n"
        rf"({MARKER_LABEL[res[s][c]['marker_a']]}$\leftrightarrow${MARKER_LABEL[res[s][c]['marker_b']]})"
        for s, c in strata
    ], fontsize=TICK_SIZE, rotation=25, ha="right", rotation_mode="anchor")
    ax_a.set_ylabel("Coupling strength (Spearman ρ)")
    _style_ax(ax_a)
    _panel_label(ax_a, "(a)", x=-0.087)

    # ── (b) does the coupling conclusion survive less residualization too? ──
    # Grouped bars, one cluster per stratum, three tiers (raw / timepoint /
    # timepoint+count) read left to right in the order actually applied. The
    # raw/pooled tier is descriptive only (confounded by shared timepoint
    # response), so it's drawn hatched and faint rather than as a fourth
    # test on equal footing.
    tier_names = ["Raw", "Timepoint-\nresidualized", "+Count\n(audited)"]
    n_tiers = len(tier_names)
    bar_w = 0.24
    tier_offsets = np.linspace(-(n_tiers - 1) / 2, (n_tiers - 1) / 2, n_tiers) * (bar_w + 0.03)
    TIER_STYLE = [
        dict(hatch="////", alpha=0.40, edge_is_face=True, lw=1.3),
        dict(hatch=None, alpha=0.55, edge_is_face=False, lw=0.0),
        dict(hatch=None, alpha=1.00, edge_is_face=False, lw=0.0),
    ]
    ARCHIVE_GAP = 0.9

    def _archive_group_x(ordered_keys: list, archive_of) -> Dict:
        gx, cursor, prev = {}, 0.0, None
        for k in ordered_keys:
            a = archive_of(k)
            if prev is not None and a != prev:
                cursor += ARCHIVE_GAP
            gx[k] = cursor
            cursor += 1.0
            prev = a
        return gx

    group_x_b = _archive_group_x(strata, lambda k: k[0])
    for src, ct in strata:
        entry = res[src][ct]["primary_residualized_test"]
        col = GENOTYPE_COLOR[CELL_TYPE_GENOTYPE[ct]]
        gx = group_x_b[(src, ct)]
        rhos = [entry["raw_pooled_spearman_rho"], entry["residual_spearman_rho"],
                entry["residual_with_count_spearman_rho"]]
        for off, rho, style in zip(tier_offsets, rhos, TIER_STYLE):
            ax_b.bar(gx + off, rho, width=bar_w, color=col, alpha=style["alpha"],
                     hatch=style["hatch"], edgecolor=col if style["edge_is_face"] else "white",
                     linewidth=style["lw"], zorder=3)
    ax_b.axhline(0, color=SPINE_COLOR, lw=1.0, zorder=2)
    ax_b.set_xticks([group_x_b[k] for k in strata])
    ax_b.set_xticklabels([
        rf"{DATASET_LABEL[s].split(',')[0]}" "\n" rf"{CELL_TYPE_LABEL[c]}" for s, c in strata
    ], fontsize=TICK_SIZE, rotation=20, ha="right", rotation_mode="anchor")
    ax_b.set_ylabel("Coupling strength (Spearman ρ)")
    _style_ax(ax_b)
    _panel_label(ax_b, "(b)")

    # ── (c) does the same coupling claim hold under a second, independent
    # View? (a)/(b) are View 3 (whole persistence diagram, Wasserstein
    # displacement from control). (c) is View 2 (the persistence landscape's
    # value at one specific, seed-confirmed grid position -- the same
    # position Figure 5 plots the location of). Not split by genotype here:
    # both cell lines are already pooled upstream, so color encodes marker
    # identity instead, as in Figures 4-5.
    position_coupling_panels = [
        ("kuentzelmann", "yH2AX", "53BP1"),
        ("hahn", "Mre11", "yH2AX"),
        ("hahn", "yH2AX", "Mre11"),
    ]
    position_coupling = d.get("position_coupling")
    if position_coupling is None:
        logger.warning("[Fig.3] view2_position_coupling.json not found -- panel (c) skipped.")
        ax_c.text(0.5, 0.5, "data not found", ha="center", va="center",
                   transform=ax_c.transAxes, color=NS_COLOR)
    else:
        group_x_c = _archive_group_x(position_coupling_panels, lambda k: k[0])
        tick_x, tick_labels = [], []
        for src, marker_a, marker_b in position_coupling_panels:
            row = next((r for r in position_coupling if r.get("source") == src
                        and r.get("marker_a") == marker_a and r.get("marker_b") == marker_b), None)
            if row is None:
                continue
            col = MARKER_COLOR[marker_a]
            gx = group_x_c[(src, marker_a, marker_b)]
            rhos = [row["raw_pooled_spearman_rho"], row["timepoint_resid_spearman_rho"],
                    row["timepoint_count_resid_spearman_rho"]]
            for off, rho, style in zip(tier_offsets, rhos, TIER_STYLE):
                ax_c.bar(gx + off, rho, width=bar_w, color=col, alpha=style["alpha"],
                         hatch=style["hatch"], edgecolor=col if style["edge_is_face"] else "white",
                         linewidth=style["lw"], zorder=3)
            tick_x.append(gx)
            tick_labels.append(rf"{DATASET_LABEL[src].split(',')[0]}" "\n"
                                rf"{MARKER_LABEL[marker_a]}$\leftrightarrow${MARKER_LABEL[marker_b]}")
        ax_c.set_xticks(tick_x)
        ax_c.set_xticklabels(tick_labels, fontsize=TICK_SIZE, rotation=20, ha="right",
                             rotation_mode="anchor")
    ax_c.axhline(0, color=SPINE_COLOR, lw=1.0, zorder=2)
    ax_c.set_ylabel("Coupling strength (Spearman ρ)")
    _style_ax(ax_c)
    _panel_label(ax_c, "(c)")

    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=GENOTYPE_COLOR["normal"],
                      markeredgecolor="white", ms=14, label="Normal"),
               Line2D([0], [0], marker="s", color="w", markerfacecolor=GENOTYPE_COLOR["cancer"],
                      markeredgecolor="white", ms=14, label="Cancer")]
    if any_ns:
        handles.append(Line2D([0], [0], marker="s", color="w", markerfacecolor="white",
                              markeredgecolor=NS_COLOR, markeredgewidth=2.2, ms=12,
                              label="not significant, timepoint + count residualized"))
    handles += [
        Patch(facecolor=TEXT_COLOR, edgecolor=TEXT_COLOR, hatch="////", alpha=0.40,
              label="Raw (descriptive only, not a test)"),
        Patch(facecolor=TEXT_COLOR, edgecolor="white", alpha=0.55, label="Timepoint-residualized"),
        Patch(facecolor=TEXT_COLOR, edgecolor="white", alpha=1.00, label="+ Count-residualized (audited)"),
    ]
    handles += [Line2D([0], [0], color=MARKER_COLOR[m], marker="o", ms=9, lw=2.2,
                       label=f"(c) {MARKER_LABEL[m]}") for m in ("Mre11", "yH2AX")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=3, framealpha=1.0, edgecolor="#CCCCCC", fontsize=LEGEND_SIZE)
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    fig.subplots_adjust(bottom=0.24)
    _savefig(fig, out_dir / "fig3_hero_coupling")

    _write_notes(out_dir, "fig3_hero_coupling",
        "(a) Within-nucleus cross-marker coupling (Spearman rho between each marker's own distance-to-"
        "control, same nucleus): Kuntzelmann is 53BP1<->gammaH2AX, Hahn is MRE11<->gammaH2AX. "
        "Residualized against timepoint AND each marker's own localization count -- the strictest tier "
        "tested. Cancer couples more strongly than normal in both independent archives; this asymmetry "
        "is the paper's one central, novel claim. (b) Confirms (a) is not fragile to the specific "
        "correction chosen: the same four strata across all three residualization tiers, same n_pairs "
        "throughout each stratum. (c) Does the claim in (a)/(b) hold under a second, independent view of "
        "the same diagrams? (a)/(b) use the whole persistence diagram's Wasserstein displacement from "
        "control; (c) uses the persistence landscape's value at one specific, seed-confirmed grid "
        "position -- a structurally different mathematical object answering the same biological "
        "question. Not split by genotype: both cell lines are already pooled upstream of this analysis.")


# ==============================================================================
# FIG. 4 -- leading discriminative descriptor per stratum, over time
# ==============================================================================
# Each panel plots whichever of the tested candidate descriptors (Betti
# curve variance/mean, landscape peak-time, landscape integral, persistent
# entropy; H0 or H1) is that stratum's own strongest, most defensible
# choice, rather than one statistic forced across all four panels for
# uniformity.

FIG4_FEATURE_BY_PANEL = {
    ("kuentzelmann", "53BP1"): "noise_h1_betti_mean",
    ("kuentzelmann", "yH2AX"): "scaleC_h1_betti_var",
    ("hahn", "Mre11"):        "scaleC_h1_landscape_peak_t",
    ("hahn", "yH2AX"):        "scaleC_h0_betti_var",
}
FIG4_PANELS = [
    ("kuentzelmann", "53BP1", ["NHDF", "U87"]),
    ("kuentzelmann", "yH2AX", ["NHDF", "U87"]),
    ("hahn", "Mre11", ["HGF", "MCF7"]),
    ("hahn", "yH2AX", ["HGF", "MCF7"]),
]


def _feature_short_label(feat: str) -> str:
    hom = "H1" if "_h1_" in feat else "H0"
    if feat.endswith("betti_var"):
        return f"{hom} Betti variance"
    if feat.endswith("betti_mean"):
        return f"{hom} Betti mean"
    if "landscape_peak_t" in feat:
        return f"{hom} landscape peak-time"
    if feat.endswith("landscape_integral"):
        return f"{hom} landscape integral"
    if feat.endswith("persistent_entropy"):
        return f"{hom} persistent entropy"
    return f"{hom} {feat}"


def _feature_ylabel(feat: str) -> str:
    if feat.endswith("betti_var"):
        return "Betti curve variance\n(nuclear-field scale)"
    if feat.endswith("betti_mean"):
        return "Betti curve mean\n(nuclear-field scale)"
    if "landscape_peak_t" in feat:
        return "Landscape peak-time\n(nuclear-field scale)"
    if feat.endswith("landscape_integral"):
        return "Landscape integral\n(nuclear-field scale)"
    if feat.endswith("persistent_entropy"):
        return "Persistent entropy\n(nuclear-field scale)"
    return feat.replace("_", " ")


def make_fig4(d: Dict, out_dir: Path) -> None:
    if d.get("temporal_structure") is None:
        logger.warning("[Fig.4] temporal_structure_results.json not found -- skipping.")
        return
    probe1 = d["temporal_structure"]["probe1_leading_feature_temporal"]

    fig, axes = plt.subplots(2, 2, figsize=GRID_2x2, gridspec_kw={"hspace": 0.55, "wspace": 0.3})
    axes = axes.flatten()
    panel_feature_used = {}
    panel_feature_rank = {}

    for ax, (src, marker, cell_types) in zip(axes, FIG4_PANELS):
        feat = FIG4_FEATURE_BY_PANEL[(src, marker)]
        panel_feature_used[(src, marker)] = feat
        available = probe1.get(src, {}).get(marker, {}).get("features_tested", [])
        if feat in available:
            panel_feature_rank[(src, marker)] = (available.index(feat) + 1, len(available))
        else:
            logger.warning(f"[Fig.4] {feat} not tested for {src}/{marker} -- available: {available}")
            ax.text(0.5, 0.5, f"{feat}\nnot tested for this stratum", ha="center", va="center",
                     transform=ax.transAxes, color=NS_COLOR)
            continue
        res = probe1[src][marker]["results"][feat]
        for ct in cell_types:
            rows = [r for r in res["trajectory_raw"] if r["cell_type"] == ct]
            rows.sort(key=lambda r: r["timepoint_h"])
            x = [r["timepoint_h"] for r in rows]
            y = [r["mean"] for r in rows]
            se = [r["se"] for r in rows]
            genotype = CELL_TYPE_GENOTYPE[ct]
            ax.errorbar(x, y, yerr=se, marker=DATASET_MARKER_SHAPE[src], ms=8, lw=2.2,
                        elinewidth=1.4, capsize=3, color=GENOTYPE_COLOR[genotype], zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("Time post-irradiation (h)")
        _style_ax(ax)
        ax.set_title(f"{DATASET_LABEL[src].split(',')[0]}, {MARKER_LABEL[marker]}\n"
                     f"{_feature_short_label(feat)}", fontsize=TICK_SIZE, color=TEXT_COLOR, pad=8)

    for ax, (src, marker, _cts) in zip(axes, FIG4_PANELS):
        ax.set_ylabel(_feature_ylabel(panel_feature_used[(src, marker)]))

    for i, ax in enumerate(axes):
        _panel_label(ax, f"({chr(97 + i)})")

    handles = [Line2D([0], [0], color=GENOTYPE_COLOR["normal"], marker="o", ms=9, lw=2.2, label="Normal"),
               Line2D([0], [0], color=GENOTYPE_COLOR["cancer"], marker="o", ms=9, lw=2.2, label="Cancer")]
    handles += [Line2D([0], [0], marker=DATASET_MARKER_SHAPE[s], color="w", markerfacecolor=TEXT_COLOR,
                       markeredgecolor="white", ms=10, label=DATASET_LABEL[s]) for s in ("kuentzelmann", "hahn")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=2, framealpha=1.0, edgecolor="#CCCCCC", fontsize=LEGEND_SIZE)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.13, hspace=0.55, wspace=0.3)
    _savefig(fig, out_dir / "fig4_leading_descriptor")

    rank_bits = []
    for label, (src, marker) in zip("abcd", [(s, m) for s, m, _ in FIG4_PANELS]):
        r = panel_feature_rank.get((src, marker))
        if r is None:
            continue
        rank, total = r
        rank_bits.append(f"({label}) rank {rank} of {total} tested" if rank > 1
                          else f"({label}) the top-ranked (rank 1 of {total}) feature")
    rank_sentence = "; ".join(rank_bits) + "."

    _write_notes(out_dir, "fig4_leading_descriptor",
        "Leading discriminative descriptor by stratum -- each panel plots whichever tested candidate "
        "(Betti-curve variance/mean, landscape peak-time, landscape integral, persistent entropy; H0 or "
        "H1) is this stratum's own strongest choice, ranked by signal-to-noise and monotonicity over the "
        "timecourse, not forced into one statistic for cross-panel uniformity. Rank of the plotted "
        "feature within the tested candidate list, per panel: " + rank_sentence + " Raw values with "
        "standard-error bars (field-standard tier: neither source dataset's own published analysis "
        "residualizes against localization count, so this figure is held to that same standard).")


# ==============================================================================
# FIG. 5 -- does the discriminative spatial scale move with time?
# ==============================================================================

HEADLINE_PANELS = [
    ("hahn", "Mre11", "scaleC_h0", "lambda1"),
    ("kuentzelmann", "yH2AX", "scaleC_h1", "lambda1"),
    ("hahn", "yH2AX", "noise_h1", "lambda2"),
]

N_GRID_POINTS = 300  # fixed persistence-landscape grid size used throughout


def _grid_nm_lookup(localization_fl: Optional[List[Dict]]) -> Dict[str, Tuple[float, float]]:
    if localization_fl is None:
        return {}
    return {row["label"]: (row["grid_min_nm"], row["grid_max_nm"]) for row in localization_fl}


def _idx_to_nm(idx: float, grid_min_nm: float, grid_max_nm: float) -> float:
    return grid_min_nm + (idx / (N_GRID_POINTS - 1)) * (grid_max_nm - grid_min_nm)


def make_fig5(d: Dict, out_dir: Path) -> None:
    if d.get("temporal_trajectory") is None:
        logger.warning("[Fig.5] temporal_trajectory_position.json not found -- skipping.")
        return
    trajectories = d["temporal_trajectory"]
    grid_nm = _grid_nm_lookup(d.get("localization_fl"))

    fig, axes = plt.subplots(1, 3, figsize=WIDE_3)

    for ax, (src, marker, scale, lam) in zip(axes, HEADLINE_PANELS):
        key = f"{src}/{marker}/{scale}/{lam}"
        entry = trajectories.get(key)
        if entry is None:
            ax.text(0.5, 0.5, f"{key}\nnot found", ha="center", va="center",
                     transform=ax.transAxes, color=NS_COLOR)
            continue
        col = MARKER_COLOR[marker]
        shape = DATASET_MARKER_SHAPE[src]
        nm_range = grid_nm.get(key)

        # Three states per timepoint:
        #   resolved  -- full seed-stable consensus position.
        #   contested -- informative seeds found signal but disagreed on the
        #                exact grid bin; position is the mean of their picks,
        #                drawn open with an error bar spanning their range.
        #   silent    -- no seed found any signal at this timepoint.
        resolved_tp, resolved_pos, resolved_spread = [], [], []
        contested_tp, contested_pos, contested_spread = [], [], []
        silent_tps = []
        for row in entry["trajectory"]:
            if row.get("timepoint") == "control":
                continue
            tp = float(str(row["timepoint"]).replace("h", ""))
            core = row.get("seed_stable_core", [])
            if core:
                resolved_tp.append(tp)
                resolved_pos.append(float(np.mean(core)))
                resolved_spread.append((min(core), max(core)) if len(core) > 1 else None)
            elif row.get("all_seeds_selected_zero", False):
                silent_tps.append(tp)
            else:
                picks = [i for ps in row.get("per_seed", []) if ps.get("n_nz", 0) > 0
                         for i in ps.get("selected_indices", [])]
                if picks:
                    contested_tp.append(tp)
                    contested_pos.append(float(np.mean(picks)))
                    contested_spread.append((min(picks), max(picks)))
                else:
                    silent_tps.append(tp)

        # Grid index -> physical birth radius (nm) via this stratum's own
        # calibration.
        if nm_range is not None:
            g_min, g_max = nm_range
            resolved_pos = [_idx_to_nm(p, g_min, g_max) for p in resolved_pos]
            resolved_spread = [None if s is None else
                                (_idx_to_nm(s[0], g_min, g_max), _idx_to_nm(s[1], g_min, g_max))
                                for s in resolved_spread]
            contested_pos = [_idx_to_nm(p, g_min, g_max) for p in contested_pos]
            contested_spread = [(_idx_to_nm(s[0], g_min, g_max), _idx_to_nm(s[1], g_min, g_max))
                                 for s in contested_spread]
        else:
            logger.warning(f"[Fig.5] no grid calibration found for {key} -- plotting raw grid index.")

        all_tp = np.array(resolved_tp + contested_tp)
        all_pos = np.array(resolved_pos + contested_pos)
        if len(all_tp) > 1:
            order = np.argsort(all_tp)
            ax.plot(all_tp[order], all_pos[order], color=col, lw=2.2, zorder=3)

        for tp, pos, spread in zip(resolved_tp, resolved_pos, resolved_spread):
            if spread is not None:
                ax.errorbar([tp], [pos], yerr=[[pos - spread[0]], [spread[1] - pos]],
                            fmt="none", ecolor=col, elinewidth=1.4, capsize=3, zorder=4)
            ax.scatter([tp], [pos], marker=shape, s=90, facecolor=col, edgecolor="white",
                       linewidth=1.0, zorder=5)

        for tp, pos, spread in zip(contested_tp, contested_pos, contested_spread):
            ax.errorbar([tp], [pos], yerr=[[pos - spread[0]], [spread[1] - pos]],
                        fmt="none", ecolor=col, elinewidth=1.4, capsize=3, zorder=4)
            ax.scatter([tp], [pos], marker=shape, s=90, facecolor="white", edgecolor=col,
                       linewidth=2.0, zorder=5)

        # Silent timepoints are plotted at an offset below the panel's own
        # real data range (not at y=0, which is a real physical value here),
        # so the "no signal" zone reads as its own region, not a measurement.
        real_vals = list(resolved_pos) + list(contested_pos)
        for spread in list(resolved_spread) + list(contested_spread):
            if spread is not None:
                real_vals.extend(spread)
        if real_vals:
            vmin, vmax = min(real_vals), max(real_vals)
            vrange = (vmax - vmin) if vmax > vmin else max(vmax * 0.1, 1.0)
        else:
            vmin, vmax, vrange = 0.0, 100.0, 100.0
        silent_y = vmin - 0.22 * vrange
        for stp in sorted(silent_tps):
            ax.scatter([stp], [silent_y], marker="x", s=90, color=NS_COLOR, zorder=5, linewidth=2.2)
            ax.annotate("silent", xy=(stp, silent_y), xytext=(0, 10), textcoords="offset points",
                        fontsize=TICK_SIZE, color=NS_COLOR, ha="center", va="bottom")
        if silent_tps:
            ax.axhline(vmin - 0.08 * vrange, color=SPINE_COLOR, lw=0.8, ls=":", zorder=1)
            ax.set_ylim(silent_y - 0.30 * vrange, vmax + 0.20 * vrange)

        ax.set_xscale("log")
        ax.set_xlabel("Time post-irradiation (h)")
        _style_ax(ax)

    axes[0].set_ylabel("Discriminative loop-size window\n(nm)")

    for i, (ax, (src, marker, _s, _l)) in enumerate(zip(axes, HEADLINE_PANELS)):
        _panel_label(ax, f"({chr(97 + i)})")
        ax.set_title(f"{DATASET_LABEL[src].split(',')[0]}, {MARKER_LABEL[marker]}",
                     fontsize=TICK_SIZE, color=TEXT_COLOR, pad=6)

    handles = [Line2D([0], [0], color=MARKER_COLOR[m], marker="o", ms=9, lw=2.2, label=MARKER_LABEL[m])
               for m in ["Mre11", "yH2AX"]]
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                          markeredgecolor=TEXT_COLOR, markeredgewidth=2.0, ms=10,
                          label="contested position (signal present, seeds disagree)"))
    handles.append(Line2D([0], [0], marker="x", color="w", markerfacecolor=NS_COLOR,
                          markeredgecolor=NS_COLOR, ms=10, markeredgewidth=2.2,
                          label="no seed-stable signal at this timepoint"))
    handles.append(Line2D([0], [0], color=SPINE_COLOR, lw=0.8, ls=":",
                          label="floor: separates real data from the 'no signal' zone below"))
    handles += [Line2D([0], [0], marker=DATASET_MARKER_SHAPE[s], color="w", markerfacecolor=TEXT_COLOR,
                       markeredgecolor="white", ms=10, label=DATASET_LABEL[s]) for s in ("kuentzelmann", "hahn")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=3, framealpha=1.0, edgecolor="#CCCCCC", fontsize=LEGEND_SIZE)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    _savefig(fig, out_dir / "fig5_spatial_trajectory")

    _write_notes(out_dir, "fig5_spatial_trajectory",
        "Does the spatial scale that discriminates cancer from normal itself move with time? Selected "
        "persistence-landscape grid position, converted to nanometres via each stratum's own grid "
        "calibration, plotted per timepoint for three headline stratum/marker panels. Resolved points "
        "(filled) reach full seed-stable consensus; contested points (open) have signal but disagree on "
        "the exact grid bin (error bar spans the seeds' range); silent timepoints (x, below the real "
        "data) have no seed finding any signal at all. Panel (b) carries this figure's only true "
        "silence, a timepoint where classification and other analyses in this pipeline independently "
        "flag the same signal dropout.")


# ==============================================================================
# MAIN
# ==============================================================================

FIGURE_FUNCS = {
    "2": ("Population-shape separation by genotype", make_fig2),
    "3": ("Hero: within-nucleus cross-marker coupling", make_fig3),
    "4": ("Leading discriminative descriptor by stratum", make_fig4),
    "5": ("Discriminative spatial scale over time", make_fig5),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce paper Figures 2-5 from analysis-module output.")
    parser.add_argument("--basedir", default=".", help="Project root containing results/.")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated figure IDs, e.g. '2,3'.")
    args = parser.parse_args()

    base_dir = Path(args.basedir).resolve()
    out_dir = base_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    only_set = None
    if args.only:
        only_set = {s.strip() for s in args.only.split(",") if s.strip()}

    def _want(n: str) -> bool:
        return only_set is None or n in only_set

    logger.info("Loading data files ...")
    d = load_all(base_dir)

    for n in sorted(FIGURE_FUNCS):
        if _want(n):
            title, fn = FIGURE_FUNCS[n]
            logger.info(f"\n[Fig. {n}] {title} ...")
            fn(d, out_dir)

    logger.info(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
