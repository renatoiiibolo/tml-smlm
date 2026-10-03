#!/usr/bin/env python3
"""
Persistent-homology feature extraction for DNA-damage SMLM nuclei.

For each nucleus, this computes Vietoris-Rips persistence diagrams (H0 and
H1) at four spatial scales:

  Scale A -- one diagram per DBSCAN cluster, treating each focus as its own
  point cloud, aggregated to a mean and SD across a nucleus's clusters.
  Scale B -- each cluster collapsed to its centroid, capturing how foci are
  arranged relative to one another across the whole nucleus.
  Scale C -- every localization in the nucleus (subsampled if large),
  capturing the shape of the field as a whole.
  Noise   -- only the unclustered (DBSCAN label -1) localizations, since the
  background field itself can carry a real signal rather than being inert.

Each diagram is reduced, in the same pass it's computed, to a fixed set of
scalar summaries per homology degree (Betti-curve mean/variance, persistent
entropy, and the integral and peak location of the top persistence-landscape
layer). The raw diagrams for scales B, C, and noise are also written to a
compressed .npz alongside the feature matrix, so later analyses can reuse
them without recomputing anything from ripser.

Correctness note: every Vietoris-Rips H0 diagram contains at least one bar
with infinite persistence (the connected-component class that never dies).
If that bar isn't removed before building a persistence landscape, it
dominates the top landscape layer entirely, and any feature derived from it
ends up measuring the filtration cutoff (i.e. nucleus size) rather than how
the points are actually arranged. `_clean()` filters out infinite and
zero-lifetime bars before every landscape computation to avoid this.

Processing runs one nucleus at a time and each nucleus depends only on its
own two source files, so the work is embarrassingly parallel; --n-jobs
(joblib, loky backend) hands nuclei out across worker processes and
defaults to 1 (serial). Progress is checkpointed by appending each
finished row to the output CSV immediately, and by reading back which
(source, cell_type, marker, condition, replicate) keys already exist on
disk at startup, so an interrupted run can be resumed by simply rerunning
the same command.

Usage:
  python -m tml_smlm.features.persistence_features \
      --manifest data/manifest.csv --out data/feature_matrix.csv \
      --diagrams diagrams/ --results-dir results/

  # Parallel run. Set these two env vars first: ripser's underlying BLAS
  # calls can otherwise each try to claim more threads than the machine
  # has cores once several worker processes are running at once.
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  python -m tml_smlm.features.persistence_features \
      --manifest data/manifest.csv --out data/feature_matrix.csv \
      --diagrams diagrams/ --results-dir results/ --n-jobs -1

Requires: numpy, pandas, ripser, joblib
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs

from tml_smlm.io.kip_archive_loader import (
    _read_csv_rows,
    NUCLEUS_SUMMARY_COLS,
    NUCLEUS_POINT_COLS,
    CLUSTER_PARAM_COLS,
    CLUSTER_POINT_COLS,
)

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("tml_smlm.features.persistence_features")

try:
    from ripser import ripser as _ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False


# Scale A operates on a single focus, so its filtration cap is much tighter
# than the whole-nucleus scales below it -- 500 nm matches the biological
# cluster scale used elsewhere in this analysis.
SCALE_A_MAX_NM = 500.0
SCALE_A_MIN_PTS = 4          # below this, a per-cluster diagram isn't saying much
SCALE_A_SUBSAMPLE_CAP = 2_000

# Scale C and the noise scale both draw from potentially large point
# clouds -- 2000 and 5000 points respectively keep ripser's runtime
# reasonable without throwing away the shape of the distribution.
SCALE_C_MAX_PTS = 2_000
SCALE_C_SUBSAMPLE_SEED = 42
NOISE_MAX_PTS = 5_000
NOISE_SUBSAMPLE_SEED = 43

LANDSCAPE_N_T = 200
BETTI_N_T = 200
LANDSCAPE_K_MAX = 3   # only the top layer (lambda_1) becomes a feature

SCALE_A_SD_DEGENERACY_THRESHOLD = 0.30
COORD_MATCH_TOL_NM = 1e-3

FEATURE_NAMES_PER_DEGREE = [
    "landscape_integral", "landscape_peak_t",
    "persistent_entropy", "betti_mean", "betti_var",
]


# ══════════════════════════════════════════════════════════════════════════
# LOADING A NUCLEUS'S FULL POINT TABLES
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class LoadedNucleus:
    """Everything a scale function needs about one nucleus, gathered in
    one place so the scale functions themselves don't have to know
    anything about manifest columns or file formats."""
    source: str
    cell_type: str
    marker: str
    condition: str
    replicate: str
    is_control: bool
    timepoint_h: Optional[float]
    dose_Gy: Optional[float]
    points: pd.DataFrame     # columns: x_nm, y_nm, cluster_id
    diagonal_nm: float       # nucleus bounding-box diagonal, the filtration cap for B/C/noise
    n_clusters: int
    n_localisations: int
    noise_fraction: float


def _read_point_block(path: Path, header_row_offset: int, col_names: List[str]) -> pd.DataFrame:
    """Reads the trailing point table out of a nucleus or cluster CSV.
    header_row_offset is where the table's own header line sits (4 for a
    nucleus file, 2 for a cluster file, matching the fixed block layout the
    archive loader uses) -- the data starts one row after that."""
    rows = _read_csv_rows(path)
    data_rows = rows[header_row_offset + 1:]
    if not data_rows:
        return pd.DataFrame(columns=col_names)
    return pd.DataFrame(np.array(data_rows, dtype=float), columns=col_names)


def load_nucleus(source: str, cell_type: str, marker: str, condition: str,
                  replicate: str, is_control: bool, timepoint_h: Optional[float],
                  dose_Gy: Optional[float], nucleus_file: Path,
                  cluster_file: Path) -> LoadedNucleus:
    """Reads both point tables for one nucleus, checks that they actually
    describe the same localizations, and merges them into the single
    (x_nm, y_nm, cluster_id) table the scale functions work from.

    The nucleus and cluster files are two independently exported views of
    the same underlying localization list -- one carries acquisition
    detail we don't need for topology, the other carries the DBSCAN
    cluster assignment we do. Rather than trust that they line up, we
    check: same row count, and coordinates matching to within a nanometer
    fraction. If they don't, something upstream is broken and we'd rather
    fail loudly here than silently merge two nuclei that aren't the same
    nucleus.
    """
    nuc_summary_rows = _read_csv_rows(nucleus_file)
    summary_values = nuc_summary_rows[1]
    if len(summary_values) != len(NUCLEUS_SUMMARY_COLS):
        raise ValueError(
            f"{nucleus_file}: summary row has {len(summary_values)} fields, "
            f"expected {len(NUCLEUS_SUMMARY_COLS)}"
        )
    summary = {name: float(v) for name, v in zip(NUCLEUS_SUMMARY_COLS, summary_values)}
    diagonal_nm = float(np.hypot(summary["size_x_nm"], summary["size_y_nm"]))

    nuc_points = _read_point_block(nucleus_file, header_row_offset=4, col_names=NUCLEUS_POINT_COLS)
    clu_points = _read_point_block(cluster_file, header_row_offset=2, col_names=CLUSTER_POINT_COLS)

    if len(nuc_points) != len(clu_points):
        raise ValueError(
            f"{nucleus_file.name} vs {cluster_file.name}: row count mismatch "
            f"({len(nuc_points)} vs {len(clu_points)}) -- these should be the same "
            f"localizations exported twice, so this isn't safe to merge positionally"
        )
    if len(nuc_points) > 0:
        dx = (nuc_points["x_nm"] - clu_points["x_nm"]).abs().max()
        dy = (nuc_points["y_nm"] - clu_points["y_nm"]).abs().max()
        if max(dx, dy) > COORD_MATCH_TOL_NM:
            raise ValueError(
                f"{nucleus_file.name} vs {cluster_file.name}: coordinates disagree by "
                f"up to {max(dx, dy):.4f} nm -- positional merge is unsafe here"
            )

    points = pd.DataFrame({
        "x_nm": clu_points["x_nm"],
        "y_nm": clu_points["y_nm"],
        "cluster_id": clu_points["cluster_id"].astype(int),
    })
    n_localisations = len(points)
    n_clusters = int((points["cluster_id"].unique() >= 0).sum())
    noise_fraction = float((points["cluster_id"] == -1).mean()) if n_localisations else float("nan")

    return LoadedNucleus(
        source=source, cell_type=cell_type, marker=marker, condition=condition,
        replicate=replicate, is_control=is_control, timepoint_h=timepoint_h,
        dose_Gy=dose_Gy, points=points, diagonal_nm=diagonal_nm,
        n_clusters=n_clusters, n_localisations=n_localisations,
        noise_fraction=noise_fraction,
    )


# ══════════════════════════════════════════════════════════════════════════
# THE PERSISTENT-HOMOLOGY MATH ITSELF
# ══════════════════════════════════════════════════════════════════════════

def _clean(dgm: np.ndarray) -> np.ndarray:
    """Drops infinite-persistence bars and zero-lifetime bars before a
    diagram goes anywhere near a landscape computation -- see the module
    docstring for why an H0 landscape needs this filter."""
    if len(dgm) == 0:
        return dgm
    dgm = dgm[np.isfinite(dgm[:, 1])]
    if len(dgm) == 0:
        return dgm
    return dgm[(dgm[:, 1] - dgm[:, 0]) > 0]


def _dgm_to_landscape(dgm: np.ndarray, t_vals: np.ndarray) -> np.ndarray:
    dgm = _clean(dgm)
    k_max, n_t = LANDSCAPE_K_MAX, len(t_vals)
    if len(dgm) == 0:
        return np.zeros((k_max, n_t))
    tents = np.maximum(0.0, np.minimum(
        t_vals[None, :] - dgm[:, 0, None],
        dgm[:, 1, None] - t_vals[None, :],
    ))
    stacked = np.sort(tents, axis=0)[::-1]
    if stacked.shape[0] < k_max:
        stacked = np.vstack([stacked, np.zeros((k_max - stacked.shape[0], n_t))])
    return stacked[:k_max]


def _persistent_entropy(dgm: np.ndarray) -> float:
    if len(dgm) == 0:
        return 0.0
    lifetimes = dgm[:, 1] - dgm[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > 0)]
    if len(lifetimes) == 0:
        return 0.0
    total = lifetimes.sum()
    p = lifetimes / total
    return float(-np.sum(p * np.log(p + 1e-300)))


def _betti_curve(dgm: np.ndarray, t_vals: np.ndarray) -> np.ndarray:
    if len(dgm) == 0:
        return np.zeros(len(t_vals))
    births = dgm[:, 0][:, None]
    deaths = dgm[:, 1][:, None]
    deaths = np.where(np.isfinite(deaths), deaths, t_vals[-1] + 1.0)
    return ((births <= t_vals[None, :]) & (t_vals[None, :] < deaths)).sum(axis=0).astype(float)


def _compute_diagrams(coords_nm: np.ndarray, thresh_nm: float) -> Dict[str, np.ndarray]:
    """One ripser call, H0 and H1 together. The threshold caps the
    filtration at thresh_nm -- without it ripser would enumerate the
    complex out to the cloud's full pairwise diameter, most of which
    nothing downstream ever looks at."""
    if not HAS_RIPSER:
        raise ImportError("ripser is required: pip install ripser")
    if len(coords_nm) < 2:
        return {"h0": np.empty((0, 2)), "h1": np.empty((0, 2))}
    result = _ripser(coords_nm, maxdim=1, thresh=thresh_nm)["dgms"]
    h0 = result[0]
    h1 = result[1] if len(result) > 1 else np.empty((0, 2))
    return {"h0": h0, "h1": h1}


def _scale_features(coords_nm: np.ndarray, max_filtration_nm: float) -> Dict[str, float]:
    """The ten summary features -- five per homology dimension -- for one
    point cloud at one scale."""
    t_land = np.linspace(0, max_filtration_nm / 2, LANDSCAPE_N_T)
    t_betti = np.linspace(0, max_filtration_nm, BETTI_N_T)
    dt = t_land[1] - t_land[0]

    dgms = _compute_diagrams(coords_nm, thresh_nm=max_filtration_nm)
    feats: Dict[str, float] = {}
    for degree, key in [(0, "h0"), (1, "h1")]:
        dgm = dgms[key]
        top_layer = _dgm_to_landscape(dgm, t_land)[0]
        feats[f"{key}_landscape_integral"] = float(np.trapezoid(top_layer, dx=dt))
        feats[f"{key}_landscape_peak_t"] = (
            float(t_land[np.argmax(top_layer)]) if top_layer.max() > 0 else 0.0
        )
        feats[f"{key}_persistent_entropy"] = _persistent_entropy(dgm)
        betti = _betti_curve(dgm, t_betti)
        feats[f"{key}_betti_mean"] = float(betti.mean())
        feats[f"{key}_betti_var"] = float(betti.var())
    return feats


def _empty_scale_features() -> Dict[str, float]:
    return {f"{key}_{name}": float("nan") for key in ("h0", "h1") for name in FEATURE_NAMES_PER_DEGREE}


# ══════════════════════════════════════════════════════════════════════════
# THE FOUR SCALES
# ══════════════════════════════════════════════════════════════════════════

def scale_a_features(nucleus: LoadedNucleus) -> Dict[str, float]:
    """One diagram per cluster, then a mean and an SD across whatever
    clusters the nucleus has. We seed each cluster's subsample draw off
    its own cluster_id rather than off its position in a loop, so adding
    or dropping a cluster elsewhere in the nucleus never perturbs another
    cluster's draw."""
    pts = nucleus.points
    cluster_ids = sorted(int(c) for c in pts["cluster_id"].unique() if c >= 0)

    per_cluster: List[Dict[str, float]] = []
    for cid in cluster_ids:
        sub = pts[pts["cluster_id"] == cid]
        if len(sub) < SCALE_A_MIN_PTS:
            continue
        coords = sub[["x_nm", "y_nm"]].to_numpy()
        if len(coords) > SCALE_A_SUBSAMPLE_CAP:
            rng = np.random.default_rng(SCALE_C_SUBSAMPLE_SEED + cid)
            idx = rng.choice(len(coords), size=SCALE_A_SUBSAMPLE_CAP, replace=False)
            coords = coords[idx]
        per_cluster.append(_scale_features(coords, SCALE_A_MAX_NM))

    if not per_cluster:
        return {
            f"scaleA_{key}_{name}_{stat}": float("nan")
            for key in ("h0", "h1") for name in FEATURE_NAMES_PER_DEGREE for stat in ("mean", "sd")
        }

    df = pd.DataFrame(per_cluster)
    out: Dict[str, float] = {}
    for col in df.columns:
        out[f"scaleA_{col}_mean"] = float(df[col].mean())
        out[f"scaleA_{col}_sd"] = float(df[col].std(ddof=1)) if len(df) > 1 else 0.0
    return out


def scale_b_features(nucleus: LoadedNucleus) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Collapse each cluster to its centroid and look at how the foci sit
    relative to each other, at the scale of the whole nucleus."""
    clustered = nucleus.points[nucleus.points["cluster_id"] >= 0]
    if clustered.empty:
        return (
            {f"scaleB_{k}": v for k, v in _empty_scale_features().items()},
            {"scaleB_h0": np.empty((0, 2)), "scaleB_h1": np.empty((0, 2))},
        )
    centroids = clustered.groupby("cluster_id")[["x_nm", "y_nm"]].mean().to_numpy()
    dgms = _compute_diagrams(centroids, thresh_nm=nucleus.diagonal_nm)
    feats = _scale_features(centroids, nucleus.diagonal_nm)
    return (
        {f"scaleB_{k}": v for k, v in feats.items()},
        {"scaleB_h0": dgms["h0"], "scaleB_h1": dgms["h1"]},
    )


def scale_c_features(nucleus: LoadedNucleus, max_pts: int = SCALE_C_MAX_PTS,
                      seed: int = SCALE_C_SUBSAMPLE_SEED) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Every localization in the nucleus, clustered or not, subsampled
    down if there are more points than ripser can comfortably chew
    through."""
    coords = nucleus.points[["x_nm", "y_nm"]].to_numpy()
    if len(coords) > max_pts:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(coords), size=max_pts, replace=False)
        coords = coords[idx]
    dgms = _compute_diagrams(coords, thresh_nm=nucleus.diagonal_nm)
    feats = _scale_features(coords, nucleus.diagonal_nm)
    return (
        {f"scaleC_{k}": v for k, v in feats.items()},
        {"scaleC_h0": dgms["h0"], "scaleC_h1": dgms["h1"]},
    )


def noise_features(nucleus: LoadedNucleus, max_pts: int = NOISE_MAX_PTS,
                    seed: int = NOISE_SUBSAMPLE_SEED) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Whatever DBSCAN called background, treated on its own -- the
    unclustered field can carry real signal rather than being pure noise
    in the colloquial sense, which is why it gets its own scale instead of
    being dropped."""
    noise_pts = nucleus.points[nucleus.points["cluster_id"] == -1]
    if noise_pts.empty:
        return (
            {f"noise_{k}": v for k, v in _empty_scale_features().items()},
            {"noise_h0": np.empty((0, 2)), "noise_h1": np.empty((0, 2))},
        )
    coords = noise_pts[["x_nm", "y_nm"]].to_numpy()
    if len(coords) > max_pts:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(coords), size=max_pts, replace=False)
        coords = coords[idx]
    dgms = _compute_diagrams(coords, thresh_nm=nucleus.diagonal_nm)
    feats = _scale_features(coords, nucleus.diagonal_nm)
    return (
        {f"noise_{k}": v for k, v in feats.items()},
        {"noise_h0": dgms["h0"], "noise_h1": dgms["h1"]},
    )


# ══════════════════════════════════════════════════════════════════════════
# ONE NUCLEUS, START TO FINISH
# ══════════════════════════════════════════════════════════════════════════

_UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_.\-]")


def process_nucleus(nucleus: LoadedNucleus, diagrams_dir: Optional[Path]) -> Tuple[Dict, Dict]:
    """Runs all four scales on one nucleus and, if a diagrams directory
    was given, writes the raw diagrams to an .npz right here -- the same
    pass that computed them, not a second pass that recomputes them
    later. Returns the feature-matrix row and the diagram-manifest row
    (the latter empty if diagrams weren't requested)."""
    feats_a = scale_a_features(nucleus)
    feats_b, dgms_b = scale_b_features(nucleus)
    feats_c, dgms_c = scale_c_features(nucleus)
    feats_n, dgms_n = noise_features(nucleus)

    row: Dict = {
        "source": nucleus.source, "cell_type": nucleus.cell_type, "marker": nucleus.marker,
        "condition": nucleus.condition, "replicate": nucleus.replicate,
        "is_control": nucleus.is_control, "timepoint_h": nucleus.timepoint_h,
        "dose_Gy": nucleus.dose_Gy, "n_clusters": nucleus.n_clusters,
        "n_localisations": nucleus.n_localisations, "noise_fraction": nucleus.noise_fraction,
    }
    row.update(feats_a)
    row.update(feats_b)
    row.update(feats_c)
    row.update(feats_n)

    diagram_row: Dict = {}
    if diagrams_dir is not None:
        stem = _UNSAFE_CHARS_RE.sub("_", (
            f"{nucleus.source}__{nucleus.cell_type}__{nucleus.marker}__"
            f"{nucleus.condition}__{nucleus.replicate}"
        ))
        npz_path = diagrams_dir / f"{stem}.npz"
        np.savez_compressed(
            npz_path,
            scaleB_h0=dgms_b["scaleB_h0"], scaleB_h1=dgms_b["scaleB_h1"],
            scaleC_h0=dgms_c["scaleC_h0"], scaleC_h1=dgms_c["scaleC_h1"],
            noise_h0=dgms_n["noise_h0"], noise_h1=dgms_n["noise_h1"],
        )
        diagram_row = {
            "source": nucleus.source, "cell_type": nucleus.cell_type, "marker": nucleus.marker,
            "condition": nucleus.condition, "replicate": nucleus.replicate,
            "n_clusters": nucleus.n_clusters, "n_localisations": nucleus.n_localisations,
            "noise_fraction": nucleus.noise_fraction,
            "scaleB_n_h0": len(dgms_b["scaleB_h0"]), "scaleB_n_h1": len(dgms_b["scaleB_h1"]),
            "scaleC_n_h0": len(dgms_c["scaleC_h0"]), "scaleC_n_h1": len(dgms_c["scaleC_h1"]),
            "noise_n_h0": len(dgms_n["noise_h0"]), "noise_n_h1": len(dgms_n["noise_h1"]),
            "npz_file": npz_path.name,
        }
    return row, diagram_row


# ══════════════════════════════════════════════════════════════════════════
# ONE ROW, AS SEEN FROM A WORKER PROCESS
# ══════════════════════════════════════════════════════════════════════════

def _process_one_row(idx: int, row: Dict, diagrams_dir: Optional[Path]) -> Dict:
    """Everything load_nucleus() and process_nucleus() already do for one
    manifest row, wrapped so it can cross a process boundary cleanly under
    loky. The only things this function hands back are numbers, strings,
    and small dicts -- no logger, no open file handle. That's deliberate:
    it's why we don't log from inside here at all, and instead return an
    error string for the main process to log once the row is back,
    single-threaded, on the process that owns the log file and stdout.

    Nothing here reaches into any state another row might also be
    touching -- every row reads its own two files and, if a diagram gets
    written, writes its own uniquely-named .npz -- so this function runs
    correctly whether it's called from a plain loop (--n-jobs 1) or from
    inside a worker process (--n-jobs > 1); only the wall-clock time
    differs, never the result.
    """
    nucleus_file = Path(row["nucleus_file"])
    cluster_file = Path(row["cluster_file"])
    label = f"{row['source']}/{row['cell_type']}/{row['marker']}/{row['condition']}/{row['replicate']}"

    if not nucleus_file.exists() or not cluster_file.exists():
        return {
            "idx": idx, "status": "failed", "feat_row": None, "diag_row": None,
            "error": f"Row {idx} ({label}): file(s) missing — {nucleus_file} / {cluster_file}",
        }

    try:
        nucleus = load_nucleus(
            source=row["source"], cell_type=row["cell_type"], marker=row["marker"],
            condition=row["condition"], replicate=row["replicate"],
            is_control=bool(row["is_control"]),
            timepoint_h=row["timepoint_h"] if pd.notna(row["timepoint_h"]) else None,
            dose_Gy=row["dose_Gy"] if pd.notna(row["dose_Gy"]) else None,
            nucleus_file=nucleus_file, cluster_file=cluster_file,
        )
    except Exception as e:
        return {
            "idx": idx, "status": "failed", "feat_row": None, "diag_row": None,
            "error": f"Row {idx} ({label}): {e}",
        }

    try:
        feat_row, diag_row = process_nucleus(nucleus, diagrams_dir)
    except Exception as e:
        return {
            "idx": idx, "status": "failed", "feat_row": None, "diag_row": None,
            "error": f"Row {idx} ({label}) feature extraction failed: {e}",
        }

    return {
        "idx": idx, "status": "ok", "feat_row": feat_row,
        "diag_row": diag_row or None, "error": None,
    }


# ══════════════════════════════════════════════════════════════════════════
# CHECKPOINTING — write as we go, and know what's already done
# ══════════════════════════════════════════════════════════════════════════

_KEY_COLS = ["source", "cell_type", "marker", "condition", "replicate"]


def _existing_keys(csv_path: Path) -> set:
    """If a previous run already wrote some rows to csv_path, read back
    which nuclei they cover so we don't redo them. An empty or missing
    file just means there's nothing to resume from -- not an error."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    existing = pd.read_csv(csv_path, usecols=_KEY_COLS, dtype=str)
    return set(existing.itertuples(index=False, name=None))


def _append_row(csv_path: Path, row: Dict) -> None:
    """Writes one row to csv_path, adding the header only if the file is
    new. This is the whole checkpoint: by the time this call returns, that
    nucleus's result is safe on disk regardless of what happens next."""
    row_df = pd.DataFrame([row])
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    row_df.to_csv(csv_path, mode="a", header=write_header, index=False)


# ══════════════════════════════════════════════════════════════════════════
# STANDING CHECKS — run every time, not held for a separate audit script
# ══════════════════════════════════════════════════════════════════════════

def compute_degeneracy_report(feature_df: pd.DataFrame) -> Dict:
    """Flags any (source, cell_type) stratum where more than 30% of
    nuclei have fewer than three clusters — below that, Scale A's SD
    features are either undefined, exactly zero, or resting on a single
    pair of clusters, and shouldn't be read as a real spread estimate."""
    report: Dict = {}
    any_flag = False
    for (src, ct), grp in feature_df.groupby(["source", "cell_type"]):
        n_total = len(grp)
        n_degenerate = int((grp["n_clusters"] < 3).sum())
        frac = n_degenerate / n_total if n_total else float("nan")
        flagged = frac > SCALE_A_SD_DEGENERACY_THRESHOLD
        any_flag = any_flag or flagged
        report[f"{src}__{ct}"] = {
            "source": src, "cell_type": ct, "n_nuclei": n_total,
            "n_clusters_lt3": n_degenerate, "frac_degenerate": round(frac, 4),
            "flagged": flagged, "threshold": SCALE_A_SD_DEGENERACY_THRESHOLD,
        }
    report["_any_flag"] = any_flag
    report["_interpretation"] = (
        "flagged strata have >30% of nuclei with fewer than 3 clusters -- treat "
        "Scale A _sd features from those strata as uninformative; _mean features "
        "are unaffected."
    )
    return report


def compute_count_summary(feature_df: pd.DataFrame) -> Dict:
    """The count-covariate landscape (n_clusters, n_localisations,
    noise_fraction) per (source, cell_type, marker), written out before
    any classifier gets near this data, so a later result can be checked
    against what the raw counts looked like going in."""
    summary: Dict = {}
    for (src, ct, mk), grp in feature_df.groupby(["source", "cell_type", "marker"]):
        entry: Dict = {"source": src, "cell_type": ct, "marker": mk, "n_nuclei": len(grp)}
        for col in ("n_clusters", "n_localisations", "noise_fraction"):
            entry[f"{col}_mean"] = round(float(grp[col].mean()), 4)
            entry[f"{col}_sd"] = round(float(grp[col].std(ddof=1)), 4)
            entry[f"{col}_min"] = round(float(grp[col].min()), 4)
            entry[f"{col}_max"] = round(float(grp[col].max()), 4)
        entry["frac_zero_clusters"] = round(float((grp["n_clusters"] == 0).mean()), 4)
        summary[f"{src}__{ct}__{mk}"] = entry
    return summary


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def setup_logging() -> None:
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True, help="Path to the archive loader's manifest.csv")
    ap.add_argument("--out", type=Path, required=True, help="Output path for the feature matrix CSV")
    ap.add_argument("--diagrams", type=Path, default=Path("diagrams"), help="Directory for per-nucleus .npz diagrams")
    ap.add_argument("--no-diagrams", action="store_true", help="Skip diagram export; feature matrix is still written")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--limit", type=int, default=None, help="Cap the number of nuclei processed — for a smoke test")
    ap.add_argument("--n-jobs", type=int, default=1,
                     help="Worker processes (joblib, loky backend). Default 1 = serial. "
                          "-1 = use all available cores. Per-nucleus work is embarrassingly "
                          "parallel (see module docstring), so the feature matrix and "
                          "diagrams this produces are the same regardless of this value -- "
                          "only wall-clock time and checkpoint granularity change.")
    ap.add_argument("--batch-size", type=int, default=None,
                     help="Nuclei per checkpoint write. Default: one wave of --n-jobs "
                          "workers, so a batch and a round of parallel work are the same "
                          "size. Set lower for a smaller at-risk window on an unreliable "
                          "connection, or higher to reduce CSV-append overhead on a very "
                          "fast run. Ignored when --n-jobs is 1, since every row is its "
                          "own checkpoint in that case.")
    args = ap.parse_args()

    if not HAS_RIPSER:
        logger.error("ripser isn't installed -- pip install ripser")
        return 1
    if not args.manifest.exists():
        logger.error(f"Manifest not found: {args.manifest}")
        return 1

    setup_logging()
    diagrams_dir = None if args.no_diagrams else args.diagrams
    if diagrams_dir is not None:
        diagrams_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest, dtype={"replicate": str})
    logger.info("=" * 70)
    logger.info("persistence_features — persistent-homology extraction")
    logger.info("=" * 70)
    logger.info(f"Manifest: {args.manifest} ({len(manifest)} nuclei)")
    logger.info(f"Diagrams: {'SKIP (--no-diagrams)' if diagrams_dir is None else diagrams_dir}")

    diag_manifest_path = diagrams_dir / "diagram_manifest.csv" if diagrams_dir is not None else None

    # A previous attempt may have gotten partway through before something
    # interrupted it. Whatever's already sitting in args.out is work we
    # don't need to redo, so we read its keys back and skip past them --
    # this is what makes rerunning this script after a crash a resume
    # rather than a restart.
    already_done = _existing_keys(args.out)
    if already_done:
        logger.info(f"Resuming: {len(already_done)} nuclei already present in {args.out}, will be skipped")

    # Decide up front, in one pass, exactly which rows this run is
    # responsible for -- skipping whatever a previous run already finished
    # and stopping at --limit -- so the batching logic below only has to
    # think about "the rows we're doing," never about resume/skip/limit
    # bookkeeping mixed in with the parallel dispatch itself.
    pending: List[Tuple[int, Dict]] = []
    n_skipped = 0
    for idx, row in manifest.iterrows():
        key = tuple(str(row[c]) for c in _KEY_COLS)
        if key in already_done:
            n_skipped += 1
            continue
        pending.append((idx, row.to_dict()))
        if args.limit is not None and len(pending) >= args.limit:
            break

    n_workers = effective_n_jobs(args.n_jobs)
    batch_size = args.batch_size if args.batch_size is not None else n_workers
    logger.info(
        f"n_jobs={args.n_jobs} -> {n_workers} worker process(es), "
        f"batch size {batch_size}  ({'serial' if n_workers == 1 else 'parallel, loky backend'})"
    )
    if n_workers > 1:
        # "not set" is a real, informative value here, not a placeholder --
        # it means ripser's BLAS calls are free to spawn as many threads as
        # they like in each of the n_workers worker processes simultaneously,
        # which is worth knowing if a parallel run's wall-clock time looks
        # off.
        omp = os.environ.get("OMP_NUM_THREADS", "not set")
        openblas = os.environ.get("OPENBLAS_NUM_THREADS", "not set")
        logger.info(f"OMP_NUM_THREADS={omp}  OPENBLAS_NUM_THREADS={openblas}")
    logger.info(f"Rows to process this run: {len(pending)}  (skipped {n_skipped} already-done, {len(manifest) - n_skipped - len(pending)} beyond --limit)")

    n_ok = n_failed = 0

    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start: batch_start + batch_size]

        # n_jobs=1 still goes through joblib.Parallel rather than a plain
        # loop -- with n_jobs=1, joblib runs everything in-process with no
        # subprocess spawned at all, so this is the same code path either
        # way, just phrased once instead of twice.
        batch_results = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(_process_one_row)(idx, row, diagrams_dir) for idx, row in batch
        )

        # Batch is back, in the same order it was submitted (joblib's
        # guarantee) -- now the single-writer part: append everything in
        # this batch to disk before looking at the next one.
        n_ok_before_batch = n_ok
        for result in batch_results:
            if result["status"] == "ok":
                _append_row(args.out, result["feat_row"])
                if result["diag_row"] and diag_manifest_path is not None:
                    _append_row(diag_manifest_path, result["diag_row"])
                n_ok += 1
            else:
                n_failed += 1
                logger.error(result["error"])

        if n_ok // 25 > n_ok_before_batch // 25:
            logger.info(f"  {n_ok} nuclei processed this run ({n_ok + len(already_done)} total on disk)...")

    logger.info(
        f"\nProcessing complete: {n_ok} OK | {n_failed} failed | "
        f"{n_skipped} skipped (already done) — of {n_ok + n_failed + n_skipped} considered this run"
    )

    if not args.out.exists():
        logger.error("Nothing was ever written — nothing to report on.")
        return 1

    # The standing checks below need to see everything on disk, not just
    # what this particular run added -- if this is a resumed run, most of
    # the data they should be summarizing was written by an earlier,
    # interrupted invocation.
    feature_df = pd.read_csv(args.out)
    logger.info(f"Feature matrix -> {args.out}  shape={feature_df.shape} (cumulative across all runs)")

    if diag_manifest_path is not None and diag_manifest_path.exists():
        logger.info(f"Diagram manifest -> {diag_manifest_path} (cumulative)")

    degeneracy = compute_degeneracy_report(feature_df)
    deg_path = args.results_dir / "persistence_features_degeneracy_report.json"
    with open(deg_path, "w") as fh:
        json.dump(degeneracy, fh, indent=2)
    if degeneracy["_any_flag"]:
        logger.warning(f"Scale A _sd degeneracy flagged in at least one stratum — see {deg_path}")
    else:
        logger.info(f"Scale A _sd degeneracy: nothing flagged -> {deg_path}")

    count_summary = compute_count_summary(feature_df)
    count_path = args.results_dir / "persistence_features_count_summary.json"
    with open(count_path, "w") as fh:
        json.dump(count_summary, fh, indent=2)
    logger.info(f"Count summary -> {count_path}")

    logger.info("=" * 70)
    logger.info("persistence_features complete.")
    logger.info("=" * 70)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
