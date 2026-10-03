#!/usr/bin/env python3
"""
Loader for the small public demo dataset bundled with this repository
(`data/weidner_sample/`, from Weidner et al. 2023 — see `data/README.md`).

This is a *different* loader from `kip_archive_loader.py`, not a more
general version of it. The two datasets share a file convention (the same
fixed-row-position CSV layout, header rows never parsed by name) because
they come out of the same lab's acquisition/export software, so this
loader reuses `kip_archive_loader`'s column definitions and row-reading
logic directly rather than redefining them. But the demo data itself is
structurally simpler than the real archives in one important way: it
carries no DBSCAN cluster assignment for each localization, because no
separate "cluster file" ships with it. Concretely, that means:

  - Scale C (the whole point cloud, clustered or not) works fine — it
    never looks at cluster_id.
  - Scale A (per-cluster), Scale B (cluster centroids), and the noise
    scale (unclustered points) all group localizations by cluster_id, and
    have nothing meaningful to group here. Rather than inventing a
    clustering step this dataset was never meant to carry, this loader
    sets cluster_id to -1 (unclustered) for every point and leaves it at
    that — an honest "we don't have this," not a fabricated substitute.

Every `LoadedNucleus` this loader returns therefore has `n_clusters = 0`,
and only `scale_c_features()` from `tml_smlm.features.persistence_features`
should be called on it. `scale_a_features()`, `scale_b_features()`, and
`noise_features()` will just return their empty/NaN-filled defaults if you
call them anyway — they won't error, but they won't tell you anything
either.

`cell_type` is set to "cancer" or "normal" from the two folder names
(`cancerCells`, `skinCells`), to keep the same two-group comparison shape
the rest of this package expects. These are not the same cell lines as
either archive in the paper — this is a demonstration of the method on a
public dataset, not a reproduction of the paper's own comparison.

Usage:
  from tml_smlm.io.demo_loader import load_demo_dataset
  nuclei = load_demo_dataset(Path("data/weidner_sample"))
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from tml_smlm.io.kip_archive_loader import (
    _read_csv_rows,
    NUCLEUS_SUMMARY_COLS,
    NUCLEUS_POINT_COLS,
)
from tml_smlm.features.persistence_features import LoadedNucleus

# Matches the demo files' own naming convention, e.g. "40_3_Orte_in_nucleus.csv":
# a leading replicate number, then a secondary index, then a fixed suffix.
_REPLICATE_RE = re.compile(r"^(\d+)_")

_FOLDER_TO_CELL_TYPE = {
    "cancerCells": "cancer",
    "skinCells": "normal",
}

# Point data starts after two 2-row blocks (summary, then camera/acquisition
# settings) plus the point table's own header row — four rows total, matching
# the row offset kip_archive_loader's real nucleus files use for the same
# reason: same export format.
_POINT_HEADER_ROW_OFFSET = 4


def _read_point_block(path: Path, header_row_offset: int, col_names: List[str]) -> pd.DataFrame:
    """Same logic as persistence_features._read_point_block, kept as its own
    small copy here rather than imported, since that function is private to
    a module whose job is the real archives, and duplicating four lines is
    cheaper than implying a dependency that isn't really there."""
    rows = _read_csv_rows(path)
    data_rows = rows[header_row_offset + 1:]
    if not data_rows:
        return pd.DataFrame(columns=col_names)
    return pd.DataFrame(np.array(data_rows, dtype=float), columns=col_names)


def load_demo_nucleus(path: Path, cell_type: str) -> LoadedNucleus:
    """Reads one demo CSV into the same LoadedNucleus shape the real
    pipeline's scale functions expect, minus a real cluster_id."""
    rows = _read_csv_rows(path)
    summary_values = rows[1]
    if len(summary_values) != len(NUCLEUS_SUMMARY_COLS):
        raise ValueError(
            f"{path}: summary row has {len(summary_values)} fields, "
            f"expected {len(NUCLEUS_SUMMARY_COLS)} — this file may not be in "
            f"the format load_demo_nucleus expects"
        )
    summary = {name: float(v) for name, v in zip(NUCLEUS_SUMMARY_COLS, summary_values)}
    diagonal_nm = float(np.hypot(summary["size_x_nm"], summary["size_y_nm"]))

    point_table = _read_point_block(path, header_row_offset=_POINT_HEADER_ROW_OFFSET, col_names=NUCLEUS_POINT_COLS)
    points = pd.DataFrame({
        "x_nm": point_table["x_nm"],
        "y_nm": point_table["y_nm"],
        "cluster_id": -1,  # no DBSCAN labels ship with this dataset — see module docstring
    })

    m = _REPLICATE_RE.match(path.name)
    replicate = m.group(1) if m else path.stem

    return LoadedNucleus(
        source="weidner_2023_demo",
        cell_type=cell_type,
        marker="unspecified",  # this dataset doesn't name a marker; single channel only
        condition="demo",
        replicate=replicate,
        is_control=False,
        timepoint_h=None,
        dose_Gy=None,
        points=points,
        diagonal_nm=diagonal_nm,
        n_clusters=0,
        n_localisations=len(points),
        noise_fraction=1.0,  # every point is cluster_id=-1 by construction here
    )


def load_demo_dataset(root: Path) -> List[LoadedNucleus]:
    """Loads every nucleus under root/{cancerCells,skinCells}/*.csv.

    root defaults, in the tutorial notebook, to data/weidner_sample — see
    data/README.md for what's there and where it came from.
    """
    nuclei: List[LoadedNucleus] = []
    for folder_name, cell_type in _FOLDER_TO_CELL_TYPE.items():
        folder = root / folder_name
        if not folder.exists():
            continue
        for csv_path in sorted(folder.glob("*.csv")):
            nuclei.append(load_demo_nucleus(csv_path, cell_type))
    return nuclei


def demo_manifest(nuclei: List[LoadedNucleus]) -> pd.DataFrame:
    """A tidy summary table, one row per nucleus, for a quick look before
    running anything expensive on the full set — the demo-scale analogue of
    kip_archive_loader's manifest.csv."""
    return pd.DataFrame([
        {
            "cell_type": n.cell_type,
            "replicate": n.replicate,
            "n_localisations": n.n_localisations,
            "diagonal_nm": round(n.diagonal_nm, 1),
        }
        for n in nuclei
    ])
