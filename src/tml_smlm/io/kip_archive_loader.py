#!/usr/bin/env python3
"""
Loader for raw SMLM localization data from two published DNA-damage archives:
Kuentzelmann et al. (2026) and Hahn et al. (2021).

It discovers each archive's per-nucleus files, pairs a nucleus's "nucleus" file
with its "cluster" file by replicate ID, verifies co-staining pairs (two markers
imaged in the same nucleus), and cross-checks the DBSCAN clustering parameters
recorded in each cluster file's filename against the parameters recorded inside
the file itself and against the values expected for that archive.

It writes two outputs that the rest of the pipeline builds on: a tidy per-nucleus
manifest (manifest.csv), and a capability manifest (capability_manifest.json)
describing which statistical comparisons each archive's structure supports —
available timepoints, dose levels, control structure, and co-staining coverage.

This loader is written against the specific directory layout and file-naming
conventions of these two archives, so it will not run against other SMLM
datasets (including the small public demo dataset included elsewhere in this
repository). It is included here for transparency about how the results in the
accompanying paper were produced, not as a general-purpose SMLM loader.

Logging goes to the console only; redirect it to a file yourself (e.g. via
`tee`) if you want a persistent log.

Usage:
  python kip_archive_loader.py --root ../data --out-dir data/ --results-dir results/
  python kip_archive_loader.py --root ../data --out-dir data/ --results-dir results/ --smoke-test

Requires: pandas
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = logging.getLogger("tml_smlm.io.kip_archive_loader")


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

SOURCE_CONFIGS: Dict[str, dict] = {
    "kuentzelmann": {
        "cell_types": ["NHDF", "U87"],
        "markers": {
            "yH2AX": {"r_nm": 200, "nmin": 70},
            "53BP1": {"r_nm": 200, "nmin": 80},
        },
        "default_dose_Gy": 1.25,  # fixed heavy-ion dose; not encoded in the folder name itself
    },
    "hahn": {
        "cell_types": ["HGF", "MCF7"],
        "markers": {
            "yH2AX": {"r_nm": 200, "nmin": 90},
            "Mre11": {"r_nm": 200, "nmin": 90},
        },
        "default_dose_Gy": None,  # not needed: this archive's folders always carry an explicit '<n>Gy' token
    },
}

# Leading-digit replicate ID. The optional '561nm_' prefix handles a channel-tag
# naming variant seen in parts of the archive, kept here for robustness even
# where it isn't strictly required by the two sources above.
_REPLICATE_RE = re.compile(r"^(?:561nm_)?(\d+)_")

# These column names are hardcoded and matched by fixed row position, not
# parsed from each file's own header text. CSV headers here are read with
# csv.reader rather than a plain split on commas, since a quoted header field
# can itself contain a comma — but even with correct quote handling, a header
# row can carry a different field count than its own data row. Rather than
# trying to reconcile that, the header row is never parsed for names: row 0 is
# skipped, row 1 is validated against the expected length and read positionally.
NUCLEUS_SUMMARY_COLS = [
    "n_events_total", "mean_loc_error_nm", "size_x_nm", "size_y_nm",
    "stack_size", "density_blinks_per_nm2", "masked_region_area_nm2",
]
CLUSTER_PARAM_COLS = [
    "pixelsize_output_nm", "max_hist_distance_nm",
    "density_image_radius_nm", "cluster_radius_nm", "cluster_n_min",
]
# The per-point column layout, declared here rather than re-derived elsewhere,
# so this loader and the feature-extraction code downstream share one
# definition of the fixed CSV layout for these archives.
NUCLEUS_POINT_COLS = [
    "max_photoelectron", "x_nm", "y_nm",
    "loc_err_x_nm", "loc_err_y_nm", "std_x_nm", "std_y_nm",
    "total_photoelectron", "frame_index",
]
CLUSTER_POINT_COLS = ["x_nm", "y_nm", "density", "cluster_id"]


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NucleusRecord:
    source: str
    cell_type: str
    marker: str
    condition: str
    replicate: str
    is_control: bool
    timepoint_h: Optional[float]
    dose_Gy: Optional[float]
    nucleus_file: str
    cluster_file: str
    dbscan_r_nm: Optional[int] = None
    dbscan_nmin: Optional[int] = None
    dbscan_validated: bool = False
    size_x_nm: Optional[float] = None
    size_y_nm: Optional[float] = None
    co_stain_partner: Optional[str] = None
    co_stain_size_check: Optional[str] = None  # "agree" | "disagree" | "unavailable"


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging() -> None:
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logger.info(f"Run {RUN_ID} started")


# ══════════════════════════════════════════════════════════════════════════════
# CSV READING -- csv.reader, fixed row positions, no header-text parsing
# ══════════════════════════════════════════════════════════════════════════════

def _read_csv_rows(path: Path) -> List[List[str]]:
    """Read all non-empty rows with proper quote handling. utf-8-sig strips
    a BOM if present."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [row for row in csv.reader(fh) if row]


def _read_nucleus_summary(path: Path) -> Optional[Dict[str, float]]:
    """Returns the 7-field summary block (rows[0]=header, rows[1]=values)
    as a dict keyed by NUCLEUS_SUMMARY_COLS, or None if the file doesn't
    have the expected shape. Only the summary block is needed here (nucleus
    size, for the co-staining QC check below) -- the full per-point table
    is read later by feature extraction, using NUCLEUS_POINT_COLS."""
    try:
        rows = _read_csv_rows(path)
    except Exception as e:
        logger.warning(f"  Could not read {path.name}: {e}")
        return None
    if len(rows) < 2:
        logger.warning(f"  {path.name}: expected >=2 rows for the summary block, got {len(rows)}")
        return None
    value_row = rows[1]
    if len(value_row) != len(NUCLEUS_SUMMARY_COLS):
        logger.warning(
            f"  {path.name}: summary value row has {len(value_row)} fields, "
            f"expected {len(NUCLEUS_SUMMARY_COLS)} -- header was {rows[0]}"
        )
        return None
    try:
        return {name: float(v) for name, v in zip(NUCLEUS_SUMMARY_COLS, value_row)}
    except ValueError as e:
        logger.warning(f"  {path.name}: could not parse summary values as floats: {e}")
        return None


def _read_cluster_params(path: Path) -> Optional[Dict[str, float]]:
    """Returns the 5-field DBSCAN parameter block (rows[0]=header,
    rows[1]=values) as a dict keyed by CLUSTER_PARAM_COLS, or None."""
    try:
        rows = _read_csv_rows(path)
    except Exception as e:
        logger.warning(f"  Could not read {path.name}: {e}")
        return None
    if len(rows) < 2:
        logger.warning(f"  {path.name}: expected >=2 rows for the param block, got {len(rows)}")
        return None
    value_row = rows[1]
    if len(value_row) != len(CLUSTER_PARAM_COLS):
        logger.warning(
            f"  {path.name}: param value row has {len(value_row)} fields, "
            f"expected {len(CLUSTER_PARAM_COLS)} -- header was {rows[0]}"
        )
        return None
    try:
        return {name: float(v) for name, v in zip(CLUSTER_PARAM_COLS, value_row)}
    except ValueError as e:
        logger.warning(f"  {path.name}: could not parse param values as floats: {e}")
        return None


def _parse_dbscan_suffix_from_filename(fname: str) -> Optional[Tuple[int, int]]:
    """Extracts (r_nm, nmin) from a cluster filename's own suffix
    (..._Orte_200_80.csv), independent of whatever precedes it."""
    marker_str = "_Orte_in_mask_cluster_Orte_"
    if marker_str not in fname:
        return None
    _, tail = fname.split(marker_str, 1)
    tail = tail.replace(".csv", "")
    try:
        r_str, nmin_str = tail.split("_")
        return int(r_str), int(nmin_str)
    except ValueError:
        return None


def _replicate_id(filename: str) -> Optional[str]:
    """Leading-digit replicate ID parsed from the start of a filename."""
    m = _REPLICATE_RE.match(filename)
    return m.group(1) if m else None


# ══════════════════════════════════════════════════════════════════════════════
# CONDITION PARSING
# ══════════════════════════════════════════════════════════════════════════════

def _condition_to_timepoint_dose(condition: str, default_dose_Gy: Optional[float] = None) -> Tuple[bool, Optional[float], Optional[float]]:
    """Parses a condition folder name into (is_control, timepoint_h, dose_Gy).
      Kuentzelmann: 'NHDF_N_0.08h' (irradiated, no dose token at all),
                    'NHDF_sham' (control, no timepoint token at all)
      Hahn:         '2Gy_0.16h' (irradiated), 'sham_0.16h' (control, WITH a
                    timepoint token -- this archive's shams are timepoint-matched
                    to their irradiated counterparts).
    default_dose_Gy is substituted only when no explicit '<n>Gy' token is
    present and the condition is irradiated (Kuentzelmann's case).
    """
    is_control = "sham" in condition.lower()
    dose_match = re.search(r"([\d.]+)Gy", condition)
    time_match = re.search(r"([\d.]+)h", condition)
    timepoint_h = float(time_match.group(1)) if time_match else None

    if dose_match:
        dose_Gy = float(dose_match.group(1))
        is_control = is_control or dose_Gy == 0.0
    else:
        dose_Gy = 0.0 if is_control else default_dose_Gy

    return is_control, timepoint_h, dose_Gy


# ══════════════════════════════════════════════════════════════════════════════
# NMIN-VARIANT SUBFOLDER RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_condition_root(cluster_dir: Path, expected_nmin: int) -> Optional[Path]:
    """Most marker/cell-type combinations keep their condition folders
    directly under cluster/. One combination in the Hahn archive
    (MCF7/Mre11) instead has Nmin60/ and Nmin90/ subfolders sitting where a
    condition folder would normally be, each holding a full set of
    conditions underneath. Rather than special-case that one path, this
    detects the pattern by folder name, so any other marker/cell-type
    combination with the same layout is still picked up instead of
    silently dropping out of the manifest."""
    children = [c for c in cluster_dir.iterdir() if c.is_dir() and not c.name.startswith(".")]
    nmin_dirs = [c for c in children if re.fullmatch(r"Nmin\d+", c.name)]
    if not nmin_dirs:
        return cluster_dir
    match = [c for c in nmin_dirs if c.name == f"Nmin{expected_nmin}"]
    if not match:
        logger.error(
            f"  {cluster_dir}: Nmin-variant subfolders found ({[c.name for c in nmin_dirs]}) "
            f"but none match the expected Nmin{expected_nmin} — skipping this marker/cell_type"
        )
        return None
    logger.info(f"  {cluster_dir}: Nmin-variant layout detected -> using {match[0].name}/")
    return match[0]


# ══════════════════════════════════════════════════════════════════════════════
# PER-SOURCE ITERATION -- independent discovery of nucleus/ and cluster/,
# joined by replicate ID, not by filename reconstruction
# ══════════════════════════════════════════════════════════════════════════════

def iter_source(root: Path, source: str, limit: Optional[int] = None) -> List[NucleusRecord]:
    config = SOURCE_CONFIGS[source]
    records: List[NucleusRecord] = []

    for cell_type in config["cell_types"]:
        for marker, expected in config["markers"].items():
            nucleus_dir = root / source / cell_type / marker / "nucleus"
            cluster_dir = root / source / cell_type / marker / "cluster"
            if not cluster_dir.exists() or not nucleus_dir.exists():
                logger.info(f"  {source}/{cell_type}/{marker}: missing nucleus/ or cluster/ directory, skipping")
                continue
            cluster_dir = _resolve_condition_root(cluster_dir, expected["nmin"])
            if cluster_dir is None:
                continue

            n_seen_this_marker = 0

            for condition_dir in sorted(cluster_dir.iterdir()):
                if not condition_dir.is_dir():
                    continue
                if limit is not None and n_seen_this_marker >= limit:
                    break
                condition = condition_dir.name
                nucleus_condition_dir = nucleus_dir / condition
                if not nucleus_condition_dir.exists():
                    logger.warning(f"  {source}/{cell_type}/{marker}: no nucleus/{condition}/ directory — skipping this condition entirely")
                    continue

                is_control, timepoint_h, dose_Gy = _condition_to_timepoint_dose(
                    condition, default_dose_Gy=config.get("default_dose_Gy")
                )

                # Independent discovery, keyed by leading-digit replicate ID.
                nucleus_by_rep = {
                    _replicate_id(f.name): f
                    for f in sorted(nucleus_condition_dir.glob("*.csv"))
                    if _replicate_id(f.name) is not None
                }

                for cluster_path in sorted(condition_dir.glob("*.csv")):
                    if limit is not None and n_seen_this_marker >= limit:
                        break
                    rep = _replicate_id(cluster_path.name)
                    if rep is None:
                        logger.warning(f"  Could not parse replicate id from: {cluster_path.name}")
                        continue

                    dbscan_suffix = _parse_dbscan_suffix_from_filename(cluster_path.name)
                    if dbscan_suffix is None:
                        logger.warning(f"  Could not parse DBSCAN suffix from: {cluster_path.name}")
                        continue
                    r_from_name, nmin_from_name = dbscan_suffix

                    params = _read_cluster_params(cluster_path)
                    from_content = (
                        (int(round(params["cluster_radius_nm"])), int(round(params["cluster_n_min"])))
                        if params is not None else None
                    )
                    validated = (
                        from_content is not None
                        and from_content == (r_from_name, nmin_from_name)
                        and (r_from_name, nmin_from_name) == (expected["r_nm"], expected["nmin"])
                    )
                    if not validated:
                        logger.error(
                            f"  DBSCAN mismatch for {cluster_path}: "
                            f"filename=({r_from_name},{nmin_from_name}) "
                            f"file_content={from_content} "
                            f"expected=({expected['r_nm']},{expected['nmin']}) — EXCLUDING from manifest"
                        )
                        n_seen_this_marker += 1
                        continue

                    nucleus_path = nucleus_by_rep.get(rep)
                    if nucleus_path is None:
                        logger.warning(
                            f"  {cluster_path}: no matching nucleus/ file found for "
                            f"replicate '{rep}' in {nucleus_condition_dir} — excluding"
                        )
                        n_seen_this_marker += 1
                        continue

                    size_x = size_y = None
                    summary = _read_nucleus_summary(nucleus_path)
                    if summary is not None:
                        size_x, size_y = summary["size_x_nm"], summary["size_y_nm"]

                    records.append(NucleusRecord(
                        source=source, cell_type=cell_type, marker=marker,
                        condition=condition, replicate=rep, is_control=is_control,
                        timepoint_h=timepoint_h, dose_Gy=dose_Gy,
                        nucleus_file=str(nucleus_path), cluster_file=str(cluster_path),
                        dbscan_r_nm=r_from_name, dbscan_nmin=nmin_from_name,
                        dbscan_validated=True, size_x_nm=size_x, size_y_nm=size_y,
                    ))
                    n_seen_this_marker += 1

    logger.info(f"  {source}: {len(records)} validated nuclei loaded")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# CO-STAINING VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def verify_co_staining(records: List[NucleusRecord], source: str) -> None:
    """Groups nuclei by (cell_type, condition, replicate). When a group has
    exactly two rows carrying different markers, they are the same physical
    nucleus imaged twice -- once per marker -- and each is tagged with the
    other's cluster file as its co-stain partner. This relies on replicate
    IDs having been parsed independently from each file's own name in
    iter_source(), so the same number found in both the nucleus/ and
    cluster/ listings is trustworthy on its own, without inspecting
    anything else about the filenames.

    Nucleus size agreement between paired partners is checked as a sanity
    check only, not as part of what decides a pair is real: a disagreement
    is logged as a warning but does not break the pair.
    """
    by_replicate: Dict[Tuple[str, str, str], List[NucleusRecord]] = {}
    for r in records:
        if r.source != source:
            continue
        by_replicate.setdefault((r.cell_type, r.condition, r.replicate), []).append(r)

    n_paired = n_size_agree = n_size_disagree = n_size_unavailable = 0
    for group in by_replicate.values():
        if len(group) != 2 or group[0].marker == group[1].marker:
            continue
        a, b = group
        a.co_stain_partner, b.co_stain_partner = b.cluster_file, a.cluster_file
        if a.size_x_nm is not None and b.size_x_nm is not None:
            agree = abs(a.size_x_nm - b.size_x_nm) <= 10 and abs(a.size_y_nm - b.size_y_nm) <= 10
            tag = "agree" if agree else "disagree"
            if not agree:
                logger.warning(
                    f"  co-stain size mismatch: {a.cluster_file} "
                    f"({a.size_x_nm},{a.size_y_nm}) vs {b.cluster_file} "
                    f"({b.size_x_nm},{b.size_y_nm})"
                )
                n_size_disagree += 1
            else:
                n_size_agree += 1
        else:
            tag = "unavailable"
            n_size_unavailable += 1
        a.co_stain_size_check = b.co_stain_size_check = tag
        n_paired += 2

    logger.info(
        f"  {source} co-staining: {n_paired}/{len(records)} nuclei paired "
        f"(size QC: {n_size_agree} agree, {n_size_disagree} disagree, {n_size_unavailable} unavailable)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CAPABILITY MANIFEST
# ══════════════════════════════════════════════════════════════════════════════

def build_capability_manifest(records: List[NucleusRecord]) -> dict:
    if not records:
        logger.error(
            "  build_capability_manifest: zero records survived loading — "
            "returning an empty capability manifest rather than crashing on groupby()."
        )
        return {}
    df = pd.DataFrame([asdict(r) for r in records])
    manifest = {}
    for (source, marker), group in df.groupby(["source", "marker"]):
        irradiated = group[~group["is_control"]]
        size_qc = group["co_stain_size_check"].value_counts().to_dict()
        manifest[f"{source}/{marker}"] = {
            "n_real_timepoints": int(irradiated["timepoint_h"].nunique()),
            "n_dose_levels": int(irradiated["dose_Gy"].nunique()),
            "control_structure": (
                "timepoint_matched"
                if group[group["is_control"]]["timepoint_h"].notna().any()
                else "pooled"
            ),
            "n_nuclei": int(len(group)),
            "co_stained_fraction": float((group["co_stain_partner"].notna()).mean()),
            "co_stain_size_qc": {k: int(v) for k, v in size_qc.items()},
        }
    return manifest


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="Path to the shared data/ directory.")
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--limit", type=int, default=None, help="Cap files processed per source — smoke-test before a full run.")
    ap.add_argument("--smoke-test", action="store_true", help="Alias for --limit 5 if --limit not otherwise set.")
    args = ap.parse_args()

    if args.smoke_test and args.limit is None:
        args.limit = 5

    setup_logging()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("kip_archive_loader — archive discovery and structural audit")
    logger.info("=" * 70)

    all_records: List[NucleusRecord] = []
    per_source_counts = {}
    for source in SOURCE_CONFIGS:
        logger.info(f"\nLoading source: {source}")
        recs = iter_source(args.root, source, limit=args.limit)
        verify_co_staining(recs, source)
        all_records.extend(recs)
        per_source_counts[source] = len(recs)

    manifest_path = args.out_dir / "manifest.csv"
    pd.DataFrame([asdict(r) for r in all_records]).to_csv(manifest_path, index=False)
    if not all_records:
        logger.warning(f"  manifest.csv written with ZERO rows — check per-source warnings/errors above")
    logger.info(f"\nManifest -> {manifest_path} ({len(all_records)} nuclei)")

    capability = build_capability_manifest(all_records)
    capability_path = args.out_dir / "capability_manifest.json"
    with open(capability_path, "w") as fh:
        json.dump(capability, fh, indent=2)
    logger.info(f"Capability manifest -> {capability_path}")
    for key, facts in capability.items():
        logger.info(f"  {key}: {facts}")

    run_manifest = {
        "run_id": RUN_ID,
        "started_at": RUN_ID,
        "args": {k: str(v) for k, v in vars(args).items()},
        "per_source_nucleus_counts": per_source_counts,
        "note": "log output is console-only; redirect via the run command (e.g. tee) to persist it",
    }
    run_manifest_path = args.results_dir / f"run_manifest_{RUN_ID}.json"
    with open(run_manifest_path, "w") as fh:
        json.dump(run_manifest, fh, indent=2)
    logger.info(f"Run manifest -> {run_manifest_path}")

    logger.info("=" * 70)
    logger.info("kip_archive_loader complete.")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
