"""
Stage 1: convert user-supplied stay-point DataFrame into the pipeline's
canonical parquet format.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h3
import pandas as pd

from sparkmobility.models.timegeo.paths import TimeGeoPaths

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("caid", "stay_start_timestamp", "type", "h3_id_region")


def align(
    stay_df: pd.DataFrame,
    paths: TimeGeoPaths,
    user_column: str = "caid",
    timestamp_column: str = "stay_start_timestamp",
    location_column: str = "h3_id_region",
    type_column: str = "type",
) -> Path:
    """Normalize a stay-point DataFrame and write it to parquet.

    Produces the canonical layout the rest of the pipeline expects:
    `caid`, `timestamp` (UNIX seconds), `type` (string), `zero`,
    `h3_id_region`, `Longitude`, `Latitude`, plus optional
    `home_h3_index` / `work_h3_index` / `day_of_week` when present.

    Returns the path of the written parquet.
    """
    rename = {
        user_column: "caid",
        timestamp_column: "stay_start_timestamp",
        location_column: "h3_id_region",
        type_column: "type",
    }
    df = stay_df.rename(columns={k: v for k, v in rename.items() if k != v})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns (after rename): {missing}")

    logger.info(
        "aligning %d stay points across %d users", len(df), df["caid"].nunique()
    )

    # sparkmobility's FilteredUserStayPoints encodes `type` as int32
    # {0=other, 1=home, 2=work}; downstream code expects string labels.
    if pd.api.types.is_integer_dtype(df["type"]):
        df = df.copy()
        df["type"] = df["type"].map({0: "other", 1: "home", 2: "work"})

    # H3 integer -> hex string + lat/lng. Dedupe first: 45M rows collapse to
    # O(1e5) unique cells, so per-cell work beats per-row by orders of magnitude.
    ints = df["h3_id_region"].astype("int64")
    int_to_hex = {i: format(i, "x") for i in ints.unique()}
    df = df.assign(h3_id_region_16=ints.map(int_to_hex))

    hex_to_latlng = {
        h: h3.cell_to_boundary(h)[0] for h in df["h3_id_region_16"].unique()
    }
    df["Latitude"] = df["h3_id_region_16"].map(
        {h: v[0] for h, v in hex_to_latlng.items()}
    )
    df["Longitude"] = df["h3_id_region_16"].map(
        {h: v[1] for h, v in hex_to_latlng.items()}
    )

    df["timestamp"] = pd.to_datetime(df["stay_start_timestamp"]).astype(int) // 10**9
    df["zero"] = 0
    df["caid"] = df["caid"].astype(str)

    # Keep home_h3_index / work_h3_index / day_of_week if the input has them —
    # the C++ parameter-estimation binary looks them up by name.
    candidate_cols = [
        "caid",
        "timestamp",
        "type",
        "zero",
        "h3_id_region",
        "Longitude",
        "Latitude",
        "home_h3_index",
        "work_h3_index",
        "day_of_week",
    ]
    out_cols = [c for c in candidate_cols if c in df.columns]

    output_path = paths.data_cdr / "StayRegionsFiltered.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_parquet(output_path, index=False)
    logger.info("aligned data saved to %s", output_path)
    return output_path


def remove_redundant_stays(input_path: Path, output_path: Path) -> Path:
    """Drop duplicates on (caid, h3_id_region, timestamp)."""
    df = pd.read_parquet(input_path)
    before = len(df)
    df = df.drop_duplicates(subset=["caid", "h3_id_region", "timestamp"], keep="first")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("removed %d redundant stays (%d remaining)", before - len(df), len(df))
    return output_path
