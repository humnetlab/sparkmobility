"""
Stage 1: convert user-supplied stay-point DataFrame into the pipeline's
canonical parquet format.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Union

import h3
import pandas as pd

from sparkmobility.models.timegeo.paths import TimeGeoPaths

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame
else:
    SparkDataFrame = None

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("caid", "stay_start_timestamp", "type", "h3_id_region")


def _is_spark_df(df) -> bool:
    """True iff ``df`` is a pyspark.sql.DataFrame, without importing pyspark
    unless it's already loaded."""
    try:
        from pyspark.sql import DataFrame as _SparkDF  # noqa: WPS433
    except ImportError:
        return False
    return isinstance(df, _SparkDF)


def align(
    stay_df: "Union[pd.DataFrame, SparkDataFrame]",
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

    Accepts a pandas DataFrame or a pyspark.sql.DataFrame. Spark inputs
    stay in Spark end-to-end in this stage (renames, type labelling,
    H3 → lat/lng via broadcast-join, timestamp conversion) and write a
    multi-part parquet that the downstream pandas / pyarrow stages
    consume transparently.
    """
    if _is_spark_df(stay_df):
        return _align_spark(
            stay_df,
            paths,
            user_column=user_column,
            timestamp_column=timestamp_column,
            location_column=location_column,
            type_column=type_column,
        )

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


def _align_spark(
    stay_df: "SparkDataFrame",
    paths: TimeGeoPaths,
    user_column: str,
    timestamp_column: str,
    location_column: str,
    type_column: str,
) -> Path:
    """Spark-native align. Mirrors :func:`align`'s pandas path."""
    from pyspark.sql import functions as F

    rename = {
        user_column: "caid",
        timestamp_column: "stay_start_timestamp",
        location_column: "h3_id_region",
        type_column: "type",
    }
    df = stay_df
    for old, new in rename.items():
        if old != new and old in df.columns:
            df = df.withColumnRenamed(old, new)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns (after rename): {missing}")

    # int {0,1,2} → string labels, matching the pandas path.
    type_dtype = dict(df.dtypes)["type"]
    if type_dtype in ("tinyint", "smallint", "int", "bigint"):
        df = df.withColumn(
            "type",
            F.when(F.col("type") == 0, F.lit("other"))
            .when(F.col("type") == 1, F.lit("home"))
            .when(F.col("type") == 2, F.lit("work")),
        )

    df = df.withColumn("h3_id_region", F.col("h3_id_region").cast("long"))

    # Collect unique h3 cells to the driver (O(1e5) even for 46M-row input),
    # compute lat/lng once, broadcast-join back. Avoids a per-row Python UDF.
    spark = df.sparkSession
    unique_ids = [r[0] for r in df.select("h3_id_region").distinct().collect()]
    latlng_rows = []
    for i in unique_ids:
        lat, lng = h3.cell_to_boundary(format(int(i), "x"))[0]
        latlng_rows.append((int(i), float(lat), float(lng)))
    latlng_df = spark.createDataFrame(
        latlng_rows, schema=["h3_id_region", "Latitude", "Longitude"]
    )
    df = df.join(F.broadcast(latlng_df), on="h3_id_region", how="left")

    df = (
        df.withColumn(
            "timestamp",
            F.unix_timestamp(F.col("stay_start_timestamp").cast("timestamp")),
        )
        .withColumn("zero", F.lit(0))
        .withColumn("caid", F.col("caid").cast("string"))
    )

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
    df.select(*out_cols).write.mode("overwrite").parquet(str(output_path))
    logger.info("aligned spark data saved to %s", output_path)
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
