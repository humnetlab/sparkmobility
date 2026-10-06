"""Density-based stay detection: threshold stop detection + DBSCAN clustering.

The alternative to :class:`~sparkmobility.processing.grid_stay_detection.GridStayDetection`.
Both detect stops and then group them into places; they differ in how:

* ``GridStayDetection`` walks a running centroid (Zheng et al. 2010) and snaps
  the result onto H3 cells at a fixed resolution.
* ``ClusterStayDetection`` flags stationary runs by pairwise distance/time
  thresholds, refines each run around its median centroid, then clusters a
  device's stops with DBSCAN so recurring visits to the same place collapse
  into one cluster without being quantised to a grid.

Output matches the canonical stay schema the rest of the package expects, so
``MobilityDataset.load_stays()``, ``UserSelection`` and ``TimeGeo.fit()``
consume it unchanged. See :func:`to_canonical_schema` for the column contract.
"""

import logging

import h3
import h3.api.basic_int as h3int
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from sparkmobility.processing._geo import haversine_sql
from sparkmobility.utils import spark_session

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6371008.0

# Canonical location-type encoding, shared with the Scala backend and the
# TimeGeo C++ binary ("1 = home, 2 = work, 0 = other").
TYPE_OTHER, TYPE_HOME, TYPE_WORK = 0, 1, 2


# --------------------------------------------------------------------------
# Stage 1: stop detection
# --------------------------------------------------------------------------

# Timestamps are normalised to milliseconds on the way in and divided back
# out at the end, so `time_threshold` and `min_stop_duration` stay in seconds
# for the caller regardless of the input resolution.
_STOPS_SQL = """
WITH base AS (
    SELECT DISTINCT caid, latitude, longitude, utc_timestamp * 1000 AS ts
    FROM raw_pings
),
ordered AS (
    SELECT *,
        LEAD(latitude)  OVER w AS next_lat,
        LEAD(longitude) OVER w AS next_lon,
        LEAD(ts)        OVER w AS next_ts
    FROM base
    WINDOW w AS (PARTITION BY caid ORDER BY ts)
),
calc AS (
    SELECT *,
        CASE WHEN next_lat IS NOT NULL THEN {dist_next} END AS dist,
        CASE WHEN next_ts  IS NOT NULL THEN (next_ts - ts) / 1000 END AS tdiff
    FROM ordered
),
flags AS (
    SELECT *,
        ((dist <= {distance_threshold} AND (tdiff <= {time_threshold} OR tdiff IS NULL))
         OR (LAG(dist) OVER w <= {distance_threshold}
             AND (LAG(tdiff) OVER w <= {time_threshold} OR LAG(tdiff) OVER w IS NULL))
        ) AS stationary
    FROM calc
    WINDOW w AS (PARTITION BY caid ORDER BY ts)
),
events AS (
    SELECT *,
        CASE WHEN stationary AND (LAG(stationary) OVER w IS NULL OR NOT LAG(stationary) OVER w
                                  OR LAG(dist) OVER w > {distance_threshold})
            THEN 1 ELSE 0
        END AS ev_start
    FROM flags
    WINDOW w AS (PARTITION BY caid ORDER BY ts)
),
cumul AS (
    SELECT *,
        SUM(ev_start) OVER (PARTITION BY caid ORDER BY ts
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS ev_id
    FROM events
),
stationary_points AS (
    SELECT caid, ts, latitude, longitude, ev_id
    FROM cumul
    WHERE stationary AND ev_id > 0
),
-- First pass: a provisional median centroid per candidate stop.
initial_stops AS (
    SELECT caid, ev_id AS stop_id,
        MIN(ts) AS start_ts, MAX(ts) AS end_ts,
        percentile_approx(latitude,  0.5) AS median_latitude,
        percentile_approx(longitude, 0.5) AS median_longitude
    FROM stationary_points
    GROUP BY caid, ev_id
    HAVING (MAX(ts) - MIN(ts)) >= {min_stop_duration} * 1000
),
-- Second pass: drop points that drifted outside the radius, then recompute.
-- This keeps a slow walk away from a stop from dragging the centroid with it.
filtered_points AS (
    SELECT sp.caid, sp.ts, sp.latitude, sp.longitude, sp.ev_id
    FROM stationary_points sp
    JOIN initial_stops ist ON sp.caid = ist.caid AND sp.ev_id = ist.stop_id
    WHERE {dist_centroid} <= {distance_threshold}
),
stops AS (
    SELECT caid, ev_id AS stop_id,
        MIN(ts) AS start_ts, MAX(ts) AS end_ts,
        percentile_approx(latitude,  0.5) AS median_latitude,
        percentile_approx(longitude, 0.5) AS median_longitude
    FROM filtered_points
    GROUP BY caid, ev_id
    HAVING (MAX(ts) - MIN(ts)) >= {min_stop_duration} * 1000
)
SELECT caid,
    start_ts DIV 1000 AS start_timestamp,
    end_ts   DIV 1000 AS end_timestamp,
    median_latitude, median_longitude,
    ROW_NUMBER() OVER (PARTITION BY caid ORDER BY start_ts) AS stop_events
FROM (SELECT DISTINCT * FROM stops)
"""


def detect_stops(
    spark, pings, distance_threshold=20, time_threshold=3600, min_stop_duration=300
):
    """Flag stationary runs in a ping stream and reduce each to one stop."""
    pings.createOrReplaceTempView("raw_pings")
    sql = _STOPS_SQL.format(
        distance_threshold=distance_threshold,
        time_threshold=time_threshold,
        min_stop_duration=min_stop_duration,
        dist_next=haversine_sql("latitude", "longitude", "next_lat", "next_lon"),
        dist_centroid=haversine_sql(
            "sp.latitude", "sp.longitude", "ist.median_latitude", "ist.median_longitude"
        ),
    )
    return spark.sql(sql)


# --------------------------------------------------------------------------
# Stage 2: per-device DBSCAN over stop centroids
# --------------------------------------------------------------------------

_CLUSTER_SCHEMA = StructType(
    [
        StructField("caid", StringType(), False),
        StructField("stop_event", IntegerType(), False),
        StructField("start_timestamp", LongType(), False),
        StructField("end_timestamp", LongType(), False),
        StructField("stay_duration", LongType(), False),
        StructField("stop_latitude", DoubleType(), False),
        StructField("stop_longitude", DoubleType(), False),
        StructField("cluster_label", IntegerType(), False),
        StructField("cluster_counts", IntegerType(), False),
        StructField("cluster_latitude", DoubleType(), False),
        StructField("cluster_longitude", DoubleType(), False),
    ]
)


def _dbscan_device(pdf, eps_m, min_pts):
    """DBSCAN one device's stops. Runs inside applyInPandas, one task per caid."""
    from sklearn.cluster import DBSCAN

    pdf = (
        pdf.sort_values(["start_timestamp", "stop_events"], kind="mergesort")
        .drop_duplicates(["caid", "stop_events"])
        .reset_index(drop=True)
    )

    # haversine metric wants radians and an eps in radians of arc.
    coords = np.radians(pdf[["median_latitude", "median_longitude"]].to_numpy(float))
    labels = DBSCAN(
        eps=eps_m / EARTH_RADIUS_M,
        min_samples=min_pts,
        metric="haversine",
        algorithm="ball_tree",
    ).fit_predict(coords)
    pdf["cluster_label"] = labels

    # Relabel clusters by position so ids are stable across runs rather than
    # depending on the order DBSCAN happened to encounter points.
    clustered = pdf["cluster_label"] >= 0
    if clustered.any():
        order = (
            pdf[clustered]
            .groupby("cluster_label")[["median_latitude", "median_longitude"]]
            .mean()
            .sort_values(["median_latitude", "median_longitude"], kind="mergesort")
        )
        relabel = dict(zip(order.index, range(len(order))))
        pdf.loc[clustered, "cluster_label"] = pdf.loc[clustered, "cluster_label"].map(
            relabel
        )

    groups = pdf[clustered].groupby("cluster_label")
    pdf["cluster_counts"] = pdf["cluster_label"].map(groups.size()).fillna(1)
    pdf["cluster_latitude"] = pdf["cluster_label"].map(
        groups["median_latitude"].median()
    )
    pdf["cluster_longitude"] = pdf["cluster_label"].map(
        groups["median_longitude"].median()
    )
    # Noise points (label -1) are their own location: fall back to the stop.
    pdf["cluster_latitude"] = pdf["cluster_latitude"].fillna(pdf["median_latitude"])
    pdf["cluster_longitude"] = pdf["cluster_longitude"].fillna(pdf["median_longitude"])

    return pd.DataFrame(
        {
            "caid": pdf["caid"].astype(str),
            "stop_event": pdf["stop_events"].astype(int),
            "start_timestamp": pdf["start_timestamp"].astype("int64"),
            "end_timestamp": pdf["end_timestamp"].astype("int64"),
            "stay_duration": (pdf["end_timestamp"] - pdf["start_timestamp"]).astype(
                "int64"
            ),
            "stop_latitude": pdf["median_latitude"].astype(float),
            "stop_longitude": pdf["median_longitude"].astype(float),
            "cluster_label": pdf["cluster_label"].astype(int),
            "cluster_counts": pdf["cluster_counts"].astype(int),
            "cluster_latitude": pdf["cluster_latitude"].astype(float),
            "cluster_longitude": pdf["cluster_longitude"].astype(float),
        }
    )


def cluster_stops(stops, distance=20, min_points=2):
    """Group each device's stops into places with DBSCAN."""
    eps_m, min_pts = float(distance), int(min_points)
    return stops.groupBy("caid").applyInPandas(
        lambda pdf: _dbscan_device(pdf, eps_m, min_pts), schema=_CLUSTER_SCHEMA
    )


# --------------------------------------------------------------------------
# Stage 3: local time + home/work
# --------------------------------------------------------------------------


def add_local_time(clusters, time_zone):
    """Attach local-time fields used by the home/work rules.

    Uses the dataset's single configured timezone. (sedona-lbs inferred a
    per-stop timezone by point-in-polygon against a tz shapefile; that needs
    a geometry engine and is deliberately out of scope here.)
    """
    local = F.from_utc_timestamp(F.timestamp_seconds("start_timestamp"), time_zone)
    return (
        clusters.withColumn("local_time", local)
        .withColumn("date", F.to_date(local))
        .withColumn("hour_of_day", F.hour(local))
        .withColumn("day_of_week", F.dayofweek(local))
        .withColumn("weekend", F.dayofweek(local).isin(1, 7))
    )


_HOME_WORK_SQL = """
WITH base AS (SELECT * FROM local_stops),
cluster_day_counts AS (
    SELECT caid, cluster_label, COUNT(DISTINCT date) AS unique_days_visited
    FROM base WHERE cluster_label >= 0
    GROUP BY caid, cluster_label
),
home_candidates AS (
    SELECT caid, cluster_label,
        AVG(cluster_latitude)  AS cluster_latitude,
        AVG(cluster_longitude) AS cluster_longitude,
        COUNT(*) AS home_time_visits,
        COUNT(DISTINCT date) AS home_time_unique_days
    FROM base
    WHERE cluster_label >= 0
      AND (weekend OR hour_of_day >= {home_hour_evening} OR hour_of_day < {home_hour_morning})
    GROUP BY caid, cluster_label
),
home_locations AS (
    SELECT h.*,
        ROW_NUMBER() OVER (PARTITION BY h.caid
            ORDER BY h.home_time_visits DESC, h.home_time_unique_days DESC, h.cluster_label
        ) AS home_rank
    FROM home_candidates h
    JOIN cluster_day_counts dc ON h.caid = dc.caid AND h.cluster_label = dc.cluster_label
    WHERE dc.unique_days_visited >= {home_min_days}
),
final_home_locations AS (
    SELECT caid, cluster_label AS home_cluster_label,
        cluster_latitude AS home_latitude, cluster_longitude AS home_longitude
    FROM home_locations WHERE home_rank = 1
),
work_candidates AS (
    SELECT caid, cluster_label,
        AVG(cluster_latitude)  AS cluster_latitude,
        AVG(cluster_longitude) AS cluster_longitude,
        COUNT(*) AS work_time_visits,
        COUNT(DISTINCT date) AS work_time_unique_days
    FROM base
    WHERE cluster_label >= 0 AND NOT weekend
      AND hour_of_day >= {work_hour_start} AND hour_of_day <= {work_hour_end}
    GROUP BY caid, cluster_label
),
work_locations AS (
    SELECT w.caid, w.cluster_label AS work_cluster_label,
        ROW_NUMBER() OVER (PARTITION BY w.caid
            ORDER BY w.work_time_visits DESC, w.work_time_unique_days DESC, w.cluster_label
        ) AS work_rank
    FROM work_candidates w
    JOIN final_home_locations h ON w.caid = h.caid
    WHERE w.cluster_label <> h.home_cluster_label
      AND w.work_time_unique_days >= {work_min_days}
      AND {dist_home_work} >= {min_distance_home_work}
),
final_work_locations AS (
    SELECT caid, work_cluster_label FROM work_locations WHERE work_rank = 1
)
SELECT b.*,
    CASE
        WHEN b.cluster_label = -1 THEN {TYPE_OTHER}
        WHEN b.cluster_label = h.home_cluster_label THEN {TYPE_HOME}
        WHEN b.cluster_label = w.work_cluster_label THEN {TYPE_WORK}
        ELSE {TYPE_OTHER}
    END AS type,
    h.home_latitude, h.home_longitude,
    w.work_cluster_label
FROM base b
LEFT JOIN final_home_locations h ON b.caid = h.caid
LEFT JOIN final_work_locations w ON b.caid = w.caid
"""


def classify_home_work(
    spark,
    local_stops,
    home_min_days=7,
    home_hour_evening=22,
    home_hour_morning=7,
    work_min_days=4,
    work_hour_start=8,
    work_hour_end=18,
    min_distance_home_work=50,
):
    """Label each stay home / work / other from cluster visit patterns."""
    local_stops.createOrReplaceTempView("local_stops")
    sql = _HOME_WORK_SQL.format(
        home_min_days=home_min_days,
        home_hour_evening=home_hour_evening,
        home_hour_morning=home_hour_morning,
        work_min_days=work_min_days,
        work_hour_start=work_hour_start,
        work_hour_end=work_hour_end,
        min_distance_home_work=min_distance_home_work,
        dist_home_work=haversine_sql(
            "w.cluster_latitude",
            "w.cluster_longitude",
            "h.home_latitude",
            "h.home_longitude",
        ),
        TYPE_OTHER=TYPE_OTHER,
        TYPE_HOME=TYPE_HOME,
        TYPE_WORK=TYPE_WORK,
    )
    return spark.sql(sql)


# --------------------------------------------------------------------------
# Stage 4: canonical schema
# --------------------------------------------------------------------------

_h3_cell_udf = F.udf(
    lambda lat, lon, res: (
        None if lat is None or lon is None else str(h3int.latlng_to_cell(lat, lon, res))
    ),
    StringType(),
)
_h3_hex_udf = F.udf(
    lambda lat, lon, res: (
        None if lat is None or lon is None else h3.latlng_to_cell(lat, lon, res).upper()
    ),
    StringType(),
)


def to_canonical_schema(labelled, hex_resolution=8):
    """Reshape to the stay schema the rest of sparkmobility consumes.

    Matches the Scala backend's ``StayPointsWithHomeWork`` output column for
    column, including the dtypes TimeGeo's C++ stage is strict about:

    * ``h3_id_region`` -- decimal digits of the H3 cell (``align()`` casts it
      to int64; a hex string would raise).
    * ``h3_index`` / ``home_h3_index`` / ``work_h3_index`` -- uppercase HEX
      strings. The C++ reads these as an Arrow StringArray and silently drops
      home/work on any other type, which makes it skip the user entirely.
    * ``type`` -- int32 {0, 1, 2}, read as an Int32Array.

    The cluster centroid is what gets mapped to an H3 cell, so DBSCAN decides
    which stays share a place and the cell is only a label for it. The raw
    ``cluster_label`` cannot be used here: it is a small per-device integer,
    while downstream code calls ``h3.cell_to_boundary()`` on this value.
    """
    res = F.lit(hex_resolution)
    lat, lon = F.col("cluster_latitude"), F.col("cluster_longitude")

    df = (
        labelled.withColumn("h3_id_region", _h3_cell_udf(lat, lon, res))
        .withColumn("h3_index", _h3_hex_udf(lat, lon, res))
        .withColumn(
            "home_h3_index",
            _h3_hex_udf(F.col("home_latitude"), F.col("home_longitude"), res),
        )
    )

    # Work centroid: the cluster flagged as work for this user.
    work = (
        labelled.filter(F.col("cluster_label") == F.col("work_cluster_label"))
        .groupBy("caid")
        .agg(
            F.first("cluster_latitude").alias("w_lat"),
            F.first("cluster_longitude").alias("w_lon"),
        )
    )
    df = df.join(work, on="caid", how="left").withColumn(
        "work_h3_index", _h3_hex_udf(F.col("w_lat"), F.col("w_lon"), res)
    )

    return df.select(
        F.col("caid").cast("string"),
        F.col("stop_event").cast("long").alias("h3_region_stay_id"),
        F.timestamp_seconds("start_timestamp").alias("stay_start_timestamp"),
        F.timestamp_seconds("end_timestamp").alias("stay_end_timestamp"),
        # MICROSECONDS, not seconds -- matching the Scala backend, whose
        # stay_duration is a timestamp difference cast to long (Spark yields
        # microseconds). Emitting seconds here would make the two algorithms
        # disagree by 1e6 in any stay-duration plot. The unit is arguably
        # wrong on both sides; it is wrong consistently, which is what keeps
        # the two interchangeable.
        (F.col("stay_duration") * 1_000_000).cast("long").alias("stay_duration"),
        F.col("h3_id_region"),
        F.col("local_time"),
        F.col("h3_index"),
        F.col("day_of_week").cast("int"),
        F.col("hour_of_day").cast("int"),
        F.col("home_h3_index"),
        F.col("type").cast("int"),
        F.col("work_h3_index"),
        F.col("cluster_label").cast("int"),
        F.col("stop_latitude").cast("double").alias("latitude"),
        F.col("stop_longitude").cast("double").alias("longitude"),
    )


# --------------------------------------------------------------------------
# Public class
# --------------------------------------------------------------------------


class ClusterStayDetection:
    """Density-based stay detection over a :class:`MobilityDataset`.

    Mirrors :class:`GridStayDetection`'s shape -- same constructor, same
    ``get_stays()`` entry point, same output paths -- so the two are
    swappable. The parameters differ because the algorithms differ.
    """

    def __init__(self, MobilityDataset):
        self.dataset = MobilityDataset

    def _read_pings(self, spark):
        """Load raw pings, apply the dataset's column map and filters."""
        df = spark.read.parquet(self.dataset.input_path)

        for original, canonical in (self.dataset.column_names or {}).items():
            if original != canonical and original in df.columns:
                df = df.withColumnRenamed(original, canonical)

        ts = F.col("utc_timestamp")
        if self.dataset.time_format != "UNIX":
            # Parse as UTC wall-clock explicitly. Spark's unix_timestamp()
            # would parse in the session timezone and shift the window.
            ts = F.to_unix_timestamp(
                F.to_timestamp(ts.cast("string"), self.dataset.time_format)
            )

        df = df.select(
            F.col("caid").cast("string"),
            F.col("latitude").cast("double"),
            F.col("longitude").cast("double"),
            ts.cast("long").alias("utc_timestamp"),
        ).na.drop(subset=["caid", "latitude", "longitude", "utc_timestamp"])

        start = F.unix_timestamp(F.lit(self.dataset.start_datetime).cast("timestamp"))
        end = F.unix_timestamp(F.lit(self.dataset.end_datetime).cast("timestamp"))
        df = df.filter(F.col("utc_timestamp").between(start, end))

        if self.dataset.longitude and self.dataset.latitude:
            lon0, lon1 = sorted(self.dataset.longitude)
            lat0, lat1 = sorted(self.dataset.latitude)
            df = df.filter(
                F.col("longitude").between(lon0, lon1)
                & F.col("latitude").between(lat0, lat1)
            )
        return df

    @spark_session
    def get_stays(
        spark,
        self,
        distance_threshold=20,
        time_threshold=3600,
        min_stop_duration=300,
        dbscan_distance=20,
        dbscan_min_points=2,
        hex_resolution=8,
        find_home_and_work=True,
        home_min_days=7,
        home_hour_evening=22,
        home_hour_morning=7,
        work_min_days=4,
        work_hour_start=8,
        work_hour_end=18,
        min_distance_home_work=50,
    ):
        """Run the full pipeline and write stays under the dataset's output path.

        Writes ``StayPoints`` always, and ``StayPointsWithHomeWork`` when
        ``find_home_and_work`` is set -- the same layout GridStayDetection
        uses, so ``MobilityDataset.load_stays()`` works either way.
        """
        out = self.dataset.output_path

        pings = self._read_pings(spark)
        stops = detect_stops(
            spark,
            pings,
            distance_threshold=distance_threshold,
            time_threshold=time_threshold,
            min_stop_duration=min_stop_duration,
        )
        clusters = cluster_stops(
            stops, distance=dbscan_distance, min_points=dbscan_min_points
        )
        local = add_local_time(clusters, self.dataset.time_zone)

        if not find_home_and_work:
            local.write.mode("overwrite").parquet(f"{out}/StayPoints")
            return "Stay detection completed"

        # Reused by the home/work aggregations and then by the final join;
        # without this the whole stop+DBSCAN pipeline recomputes each time.
        local = local.cache()
        labelled = classify_home_work(
            spark,
            local,
            home_min_days=home_min_days,
            home_hour_evening=home_hour_evening,
            home_hour_morning=home_hour_morning,
            work_min_days=work_min_days,
            work_hour_start=work_hour_start,
            work_hour_end=work_hour_end,
            min_distance_home_work=min_distance_home_work,
        )
        canonical = to_canonical_schema(labelled, hex_resolution=hex_resolution)
        canonical.write.mode("overwrite").parquet(f"{out}/StayPointsWithHomeWork")
        canonical.drop("home_h3_index", "work_h3_index", "type").write.mode(
            "overwrite"
        ).parquet(f"{out}/StayPoints")
        local.unpersist()
        return "Stay detection completed with home and work locations labeled"
