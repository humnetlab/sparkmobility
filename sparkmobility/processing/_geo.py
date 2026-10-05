"""Great-circle distance helpers for Spark SQL expressions.

Kept separate from :mod:`sparkmobility.dataset` so stay-detection code can
use metre-valued distances without importing the plotting stack.
"""

from pyspark.sql import Column
from pyspark.sql import functions as F

# Mean Earth radius. This specific value matters: the stop-detection and
# home/work thresholds in ClusterStayDetection were calibrated against
# distances computed with it, so changing it shifts which pings merge into
# a stop. MobilityDataset.haversine uses 6371000.0 and returns kilometres --
# do not swap one for the other.
EARTH_RADIUS_M = 6371008.0


def haversine_meters(lat1: Column, lon1: Column, lat2: Column, lon2: Column) -> Column:
    """Great-circle distance in metres between two lat/lon Columns.

    Clamped before the square roots so floating-point error on identical or
    antipodal points can't produce a NaN.
    """
    phi1, phi2 = F.radians(lat1), F.radians(lat2)
    d_phi = F.radians(lat2 - lat1)
    d_lambda = F.radians(lon2 - lon1)

    a = F.pow(F.sin(d_phi / 2), 2) + F.cos(phi1) * F.cos(phi2) * F.pow(
        F.sin(d_lambda / 2), 2
    )
    c = F.lit(2.0) * F.atan2(
        F.sqrt(F.least(F.lit(1.0), a)),
        F.sqrt(F.greatest(F.lit(0.0), F.lit(1.0) - a)),
    )
    return F.lit(EARTH_RADIUS_M) * c


def haversine_sql(lat1: str, lon1: str, lat2: str, lon2: str) -> str:
    """Same formula as a SQL fragment, for use inside spark.sql() strings.

    The stop-detection pipeline is written as SQL so the whole thing stays
    one Catalyst plan; this keeps the distance expression in one place
    rather than repeating the trig at every call site.
    """
    return f"""(
        {EARTH_RADIUS_M} * 2 * ATAN2(
            SQRT(LEAST(1.0,
                POW(SIN(RADIANS({lat2} - {lat1}) / 2), 2)
                + COS(RADIANS({lat1})) * COS(RADIANS({lat2}))
                  * POW(SIN(RADIANS({lon2} - {lon1}) / 2), 2)
            )),
            SQRT(GREATEST(0.0, 1.0 - (
                POW(SIN(RADIANS({lat2} - {lat1}) / 2), 2)
                + COS(RADIANS({lat1})) * COS(RADIANS({lat2}))
                  * POW(SIN(RADIANS({lon2} - {lon1}) / 2), 2)
            )))
        )
    )"""
