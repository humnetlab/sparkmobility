"""Stay detection and user filtering.

Two stay-detection algorithms are available. Both reduce a ping stream to
stays and then group those stays into places; they differ in how places are
formed, which is the choice that matters for downstream analysis:

``GridStayDetection``
    Sequential-centroid detection (Zheng et al. 2010) with places snapped to
    H3 cells at a fixed resolution. Compute runs in the Scala/Spark backend.

``ClusterStayDetection``
    Distance/time-threshold detection with places formed by per-device DBSCAN
    clustering, so recurring visits collapse without being quantised to a
    grid. Pure PySpark.

Both write the same output schema, so ``MobilityDataset``, ``UserSelection``
and the models consume either interchangeably.
"""

import warnings

from .cluster_stay_detection import ClusterStayDetection
from .grid_stay_detection import GridStayDetection
from .user_selection import UserSelection

__all__ = [
    "ClusterStayDetection",
    "GridStayDetection",
    "StayDetection",
    "UserSelection",
]


def __getattr__(name):
    # `StayDetection` was the only stay detector before ClusterStayDetection
    # existed, so the bare name no longer says which algorithm is meant.
    # Kept working for code written against v1.0.0.
    if name == "StayDetection":
        warnings.warn(
            "StayDetection has been renamed to GridStayDetection, now that "
            "ClusterStayDetection offers a second algorithm. The old name "
            "still works but will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return GridStayDetection
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
