"""
DataFrame-facing plot helpers for TimeGeo output.

The underlying implementations in ``_pipeline.s5_aggregated_plots`` read
parquet files; these wrappers materialize the caller's DataFrames to a
scratch tempdir and delegate.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from sparkmobility.models.timegeo._pipeline import s5_aggregated_plots as _plots

# TimeGeo.generate() returns a DataFrame keyed on ``user`` / ``datetime``,
# but the parquet helpers in s5_aggregated_plots expect the raw simulation
# schema (``user_id`` / ``timeslot``). ``_normalize_schema`` converts back
# so callers can pass either flavor transparently.
_EPOCH = pd.Timestamp("1970-01-01")


def _normalize_schema(df: pd.DataFrame, slot_interval_seconds: int) -> pd.DataFrame:
    if "user_id" in df.columns and "timeslot" in df.columns:
        return df
    df = df.rename(columns={"user": "user_id"})
    df["timeslot"] = (
        ((df["datetime"] - _EPOCH).dt.total_seconds() / slot_interval_seconds)
        .round()
        .astype("int64")
    )
    return df


def _to_parquet(df: pd.DataFrame, path: Path) -> Path:
    df.to_parquet(path, index=False)
    return path


def plot_hourly_trips(
    mapped_dir: str | Path,
    output: str | Path,
) -> None:
    """Plot hourly trip counts from the raw Mapped/ directory of a fitted model."""
    _plots.plot_hourly_trip_counts(mapped_dir=str(mapped_dir), output=str(output))


def plot_stay_durations(
    traj_df: pd.DataFrame,
    observed: Optional[pd.DataFrame] = None,
    output: str | Path = "stay_durations.png",
    slot_interval_seconds: int = 600,
) -> None:
    """Plot stay-duration CDF for simulation (and optional observed) data."""
    with tempfile.TemporaryDirectory() as tmp:
        sim_p = _to_parquet(
            _normalize_schema(traj_df, slot_interval_seconds),
            Path(tmp) / "sim.parquet",
        )
        obs_p = (
            _to_parquet(
                _normalize_schema(observed, slot_interval_seconds),
                Path(tmp) / "obs.parquet",
            )
            if observed is not None
            else None
        )
        _plots.plot_stay_durations_parquet(
            sim_parquet=str(sim_p),
            cdr_parquet=str(obs_p) if obs_p else None,
            output_file=str(output),
        )


def plot_trip_distance(
    traj_df: pd.DataFrame,
    observed: Optional[pd.DataFrame] = None,
    output: str | Path = "trip_distance.png",
    slot_interval_seconds: int = 600,
) -> None:
    """Plot trip-distance CDF."""
    with tempfile.TemporaryDirectory() as tmp:
        sim_p = _to_parquet(
            _normalize_schema(traj_df, slot_interval_seconds),
            Path(tmp) / "sim.parquet",
        )
        obs_p = (
            _to_parquet(
                _normalize_schema(observed, slot_interval_seconds),
                Path(tmp) / "obs.parquet",
            )
            if observed is not None
            else None
        )
        _plots.plot_trip_distance_parquet(
            sim_parquet=str(sim_p),
            cdr_parquet=str(obs_p) if obs_p else None,
            output_file=str(output),
        )


def plot_daily_locations(
    traj_df: pd.DataFrame,
    observed: Optional[pd.DataFrame] = None,
    output: str | Path = "daily_locations.png",
    slot_interval_seconds: int = 600,
) -> None:
    """Plot daily-visited-locations distribution."""
    with tempfile.TemporaryDirectory() as tmp:
        sim_p = _to_parquet(
            _normalize_schema(traj_df, slot_interval_seconds),
            Path(tmp) / "sim.parquet",
        )
        obs_p = (
            _to_parquet(
                _normalize_schema(observed, slot_interval_seconds),
                Path(tmp) / "obs.parquet",
            )
            if observed is not None
            else None
        )
        _plots.plot_daily_visited_locations_parquet(
            sim_parquet=str(sim_p),
            cdr_parquet=str(obs_p) if obs_p else None,
            output_file=str(output),
        )


def plot_location_rank(
    traj_df: pd.DataFrame,
    observed: Optional[pd.DataFrame] = None,
    output: str | Path = "location_rank.png",
    top_n: int = 50,
    slot_interval_seconds: int = 600,
) -> None:
    """Plot location-rank frequency."""
    with tempfile.TemporaryDirectory() as tmp:
        sim_p = _to_parquet(
            _normalize_schema(traj_df, slot_interval_seconds),
            Path(tmp) / "sim.parquet",
        )
        obs_p = (
            _to_parquet(
                _normalize_schema(observed, slot_interval_seconds),
                Path(tmp) / "obs.parquet",
            )
            if observed is not None
            else None
        )
        _plots.plot_location_rank_parquet(
            sim_parquet=str(sim_p),
            cdr_parquet=str(obs_p) if obs_p else None,
            output_file=str(output),
            top_n=top_n,
        )
