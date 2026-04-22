import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


# Configure parquet engine to handle PyArrow issues
def configure_parquet_engine():
    """Configure the best available parquet engine."""
    try:
        pass

        return "pyarrow"
    except (ImportError, ValueError):
        try:
            pass

            return "fastparquet"
        except ImportError:
            import os

            os.system("conda install fastparquet -y")
            return "fastparquet"


PARQUET_ENGINE = configure_parquet_engine()


def read_parquet_robust(file_path):
    """Read parquet file with fallback engine support."""
    try:
        return pd.read_parquet(file_path, engine=PARQUET_ENGINE)
    except Exception as e:
        logger.error(f"Error reading {file_path} with {PARQUET_ENGINE}: {e}")
        alt_engine = "fastparquet" if PARQUET_ENGINE == "pyarrow" else "pyarrow"
        logger.info(f"Trying alternative engine: {alt_engine}")
        return pd.read_parquet(file_path, engine=alt_engine)


def to_parquet_robust(df, file_path, **kwargs):
    """Write parquet file with fallback engine support."""
    try:
        return df.to_parquet(file_path, engine=PARQUET_ENGINE, **kwargs)
    except Exception as e:
        logger.error(f"Error writing {file_path} with {PARQUET_ENGINE}: {e}")
        alt_engine = "fastparquet" if PARQUET_ENGINE == "pyarrow" else "pyarrow"
        logger.info(f"Trying alternative engine: {alt_engine}")
        return df.to_parquet(file_path, engine=alt_engine, **kwargs)


##--------------------------2-Parameters---------------------------
## 3-ParameterValues.py
def decode_and_write_parameters(
    b1_array,
    b2_array,
    commuter_input_path,
    noncommuter_input_path,
    commuter_output_path,
    noncommuter_output_path,
):
    """
    Decode indexed b1 and b2 values in commuter and non-commuter parameter files,
    replacing them with actual numeric values, and write the results to new files.

    Parameters:
        b1_array (list[int]): Mapping array for b1 values.
        b2_array (list[int]): Mapping array for b2 values.
        commuter_input_path (str): Path to the original commuter parameter file.
        noncommuter_input_path (str): Path to the original non-commuter parameter file.
        commuter_output_path (str): Path to the output commuter parameter file.
        noncommuter_output_path (str): Path to the output non-commuter parameter file.
    """

    def process_file(input_path, output_path, b1_array, b2_array):
        with open(output_path, "w") as g:
            with open(input_path, "r") as f:
                for line in f:
                    parts = line.strip().split(" ")
                    if len(parts) < 3:
                        continue  # Skip malformed lines
                    try:
                        # Replace index with actual b1 and b2 values
                        parts[0] = str(b1_array[int(parts[0])])
                        parts[1] = str(b2_array[int(parts[1])])
                    except IndexError:
                        logger.error(f"IndexError in line: {line}")
                        continue
                    g.write(" ".join(parts) + "\n")

    # Process both commuter and non-commuter parameter files
    process_file(commuter_input_path, commuter_output_path, b1_array, b2_array)
    process_file(noncommuter_input_path, noncommuter_output_path, b1_array, b2_array)

    logger.info(
        f"Decoded parameters written to:\n  {commuter_output_path}\n  {noncommuter_output_path}"
    )


##--------------------------3_SRFiltered_to_SimInput----------------
## 0_removeRedundance.py


def remove_redundant_stays_parquet(input_path, output_path):
    """
    Remove consecutive stays at the same location from parquet data.

    Args:
        input_path (str): Path to input parquet file
        output_path (str): Path to output parquet file
    """
    # Load data
    df = read_parquet_robust(input_path)

    # Sort by user and timestamp
    df = df.sort_values(["caid", "timestamp"]).reset_index(drop=True)

    # Create location identifier combining longitude and latitude
    df["location_id"] = df["Longitude"].astype(str) + "_" + df["Latitude"].astype(str)

    # Remove consecutive stays at same location
    df["prev_user"] = df["caid"].shift(1)
    df["prev_location"] = df["location_id"].shift(1)

    # Keep first record for each user and records where location changed
    mask = (df["caid"] != df["prev_user"]) | (df["location_id"] != df["prev_location"])
    df_filtered = df[mask].copy()

    # Drop helper columns
    df_filtered = df_filtered.drop(
        ["location_id", "prev_user", "prev_location"], axis=1
    )

    # Save as parquet
    to_parquet_robust(df_filtered, output_path, index=False)
    logger.info(
        f"Removed redundant stays, saved {len(df_filtered)} records to {output_path}"
    )


def parse_parquet_line(row):
    """Parse a parquet row into the format expected by downstream functions"""
    user = str(row["caid"])
    timestamp = int(row["timestamp"])
    trip_purpose = row["type"]
    longitude = str(row["Longitude"])
    latitude = str(row["Latitude"])

    LAtime = datetime.utcfromtimestamp(timestamp)
    date = datetime.strftime(LAtime, "%Y-%m-%d")
    time = float(LAtime.hour) + float(LAtime.minute) / 60 + float(LAtime.second) / 3600

    return [user, date, time, trip_purpose, longitude, latitude]


def extract_frequent_users_parquet(input_path, output_path, num_stays_threshold=15):
    """
    Extract frequent users from parquet data and save as parquet.

    Args:
        input_path (str): Path to input parquet file
        output_path (str): Path to output parquet file
        num_stays_threshold (int): Minimum number of distinct stays
    """
    # Load data
    df = read_parquet_robust(input_path)

    # Create location identifier
    df["location_id"] = df["Longitude"].astype(str) + "_" + df["Latitude"].astype(str)

    # Count unique locations per user
    user_location_counts = df.groupby("caid")["location_id"].nunique().reset_index()
    user_location_counts.columns = ["caid", "num_locations"]

    # Filter frequent users
    frequent_users = user_location_counts[
        user_location_counts["num_locations"] > num_stays_threshold
    ]["caid"]

    # Save as parquet
    frequent_users_df = pd.DataFrame({"caid": frequent_users})
    to_parquet_robust(frequent_users_df, output_path, index=False)

    logger.info(f"Saved {len(frequent_users)} frequent users to {output_path}")
    return frequent_users.tolist()


def extract_stay_regions_for_frequent_users_parquet(
    fa_users_path, input_path, output_path
):
    """
    Filter stay regions for frequent users using parquet files.

    Args:
        fa_users_path (str): Path to frequent users parquet file
        input_path (str): Path to input stay regions parquet file
        output_path (str): Path to output parquet file
    """
    # Load frequent users
    frequent_users_df = read_parquet_robust(fa_users_path)
    frequent_users = set(frequent_users_df["caid"].tolist())

    # Load stay regions data
    df = read_parquet_robust(input_path)

    # Filter for frequent users only
    df_filtered = df[df["caid"].isin(frequent_users)].copy()

    # Sort by user and timestamp
    df_filtered = df_filtered.sort_values(["caid", "timestamp"]).reset_index(drop=True)

    # Create location identifier for consecutive duplicate removal
    df_filtered["location_id"] = (
        df_filtered["Longitude"].astype(str) + "_" + df_filtered["Latitude"].astype(str)
    )

    # Remove consecutive duplicates within each user
    df_filtered["prev_user"] = df_filtered["caid"].shift(1)
    df_filtered["prev_location"] = df_filtered["location_id"].shift(1)

    mask = (df_filtered["caid"] != df_filtered["prev_user"]) | (
        df_filtered["location_id"] != df_filtered["prev_location"]
    )
    df_final = df_filtered[mask].copy()

    # Drop helper columns
    df_final = df_final.drop(["location_id", "prev_user", "prev_location"], axis=1)

    # Add derived columns for compatibility
    df_final["date"] = pd.to_datetime(df_final["timestamp"], unit="s").dt.strftime(
        "%Y-%m-%d"
    )
    df_final["time"] = (
        pd.to_datetime(df_final["timestamp"], unit="s").dt.hour
        + pd.to_datetime(df_final["timestamp"], unit="s").dt.minute / 60
        + pd.to_datetime(df_final["timestamp"], unit="s").dt.second / 3600
    )

    # Save as parquet
    to_parquet_robust(df_final, output_path, index=False)
    logger.info(f"Filtered stay regions written to: {output_path}")


def clean_and_format_fa_users_parquet(input_path, output_path):
    """
    Clean and format frequent user data using parquet files.
    Vectorized with groupby; equivalent to the per-user iterrows loop:
    within each user, a (lon, lat) location is labeled by its first stay's
    trip_purpose — 'h' for home, 'w' for work, else a sequential integer
    (1, 2, ...) in order of first appearance.
    """
    df = read_parquet_robust(input_path)

    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%d")
    if "time" not in df.columns:
        ts = pd.to_datetime(df["timestamp"], unit="s")
        df["time"] = ts.dt.hour + ts.dt.minute / 60 + ts.dt.second / 3600

    # Map `type` → trip_purpose. Post-align_data `type` is a string; keep
    # legacy int fallbacks for standalone callers.
    type_map = {
        "home": "h",
        "work": "w",
        "other": "o",
        0: "h",
        1: "w",
        "0": "h",
        "1": "w",
    }
    df["trip_purpose"] = df["type"].map(type_map).fillna("o")

    df["location_key"] = df["Longitude"].astype(str) + "_" + df["Latitude"].astype(str)

    df = df.sort_values(["caid", "timestamp"]).reset_index(drop=True)

    # Per (caid, location_key), take the first stay's trip_purpose + timestamp.
    first_stays = df.groupby(["caid", "location_key"], sort=False, as_index=False).agg(
        first_purpose=("trip_purpose", "first"),
        first_ts=("timestamp", "first"),
    )
    first_stays["location_index"] = first_stays["first_purpose"]

    # For non-h/w locations, assign a per-user sequential integer ordered by
    # first_ts (matches the original increment-on-new-other-location behavior).
    other = first_stays[~first_stays["first_purpose"].isin(["h", "w"])].copy()
    other = other.sort_values(["caid", "first_ts"])
    other_idx = (other.groupby("caid").cumcount() + 1).astype(str)
    first_stays.loc[other.index, "location_index"] = other_idx.values

    df = df.merge(
        first_stays[["caid", "location_key", "location_index"]],
        on=["caid", "location_key"],
        how="left",
    )

    df = df[
        [
            "caid",
            "date",
            "time",
            "trip_purpose",
            "Longitude",
            "Latitude",
            "location_index",
            "timestamp",
        ]
    ]

    # Collapse consecutive rows with same (caid, location_index).
    prev_user = df["caid"].shift(1)
    prev_loc = df["location_index"].shift(1)
    df = df[(df["caid"] != prev_user) | (df["location_index"] != prev_loc)].copy()

    to_parquet_robust(df, output_path, index=False)
    logger.info(f"Cleaned and formatted data written to: {output_path}")
