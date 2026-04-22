import logging
import os
import random

import h3
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

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


# 5_Simulation/1_formatStays.py
def generate_simulation_input_parquet(
    input_path="./results/SRFiltered_to_SimInput/FAUsers_Cleaned_Formatted.parquet",
    output_path="./results/Simulation/simulation_location.txt",
):
    """
    Generate simulation input file from parquet data.
    Handles both aligned data format (with Longitude/Latitude columns) and original format.
    """
    df = read_parquet_robust(input_path)
    logger.info(f"Loaded {len(df)} records from {input_path}")
    logger.info(f"Data columns: {df.columns.tolist()}")

    # Check if this is aligned data format (has Longitude/Latitude) or original data format
    if "Longitude" in df.columns and "Latitude" in df.columns:
        # Use aligned data format - coordinates are already available
        return _generate_simulation_input_aligned_coords(df, output_path)
    elif "trip_purpose" in df.columns:
        # Use aligned data format
        return _generate_simulation_input_aligned(df, output_path)
    else:
        # Use original data format
        return _generate_simulation_input_original(df, output_path)


def _write_user_location_blocks(df, output_path):
    """Write per-user location blocks from a df with columns
    [caid, location_id, Longitude, Latitude], already sorted within each
    user by time. Counts per (caid, location_id) and writes the first-seen
    coords, preserving the order each location_id first appears."""
    df = df.assign(_pos=df.groupby("caid").cumcount())
    agg = (
        df.groupby(["caid", "location_id"], sort=False)
        .agg(
            count=("location_id", "size"),
            lon=("Longitude", "first"),
            lat=("Latitude", "first"),
            first_pos=("_pos", "min"),
        )
        .reset_index()
        .sort_values(["caid", "first_pos"], kind="stable")
    )
    agg["line"] = (
        agg["location_id"].astype(str)
        + " "
        + agg["count"].astype(str)
        + " "
        + agg["lon"].astype(str)
        + " "
        + agg["lat"].astype(str)
        + "\n"
    )
    with open(output_path, "w") as g:
        for caid, group in agg.groupby("caid", sort=False):
            g.write(f"{caid}\n")
            g.writelines(group["line"].tolist())


def _location_id_from_trip_purpose(trip_purpose, location_index):
    """Vectorized: 'h'/'w' → themselves; other → str(int(location_index)+1),
    with non-numeric location_index mapped to '1'."""
    idx = pd.to_numeric(location_index, errors="coerce").fillna(0).astype(int) + 1
    loc = idx.astype(str)
    loc = loc.mask(trip_purpose == "h", "h").mask(trip_purpose == "w", "w")
    return loc


def _generate_simulation_input_aligned_coords(df, output_path):
    """Generate simulation input from aligned data format with Longitude/Latitude columns."""
    df = df.sort_values(["caid", "timestamp"], kind="stable").reset_index(drop=True)
    df = df.assign(
        caid=df["caid"].astype(str),
        location_id=_location_id_from_trip_purpose(
            df["trip_purpose"], df["location_index"]
        ),
        Longitude=df["Longitude"].astype(str),
        Latitude=df["Latitude"].astype(str),
    )
    _write_user_location_blocks(
        df[["caid", "location_id", "Longitude", "Latitude"]], output_path
    )
    logger.info(f"Simulation input written to: {output_path}")
    logger.info("Used aligned data format with direct Longitude/Latitude coordinates")


def _generate_simulation_input_aligned(df, output_path):
    """Generate simulation input from aligned data format."""
    df = df.sort_values(["caid", "timestamp"], kind="stable").reset_index(drop=True)
    df = df.assign(
        caid=df["caid"].astype(str),
        location_id=_location_id_from_trip_purpose(
            df["trip_purpose"], df["location_index"]
        ),
        Longitude=df["Longitude"].astype(str),
        Latitude=df["Latitude"].astype(str),
    )
    _write_user_location_blocks(
        df[["caid", "location_id", "Longitude", "Latitude"]], output_path
    )
    logger.info(f"Simulation input written to: {output_path}")


def _generate_simulation_input_original(df, output_path):
    """Generate simulation input from original data format."""

    def h3_to_lat_lng(h3_id):
        try:
            if isinstance(h3_id, (int, float)):
                h3_hex = format(int(h3_id), "x")
            else:
                h3_hex = str(h3_id)
            return h3.cell_to_boundary(h3_hex)
        except Exception as e:
            logger.warning(
                f"Warning: Failed to convert H3 ID {h3_id} to coordinates: {e}"
            )
            return (0.0, 0.0)

    df = df.sort_values(["caid", "stay_start_timestamp"], kind="stable").reset_index(
        drop=True
    )

    # Convert each unique H3 cell once, then map back.
    unique_h3 = df["h3_id_region"].drop_duplicates()
    h3_coords = {h: h3_to_lat_lng(h) for h in unique_h3}
    lat_lng = df["h3_id_region"].map(h3_coords)
    longitude = lat_lng.map(lambda p: p[1]).astype(str)
    latitude = lat_lng.map(lambda p: p[0]).astype(str)

    # location_id: 0 → 'h', 1 → 'w', else str(h3_region_stay_id+1) with '1' fallback
    if "h3_region_stay_id" in df.columns:
        stay_idx = (
            pd.to_numeric(df["h3_region_stay_id"], errors="coerce")
            .fillna(0)
            .astype(int)
            + 1
        )
        location_id = stay_idx.astype(str)
    else:
        location_id = pd.Series(["1"] * len(df), index=df.index)
    location_id = location_id.mask(df["type"] == 0, "h").mask(df["type"] == 1, "w")

    out = pd.DataFrame(
        {
            "caid": df["caid"].astype(str),
            "location_id": location_id,
            "Longitude": longitude,
            "Latitude": latitude,
        }
    )
    _write_user_location_blocks(out, output_path)
    logger.info(f"Simulation input written to: {output_path}")


def gen_gmm_1sample(cen, cov, mc):
    """
    Replace pypr_gmm.sample_gaussian_mixture using sklearn's GMM.

    Inputs:
        cen: list of 2D means, e.g., [ [x1, y1], [x2, y2], [x3, y3] ]
        cov: list of 2x2 covariance matrices, one per component
        mc: list of mixing coefficients, e.g., [0.3, 0.4, 0.3]

    Output:
        [x, y] where x and y are in range (0, 1440), then divided by 10
    """
    n_components = len(cen)
    gmm = GaussianMixture(n_components=n_components, covariance_type="full")

    # sklearn expects:
    # - means_ → array of shape (n_components, 2)
    # - covariances_ → array of shape (n_components, 2, 2)
    # - weights_ → array of shape (n_components,)
    gmm.weights_ = np.array(mc)
    gmm.means_ = np.array(cen)
    gmm.covariances_ = np.array(cov)
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))

    while True:
        s = gmm.sample(1)[0]  # shape (1, 2)
        if 0 < s[0, 0] < 1440 and 0 < s[0, 1] < 1440:
            break

    return [int(s[0, 0]), int(s[0, 1])]


def get_parameters(file_path):
    """
    Get parameter values from parameter file.
    """
    parameters = []
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return parameters

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) < 4:
                continue
            b1 = float(parts[0])
            b2 = float(parts[1])
            nw = float(parts[2])
            user = parts[-1]
            # Don't try to convert user to int - just check if it exists
            if user and user.strip():  # Check if user ID is not empty
                parameters.append([b1, b2, nw])
    return parameters


def generate_simulation_parameters(
    commuter_path="./results/Parameters/ParametersCommuters.txt",
    noncommuter_path="./results/Parameters/ParametersNonCommuters.txt",
    output_path="./results/Simulation/simulation_parameter.txt",
    work_prob_weekday=0.829,
    work_prob_weekend=0.354,
    num_days=1,
    reg_prob=0.846,  # prob of regular commuters
    gmm_group_index=0,  # 0: Regular commuters, 1: Flexible commuters
):
    """
    # This function builds a single flat file of "simulation-ready" parameter records for both commuter and non-commuter users. It draws each user's time-of-departure and trip-duration from a Gaussian mixture (GMM) model, assigns a random "regular vs. flexible" label, and encodes their daily work schedule over a specified number of days.

    gmm_group_index: int
        Index of GMM behavior group used to generate (ts, dur).
        - 0 = Regular commuters (default)
        - 1 = Flexible commuters (more variance)
    """

    list_par = [
        [
            [
                ([474.17242116, 450.36415361]),
                ([454.76611398, 540.17150463]),
                ([770.23785795, 396.00232714]),
            ],
            [
                ([[8458.74571565, -9434.51634444], [-9434.51634444, 36040.22889202]]),
                ([[3367.38775228, -1123.19558628], [-1123.19558628, 2680.86063147]]),
                ([[48035.69002421, -15435.34143709], [-15435.34143709, 68729.0782976]]),
            ],
            [0.29480400442746502, 0.53352099305834633, 0.17167500251418938],
        ],
        [
            [
                ([453.1255362, 544.63138923]),
                ([722.8546238, 326.65475739]),
                ([445.33662957, 550.82705344]),
            ],
            [
                ([[3748.5636386, -1087.7059591], [-1087.7059591, 2962.05884783]]),
                ([[53499.35557041, 2503.97833801], [2503.97833801, 34339.63653221]]),
                ([[6649.97753593, -6920.24538877], [-6920.24538877, 34135.84244881]]),
            ],
            [0.47180641829031889, 0.25403472847233199, 0.27415885323734995],
        ],
    ]

    cen, cov, mc = list_par[
        gmm_group_index
    ]  # list_par[0]: Regular commuters; list_par[1]: Flexible commuters

    WORK_PROB_WEEKDAY = work_prob_weekday
    WORK_PROB_WEEKEND = work_prob_weekend
    NUM_DAYS = num_days

    with open(output_path, "w") as g:
        # Handle commuters
        with open(commuter_path, "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) < 9:
                    continue
                user_id = parts[8]
                reg = 1 if random.random() < reg_prob else 0
                work_flag = 1
                home_tract = "homeTract"
                work_tract = "workTract"

                # Get parameters from the line
                b1 = float(parts[0])
                b2 = float(parts[1])
                nw = float(parts[2])

                # Generate time and duration using GMM
                time_dur = gen_gmm_1sample(cen, cov, mc)
                time_work = time_dur[0]
                dur_work = time_dur[1]

                # Generate work schedule
                work_slots = []
                # time_work and dur_work are BOTH sampled from the GMM in
                # minutes (centers ~450–770). 10-min slots → divide by 10.
                for day in range(NUM_DAYS):
                    if day < 5:  # Weekday
                        if random.random() < WORK_PROB_WEEKDAY:
                            start_slot = int(time_work / 10) + day * 144
                            end_slot = start_slot + int(dur_work / 10)
                            work_slots.extend(range(start_slot, end_slot + 1))
                    else:  # Weekend
                        if random.random() < WORK_PROB_WEEKEND:
                            start_slot = int(time_work / 10) + day * 144
                            end_slot = start_slot + int(dur_work / 10)
                            work_slots.extend(range(start_slot, end_slot + 1))

                work_slots_str = " ".join(map(str, work_slots)) if work_slots else ""

                # Write output line
                g.write(
                    f"{user_id} {b1} {b2} {nw} {reg} {work_flag} {home_tract} {work_tract} {work_slots_str}\n"
                )

        # Handle non-commuters
        with open(noncommuter_path, "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) < 9:
                    continue
                user_id = parts[8]
                reg = 1 if random.random() < reg_prob else 0
                work_flag = 0
                home_tract = "homeTract"
                work_tract = "workTract"

                # Get parameters from the line
                b1 = float(parts[0])
                b2 = float(parts[1])
                nw = float(parts[2])

                # Non-commuters don't have work slots
                work_slots_str = ""

                # Write output line
                g.write(
                    f"{user_id} {b1} {b2} {nw} {reg} {work_flag} {home_tract} {work_tract} {work_slots_str}\n"
                )

    logger.info(f"Simulation parameters written to: {output_path}")


def split_simulation_inputs(
    parameter_path="./results/Simulation/simulation_parameter.txt",
    location_path="./results/Simulation/simulation_location.txt",
    formatted_user_path="./results/SRFiltered_to_SimInput/FAUsers_Cleaned_Formatted.parquet",
    output_dir="./results/Simulation",
    num_cpus=16,
):
    """
    Split simulation inputs across multiple CPU cores for parallel processing.
    Ensures both location and parameter data exist for each user.
    Optimized for better load balancing and CPU utilization.
    """

    # Create output directories
    locations_dir = os.path.join(output_dir, "Locations")
    parameters_dir = os.path.join(output_dir, "Parameters")

    for directory in [locations_dir, parameters_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Directory created: {directory}")

    # Check if input files exist and have content
    for file_path in [parameter_path, location_path]:
        if not os.path.exists(file_path):
            logger.error(f"Error: Input file does not exist: {file_path}")
            return
        if os.path.getsize(file_path) == 0:
            logger.error(f"Error: Input file is empty: {file_path}")
            return

    # Parse location data to get home coordinates for each user
    logger.info("Reading and parsing location data...")
    user_locations = {}
    user_home_coords = {}
    current_user = None

    with open(location_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(" ")
            if len(parts) == 1:  # User ID line
                current_user = parts[0]  # Ensure user ID is always a string
                user_locations[current_user] = []
            else:  # Location data: loctype frequency lon lat
                if current_user and len(parts) >= 4:
                    try:
                        loc_type = parts[0]
                        frequency = int(parts[1])
                        lon = float(parts[2])
                        lat = float(parts[3])

                        user_locations[current_user].append(
                            [loc_type, frequency, lon, lat]
                        )

                        # Store home coordinates (first location is typically home)
                        if loc_type == "h" or len(user_locations[current_user]) == 1:
                            user_home_coords[current_user] = (lon, lat)

                    except (ValueError, IndexError) as e:
                        logger.error(f"Error parsing location line: {line}, error: {e}")

    # Parse parameter data
    logger.info("Reading and parsing parameter data...")
    user_parameters = {}

    with open(parameter_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(" ")
            if len(parts) >= 8:
                try:
                    user_id = parts[0]  # Ensure user ID is always a string
                    work_flag = int(parts[5]) if len(parts) > 5 else 0
                    has_work_slots = len(parts) > 8

                    # If this user already exists, prefer the entry with work slots
                    if user_id in user_parameters:
                        existing_parts = user_parameters[user_id].split(" ")
                        existing_work_flag = (
                            int(existing_parts[5]) if len(existing_parts) > 5 else 0
                        )
                        existing_has_work_slots = len(existing_parts) > 8

                        # Keep the entry with work slots, or the one with work_flag=1 if both have same slot status
                        if has_work_slots and not existing_has_work_slots:
                            user_parameters[user_id] = line
                        elif not has_work_slots and existing_has_work_slots:
                            # Keep existing entry with work slots
                            pass
                        elif work_flag == 1 and existing_work_flag == 0:
                            user_parameters[user_id] = line
                        elif work_flag == 0 and existing_work_flag == 1:
                            # Keep existing entry with work_flag=1
                            pass
                        else:
                            # If both have same characteristics, keep the one with more fields (more complete)
                            if len(parts) > len(existing_parts):
                                user_parameters[user_id] = line
                    else:
                        user_parameters[user_id] = line

                except (ValueError, IndexError) as e:
                    logger.error(f"Error parsing parameter line: {line}, error: {e}")

    # Find users that have both locations and parameters
    users_with_locations = set(user_locations.keys())
    users_with_parameters = set(user_parameters.keys())
    valid_users = users_with_locations & users_with_parameters

    logger.info(f"Users with locations: {len(users_with_locations)}")
    logger.info(f"Users with parameters: {len(users_with_parameters)}")
    logger.info(f"Users with both: {len(valid_users)}")

    if len(valid_users) == 0:
        logger.error("ERROR: No users have both location and parameter data!")
        return

    # Create balanced chunks for parallel processing
    valid_users_list = list(valid_users)
    chunk_size = max(1, len(valid_users_list) // num_cpus)

    logger.info(
        f"Splitting {len(valid_users_list)} users into {num_cpus} files with ~{chunk_size} users each"
    )

    files_created = 0
    for i in range(num_cpus):
        start_idx = i * chunk_size
        if i == num_cpus - 1:  # Last chunk gets remaining users
            end_idx = len(valid_users_list)
        else:
            end_idx = (i + 1) * chunk_size

        chunk_users = valid_users_list[start_idx:end_idx]

        if not chunk_users:  # Skip empty chunks
            continue

        # Write location file for this chunk
        location_file = os.path.join(locations_dir, f"usersLocations_{i}.txt")
        with open(location_file, "w") as f:
            for user_id in chunk_users:
                f.write(f"{str(user_id)}\n")  # Always write user ID as string

                # Ensure we always have a home location first
                if user_id in user_home_coords:
                    home_lon, home_lat = user_home_coords[user_id]
                else:
                    # Use first location as home if no explicit home found
                    if user_locations[user_id]:
                        home_lon = user_locations[user_id][0][2]
                        home_lat = user_locations[user_id][0][3]
                    else:
                        # Default coordinates if no location data
                        home_lon, home_lat = -96.0, 30.0

                # Write home location first
                f.write(f"h 1 {home_lon} {home_lat}\n")

                # Write other locations with appropriate IDs
                location_id = 2  # Start other locations from ID 2
                for loc in user_locations[user_id]:
                    if loc[0] != "h":  # Skip if already processed as home
                        if loc[0] == "w":  # Work location
                            f.write(f"w {loc[1]} {loc[2]} {loc[3]}\n")
                        else:  # Other location
                            f.write(f"{location_id} {loc[1]} {loc[2]} {loc[3]}\n")
                            location_id += 1

        # Write parameter file for this chunk
        parameter_file = os.path.join(parameters_dir, f"usersParameters_{i}.txt")
        with open(parameter_file, "w") as f:
            for user_id in chunk_users:
                if user_id in user_parameters:
                    f.write(f"{user_parameters[user_id]}\n")

        files_created += 1
        logger.info(f"Created chunk {i}: {len(chunk_users)} users")

    logger.info(f"Total files created: {files_created} pairs")

    # Verify no empty files were created
    for i in range(num_cpus):
        loc_file = os.path.join(locations_dir, f"usersLocations_{i}.txt")
        param_file = os.path.join(parameters_dir, f"usersParameters_{i}.txt")

        if os.path.exists(loc_file) and os.path.exists(param_file):
            loc_size = os.path.getsize(loc_file)
            param_size = os.path.getsize(param_file)

            if loc_size == 0 or param_size == 0:
                logger.warning(f"WARNING: Empty files detected for chunk {i}")
                # Remove empty files to prevent simulation issues
                if loc_size == 0:
                    os.remove(loc_file)
                    logger.info(f"Removed empty location file: {loc_file}")
                if param_size == 0:
                    os.remove(param_file)
                    logger.info(f"Removed empty parameter file: {param_file}")

    logger.info("Simulation input splitting completed successfully")


def activeness(
    noncomm_daily_path="./results/Parameters/NonComm_pt_daily.txt",
    noncomm_weekly_path="./results/Parameters/NonComm_pt_weekly.txt",
    comm_daily_path="./results/Parameters/Comm_pt_daily.txt",
    comm_weekly_path="./results/Parameters/Comm_pt_weekly.txt",
    output_path="./results/Simulation/activeness.txt",
):
    """
    Generate activeness patterns for simulation.
    If input files are empty or missing, create realistic default patterns.
    """

    def read_activeness_file(file_path):
        """Read activeness file, return default pattern if file is empty or missing."""
        activeness_data = []

        # First try the specified path
        try:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                with open(file_path, "r") as f:
                    for line in f:
                        activeness_data.append(float(line.strip()))

                # If file has meaningful data (not all zeros and not single 1.0), return it
                non_zero_count = sum(1 for x in activeness_data if x > 0)
                total_activity = sum(activeness_data)

                # Check for meaningful patterns: more than 1 non-zero value and reasonable distribution
                if non_zero_count > 1 and total_activity > 0.1:
                    logger.info(f"Using activeness file: {file_path}")
                    return activeness_data
                else:
                    logger.warning(
                        f"Warning: Activity file {file_path} has poor pattern (non-zero: {non_zero_count}, total: {total_activity:.3f})."
                    )

        except Exception as e:
            logger.error(f"Warning: Error reading activeness file {file_path}: {e}")

        # If the specified path failed, try fallback to root directory
        fallback_path = None
        if "NonComm_pt_daily" in file_path:
            fallback_path = "./Comm_pt_daily.txt"  # Use commuter file as fallback
        elif "NonComm_pt_weekly" in file_path:
            fallback_path = "./Comm_pt_weekly.txt"  # Use commuter file as fallback
        elif "Comm_pt_daily" in file_path:
            fallback_path = "./Comm_pt_daily.txt"
        elif "Comm_pt_weekly" in file_path:
            fallback_path = "./Comm_pt_weekly.txt"

        if fallback_path and os.path.exists(fallback_path):
            logger.info(f"Trying fallback activeness file: {fallback_path}")
            try:
                with open(fallback_path, "r") as f:
                    for line in f:
                        activeness_data.append(float(line.strip()))

                non_zero_count = sum(1 for x in activeness_data if x > 0)
                total_activity = sum(activeness_data)

                if non_zero_count > 1 and total_activity > 0.1:
                    logger.info(f"Using fallback activeness file: {fallback_path}")
                    return activeness_data
                else:
                    logger.warning(
                        f"Warning: Fallback activity file {fallback_path} also has poor pattern."
                    )

            except Exception as e:
                logger.error(
                    f"Warning: Error reading fallback activeness file {fallback_path}: {e}"
                )

        # If all files failed, generate default patterns
        logger.info(f"Generating default activeness pattern for: {file_path}")

        # Generate default patterns if file is empty/missing/all zeros
        if "daily" in file_path:
            # Create realistic daily activity pattern (144 time slots = 24 hours * 6 slots per hour)
            # Higher activity during day hours (6 AM to 10 PM), lower at night
            daily_pattern = []
            for slot in range(144):
                hour = slot // 6  # Convert slot to hour (0-23)
                if 6 <= hour <= 22:  # Active hours 6 AM to 10 PM
                    # Peak activity during commute times and lunch
                    if hour in [7, 8, 12, 17, 18]:
                        activity = 0.25  # Higher activity during peak times
                    elif hour in [9, 10, 11, 13, 14, 15, 16, 19, 20]:
                        activity = 0.15  # Medium activity during work/day hours
                    else:
                        activity = 0.08  # Lower activity during other day hours
                else:  # Night hours (11 PM to 5 AM)
                    activity = 0.02  # Very low activity at night
                daily_pattern.append(activity)

            # Normalize to sum to approximately 1.0
            total = sum(daily_pattern)
            return [x / total for x in daily_pattern]

        else:  # weekly pattern
            # Create weekly pattern (7 days) with higher activity on weekdays
            weekly_pattern = [
                0.16,  # Monday
                0.16,  # Tuesday
                0.16,  # Wednesday
                0.16,  # Thursday
                0.15,  # Friday (slightly lower)
                0.11,  # Saturday (lower weekend activity)
                0.10,  # Sunday (lowest activity)
            ]
            return weekly_pattern

    # Read all activeness files
    noncomm_daily = read_activeness_file(noncomm_daily_path)
    noncomm_weekly = read_activeness_file(noncomm_weekly_path)
    comm_daily = read_activeness_file(comm_daily_path)
    comm_weekly = read_activeness_file(comm_weekly_path)

    # FIX: Use original format - 2 lines with combined daily*weekly patterns
    # Non-commuter probabilities (daily * weekly for each day)
    nonCommuterPt = [d * w for w in noncomm_weekly for d in noncomm_daily]

    # Commuter probabilities (daily * weekly for each day)
    commuterPt = [d * w for w in comm_weekly for d in comm_daily]

    # Write to output file in original format (2 lines)
    with open(output_path, "w") as f:
        f.write(" ".join(map(str, nonCommuterPt)) + "\n")
        f.write(" ".join(map(str, commuterPt)) + "\n")

    logger.info(f"Activeness patterns written to: {output_path}")
    logger.info(f"  Non-commuter daily pattern: {len(noncomm_daily)} time slots")
    logger.info(f"  Commuter daily pattern: {len(comm_daily)} time slots")
    logger.info(f"  Non-commuter weekly pattern: {len(noncomm_weekly)} days")
    logger.info(f"  Commuter weekly pattern: {len(comm_weekly)} days")


def otherLocations(
    input_path="./results/Simulation/simulation_location.txt",
    output_path="./results/Simulation/otherlocation.txt",
    sample_fraction=0.02,
):
    """
    Sample locations from simulation input for "other" location generation.
    Extracts real coordinates from the location file.
    """
    coordinates = []

    try:
        with open(input_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Skip user ID lines
                if len(line.split()) == 1:
                    continue

                # Parse location lines: "location_type count longitude latitude"
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        longitude = float(parts[2])
                        latitude = float(parts[3])

                        # Only include coordinates that are not 0.0, 0.0
                        if longitude != 0.0 or latitude != 0.0:
                            coordinates.append([longitude, latitude])
                    except (ValueError, IndexError):
                        continue

    except FileNotFoundError:
        logger.warning(f"Warning: Input file {input_path} not found")
    except Exception as e:
        logger.error(f"Warning: Error reading {input_path}: {e}")

    # If no valid coordinates found, generate some default ones
    if not coordinates:
        logger.warning(
            "Warning: No valid coordinates found in input file. Generating default coordinates."
        )
        # Generate some default coordinates around Austin, TX area
        import random

        for _ in range(100):
            lng = -97.7 + random.uniform(-0.1, 0.1)  # Austin longitude ± variation
            lat = 30.3 + random.uniform(-0.1, 0.1)  # Austin latitude ± variation
            coordinates.append([lng, lat])

    # Sample a fraction of the coordinates
    num_samples = max(1, int(len(coordinates) * sample_fraction))
    if num_samples < len(coordinates):
        import random

        sampled_coords = random.sample(coordinates, num_samples)
    else:
        sampled_coords = coordinates

    # Write sampled coordinates
    with open(output_path, "w") as f:
        for coord in sampled_coords:
            f.write(f"{coord[0]} {coord[1]}\n")

    logger.info(f"Other locations written to: {output_path}")
    logger.info(f"  Total coordinates found: {len(coordinates)}")
    logger.info(f"  Sampled coordinates: {len(sampled_coords)}")
    logger.info(f"  Sample fraction: {sample_fraction}")


def create_user_id_mapping(parameter_file, location_file):
    """
    Create a mapping between hex user IDs (from parameters) and integer user IDs (from locations).
    Returns a dictionary mapping hex_id -> int_id
    """
    # Get unique hex user IDs from parameter file
    hex_user_ids = set()
    with open(parameter_file, "r") as f:
        for line in f:
            if line.strip():
                hex_id = line.strip().split()[0]
                hex_user_ids.add(hex_id)

    # Get unique integer user IDs from location file
    int_user_ids = set()
    with open(location_file, "r") as f:
        for line in f:
            if line.strip() and not line.strip().split()[0].isalpha():
                try:
                    user_id = int(line.strip())
                    int_user_ids.add(user_id)
                except ValueError:
                    continue

    # Create mapping: sort both lists and map them 1:1
    hex_ids_sorted = sorted(list(hex_user_ids))
    int_ids_sorted = sorted(list(int_user_ids))

    # Create mapping dictionary
    mapping = {}
    for i, hex_id in enumerate(hex_ids_sorted):
        if i < len(int_ids_sorted):
            mapping[hex_id] = int_ids_sorted[i]

    logger.info(f"Created user ID mapping: {len(mapping)} hex IDs -> integer IDs")
    return mapping


def align_parameter_file_user_ids(input_param_file, output_param_file, user_id_mapping):
    """
    Rewrite parameter file to use integer user IDs instead of hex user IDs.
    """
    with open(input_param_file, "r") as infile, open(output_param_file, "w") as outfile:
        for line in infile:
            if line.strip():
                parts = line.strip().split()
                hex_id = parts[0]
                if hex_id in user_id_mapping:
                    # Replace hex ID with integer ID
                    parts[0] = str(user_id_mapping[hex_id])
                    outfile.write(" ".join(parts) + "\n")
                else:
                    logger.warning(f"Warning: Hex ID {hex_id} not found in mapping")

    logger.info(f"Aligned parameter file written to: {output_param_file}")
