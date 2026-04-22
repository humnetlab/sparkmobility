import logging
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

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


def to_parquet_robust(df, file_path, **kwargs):
    """Write parquet file with fallback engine support."""
    try:
        return df.to_parquet(file_path, engine=PARQUET_ENGINE, **kwargs)
    except Exception as e:
        logger.error(f"Error writing {file_path} with {PARQUET_ENGINE}: {e}")
        alt_engine = "fastparquet" if PARQUET_ENGINE == "pyarrow" else "pyarrow"
        logger.info(f"Trying alternative engine: {alt_engine}")
        return df.to_parquet(file_path, engine=alt_engine, **kwargs)


# warnings.filterwarnings('ignore')


def compress_simulation_results(
    input_folder="./results/Simulation/Mapped/",
    output_folder="./results/Simulation/Compressed/",
    file_prefix="simulationResults_",
):
    """
    Compresses simulation result files in the specified folder that start with a given prefix.

    Parameters:
    - input_folder (str): Path to the input folder.
    - output_folder (str): Path to the folder where compressed files will be saved.
    - file_prefix (str): Prefix of the files to process.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if file_prefix in f]

    for filename in files:
        compressed_results = []
        running_location = [None, None]
        timeslot = 0

        # Read and process the input file line by line
        with open(os.path.join(input_folder, filename), "r") as f:
            for line in f:
                timeslot += 1
                parts = line.strip().split(" ")

                # Check for single-element lines (user identifiers or special markers)
                if len(parts) == 1:
                    compressed_results.append(line)
                    running_location = [None, None]
                    timeslot = 0
                # Record changes in location along with timeslot
                elif [parts[1], parts[2]] != running_location:
                    compressed_results.append(f"{timeslot} {line}")
                    running_location = [parts[1], parts[2]]

        # Write compressed results to the output file
        with open(os.path.join(output_folder, filename), "w") as f:
            for c in compressed_results:
                f.write(c)

    logger.info(f"Compressed {len(files)} simulation result files to {output_folder}")


def closest_node(node, nodes):
    return cdist([node], nodes).argmin()


def process_user_trajectories(
    datatype: str,
    modeltype: str,
    gps_path: str,
    testids_path: str,
    sim_dir: str,
    output_dir_tg: str,
    output_dir_our: str,
):
    """
    Preprocess simulated trajectory data: convert raw .txt files into a
    dictionary saved as a .npy file for model consumption.

    Parameters:
    - datatype: identifier for the dataset, used for naming the output file
    - modeltype: 'tg' to save under output_dir_tg, otherwise under output_dir_our
    - gps_path: path to GPS grid nodes file (.npy)
    - testids_path: path to test user ID list file (.npy)
    - sim_dir: directory containing simulated trajectory .txt files
    - output_dir_tg: output directory for 'tg' modeltype
    - output_dir_our: output directory for other model types
    """
    # Load reference GPS nodes and test user IDs
    gps = np.load(gps_path, allow_pickle=True)  # array of shape (N,2): [lat, lon]
    test_id = np.load(testids_path, allow_pickle=True)

    # List all simulation text files in the directory
    simFiles = [
        os.path.join(sim_dir, f)
        for f in os.listdir(sim_dir)
        if os.path.isfile(os.path.join(sim_dir, f)) and f.endswith(".txt")
    ]
    simFiles = sorted(simFiles)

    userTraj_point = {}
    userTraj_time = {}

    # Parse each simulation file line by line
    for f in simFiles:
        with open(f, "r") as data:
            for line in data:
                parts = line.strip().split(" ")
                if len(parts) == 1:
                    # New user ID line: format 'prefix-<perID>'
                    perID = int(parts[0].split("-")[1])
                    if perID not in userTraj_point and perID in test_id:
                        userTraj_point[perID] = []
                        userTraj_time[perID] = []
                else:
                    # Only record data for test users
                    if perID in test_id:
                        timestep = int(parts[0])
                        lon = float(parts[2])
                        lat = float(parts[3])
                        userTraj_point[perID].append([lat, lon])
                        userTraj_time[perID].append(timestep)

    userTraj_sim = {}

    # For each user and each day of the week, map points to GPS nodes
    # and compute stay durations and departure times
    for uid in userTraj_point.keys():
        userTraj_sim[uid] = {}
        for day in range(7):
            loc = []
            sta = []
            tim = []
            times = userTraj_time[uid]
            for i in range(len(times)):
                if times[i] > 1:
                    try:
                        # Stay duration: (next_t - current_t) * 10
                        sta.append((times[i + 1] - times[i]) * 10)
                        # Departure time in minutes: (t % 144) * 10
                        tim.append((times[i] % 144) * 10)
                        # Closest GPS node index for the point
                        loc.append(closest_node(userTraj_point[uid][i], gps))
                    except IndexError:
                        # Handle final record: use last timestamp and location
                        tim.append((times[-1] % 144) * 10)
                        loc.append(closest_node(userTraj_point[uid][-1], gps))
                        continue
            userTraj_sim[uid][day] = {
                "loc": np.array(loc),
                "sta": np.array(sta),
                "tim": np.array(tim),
            }

    # Save the processed dictionary to the appropriate output directory
    if modeltype == "tg":
        os.makedirs(output_dir_tg, exist_ok=True)
        np.save(
            os.path.join(output_dir_tg, f"{datatype}.npy"),
            userTraj_sim,
            allow_pickle=True,
        )
    else:
        os.makedirs(output_dir_our, exist_ok=True)
        np.save(
            os.path.join(output_dir_our, f"{datatype}.npy"),
            userTraj_sim,
            allow_pickle=True,
        )

    logger.info(
        f"Processed user trajectories saved to {output_dir_tg if modeltype == 'tg' else output_dir_our}"
    )


def export_simulation_results_to_parquet(
    input_folder="./results/Simulation/Mapped/",
    output_file="./results/Simulation/simulation_results.parquet",
    file_prefix="simulationResults_",
):
    """
    Export simulation results to a single parquet file for easier analysis.

    Parameters:
    - input_folder (str): Path to the input folder containing simulation results
    - output_file (str): Path to the output parquet file
    - file_prefix (str): Prefix of the files to process
    """

    all_results = []
    files = [
        f for f in os.listdir(input_folder) if file_prefix in f and f.endswith(".txt")
    ]

    for filename in files:
        filepath = os.path.join(input_folder, filename)
        current_user = None
        timeslot = 0

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                parts = line.split(" ")

                if len(parts) == 1:  # User ID line
                    current_user = parts[0]
                    timeslot = 0
                elif len(parts) >= 3:  # Location line
                    timeslot += 1
                    location_type = parts[0]
                    longitude = float(parts[1])
                    latitude = float(parts[2])

                    all_results.append(
                        {
                            "user_id": current_user,
                            "timeslot": timeslot,
                            "location_type": location_type,
                            "longitude": longitude,
                            "latitude": latitude,
                            "file_index": filename.replace(file_prefix, "").replace(
                                ".txt", ""
                            ),
                        }
                    )

    # Create DataFrame and save as parquet
    df = pd.DataFrame(all_results)
    to_parquet_robust(df, output_file, index=False)
    logger.info(f"Exported {len(all_results)} simulation records to {output_file}")

    return df


def analyze_simulation_results_parquet(
    parquet_file="./results/Simulation/simulation_results.parquet",
    output_dir="./results/Analysis/",
):
    """
    Analyze simulation results from parquet file and generate summary statistics.

    Parameters:
    - parquet_file (str): Path to the simulation results parquet file
    - output_dir (str): Directory to save analysis results
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load simulation results
    df = read_parquet_robust(parquet_file)

    # Handle empty dataframe
    if df.empty:
        logger.warning("⚠️  Warning: Simulation results are empty - no data to analyze")
        logger.info(
            "This is expected when no users meet the parameter generation criteria"
        )

        # Create empty analysis files for consistency
        empty_user_stats = pd.DataFrame()
        empty_transitions = pd.DataFrame()
        empty_hourly_activity = pd.DataFrame()

        to_parquet_robust(
            empty_user_stats, os.path.join(output_dir, "user_statistics.parquet")
        )
        to_parquet_robust(
            empty_transitions, os.path.join(output_dir, "location_transitions.parquet")
        )
        to_parquet_robust(
            empty_hourly_activity, os.path.join(output_dir, "hourly_activity.parquet")
        )

        logger.info(f"Empty analysis files created in {output_dir}")

        return {
            "user_stats": empty_user_stats,
            "transitions": empty_transitions,
            "hourly_activity": empty_hourly_activity,
        }

    # Basic statistics (only if data exists)
    logger.info(f"Total simulation records: {len(df)}")

    # Check if required columns exist
    if "user_id" not in df.columns:
        logger.warning("⚠️  Warning: Missing 'user_id' column in simulation results")
        return {
            "user_stats": pd.DataFrame(),
            "transitions": pd.DataFrame(),
            "hourly_activity": pd.DataFrame(),
        }

    logger.info(f"Unique users: {df['user_id'].nunique()}")

    if "location_type" in df.columns:
        logger.info("Location type distribution:")
        logger.info(df["location_type"].value_counts())

    # User activity analysis
    user_stats = (
        df.groupby("user_id")
        .agg(
            {
                "timeslot": "max",
                "location_type": lambda x: x.nunique(),
                "longitude": lambda x: x.nunique(),
                "latitude": lambda x: x.nunique(),
            }
        )
        .rename(
            columns={
                "timeslot": "max_timeslot",
                "location_type": "unique_location_types",
                "longitude": "unique_longitudes",
                "latitude": "unique_latitudes",
            }
        )
    )

    # Save user statistics
    to_parquet_robust(user_stats, os.path.join(output_dir, "user_statistics.parquet"))

    # Location type transitions analysis
    df_sorted = df.sort_values(["user_id", "timeslot"])
    df_sorted["prev_location_type"] = df_sorted.groupby("user_id")[
        "location_type"
    ].shift(1)

    # Count transitions
    transitions = (
        df_sorted.dropna(subset=["prev_location_type"])
        .groupby(["prev_location_type", "location_type"])
        .size()
        .reset_index(name="count")
    )

    # Save transition matrix
    to_parquet_robust(
        transitions, os.path.join(output_dir, "location_transitions.parquet")
    )

    # Hourly activity patterns
    df["hour"] = df["timeslot"] % 24
    hourly_activity = (
        df.groupby(["hour", "location_type"]).size().reset_index(name="count")
    )
    to_parquet_robust(
        hourly_activity, os.path.join(output_dir, "hourly_activity.parquet")
    )

    logger.info(f"Analysis results saved to {output_dir}")

    return {
        "user_stats": user_stats,
        "transitions": transitions,
        "hourly_activity": hourly_activity,
    }


def compress_and_export_simulation_results(
    input_folder="./results/Simulation/Mapped/",
    compressed_folder="./results/Simulation/Compressed/",
    parquet_file="./results/Simulation/simulation_results.parquet",
    analysis_dir="./results/Analysis/",
    file_prefix="simulationResults_",
):
    """
    Complete post-processing pipeline: compress, export to parquet, and analyze.

    Parameters:
    - input_folder (str): Path to raw simulation results
    - compressed_folder (str): Path to save compressed results
    - parquet_file (str): Path to save parquet export
    - analysis_dir (str): Directory to save analysis results
    - file_prefix (str): Prefix of simulation result files
    """

    logger.info("Step 1: Compressing simulation results...")
    compress_simulation_results(input_folder, compressed_folder, file_prefix)

    logger.info("Step 2: Exporting to parquet...")
    df = export_simulation_results_to_parquet(input_folder, parquet_file, file_prefix)

    logger.info("Step 3: Analyzing results...")
    analysis_results = analyze_simulation_results_parquet(parquet_file, analysis_dir)

    logger.info("Post-processing pipeline completed successfully!")

    return df, analysis_results
