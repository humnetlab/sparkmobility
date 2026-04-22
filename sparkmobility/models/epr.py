import datetime
import logging
import math
from collections import defaultdict

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import powerlaw
from shapely.geometry import Polygon
from tqdm import tqdm

from sparkmobility.logging_setup import configure_if_needed

logger = logging.getLogger(__name__)


class EPR:
    """
    Exploration and Preferential Return (EPR) model for human mobility.

    This implementation follows conventions similar to the Gravity model:
    - Uses a spatial tessellation with 'index' and 'geometry' columns
    - Has a fit() method to learn OD matrix from stay point data
    - Has a generate() method to generate synthetic trajectories
    """

    def __init__(
        self,
        rho=0.6,
        gamma=0.21,
        beta=0.8,
        tau=17,
        min_wait_time_minutes=20,
        gravity_type="single",
        name="EPR model",
        is_h3_hexagon=True,
        configure_logging_if_needed: bool = True,
    ):
        if configure_logging_if_needed:
            configure_if_needed()
        """
        Initialize the EPR model.

        Parameters
        ----------
        rho : float, optional (default=0.6)
            Parameter controlling the exploration vs return trade-off
        gamma : float, optional (default=0.21)
            Exponent for the exploration probability decay
        beta : float, optional (default=0.8)
            Exponent for the waiting time distribution
        tau : float, optional (default=17)
            Characteristic time scale for waiting time distribution (in hours)
        min_wait_time_minutes : float, optional (default=20)
            Minimum waiting time between trips in minutes
        gravity_type : str, optional (default="single")
            Type of model constraint ('single' or 'global') - kept for API compatibility
        name : str, optional (default="EPR model")
            Name of the model instance
        is_h3_hexagon : bool, optional (default=True)
            Whether locations are H3 hexagons
        """
        self.rho = rho
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.gravity_type = gravity_type
        self.name = name
        self.is_h3_hexagon = is_h3_hexagon

        # Minimum waiting time (in hours)
        self.min_wait_time = min_wait_time_minutes / 60.0

        # Model parameters learned from data
        self.od_matrix = None  # Origin-destination matrix (relevance matrix)
        self.spatial_tessellation = None  # GeoDataFrame with tessellation info
        self.tileid2index = None  # Mapping from tile ID to integer index
        self.index2tileid = None  # Mapping from integer index to tile ID

        # Agent-specific state (used during generation)
        self.location2visits = defaultdict(int)
        self.starting_loc = None
        self.trajectories_ = []

    def __str__(self):
        return (
            f'EPR(name="{self.name}", rho={self.rho}, gamma={self.gamma}, '
            f"beta={self.beta}, tau={self.tau})"
        )

    def fit(
        self,
        flow_df,
        relevance_df=None,
        user_column="caid",
        datetime_column="stay_start_timestamp",
        location_column="h3_id_region",
    ):
        """
        Fit the EPR model by computing the OD matrix from stay point data.

        This follows the Gravity model API but computes transitions from stay points.
        The spatial tessellation is automatically built from the H3 locations in flow_df.

        Parameters
        ----------
        flow_df : pd.DataFrame
            Stay points dataframe with columns:
            - caid: user ID
            - stay_start_timestamp: timestamp of stay start
            - h3_id_region: H3 hexagon ID
            - (optional) stay_end_timestamp, stay_duration, h3_region_stay_id
        relevance_df : gpd.GeoDataFrame, optional
            Pre-built spatial tessellation (if None, built from flow_df)
            Should have 'index' column with H3 IDs and 'geometry' column
        user_column : str, optional (default='caid')
            Name of the user ID column in flow_df
        datetime_column : str, optional (default='stay_start_timestamp')
            Name of the timestamp column in flow_df
        location_column : str, optional (default='h3_id_region')
            Name of the location (H3) column in flow_df

        Returns
        -------
        result : dict
            Dictionary containing fit statistics:
            - 'od_matrix_shape': shape of the OD matrix
            - 'num_locations': number of unique locations
            - 'num_users': number of unique users
            - 'num_transitions': total number of transitions
        """
        logger.info(
            "fitting EPR from %d stay points across %d users",
            len(flow_df),
            flow_df[user_column].nunique(),
        )

        # Build spatial tessellation if not provided
        if relevance_df is None:
            logger.info("building spatial tessellation from H3 locations")
            relevance_df = self._build_spatial_tessellation(flow_df, location_column)

        # Store the spatial tessellation
        self.spatial_tessellation = relevance_df.copy()

        # Create mappings between tile IDs and integer indices
        self.tileid2index = {
            tileid: i for i, tileid in enumerate(relevance_df["index"].values)
        }
        self.index2tileid = {i: tileid for tileid, i in self.tileid2index.items()}

        logger.info("computing OD matrix from stay point transitions")

        # Compute OD matrix from stay point sequences
        self.od_matrix, num_transitions = self._compute_od_matrix_from_stays(
            flow_df, user_column, datetime_column, location_column
        )

        logger.info(
            "OD matrix %s, %d locations, %d transitions",
            self.od_matrix.shape,
            len(self.tileid2index),
            num_transitions,
        )

        # Return result dictionary (similar to Gravity model's GLM result)
        result = {
            "od_matrix_shape": self.od_matrix.shape,
            "num_locations": len(self.tileid2index),
            "num_users": flow_df[user_column].nunique(),
            "num_transitions": num_transitions,
        }

        return result

    def _build_spatial_tessellation(self, stay_df, location_column):
        """
        Build spatial tessellation GeoDataFrame from stay points.

        Parameters
        ----------
        stay_df : pd.DataFrame
            Stay points dataframe
        location_column : str
            Name of the H3 location column

        Returns
        -------
        gpd.GeoDataFrame
            Spatial tessellation with 'index', 'geometry', and 'stay_count' columns
        """
        # Get unique H3 locations
        unique_h3 = stay_df[location_column].unique()

        # Count stays per location
        stay_counts = stay_df.groupby(location_column).size().to_dict()

        # Create geometries
        data = []
        for h3_id in unique_h3:
            # Get polygon boundary
            boundary = h3.cell_to_boundary(h3_id)
            polygon = Polygon([(lon, lat) for (lat, lon) in boundary])

            data.append(
                {
                    "index": h3_id,
                    "stay_count": stay_counts.get(h3_id, 0),
                    "geometry": polygon,
                }
            )

        gdf = gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:4326")
        logger.info("built tessellation with %d locations", len(gdf))

        return gdf

    def _compute_od_matrix_from_stays(
        self, stay_df, user_column, datetime_column, location_column
    ):
        """
        Compute origin-destination matrix from stay point sequences.

        The OD matrix represents the probability of moving from one location to another,
        normalized by row (each row sums to 1).

        Parameters
        ----------
        stay_df : pd.DataFrame
            Stay points dataframe
        user_column : str
            User ID column name
        datetime_column : str
            Timestamp column name
        location_column : str
            Location (H3) column name

        Returns
        -------
        od_matrix : np.ndarray
            Normalized OD probability matrix
        num_transitions : int
            Total number of transitions counted
        """
        # Sort stay points by user and time
        stay_df = stay_df.sort_values(by=[user_column, datetime_column]).reset_index(
            drop=True
        )

        num_locs = len(self.tileid2index)
        od_counts = np.zeros((num_locs, num_locs))
        num_transitions = 0

        # Group by user and compute transitions between consecutive stays
        for user_id, user_stays in tqdm(
            stay_df.groupby(user_column), desc="Processing users"
        ):
            locations = user_stays[location_column].values

            # Count transitions between consecutive stay locations
            for i in range(len(locations) - 1):
                origin = locations[i]
                destination = locations[i + 1]

                # Skip if location not in tessellation
                if (
                    origin not in self.tileid2index
                    or destination not in self.tileid2index
                ):
                    continue

                origin_idx = self.tileid2index[origin]
                dest_idx = self.tileid2index[destination]

                # Don't count self-loops (staying at same location)
                if origin_idx != dest_idx:
                    od_counts[origin_idx, dest_idx] += 1
                    num_transitions += 1

        # Normalize each row to sum to 1 (convert counts to probabilities)
        od_matrix = np.zeros_like(od_counts, dtype=float)
        for i in range(num_locs):
            row_sum = od_counts[i, :].sum()
            if row_sum > 0:
                od_matrix[i, :] = od_counts[i, :] / row_sum
            else:
                # If no outgoing trips from this location, use uniform distribution
                # excluding self-loops
                od_matrix[i, :] = 1.0 / (num_locs - 1) if num_locs > 1 else 0
                od_matrix[i, i] = 0

        # Set diagonal to zero (no self-loops)
        np.fill_diagonal(od_matrix, 0)

        # Re-normalize rows after setting diagonal to zero
        for i in range(num_locs):
            row_sum = od_matrix[i, :].sum()
            if row_sum > 0:
                od_matrix[i, :] = od_matrix[i, :] / row_sum

        return od_matrix, num_transitions

    def generate(
        self,
        start_date,
        end_date,
        n_agents=1,
        starting_locations=None,
        random_state=None,
        show_progress=True,
    ):
        """
        Generate synthetic trajectories using the fitted EPR model.

        Parameters
        ----------
        start_date : datetime
            Start date/time for trajectory generation
        end_date : datetime
            End date/time for trajectory generation
        n_agents : int, optional (default=1)
            Number of agents to simulate
        starting_locations : array-like, optional
            Array of starting location indices or IDs. If None, randomly chosen.
        random_state : int, optional
            Random seed for reproducibility
        show_progress : bool, optional (default=True)
            Whether to show progress bar

        Returns
        -------
        pd.DataFrame
            Trajectory dataframe with columns: ['user', 'datetime', 'location']
            where 'location' contains tile IDs from the spatial tessellation
        """
        if self.od_matrix is None or self.spatial_tessellation is None:
            raise ValueError(
                "Model must be fitted before generating trajectories. Call fit() first."
            )

        if random_state is not None:
            np.random.seed(random_state)

        # Initialize trajectories
        self.trajectories_ = []
        num_locs = len(self.od_matrix)

        # Set up starting locations
        if starting_locations is not None:
            # Convert to indices if tile IDs provided
            if isinstance(starting_locations[0], str):
                starting_locs_idx = [
                    self.tileid2index.get(loc, None) for loc in starting_locations
                ]
                starting_locs_idx = [
                    loc for loc in starting_locs_idx if loc is not None
                ]
            else:
                starting_locs_idx = starting_locations
        else:
            starting_locs_idx = None

        # Loop through agents
        loop = tqdm(range(1, n_agents + 1), disable=not show_progress)

        for agent_id in loop:
            # Reset agent-specific state
            self.location2visits = defaultdict(int)

            # Choose starting location
            if starting_locs_idx is not None and len(starting_locs_idx) > 0:
                if agent_id <= len(starting_locs_idx):
                    self.starting_loc = starting_locs_idx[agent_id - 1]
                else:
                    self.starting_loc = np.random.choice(starting_locs_idx)
            else:
                self.starting_loc = np.random.choice(num_locs)

            # Generate travel diary for this agent
            current_date = start_date
            starting_tileid = self.index2tileid[self.starting_loc]
            self.trajectories_.append((agent_id, current_date, starting_tileid))
            self.location2visits[self.starting_loc] += 1

            # Wait before first trip
            waiting_time = self._waiting()
            current_date += datetime.timedelta(hours=waiting_time)

            # Generate trajectory until end_date
            while current_date < end_date:
                # Choose next location
                next_location_idx = self._next_location()
                next_location_tileid = self.index2tileid[next_location_idx]

                self.trajectories_.append(
                    (agent_id, current_date, next_location_tileid)
                )
                self.location2visits[next_location_idx] += 1

                # Wait before next trip
                waiting_time = self._waiting()
                current_date += datetime.timedelta(hours=waiting_time)

        # Convert to DataFrame
        df = pd.DataFrame(self.trajectories_, columns=["user", "datetime", "location"])
        df = df.sort_values(by=["user", "datetime"]).reset_index(drop=True)

        return df

    def _waiting(self):
        """
        Sample waiting time from truncated power law distribution.

        Returns
        -------
        float
            Waiting time in hours
        """
        time_to_wait = powerlaw.Truncated_Power_Law(
            xmin=self.min_wait_time, parameters=[1.0 + self.beta, 1.0 / self.tau]
        ).generate_random()[0]
        return time_to_wait

    def _next_location(self):
        """
        Choose the next location based on EPR model logic.

        Returns
        -------
        int
            Index of the next location
        """
        n_visited_locations = len(self.location2visits)

        if n_visited_locations == 0:
            # First location after starting point
            next_loc = self._preferential_exploration(self.starting_loc)
            return next_loc

        # Get current location from last entry
        agent_id, current_time, current_location_tileid = self.trajectories_[-1]
        current_location = self.tileid2index[current_location_tileid]

        # Decide whether to explore or return
        p_new = np.random.uniform(0, 1)
        p_explore = self.rho * math.pow(n_visited_locations, -self.gamma)

        # Explore if: probability says so AND we haven't visited all locations
        # OR if we've only visited 1 location
        if (
            p_new <= p_explore and n_visited_locations < len(self.od_matrix)
        ) or n_visited_locations == 1:
            # PREFERENTIAL EXPLORATION
            next_location = self._preferential_exploration(current_location)
        else:
            # PREFERENTIAL RETURN
            next_location = self._preferential_return(current_location)

        return next_location

    def _preferential_exploration(self, current_location):
        """
        Choose a new location to explore based on OD matrix probabilities.

        Only considers locations that have not been visited yet.

        Parameters
        ----------
        current_location : int
            Index of the current location

        Returns
        -------
        int
            Index of the next location to explore
        """
        # Previously visited locations
        prev_locations = np.array(list(self.location2visits.keys()))
        all_locations = np.arange(len(self.od_matrix))

        # Find unvisited locations
        new_locs = np.setdiff1d(all_locations, prev_locations)

        if len(new_locs) == 0:
            # All locations visited, choose any location (including visited ones)
            new_locs = all_locations

        # Get probabilities from OD matrix for unvisited locations
        weights_subset = self.od_matrix[current_location][new_locs]

        # Normalize to sum to 1
        if weights_subset.sum() > 0:
            weights_subset = weights_subset / weights_subset.sum()
        else:
            # If all weights are zero, use uniform distribution
            weights_subset = np.ones(len(new_locs)) / len(new_locs)

        # Sample location
        location = np.random.choice(new_locs, size=1, p=weights_subset)[0]

        return location

    def _preferential_return(self, current_location):
        """
        Return to a previously visited location based on visitation frequency.

        Parameters
        ----------
        current_location : int
            Index of the current location

        Returns
        -------
        int
            Index of the location to return to
        """
        next_location = self._weighted_random_selection(current_location)
        return next_location

    def _weighted_random_selection(self, current_location):
        """
        Select a previously visited location weighted by visitation frequency.

        Parameters
        ----------
        current_location : int
            Index of the current location

        Returns
        -------
        int
            Index of the selected location
        """
        locations = np.fromiter(self.location2visits.keys(), dtype=int)
        weights = np.fromiter(self.location2visits.values(), dtype=float)

        # Normalize weights
        weights = weights / np.sum(weights)

        # Sample location
        location = np.random.choice(locations, size=1, p=weights)

        return int(location[0])

    def to_flow_df(
        self,
        traj_df,
        user_column="user",
        datetime_column="datetime",
        location_column="location",
    ):
        """
        Convert trajectory dataframe to flow dataframe (origin-destination format).

        Parameters
        ----------
        traj_df : pd.DataFrame
            Trajectory dataframe with columns: user, datetime, location
        user_column : str, optional
            Name of user column
        datetime_column : str, optional
            Name of datetime column
        location_column : str, optional
            Name of location column

        Returns
        -------
        pd.DataFrame
            Flow dataframe with columns: ['origin', 'destination', 'flow']
        """
        # Sort by user and time
        traj_df = traj_df.sort_values(by=[user_column, datetime_column]).reset_index(
            drop=True
        )

        flow_data = []

        # Group by user and extract transitions
        for user_id, user_traj in traj_df.groupby(user_column):
            locations = user_traj[location_column].values

            for i in range(len(locations) - 1):
                origin = locations[i]
                destination = locations[i + 1]

                if origin != destination:  # Exclude self-loops
                    flow_data.append({"origin": origin, "destination": destination})

        # Convert to DataFrame and aggregate
        flow_df = pd.DataFrame(flow_data)

        if len(flow_df) > 0:
            flow_df = flow_df.groupby(["origin", "destination"], as_index=False).size()
            flow_df = flow_df.rename(columns={"size": "flow"})
        else:
            flow_df = pd.DataFrame(columns=["origin", "destination", "flow"])

        return flow_df

    def get_od_matrix_as_df(self):
        """
        Get the OD matrix as a flow dataframe.

        Returns
        -------
        pd.DataFrame
            Flow dataframe with columns: ['origin', 'destination', 'flow']
        """
        if self.od_matrix is None:
            raise ValueError(
                "Model must be fitted first. Call fit() before accessing OD matrix."
            )

        flow_data = []

        for origin_idx in range(len(self.od_matrix)):
            for dest_idx in range(len(self.od_matrix)):
                flow = self.od_matrix[origin_idx, dest_idx]

                if flow > 0 and origin_idx != dest_idx:
                    origin_id = self.index2tileid[origin_idx]
                    dest_id = self.index2tileid[dest_idx]

                    flow_data.append(
                        {"origin": origin_id, "destination": dest_id, "flow": flow}
                    )

        flow_df = pd.DataFrame(flow_data)
        return flow_df
