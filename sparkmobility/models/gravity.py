import logging

import h3
import numpy as np
import pandas as pd
import statsmodels as sm
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon
from statsmodels.genmod.generalized_linear_model import GLM
from tqdm import tqdm

from sparkmobility.logging_setup import configure_if_needed

logger = logging.getLogger(__name__)

tqdm.pandas()


class Gravity:
    """
    Gravity model.
    """

    # -------------- Private helpers (internal use) --------------

    def _h3_index_to_centroid(self, h3_index):
        """Return centroid Point(lon, lat) for a given H3 cell (v4)."""
        lat, lng = h3.cell_to_latlng(h3_index)  # v4
        return Point(lng, lat)

    def _get_polygon_centroid(self, polygon):
        """Return centroid of a Shapely Polygon."""
        if not isinstance(polygon, Polygon):
            raise ValueError("Input must be a Shapely Polygon object.")
        return polygon.centroid

    def _get_h3_index_at_resolution(self, h3_index, resolution):
        """
        Get parent cell at a coarser resolution; raising if asked to go finer.
        """
        current_resolution = h3.get_resolution(h3_index)
        if resolution > current_resolution:
            # Convert to a finer resolution
            raise ValueError(
                "Cannot directly get a higher resolution parent. Use `h3.cell_to_children()` for this."
            )
        elif resolution < current_resolution:
            # Convert to a coarser resolution
            return h3.cell_to_parent(h3_index, resolution)
        else:
            return h3_index

    def _create_normalization_constraint_vector(self, origin_index, total_locations):
        """Create a one-hot vector used as normalization constraint."""
        normalization_vector = list(np.zeros(total_locations))
        normalization_vector[origin_index] = 1.0
        return normalization_vector

    def _compute_distance_matrix(self, spatial_tessellation, origins):
        """
        Compute pairwise geodesic distance (in km) between cell centroids.
        """
        if not hasattr(spatial_tessellation, "geometry"):
            raise ValueError(
                "The provided spatial_tessellation must have a 'geometry' column."
            )

        # get centroid lat/lon from H3 index column "index"
        centroids = (
            spatial_tessellation["index"]
            .apply(lambda h3_index: self._h3_index_to_centroid(h3_index))
            .apply(lambda point: (point.y, point.x))  # (lat, lon)
            .values
        )

        n = len(spatial_tessellation)
        distance_matrix = np.zeros((n, n))

        for id_i in tqdm(range(n)):  # origins
            lat_i, lng_i = centroids[id_i]
            for id_j in range(id_i + 1, n):
                lat_j, lng_j = centroids[id_j]
                distance = geodesic(
                    (lat_i, lng_i), (lat_j, lng_j)
                ).kilometers  # .kilometers
                distance_matrix[id_i, id_j] = distance
                distance_matrix[id_j, id_i] = distance

        return distance_matrix

    def _h3_to_polygon(self, h3_index):
        """
        Convert H3 cell to shapely Polygon using v4 boundary API.
        """
        boundary = h3.cell_to_boundary(h3_index)  # returns [(lat, lon), ...]
        polygon_coords = [(lon, lat) for (lat, lon) in boundary]
        return Polygon(polygon_coords)

    def _get_gravity_score(self, distance_matrix, relevances_orig, relevances_dest):
        """
        Gravity score matrix using the selected deterrence function and exponents.
        """
        trip_probs_matrix = self.deterrence_function(distance_matrix)

        relevances_dest_adj = np.power(relevances_dest, self.destination_exp)
        relevances_orig_adj = np.power(relevances_orig, self.origin_exp)

        trip_probs_matrix *= relevances_dest_adj
        trip_probs_matrix = (trip_probs_matrix.T * relevances_orig_adj).T

        trip_probs_matrix = np.nan_to_num(
            trip_probs_matrix, nan=0.0, posinf=0.0, neginf=0.0
        )

        np.fill_diagonal(trip_probs_matrix, 0.0)

        return trip_probs_matrix

    def _from_matrix_to_flowdf(self, flow_matrix, origins, spatial_tessellation):
        """
        Converts an origin-destination matrix into a DataFrame with columns
        ['origin', 'destination', 'flow'] using tessellation's 'index' column.
        """
        index2tileid = {
            index: spatial_tessellation.iloc[index]["index"]
            for index in range(len(spatial_tessellation))
        }

        flow_data = []
        for origin_idx in origins:
            for dest_idx, flow in enumerate(flow_matrix[origin_idx]):
                if flow > 0 and origin_idx != dest_idx:
                    origin_id = index2tileid.get(origin_idx)
                    dest_id = index2tileid.get(dest_idx)
                    flow_data.append(
                        {"origin": origin_id, "destination": dest_id, "flow": flow}
                    )
        flow_df = pd.DataFrame(flow_data)
        return flow_df

    def _update_training_set(self, flow_example):
        """
        Update the training set for fitting the gravity model parameters.

        Parameters
        ----------
        flow_example : pd.Series
            A single flow instance containing origin, destination, and flow values.
        """
        origin_id = flow_example["origin"]
        destination_id = flow_example["destination"]
        flow_value = flow_example["flow"]

        # Exclude self-loops
        if origin_id == destination_id:
            return

        if self.is_h3_hexagon:
            distance = flow_example["distance"]

        try:
            origin_relevance = self.weights[self.tileid2index[origin_id]]
            destination_relevance = self.weights[self.tileid2index[destination_id]]
        except KeyError:
            logger.warning('missing info for location "%s"; skipping', origin_id)
            return

        if origin_relevance <= 0 or destination_relevance <= 0:
            return
        if distance <= 0:
            return

        if self.gravity_type == "global":
            sc_vars = [np.log(origin_relevance)]
        elif self.gravity_type == "single":
            sc_vars = self._create_normalization_constraint_vector(
                self.tileid2index[origin_id], len(self.tileid2index)
            )

        if self.deterrence_func_type == "exponential":
            # exponential deterrence function
            self.X += [[1.0] + sc_vars + [np.log(destination_relevance), -distance]]
        elif self.deterrence_func_type == "power_law":
            # power law deterrence function
            self.X += [
                [1.0] + sc_vars + [np.log(destination_relevance), np.log(distance)]
            ]

        self.y += [float(flow_value)]

    # -------------- Public API --------------

    def __init__(
        self,
        deterrence_func_type="power_law",
        deterrence_func_args=[-2.0],
        origin_exp=1.0,
        destination_exp=1.0,
        gravity_type="single",  # 'single' or 'global'
        name="Gravity model",
        is_h3_hexagon=True,
        configure_logging_if_needed: bool = True,
    ):
        if configure_logging_if_needed:
            configure_if_needed()
        """
        Initialize the Gravity model instance.

        Parameters:
        - deterrence_func_type (str): Type of deterrence function ('power_law' or 'exponential')
        - deterrence_func_args (list): Parameters for the deterrence function
        - origin_exp (float): Exponent for origin (only for globally constrained models)
        - destination_exp (float): Exponent for destination
        - gravity_type (str): Type of gravity model ('single' or 'global')
        - name (str): Name of the gravity model instance
        """
        self.deterrence_func_type = deterrence_func_type
        self.deterrence_func_args = deterrence_func_args
        self.origin_exp = origin_exp
        self.destination_exp = destination_exp
        self.gravity_type = gravity_type
        self.name = name
        self.is_h3_hexagon = is_h3_hexagon

        # Set the deterrence function based on the type
        if self.deterrence_func_type == "power_law":
            exponent = self.deterrence_func_args[0]
            self.deterrence_function = lambda d: d**exponent
        elif self.deterrence_func_type == "exponential":
            coefficient = self.deterrence_func_args[0]
            self.deterrence_function = lambda d: np.exp(-coefficient * d)
        else:
            raise ValueError(
                f"Invalid deterrence function type: {self.deterrence_func_type}"
            )

    def __str__(self):
        return (
            'Gravity(name="%s", deterrence_func_type="%s", '
            'deterrence_func_args=%s, origin_exp=%s, destination_exp=%s, gravity_type="%s")'
            % (
                self.name,
                self.deterrence_func_type,
                self.deterrence_func_args,
                self.origin_exp,
                self.destination_exp,
                self.gravity_type,
            )
        )

    def get_str(self, *args):
        values = [getattr(self, arg, None) for arg in args]
        return values

    def generate(
        self,
        spatial_tessellation,
        tile_id_column,  # =constants.TILE_ID,
        relevance_column,  # =constants.RELEVANCE,
        tot_outflows_column="total_outflow",  # =constants.TOT_OUTFLOW,
        out_format="flows",  # 'flows', 'flows_sample', 'probabilities'
    ):
        """
        Generate flow data based on the gravity model and the given spatial tessellation.
        """
        valid_formats = ["flows", "flows_sample", "probabilities"]
        if out_format not in valid_formats:
            raise ValueError(
                f"Invalid out_format '{out_format}'. Valid options are {valid_formats}."
            )

        self._tile_id_column = tile_id_column

        relevances = spatial_tessellation[relevance_column].fillna(0).values
        if out_format in ["flows", "flows_sample"]:
            tot_outflows = spatial_tessellation[tot_outflows_column].fillna(0).values

        origins = np.arange(len(spatial_tessellation))

        # distance matrix
        distance_matrix = self._compute_distance_matrix(spatial_tessellation, origins)

        # score matrix
        trip_probs_matrix = self._get_gravity_score(
            distance_matrix, relevances, relevances
        )

        if self.gravity_type == "global":
            trip_probs_matrix /= np.sum(trip_probs_matrix)
            trip_probs_matrix = np.nan_to_num(
                trip_probs_matrix, nan=0.0, posinf=0.0, neginf=0.0
            )

            if out_format == "probabilities":
                return self._from_matrix_to_flowdf(
                    trip_probs_matrix, origins, spatial_tessellation
                )

            elif out_format == "flows":
                od_matrix = (trip_probs_matrix.T * tot_outflows).T
                return self._from_matrix_to_flowdf(
                    od_matrix, origins, spatial_tessellation
                )

            elif out_format == "flows_sample":
                total_flow = int(tot_outflows.sum())
                flow_probs_flat = trip_probs_matrix.flatten()
                flows_sampled = np.random.multinomial(total_flow, flow_probs_flat)
                flows_sampled_matrix = flows_sampled.reshape(trip_probs_matrix.shape)
                return self._from_matrix_to_flowdf(
                    flows_sampled_matrix, origins, spatial_tessellation
                )

        elif self.gravity_type == "single":
            # Normalize trip_probs_matrix for each origin
            trip_probs_matrix = np.transpose(
                trip_probs_matrix / np.sum(trip_probs_matrix, axis=0)
            )
            trip_probs_matrix = np.nan_to_num(
                trip_probs_matrix, nan=0.0, posinf=0.0, neginf=0.0
            )

            if out_format == "probabilities":
                return self._from_matrix_to_flowdf(
                    trip_probs_matrix, origins, spatial_tessellation
                )

            elif out_format == "flows":
                avg_flows_matrix = trip_probs_matrix * tot_outflows[:, np.newaxis]
                logger.debug("avg_flows_matrix:\n%s", avg_flows_matrix)
                return self._from_matrix_to_flowdf(
                    avg_flows_matrix, origins, spatial_tessellation
                )

            elif out_format == "flows_sample":
                flows_sampled_matrix = np.zeros(trip_probs_matrix.shape)
                for i, (probs, N) in enumerate(zip(trip_probs_matrix, tot_outflows)):
                    flows_sampled_matrix[i, :] = np.random.multinomial(int(N), probs)
                return self._from_matrix_to_flowdf(
                    flows_sampled_matrix, origins, spatial_tessellation
                )

        else:
            raise ValueError(
                f"Invalid model_type '{self.gravity_type}'. Expected 'global' or 'single'."
            )

    def fit(
        self,
        flow_df,
        relevance_df,  # relevance_df, with geometry info and relevance column
        relevance_column,
    ):
        """
        Fit the gravity model using observed flow data.
        """
        if self.is_h3_hexagon:
            if h3.get_resolution(flow_df["origin"][0]) != h3.get_resolution(
                relevance_df["index"][0]
            ):
                raise ValueError(
                    f"The H3 resolution of flow_df and relevance_df do not match. The resolution of the flow df is {h3.get_resolution(flow_df['origin'][0])}, "
                    f"while the resolution of the relevance df is {h3.get_resolution(relevance_df['index'][0])}."
                )

            unique_h3_index = relevance_df["index"].unique()
            h3center_mapping = {
                h3_index: self._h3_index_to_centroid(h3_index)
                for h3_index in unique_h3_index
            }
            relevance_df["center"] = relevance_df["index"].map(h3center_mapping)

        self.weights = relevance_df[relevance_column].fillna(0).values
        self.tileid2index = dict(
            [(tileid, i) for i, tileid in enumerate(relevance_df["index"].values)]
        )

        self.X, self.y = [], []
        flow_df.progress_apply(
            lambda flow_example: self._update_training_set(flow_example), axis=1
        )

        logger.info("fitting GLM model")
        model = GLM(
            self.y,
            self.X,
            family=sm.genmod.families.family.Poisson(
                link=sm.genmod.families.links.Log()
            ),
        )
        result = model.fit(disp=True)

        if self.gravity_type == "global":
            self.origin_exp = result.params[1]
            self.destination_exp = result.params[2]
            self.deterrence_func_args = [result.params[3]]
        elif self.gravity_type == "single":
            self.origin_exp = 1.0
            self.destination_exp = result.params[-2]
            self.deterrence_func_args = [result.params[-1]]

        del self.X
        del self.y

        self.deterrence_func_args = self.get_str("deterrence_func_args")[0]
        self.origin_exp = self.get_str("origin_exp")[0]
        self.destination_exp = self.get_str("destination_exp")[0]

        return result
