import json
import os

from sparkmobility.utils import spark_session


class StayDetection:
    def __init__(
        self,
        MobilityDataset,
    ):
        self.dataset = MobilityDataset

        os.makedirs(self.dataset.output_path, exist_ok=True)

        # Base config derived from the dataset. Per-call parameters (deltaT,
        # thresholds, etc.) are folded in inside each method and handed to the
        # Scala side as a JSON string — no round-trip through disk — but we
        # also drop a config.json breadcrumb after each run so methods like
        # get_home_work_od_matrix() can be called in a fresh process without
        # the in-memory state from get_stays().
        self._base_params = {
            "dataset_name": self.dataset.dataset_name,
            "startTimestamp": self.dataset.start_datetime,
            "endTimestamp": self.dataset.end_datetime,
            "longitude": self.dataset.longitude,
            "latitude": self.dataset.latitude,
            "timeZone": self.dataset.time_zone,
        }
        self._config_path = os.path.join(self.dataset.output_path, "config.json")

    @property
    def _last_params(self):
        if os.path.exists(self._config_path):
            with open(self._config_path, "r") as f:
                return json.load(f)
        return None

    def _get_entry_point(self, spark):
        return spark._jvm.pipelines.PyEntryPoint

    def _build_params(self, **overrides):
        params = dict(self._base_params)
        params.update(overrides)
        with open(self._config_path, "w") as f:
            json.dump(params, f, indent=4)
        return params

    @spark_session
    def get_stays(
        spark,
        self,
        delta_t=300,
        spatial_threshold=300,
        speed_threshold=6.0,
        temporal_threshold=300,
        hex_resolution=8,
        regional_temporal_threshold=3600,
        passing=True,
        home_to_work=8,
        work_to_home=19,
        work_distance_limit=500,
        work_freq_count_limit=3,
        find_home_and_work=True,
    ):
        params = self._build_params(
            deltaT=delta_t,
            spatialThreshold=spatial_threshold,
            speedThreshold=speed_threshold,
            temporalThreshold=temporal_threshold,
            hexResolution=hex_resolution,
            regionalTemporalThreshold=regional_temporal_threshold,
            passing=passing,
            homeToWork=home_to_work,
            workToHome=work_to_home,
            workDistanceLimit=work_distance_limit,
            workFreqCountLimit=work_freq_count_limit,
            findHomeAndWork=find_home_and_work,
        )
        config_json = json.dumps(params)
        column_names_json = json.dumps(self.dataset.column_names)

        entry = self._get_entry_point(spark)

        entry.getStays(
            self.dataset.input_path,
            self.dataset.output_path + "/StayPoints",
            self.dataset.time_format,
            "parquet",
            "",
            "true",
            column_names_json,
            config_json,
        )
        if find_home_and_work:
            entry.getHomeWorkLocation(
                self.dataset.output_path + "/StayPoints",
                self.dataset.output_path + "/StayPointsWithHomeWork",
                config_json,
            )
            return "Stay detection completed with home and work locations labeled"
        else:
            return "Stay detection completed"

    @spark_session
    def get_home_work_od_matrix(spark, self, hex_resolution) -> None:
        entry = self._get_entry_point(spark)
        if self._last_params is None:
            raise RuntimeError(
                "get_stays() must be run before get_home_work_od_matrix() so "
                "the stay-detection configuration is available."
            )
        if hex_resolution > self._last_params["hexResolution"]:
            raise ValueError(
                f"Resolution must be smaller than {self._last_params['hexResolution']}"
            )
        entry.getODMatrix(
            self.dataset.output_path + "/StayPointsWithHomeWork",
            self.dataset.output_path
            + f"""/HomeWorkODMatrix/Resolution{hex_resolution}""",
            hex_resolution,
        )
        return "OD matrix based on home and work locations"

    @spark_session
    def get_od_matrix(spark, self, input_path, output_path, resolution):
        entry = self._get_entry_point(spark)
        entry.getFullODMatrix(input_path, output_path, resolution)
        return "od matrix based on default parameters"

    @spark_session
    def summarize(spark, self):
        entry = self._get_entry_point(spark)
        # Prefer the home/work-labeled output when it exists; otherwise fall
        # back to raw stay points. This is observable from the filesystem, so
        # summarize() works in a fresh process without needing the in-memory
        # config from get_stays().
        home_work_path = self.dataset.output_path + "/StayPointsWithHomeWork"
        if os.path.isdir(home_work_path):
            input_path = home_work_path
        else:
            input_path = self.dataset.output_path + "/StayPoints"

        entry.getStayDurationDistribution(
            input_path,
            self.dataset.output_path + "/Metrics/StayDurationDistribution",
        )

        entry.getDailyVisitedLocation(
            input_path,
            self.dataset.output_path + "/Metrics/DailyVisitedLocations",
        )
        entry.getDepartureTimeDistribution(
            input_path,
            self.dataset.output_path + "/Metrics/DepartureTimeDistribution",
        )

        return None
